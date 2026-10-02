"""Select a Ridge model with only B1 development-train group CV.

Calling train_model performs 70 candidate fits and one selected-model fit.
Importing this module does not train or evaluate anything.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import time

try:
    import resource
except ImportError:  # Windows: record that this measurement is unavailable.
    resource = None

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import metrics
from .features import FEATURES, data_fingerprint
from .splits import development_data, group_folds, make_splits
from .preprocess import file_sha256


def inverse_log10(values):
    return np.power(10.0, values)


def make_model(features: list[str], alpha: float, solver: str = "svd") -> TransformedTargetRegressor:
    select_and_scale = ColumnTransformer([("selected", StandardScaler(), features)], remainder="drop")
    regression = Pipeline([("preprocess", select_and_scale), ("ridge", Ridge(alpha=alpha, solver=solver))])
    return TransformedTargetRegressor(regressor=regression, func=np.log10, inverse_func=inverse_log10)


def runtime_environment() -> dict:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "h5py", "matplotlib", "joblib")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in packages}}


def train_model(frame: pd.DataFrame, config: dict, output_dir: Path, model_dir: Path) -> dict:
    start = time.perf_counter()
    output_dir, model_dir = Path(output_dir), Path(model_dir)
    # Evaluation artifacts may otherwise look current after an unrelated retrain.
    if (output_dir / "model_performance.csv").exists() or (model_dir / "ridge.joblib").exists():
        raise FileExistsError("이 결과 폴더에 이미 모델/평가 결과가 있습니다. 별도 --run-dir로 실행하세요.")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_splits(frame, config)
    development = development_data(frame, manifest)
    folds = group_folds(development, config["cv_folds"])
    x, y = development[FEATURES], development.cycle_life.to_numpy(float)
    search_rows, fold_rows = [], []
    best_key, best_record, best_predictions = None, None, None
    fit_count = 0
    cv_start = time.perf_counter()
    for name, selected_features in config["feature_sets"].items():
        for alpha in config["alpha_grid"]:
            predictions, baseline_predictions = np.full(len(y), np.nan), np.full(len(y), np.nan)
            fold_for_row = np.full(len(y), -1, dtype=int)
            for fold, (training, valid) in enumerate(folds, 1):
                model = make_model(selected_features, alpha, config["solver"])
                model.fit(x.iloc[training], y[training])
                fit_count += 1
                pred = np.asarray(model.predict(x.iloc[valid]), float)
                score = metrics(y[valid], pred)
                base = np.full(len(valid), np.median(y[training]))
                predictions[valid], baseline_predictions[valid], fold_for_row[valid] = pred, base, fold
                fold_rows.append({"feature_set": name, "feature_count": len(selected_features), "alpha": alpha,
                                  "fold": fold, "n_train": len(training), "n_valid": len(valid), **score,
                                  "baseline_MAPE_percent": metrics(y[valid], base)["MAPE_percent"]})
            # Every row has one OOF prediction: these are n-weighted fold aggregates.
            score = metrics(y, predictions)
            row = {"feature_set": name, "feature_count": len(selected_features), "alpha": alpha,
                   "features": ";".join(selected_features), "n_cv": len(y), **score}
            search_rows.append(row)
            ordering = (score["MAPE_percent"], len(selected_features), -float(alpha))
            if best_key is None or ordering < best_key:
                best_key, best_record = ordering, row
                best_predictions = pd.DataFrame({"cell_key": development.cell_key, "fold": fold_for_row,
                                                 "actual_cycle_life": y, "predicted_cycle_life": predictions,
                                                 "baseline_cycle_life": baseline_predictions})
    cv_seconds = time.perf_counter() - cv_start
    if best_record is None or best_predictions is None:
        raise ValueError("선택할 후보 모델이 없습니다.")
    selected_features = config["feature_sets"][best_record["feature_set"]]
    refit_start = time.perf_counter()
    final_model = make_model(selected_features, best_record["alpha"], config["solver"])
    final_model.fit(x, y)
    fit_count += 1
    refit_seconds = time.perf_counter() - refit_start
    expected_fits = len(config["feature_sets"]) * len(config["alpha_grid"]) * config["cv_folds"] + 1
    if fit_count != expected_fits:
        raise RuntimeError("실제 학습 횟수가 탐색 예산과 다릅니다.")
    pd.DataFrame(search_rows).sort_values(["MAPE_percent", "feature_count", "alpha"],
                                         ascending=[True, True, False]).to_csv(output_dir / "cv_search_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "cv_fold_results.csv", index=False)
    best_predictions.to_csv(output_dir / "cv_predictions.csv", index=False)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    pd.DataFrame([{"fold": fold, "cell_key": development.iloc[index].cell_key,
                   "role": role, "policy": development.iloc[index].policy}
                  for fold, (tr, va) in enumerate(folds, 1)
                  for role, indices in (("train", tr), ("valid", va)) for index in indices]).to_csv(
                      output_dir / "cv_split_manifest.csv", index=False)
    joblib.dump(final_model, model_dir / "ridge.joblib")
    scaler = final_model.regressor_.named_steps["preprocess"].named_transformers_["selected"]
    ridge = final_model.regressor_.named_steps["ridge"]
    metadata = {"status": "trained_not_evaluated", "model": "Ridge", "selected_features": selected_features,
                "alpha": best_record["alpha"], "forecast_cycle": 100, "target": "provided cycle_life",
                "target_transform": "log10", "inverse_transform": "10**prediction",
                "train_cell_keys": development.cell_key.tolist(), "training_rows": len(y),
                "training_policy_groups": development.policy.nunique(),
                "cv_metrics": metrics(y, best_predictions.predicted_cycle_life),
                "cv_baseline_metrics": metrics(y, best_predictions.baseline_cycle_life),
                "baseline_train_median": float(np.median(y)), "fit_count": fit_count,
                "data_fingerprint": data_fingerprint(frame), "config": config,
                "model_file_sha256": file_sha256(model_dir / "ridge.joblib"),
                "environment": runtime_environment(), "same_model_for_valid_b2_b3": True,
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "standardized_coefficients_log10_target": ridge.coef_.reshape(-1).tolist(),
                "intercept_log10_target": float(np.asarray(ridge.intercept_).item()),
                "limitations": ["B2/B3 were used in DAY1 EDA", "physical cross-file cell identity unresolved",
                                "stored target ending rules differ", "tuning CV is selection-biased"]}
    (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    usage = resource.getrusage(resource.RUSAGE_SELF) if resource is not None else None
    memory_mib = usage.ru_maxrss / (1024 ** 2 if platform.system() == "Darwin" else 1024) if usage is not None else None
    runtime = {"cv_seconds": cv_seconds, "refit_seconds": refit_seconds,
               "training_total_seconds": time.perf_counter() - start, "peak_process_memory_mib": memory_mib,
               "fit_count": fit_count, "device": "CPU", "environment": metadata["environment"]}
    (output_dir / "training_runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "model_card.md").write_text(
        "# Ridge 모델 카드\n\n"
        f"- 상태: 학습 모델 저장; 평가는 별도 단계이며 결과는 model_performance.csv에 저장\n- 입력: {', '.join(selected_features)}\n"
        f"- alpha: {best_record['alpha']}\n- 학습: B1 {len(y)}개, 정책 {development.policy.nunique()}그룹\n"
        f"- 선택 CV MAPE: {best_record['MAPE_percent']:.3f}% (선택에 사용한 개발 점수)\n"
        "- 초기100사이클 후 제공 수명 cycle_life를 예측합니다. ESS 운영연수/RUL과 다릅니다.\n"
        "- 표준화·열 선택·목표 역변환이 저장 모델에 포함됩니다.\n"
        "- 모델 선택 이후 동일 모델로 Valid/B2/B3를 평가합니다. 전체 B1 재학습은 하지 않습니다.\n"
        "- 배치/라벨 규칙 차이와 파일 간 물리적 셀 동일성 미확인, 전체 EDA 노출의 한계가 있습니다.\n",
        encoding="utf-8")
    return metadata
