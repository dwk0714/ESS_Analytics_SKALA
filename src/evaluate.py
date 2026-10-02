"""Evaluate a saved, frozen model. This module never calls fit."""
from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES, check_feature_table, data_fingerprint
from .splits import make_splits
from .preprocess import file_sha256


def metrics(actual, predicted) -> dict[str, float]:
    actual, predicted = np.asarray(actual, float).reshape(-1), np.asarray(predicted, float).reshape(-1)
    if len(actual) == 0 or actual.shape != predicted.shape:
        raise ValueError("실제/예측 수명의 개수가 맞지 않거나 비어 있습니다.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (actual <= 0).any() or (predicted <= 0).any():
        raise ValueError("실제/예측 수명은 유한한 양수여야 합니다.")
    error = predicted - actual
    return {"MAPE_percent": float(np.mean(np.abs(error) / actual) * 100),
            "MAE_cycles": float(np.mean(np.abs(error))), "RMSE_cycles": float(np.sqrt(np.mean(error ** 2)))}


def summarize_cv(cv: pd.DataFrame, train_keys: set, repeats: int = 1):
    """Average per-repeat OOF metrics; repetitions are not new independent cells."""
    cv = cv.copy()
    if "repeat" not in cv:
        cv["repeat"] = 1
    if set(cv["repeat"]) != set(range(1, repeats + 1)):
        raise ValueError("CV 반복 번호가 설정과 다릅니다.")
    scores, baselines = [], []
    for _, part in cv.groupby("repeat", sort=True):
        if not part.cell_key.is_unique or set(part.cell_key) != train_keys:
            raise ValueError("각 CV 반복은 학습 셀마다 한 번의 OOF 예측이 필요합니다.")
        scores.append(metrics(part.actual_cycle_life, part.predicted_cycle_life))
        baselines.append(metrics(part.actual_cycle_life, part.baseline_cycle_life))
    def average(rows):
        return {**{key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
                "n": len(train_keys), "n_repeats": repeats}
    return average(scores), average(baselines)


def performance_table(scores: dict[str, dict], paper_target: float) -> pd.DataFrame:
    rows = []
    def measured(name, scope, note):
        rows.append({"구분": name, "MAPE (%)": scores[scope]["MAPE_percent"], "비고": note,
                     "row_type": "metric", "n": scores[scope]["n"]})
    def gap(name, value, note):
        rows.append({"구분": name, "MAPE (%)": value, "비고": note, "row_type": "gap", "n": np.nan})
    measured("Train (Batch 1 CV)", "train_cv", "B1 그룹 OOF CV; 낮을수록 예측이 더 정확함")
    measured("Valid (Batch 1 Hold-out)", "valid", "B1 holdout; 동일 저장 모델")
    measured("Test (Batch 2)", "test_b2", "필수 최종 배치 평가")
    gap("Gap (Train-Valid)", scores["valid"]["MAPE_percent"] - scores["train_cv"]["MAPE_percent"],
        "Valid−Train (%p); (+) 과적합/집단 차이 검토")
    gap("Gap (Valid-Test)", scores["test_b2"]["MAPE_percent"] - scores["valid"]["MAPE_percent"],
        "Test−Valid (%p); (+) 배치 간 일반화 저하 검토")
    gap("Gap (Target-Test)", scores["test_b2"]["MAPE_percent"] - paper_target,
        f"Test−{paper_target:g} (%p); 논문 Target과 다른 실험 구성")
    if "test_b3" in scores:
        measured("Test (Batch 3)", "test_b3", "동일 모델 추가 평가")
        gap("Gap (Batch2-Batch3)", scores["test_b3"]["MAPE_percent"] - scores["test_b2"]["MAPE_percent"],
            "B3−B2 (%p); (+) B3에서 오차 증가")
        gap("Gap (Target-Test) [Batch 3]", scores["test_b3"]["MAPE_percent"] - paper_target,
            f"B3−{paper_target:g} (%p); 추가 참고 비교")
    return pd.DataFrame(rows)


def evaluate_model(frame: pd.DataFrame, model_dir: Path, output_dir: Path, include_b3: bool = False) -> dict:
    start = time.perf_counter()
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    if (output_dir / "model_performance.csv").exists():
        raise FileExistsError("이 결과 폴더는 이미 평가했습니다. 결과를 덮어쓰지 않습니다.")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    check_feature_table(frame, config)
    if data_fingerprint(frame) != metadata["data_fingerprint"]:
        raise ValueError("학습 당시와 입력 특징/정답 표가 달라졌습니다. 저장 모델과 같은 자료를 사용하세요.")
    manifest = pd.read_csv(output_dir / "split_manifest.csv")
    if not manifest.cell_key.is_unique or set(manifest.cell_key) - set(frame.cell_key):
        raise ValueError("평가용 셀 ID 목록이 특징 표와 맞지 않습니다.")
    saved_train = set(manifest.loc[manifest.role.eq("train"), "cell_key"])
    if saved_train != set(metadata["train_cell_keys"]):
        raise ValueError("학습 셀 목록이 저장 모델과 다릅니다.")
    expected = make_splits(frame, config).sort_values("cell_key").reset_index(drop=True)
    if not manifest.sort_values("cell_key").reset_index(drop=True).equals(expected):
        raise ValueError("평가 역할·정책·셀 ID 목록이 고정 분할과 다릅니다.")
    if include_b3 and not manifest.role.eq("test_b3").any():
        raise ValueError("B3 평가를 요청했지만 특징 표에 B3가 없습니다.")
    model_path = model_dir / metadata.get("model_filename", "ridge.joblib")
    if file_sha256(model_path) != metadata["model_file_sha256"]:
        raise ValueError("저장 모델 파일이 학습 완료 이후 바뀌었습니다.")
    model = joblib.load(model_path)
    if metadata.get("cv_predictions_sha256") and file_sha256(output_dir / "cv_predictions.csv") != metadata["cv_predictions_sha256"]:
        raise ValueError("학습 완료 이후 CV 예측 파일이 바뀌었습니다.")
    cv = pd.read_csv(output_dir / "cv_predictions.csv", float_precision="round_trip")
    cv_score, cv_baseline = summarize_cv(cv, saved_train, metadata.get("cv_repeats", 1))
    scores = {"train_cv": cv_score}
    baseline = {"train_cv": cv_baseline}
    prediction_frames = []
    roles = ["valid", "test_b2"] + (["test_b3"] if include_b3 else [])
    indexed = frame.set_index("cell_key", drop=False)
    selected = metadata["selected_features"]
    train = indexed.loc[metadata["train_cell_keys"]]
    lower, upper = train[selected].min(), train[selected].max()
    for role in roles:
        keys = manifest.loc[manifest.role.eq(role), "cell_key"]
        part = indexed.loc[keys].reset_index(drop=True)
        predicted = model.predict(part)
        scores[role] = {**metrics(part.cycle_life, predicted), "n": len(part)}
        base = np.full(len(part), metadata["baseline_train_median"])
        baseline[role] = {**metrics(part.cycle_life, base), "n": len(part)}
        result = part.copy()
        result["role"], result["actual_cycle_life"], result["predicted_cycle_life"] = role, part.cycle_life, predicted
        result["error_cycles"] = result.predicted_cycle_life - result.actual_cycle_life
        result["signed_error_percent"] = result.error_cycles / result.actual_cycle_life * 100
        result["APE_percent"] = result.signed_error_percent.abs()
        result["baseline_cycle_life"] = base
        result["out_of_training_range_features"] = [
            ";".join(feature for feature in selected if row[feature] < lower[feature] or row[feature] > upper[feature])
            for _, row in part.iterrows()]
        result["target_rule_note"] = ["first Qd<0.88Ah event matches stored label" if b == "B2"
                                      else "stored label matches last observed cycle+1" for b in part.batch]
        prediction_frames.append(result)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    table = performance_table(scores, config["paper_mape_percent"])
    table.to_csv(output_dir / "model_performance.csv", index=False)
    metric_rows = [{"scope": scope, "model": name, **values}
                   for name, values_by_scope in ((metadata["model"], scores), ("Training median", baseline))
                   for scope, values in values_by_scope.items()]
    pd.DataFrame(metric_rows).to_csv(output_dir / "model_metrics.csv", index=False)
    summary = {"scores": scores, "baseline_scores": baseline, "include_b3": include_b3,
               "paper_mape_percent": config["paper_mape_percent"], "inference_and_metrics_seconds": time.perf_counter() - start,
               "same_saved_model": True, "retuned_using_test": False,
               "gap_positive_means": "later error is larger", "mape_direction": "lower is more accurate"}
    (output_dir / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary, "performance": table, "predictions": predictions, "metadata": metadata}
