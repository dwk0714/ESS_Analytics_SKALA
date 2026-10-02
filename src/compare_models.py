"""Compare OLS, the existing Ridge, and XGBoost without changing saved results.

Run with: python -m src.compare_models --output results/runs/model_comparison_20261002
All models learn log10 life. Model/parameter selection uses B1 group CV only.
The incomplete-B1 sensitivity analysis does not replace the original target table.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import metrics
from .features import FEATURES, data_fingerprint
from .pipeline import ROOT, choose_raw_dir, load_config, read_feature_csv
from .preprocess import file_sha256, reference_at
from .splits import development_data, group_folds, make_splits
from .train import inverse_log10, runtime_environment


def make_candidate(kind: str, features: list[str], parameters: dict):
    if kind == "ols":
        estimator = LinearRegression()
    elif kind == "ridge":
        estimator = Ridge(alpha=parameters.get("alpha", 1.0), solver="svd")
    elif kind == "xgb":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(
            objective="reg:squarederror", tree_method="hist", device="cpu",
            n_jobs=1, random_state=42, subsample=1.0, colsample_bytree=1.0,
            verbosity=0, **parameters,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    selection = ColumnTransformer(
        [("selected", StandardScaler(), features)], remainder="drop",
    )
    pipeline = Pipeline([("preprocess", selection), ("regressor", estimator)])
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log10, inverse_func=inverse_log10,
    )


def cross_validate(kind, features, parameters, x, y, folds):
    predictions = np.full(len(y), np.nan)
    coverage = np.zeros(len(y), dtype=int)
    fold_metrics = []
    started = time.perf_counter()
    for fold, (training, validation) in enumerate(folds, 1):
        model = make_candidate(kind, features, parameters)
        model.fit(x.iloc[training], y[training])
        predictions[validation] = model.predict(x.iloc[validation])
        coverage[validation] += 1
        fold_metrics.append({"fold": fold, "n_train": len(training),
                             "n_valid": len(validation),
                             **metrics(y[validation], predictions[validation])})
    if not np.all(coverage == 1):
        raise ValueError("OOF validation must cover every training cell exactly once")
    return {"metrics": metrics(y, predictions), "predictions": predictions,
            "fold_metrics": fold_metrics, "seconds": time.perf_counter() - started}


def audit_targets(frame, raw_dir, config):
    # Source-author loading code identifies ten unfinished B1 records:
    # c0..4 have a continuation in the unavailable 2017-06-30 experiment;
    # c8,10,12,13,22 were removed for not reaching the capacity threshold.
    # These indices are not applied to the different 2018-02-20 B2 file.
    unfinished = {f"B1c{i}" for i in (0, 1, 2, 3, 4, 8, 10, 12, 13, 22)}
    rows = []
    for batch, filename in config["batches"].items():
        with h5py.File(Path(raw_dir) / filename, "r") as handle:
            raw = handle["batch"]
            for index in range(raw["summary"].size):
                summary = handle[reference_at(raw["summary"], index)]
                capacity = np.asarray(summary["QDischarge"][()]).reshape(-1)
                cycle = np.asarray(summary["cycle"][()]).reshape(-1)
                positive = np.isfinite(capacity) & (capacity > 0)
                crossing = np.flatnonzero(positive & (capacity < 0.88))
                key = f"{batch}c{index}"
                life = float(frame.set_index("cell_key").loc[key, "cycle_life"])
                excluded = key in unfinished
                if excluded and not capacity[-1] > 0.90:
                    raise ValueError(f"{key}: unfinished-record audit no longer matches the source")
                rows.append({"cell_key": key, "batch": batch,
                             "provided_cycle_life": life,
                             "last_observed_cycle": float(cycle[-1]),
                             "last_qd_ah": float(capacity[-1]),
                             "first_observed_qd_below_088_cycle":
                                 float(cycle[crossing[0]]) if len(crossing) else np.nan,
                             "unfinished_b1_sensitivity_exclusion": excluded})
    return pd.DataFrame(rows), unfinished


def run_comparison(frame, manifest, output, scenario, parameter_candidates):
    output.mkdir(parents=True, exist_ok=False)
    manifest.to_csv(output / "split_manifest.csv", index=False)
    development = development_data(frame, manifest)
    x, y = development[FEATURES], development.cycle_life.to_numpy(float)
    folds = group_folds(development, 5)
    indexed = frame.set_index("cell_key", drop=False)
    frozen = []
    search_rows, fold_rows, cv_predictions, summary_rows = [], [], [], []
    candidates = [("OLS_delta_only", "ols", FEATURES[:1], {}),
                  ("OLS_two_features", "ols", FEATURES[:2], {}),
                  ("Ridge_existing", "ridge", FEATURES[:2], {"alpha": 1.0})]
    xgb_candidates = [("XGBoost", "xgb", selected, parameters)
                      for selected in (FEATURES[:1], FEATURES[:2], FEATURES)
                      for parameters in parameter_candidates]
    best_xgb = None
    for trial, (name, kind, selected, parameters) in enumerate(candidates + xgb_candidates):
        cv = cross_validate(kind, selected, parameters, x, y, folds)
        record = {"scenario": scenario, "model": name, "trial": trial,
                  "features": ";".join(selected), "parameters": json.dumps(parameters, sort_keys=True),
                  "CV_seconds": cv["seconds"], **cv["metrics"]}
        search_rows.append(record)
        fold_rows.extend({"model": name, "trial": trial, **row} for row in cv["fold_metrics"])
        if kind == "xgb":
            key = (cv["metrics"]["MAPE_percent"], len(selected), trial)
            if best_xgb is None or key < best_xgb[0]:
                best_xgb = (key, (name, kind, selected, parameters, cv))
        else:
            frozen.append((name, kind, selected, parameters, cv))
        if trial % 30 == 0:
            print(f"{scenario}: B1 CV candidate {trial + 1}/{len(candidates + xgb_candidates)}", flush=True)
    frozen.append(best_xgb[1])
    pd.DataFrame(search_rows).to_csv(output / "cv_search_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "cv_fold_results.csv", index=False)
    selected_rows = []
    # Freeze every model and its parameters before reading external test targets.
    trained = []
    for name, kind, selected, parameters, cv in frozen:
        began = time.perf_counter()
        final_model = make_candidate(kind, selected, parameters)
        final_model.fit(x, y)
        seconds = time.perf_counter() - began
        path = output / f"{name}.joblib"
        joblib.dump(final_model, path)
        trained.append((name, selected, parameters, cv, final_model, seconds))
        selected_rows.append({"model": name, "features": selected, "parameters": parameters,
                              "train_cell_keys": development.cell_key.tolist(),
                              "CV_MAPE_percent": cv["metrics"]["MAPE_percent"],
                              "refit_seconds": seconds, "model_sha256": file_sha256(path)})
        cv_predictions.extend({"model": name, "cell_key": key, "actual_cycle_life": actual,
                               "predicted_cycle_life": predicted}
                              for key, actual, predicted in zip(development.cell_key, y, cv["predictions"]))
    chosen_name = min(selected_rows, key=lambda r: r["CV_MAPE_percent"])["model"]
    frozen_metadata = {"scenario": scenario, "selection_scope": "B1 development group 5-fold CV",
                       "B1_CV_selected_model": chosen_name, "target_transform": "log10",
                       "same_model_for_holdout_B2_B3": True, "test_used_for_selection": False,
                       "models": selected_rows}
    (output / "selected_models.json").write_text(json.dumps(frozen_metadata, indent=2), encoding="utf-8")
    prediction_rows = []
    for name, selected, parameters, cv, final_model, refit_seconds in trained:
        summary = {"scenario": scenario, "model": name, "features": ";".join(selected),
                   "n_train": len(y), "CV_MAPE_percent": cv["metrics"]["MAPE_percent"],
                   "CV_fold_MAPE_sd": float(np.std([r["MAPE_percent"] for r in cv["fold_metrics"]], ddof=1)),
                   "CV_seconds": cv["seconds"], "refit_seconds": refit_seconds,
                   "parameters": json.dumps(parameters, sort_keys=True)}
        for role in ("valid", "test_b2", "test_b3"):
            keys = manifest.loc[manifest.role.eq(role), "cell_key"]
            part = indexed.loc[keys]
            prediction = final_model.predict(part[FEATURES])
            measured = metrics(part.cycle_life, prediction)
            summary[f"{role}_n"] = len(part)
            summary.update({f"{role}_{metric}": value for metric, value in measured.items()})
            prediction_rows.extend({"scenario": scenario, "model": name, "role": role,
                                    "cell_key": key, "actual_cycle_life": actual,
                                    "predicted_cycle_life": predicted,
                                    "APE_percent": abs(predicted - actual) / actual * 100}
                                   for key, actual, predicted in zip(part.cell_key, part.cycle_life, prediction))
        summary_rows.append(summary)
    result = pd.DataFrame(summary_rows)
    result.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(cv_predictions).to_csv(output / "cv_predictions.csv", index=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Comparison output already exists; use a new output directory")
    config = load_config()
    frame = read_feature_csv(ROOT / "data/processed/cell_features.csv", config)
    manifest = make_splits(frame, config)
    saved_manifest = pd.read_csv(ROOT / "results/split_manifest.csv")
    pd.testing.assert_frame_equal(manifest, saved_manifest)
    protected = [ROOT / path for path in ("models/ridge.joblib", "models/metadata.json",
                 "config/model.json", "data/processed/cell_features.csv",
                 "results/model_performance.csv", "results/predictions.csv")]
    before = {str(path.relative_to(ROOT)): file_sha256(path) for path in protected}
    args.output.mkdir(parents=True)
    audit, unfinished = audit_targets(frame, choose_raw_dir(ROOT, config, args.raw_dir), config)
    audit.to_csv(args.output / "target_audit.csv", index=False)
    grid = {"max_depth": [1, 2, 3], "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [50, 100, 200, 300], "min_child_weight": [3, 5],
            "reg_lambda": [1.0, 10.0, 30.0]}
    parameters = list(ParameterSampler(grid, n_iter=40, random_state=42))
    began = time.perf_counter()
    original = run_comparison(frame, manifest, args.output / "provided_labels",
                              "provided_labels", parameters)
    sensitivity_manifest = manifest.loc[~manifest.cell_key.isin(unfinished)].reset_index(drop=True)
    sensitivity = run_comparison(frame, sensitivity_manifest,
                                args.output / "unfinished_b1_excluded_sensitivity",
                                "unfinished_b1_excluded_sensitivity", parameters)
    combined = pd.concat([original, sensitivity], ignore_index=True)
    combined.to_csv(args.output / "model_comparison.csv", index=False)
    after = {str(path.relative_to(ROOT)): file_sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("An existing experiment artifact changed")
    environment = runtime_environment()
    environment["packages"]["xgboost"] = importlib.metadata.version("xgboost")
    metadata = {"data_fingerprint": data_fingerprint(frame), "elapsed_seconds": time.perf_counter() - began,
                "environment": environment, "preserved_original_artifact_sha256": after,
                "XGBoost_parameter_candidates_per_feature_set": 40,
                "XGBoost_feature_sets": [FEATURES[:1], FEATURES[:2], FEATURES],
                "XGBoost_search_grid": grid, "seed": 42,
                "model_selection": "B1 CV only; holdout/B2/B3 were evaluated after freezing settings",
                "target_policy": "Provided labels are unchanged in both experiments",
                "sensitivity_excluded_B1_cells": sorted(unfinished),
                "limitations": ["CV used for selection is optimistic; this is not nested CV",
                                "B2/B3 were already examined in EDA and earlier evaluations",
                                "B1/B3 near-threshold endpoints do not establish an exact observed crossing",
                                "Sensitivity exclusion changes B1 sample size and holdout composition",
                                "No missing target is filled and no continuation label is invented"]}
    (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shown = ["scenario", "model", "CV_MAPE_percent", "valid_MAPE_percent",
             "test_b2_MAPE_percent", "test_b3_MAPE_percent"]
    print(combined[shown].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
