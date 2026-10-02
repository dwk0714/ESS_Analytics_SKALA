"""Compare equal averaging and protocol-group OOF stacking with saved candidates.

python -m src.compare_ensembles --comparison results/runs/model_comparison_20261002
External scores never choose stacking weights, regularization, or base models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .compare_models import make_candidate
from .evaluate import metrics
from .features import FEATURES
from .pipeline import ROOT, load_config, read_feature_csv
from .preprocess import file_sha256
from .splits import development_data, group_folds
from .train import inverse_log10


def make_ensemble(kind, xgb_features, xgb_parameters, inner_folds=None, meta_alpha=1.0):
    candidates = [("OLS_delta", make_candidate("ols", FEATURES[:1], {})),
                  ("Ridge", make_candidate("ridge", FEATURES[:2], {"alpha": 1.0})),
                  ("XGBoost", make_candidate("xgb", xgb_features, xgb_parameters))]
    if kind == "voting":
        # Average the three predictions in the original cycle-life units.
        return VotingRegressor(estimators=candidates, n_jobs=1)
    if kind != "stacking" or inner_folds is None:
        raise ValueError("Stacking requires explicit group-separated OOF folds")
    # Fit every base estimator and the meta estimator on log10 life.
    # Do not nest target transformations inside the stacking base estimators.
    stack = StackingRegressor(
        estimators=[(name, candidate.regressor) for name, candidate in candidates],
        final_estimator=make_pipeline(StandardScaler(), Ridge(alpha=meta_alpha, solver="svd")),
        cv=inner_folds, n_jobs=1, passthrough=False,
    )
    return TransformedTargetRegressor(regressor=stack, func=np.log10, inverse_func=inverse_log10)


def ensemble_cv(kind, xgb_features, xgb_parameters, development, outer_folds, meta_alpha):
    x, y = development[FEATURES], development.cycle_life.to_numpy(float)
    prediction = np.full(len(y), np.nan)
    coverage = np.zeros(len(y), dtype=int)
    fold_scores = []
    started = time.perf_counter()
    for fold, (training, validation) in enumerate(outer_folds, 1):
        training_part = development.iloc[training].reset_index(drop=True)
        # The outer validation cells never train the stacking meta estimator.
        inner = group_folds(training_part, 3) if kind == "stacking" else None
        model = make_ensemble(kind, xgb_features, xgb_parameters, inner, meta_alpha)
        model.fit(x.iloc[training], y[training])
        prediction[validation] = model.predict(x.iloc[validation])
        coverage[validation] += 1
        fold_scores.append({"fold": fold, "n_train": len(training), "n_valid": len(validation),
                            **metrics(y[validation], prediction[validation])})
    if not np.all(coverage == 1):
        raise ValueError("OOF validation must cover every training cell exactly once")
    return {"metrics": metrics(y, prediction), "predictions": prediction,
            "fold_metrics": fold_scores, "seconds": time.perf_counter() - started}


def run_scenario(frame, source, output, scenario):
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((source / "selected_models.json").read_text())
    xgb = next(record for record in metadata["models"] if record["model"] == "XGBoost")
    manifest = pd.read_csv(source / "split_manifest.csv")
    development = development_data(frame, manifest)
    folds = group_folds(development, 5)
    search, frozen = [], []
    for kind in ("voting", "stacking"):
        best = None
        for alpha in ([None] if kind == "voting" else [0.1, 1.0, 10.0, 100.0]):
            cv = ensemble_cv(kind, xgb["features"], xgb["parameters"], development, folds, alpha)
            search.append({"kind": kind, "meta_alpha": alpha, "CV_seconds": cv["seconds"], **cv["metrics"]})
            key = cv["metrics"]["MAPE_percent"]
            if best is None or key < best[0]:
                best = (key, kind, alpha, cv)
        frozen.append(best[1:])
    pd.DataFrame(search).to_csv(output / "cv_search_results.csv", index=False)
    models, selected = [], []
    x, y = development[FEATURES], development.cycle_life.to_numpy(float)
    for kind, alpha, cv in frozen:
        inner = group_folds(development, 3) if kind == "stacking" else None
        model = make_ensemble(kind, xgb["features"], xgb["parameters"], inner, alpha)
        started = time.perf_counter()
        model.fit(x, y)
        refit_seconds = time.perf_counter() - started
        name = "Voting_equal" if kind == "voting" else "Stacking_Ridge"
        path = output / f"{name}.joblib"
        joblib.dump(model, path)
        record = {"model": name, "meta_alpha": alpha, "CV_MAPE_percent": cv["metrics"]["MAPE_percent"],
                  "train_cell_keys": development.cell_key.tolist(),
                  "base_XGBoost_features": xgb["features"], "base_XGBoost_parameters": xgb["parameters"],
                  "model_sha256": file_sha256(path)}
        if kind == "stacking":
            final = model.regressor_.final_estimator_
            record["meta_standardized_coefficients_log10"] = final.named_steps["ridge"].coef_.tolist()
            record["meta_intercept_log10"] = float(final.named_steps["ridge"].intercept_)
        selected.append(record)
        models.append((name, model, cv, refit_seconds))
    # This file freezes the ensemble construction before external evaluation.
    (output / "selected_models.json").write_text(json.dumps({"scenario": scenario,
        "selection_scope": "B1 group 5-fold CV only", "stacking_inner_folds": 3,
        "stacking_inner_split": "charging-policy GroupKFold", "test_used_for_selection": False,
        "models": selected}, indent=2), encoding="utf-8")
    indexed = frame.set_index("cell_key", drop=False)
    scores, predictions, cv_rows, fold_rows = [], [], [], []
    for name, model, cv, refit_seconds in models:
        row = {"scenario": scenario, "model": name,
               "features": "OLS_delta_only + Ridge_existing + XGBoost", "n_train": len(y),
               "CV_MAPE_percent": cv["metrics"]["MAPE_percent"],
               "CV_fold_MAPE_sd": float(np.std([r["MAPE_percent"] for r in cv["fold_metrics"]], ddof=1)),
               "CV_seconds": cv["seconds"], "refit_seconds": refit_seconds}
        cv_rows.extend({"model": name, "cell_key": key, "actual_cycle_life": actual,
                        "predicted_cycle_life": predicted}
                       for key, actual, predicted in zip(development.cell_key, y, cv["predictions"]))
        fold_rows.extend({"model": name, **r} for r in cv["fold_metrics"])
        for role in ("valid", "test_b2", "test_b3"):
            part = indexed.loc[manifest.loc[manifest.role.eq(role), "cell_key"]]
            pred = model.predict(part[FEATURES])
            row[f"{role}_n"] = len(part)
            row.update({f"{role}_{metric}": value for metric, value in metrics(part.cycle_life, pred).items()})
            predictions.extend({"scenario": scenario, "model": name, "role": role,
                                "cell_key": key, "actual_cycle_life": actual,
                                "predicted_cycle_life": predicted,
                                "APE_percent": abs(predicted - actual) / actual * 100}
                               for key, actual, predicted in zip(part.cell_key, part.cycle_life, pred))
        scores.append(row)
    result = pd.DataFrame(scores)
    result.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output / "cv_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "cv_fold_results.csv", index=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.comparison / "ensembles"
    if output.exists():
        raise FileExistsError("Ensemble output already exists")
    protected = [ROOT / "models/ridge.joblib", args.comparison / "model_comparison.csv"]
    before = {str(p): file_sha256(p) for p in protected}
    config = load_config()
    frame = read_feature_csv(ROOT / "data/processed/cell_features.csv", config)
    output.mkdir()
    started = time.perf_counter()
    results = []
    for scenario in ("provided_labels", "unfinished_b1_excluded_sensitivity"):
        results.append(run_scenario(frame, args.comparison / scenario, output / scenario, scenario))
    new_scores = pd.concat(results, ignore_index=True)
    new_scores.to_csv(output / "model_comparison.csv", index=False)
    combined = pd.concat([pd.read_csv(args.comparison / "model_comparison.csv"), new_scores], ignore_index=True)
    combined.to_csv(args.comparison / "all_model_comparison.csv", index=False)
    winner = combined.loc[combined.groupby("scenario").CV_MAPE_percent.idxmin(), ["scenario", "model"]]
    summary = {"elapsed_seconds": time.perf_counter() - started,
               "B1_CV_selected_models": winner.to_dict("records"),
               "test_used_for_selection": False,
               "limitations": ["Base XGBoost settings were selected using the same B1 development dataset",
                               "Meta training uses group OOF predictions; outer validation is excluded",
                               "CV still has model-selection optimism; this is not a fully nested tuning benchmark",
                               "B2/B3 were already observed in earlier experiments"]}
    (output / "experiment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if before != {str(p): file_sha256(p) for p in protected}:
        raise RuntimeError("An original experiment artifact changed")
    print(combined[["scenario", "model", "CV_MAPE_percent", "valid_MAPE_percent",
                    "test_b2_MAPE_percent", "test_b3_MAPE_percent"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
