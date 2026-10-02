"""Repeated nested protocol-group CV for small-data regression candidates.

python -m src.compare_boosting --output results/runs/boosting_selection_20261002
Feature/parameter selection uses only B1 development cells inside each outer fold.
Existing models, split manifests, labels, and evaluation files are preserved.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .compare_models import make_candidate
from .compare_ensembles import make_ensemble
from .evaluate import metrics
from .features import FEATURES, data_fingerprint
from .pipeline import ROOT, load_config, read_feature_csv
from .preprocess import file_sha256
from .splits import development_data, make_splits
from .train import inverse_log10, runtime_environment

NAMES = {"ols_delta": "OLS_delta_only", "ols_two": "OLS_two_features",
         "ridge_existing": "Ridge_existing", "ridge": "Ridge_tuned",
         "xgb": "XGBoost", "cat": "CatBoost", "lgbm": "LightGBM",
         "gbr": "GradientBoosting", "stacking": "Stacking_Ridge",
         "voting": "Voting_equal"}


def grouped_folds(part, n_splits, seed):
    folds = list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                 .split(part, groups=part.policy))
    covered = np.zeros(len(part), int)
    for train, valid in folds:
        if set(part.iloc[train].policy) & set(part.iloc[valid].policy):
            raise ValueError("Same charging protocol occurs in training and validation")
        covered[valid] += 1
    if not np.all(covered == 1):
        raise ValueError("Each cell must have one validation prediction per repeat")
    return folds


def parameter_candidates(kind, count=12):
    if kind == "xgb":
        grid = {"max_depth": [1, 2, 3], "learning_rate": [0.03, 0.07],
                "n_estimators": [100, 250], "min_child_weight": [3, 5],
                "reg_lambda": [1.0, 10.0, 30.0]}
    elif kind == "cat":
        grid = {"depth": [1, 2, 3], "learning_rate": [0.03, 0.07],
                "iterations": [100, 250], "l2_leaf_reg": [3.0, 10.0, 30.0]}
    elif kind == "lgbm":
        grid = {"max_depth": [1, 2, 3], "learning_rate": [0.03, 0.07],
                "n_estimators": [100, 250], "min_child_samples": [3, 5],
                "reg_lambda": [0.1, 1.0, 10.0]}
    elif kind == "gbr":
        grid = {"max_depth": [1, 2, 3], "learning_rate": [0.03, 0.07],
                "n_estimators": [100, 250], "min_samples_leaf": [3, 5]}
    else:
        raise ValueError(kind)
    return list(ParameterSampler(grid, n_iter=count, random_state=42))


def make_model(kind, selected, parameters):
    if kind in ("ols_delta", "ols_two", "ridge_existing", "ridge", "xgb"):
        base = "ols" if kind.startswith("ols") else "ridge" if kind.startswith("ridge") else "xgb"
        return make_candidate(base, selected, parameters)
    if kind == "cat":
        from catboost import CatBoostRegressor
        estimator = CatBoostRegressor(loss_function="RMSE", thread_count=1,
            random_seed=42, allow_writing_files=False, verbose=False,
            bootstrap_type="No", random_strength=0.0, **parameters)
    elif kind == "lgbm":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(objective="regression", n_jobs=1,
            random_state=42, verbosity=-1, deterministic=True, force_col_wise=True,
            min_data_in_bin=1, max_bin=31, num_leaves=2 ** parameters["max_depth"], **parameters)
    elif kind == "gbr":
        estimator = GradientBoostingRegressor(loss="squared_error", random_state=42,
                                               subsample=1.0, **parameters)
    else:
        raise ValueError(kind)
    pipeline = Pipeline([("preprocess", ColumnTransformer(
        [("selected", StandardScaler(), selected)], remainder="drop")),
        ("regressor", estimator)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log10, inverse_func=inverse_log10)


def candidate_settings(kind, count):
    if kind == "ols_delta":
        return [(FEATURES[:1], {})]
    if kind == "ols_two":
        return [(FEATURES[:2], {})]
    if kind == "ridge_existing":
        return [(FEATURES[:2], {"alpha": 1.0})]
    if kind == "ridge":
        return [(selected, {"alpha": alpha}) for selected in (FEATURES[:1], FEATURES[:2], FEATURES)
                for alpha in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)]
    return [(selected, params) for selected in (FEATURES[:1], FEATURES[:2], FEATURES)
            for params in parameter_candidates(kind, count)]


def tune_model(kind, part, settings, seed=42):
    folds = grouped_folds(part, 3, seed)
    x, y = part[FEATURES], part.cycle_life.to_numpy(float)
    best, records = None, []
    for trial, (selected, parameters) in enumerate(settings):
        prediction = np.full(len(y), np.nan)
        for train, valid in folds:
            model = make_model(kind, selected, parameters)
            model.fit(x.iloc[train], y[train])
            prediction[valid] = model.predict(x.iloc[valid])
        score = metrics(y, prediction)["MAPE_percent"]
        row = {"trial": trial, "features": selected, "parameters": parameters,
               "inner_MAPE_percent": score}
        records.append(row)
        key = (score, len(selected), trial)
        if best is None or key < best[0]:
            best = (key, row)
    return best[1], records


def nested_cv(kind, part, settings, seeds, cached_xgb=None):
    x, y = part[FEATURES], part.cycle_life.to_numpy(float)
    predictions, repeat_scores, fold_rows, split_rows, tuned_rows = [], [], [], [], []
    fit_sum, fit_n = 0.0, 0
    for repeat, seed in enumerate(seeds, 1):
        prediction = np.full(len(y), np.nan)
        for fold, (train, valid) in enumerate(grouped_folds(part, 5, seed), 1):
            sub = part.iloc[train].reset_index(drop=True)
            if kind in ("stacking", "voting"):
                choice = cached_xgb[(repeat, fold)]
                model = make_ensemble(kind, choice["features"], choice["parameters"],
                    grouped_folds(sub, 3, seed) if kind == "stacking" else None, meta_alpha=1.0)
            else:
                choice, search = tune_model(kind, sub, settings, seed)
                model = make_model(kind, choice["features"], choice["parameters"])
            model.fit(x.iloc[train], y[train])
            fit_prediction = model.predict(x.iloc[train])
            prediction[valid] = model.predict(x.iloc[valid])
            fit_score = metrics(y[train], fit_prediction)["MAPE_percent"]
            fit_sum += fit_score * len(train)
            fit_n += len(train)
            outer_score = metrics(y[valid], prediction[valid])["MAPE_percent"]
            fold_rows.append({"model": NAMES[kind], "repeat": repeat, "fold": fold,
                "n_train": len(train), "n_valid": len(valid), "fit_MAPE_percent": fit_score,
                "outer_MAPE_percent": outer_score, "gap_percent_points": outer_score - fit_score,
                "selected_features": ";".join(choice["features"]),
                "selected_parameters": json.dumps(choice["parameters"], sort_keys=True)})
            tuned_rows.append({"repeat": repeat, "fold": fold, **choice})
            for role, positions in (("train", train), ("valid", valid)):
                split_rows.extend({"model": NAMES[kind], "repeat": repeat, "fold": fold,
                                   "role": role, "cell_key": part.iloc[i].cell_key,
                                   "policy": part.iloc[i].policy} for i in positions)
        if not np.isfinite(prediction).all():
            raise ValueError("Missing outer OOF predictions")
        repeat_scores.append(metrics(y, prediction)["MAPE_percent"])
        predictions.extend({"model": NAMES[kind], "repeat": repeat, "cell_key": key,
                            "actual_cycle_life": actual, "predicted_cycle_life": pred}
                           for key, actual, pred in zip(part.cell_key, y, prediction))
        print(f"{NAMES[kind]} repeat {repeat}/{len(seeds)} outer MAPE={repeat_scores[-1]:.3f}%", flush=True)
    return {"nested_MAPE_percent": float(np.mean(repeat_scores)),
            "repeat_MAPE_sd": float(np.std(repeat_scores, ddof=1)),
            "repeat_MAPE_values": repeat_scores, "fit_MAPE_percent": fit_sum / fit_n,
            "overfit_gap_percent_points": float(np.mean(repeat_scores)) - fit_sum / fit_n,
            "outer_fold_MAPE_sd": float(np.std([r["outer_MAPE_percent"] for r in fold_rows], ddof=1)),
            "predictions": predictions, "fold_rows": fold_rows, "split_rows": split_rows,
            "tuned_rows": tuned_rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parameter-count", type=int, default=12)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
    if args.output.exists():
        raise FileExistsError("Use a new output path")
    config = load_config()
    frame = read_feature_csv(ROOT / "data/processed/cell_features.csv", config)
    manifest = pd.read_csv(ROOT / "results/split_manifest.csv")
    pd.testing.assert_frame_equal(manifest, make_splits(frame, config))
    part = development_data(frame, manifest)
    args.output.mkdir(parents=True)
    protected = ["models/ridge.joblib", "models/metadata.json", "config/model.json",
                 "data/processed/cell_features.csv", "results/model_performance.csv", "results/predictions.csv"]
    before = {p: file_sha256(ROOT / p) for p in protected}
    seeds = [42, 7, 2026]
    protocol = {"selection_scope": "B1 train35 only, repeated nested charging-protocol group CV",
                "outer": {"folds": 5, "seeds": seeds}, "inner": {"folds": 3},
                "parameter_candidates_per_booster_feature_set": args.parameter_count,
                "maximum_tree_depth": 3, "tree_counts": [100, 250],
                "selection_objective": "Lowest mean repeat outer-OOF MAPE; inspect fit/validation gaps separately",
                "stacking_meta_alpha_fixed_before_run": 1.0,
                "test_used_for_selection": False, "target_transform": "log10",
                "limitations": ["Repeated folds overlap; repeat SD is descriptive, not an independent confidence interval",
                    "35 development cells cannot establish absence of overfitting",
                    "B1 stored-label inconsistencies are retained for comparability",
                    "B2/B3 were already examined in prior EDA and experiments"],
                "data_fingerprint": data_fingerprint(frame), "original_sha256": before}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    started = time.perf_counter()
    summary, outer_predictions, outer_folds, outer_splits, selected, models = [], [], [], [], [], []
    xgb_choices = None
    for kind in NAMES:
        began = time.perf_counter()
        if kind in ("stacking", "voting"):
            settings = None
        else:
            settings = candidate_settings(kind, args.parameter_count)
        cv = nested_cv(kind, part, settings, seeds, xgb_choices)
        if kind == "xgb":
            xgb_choices = {(r["repeat"], r["fold"]): r for r in cv["tuned_rows"]}
        if kind in ("stacking", "voting"):
            full_choice = next(r for r in selected if r["model"] == "XGBoost")
            choice = {"features": full_choice["features"], "parameters": full_choice["parameters"]}
            model = make_ensemble(kind, choice["features"], choice["parameters"],
                grouped_folds(part, 3, 42) if kind == "stacking" else None, meta_alpha=1.0)
            search = []
        else:
            choice, search = tune_model(kind, part, settings)
            model = make_model(kind, choice["features"], choice["parameters"])
        model.fit(part[FEATURES], part.cycle_life.to_numpy(float))
        model_path = args.output / f"{NAMES[kind]}.joblib"
        joblib.dump(model, model_path)
        selected.append({"model": NAMES[kind], **choice, "nested_MAPE_percent": cv["nested_MAPE_percent"],
                         "model_sha256": file_sha256(model_path), "train_cell_keys": part.cell_key.tolist()})
        pd.DataFrame(search).to_csv(args.output / f"{NAMES[kind]}_inner_search.csv", index=False)
        models.append((NAMES[kind], model))
        row = {"model": NAMES[kind], "n_train": len(part),
               **{key: cv[key] for key in ("nested_MAPE_percent", "repeat_MAPE_sd", "fit_MAPE_percent",
                                          "overfit_gap_percent_points", "outer_fold_MAPE_sd")},
               "features": ";".join(choice["features"]),
               "parameters": json.dumps(choice["parameters"], sort_keys=True),
               "elapsed_seconds": time.perf_counter() - began}
        summary.append(row)
        outer_predictions.extend(cv["predictions"])
        outer_folds.extend(cv["fold_rows"])
        outer_splits.extend(cv["split_rows"])
        (args.output / f"{NAMES[kind]}_outer_selected_parameters.json").write_text(
            json.dumps(cv["tuned_rows"], indent=2), encoding="utf-8")
        pd.DataFrame(summary).to_csv(args.output / "nested_cv_comparison.csv", index=False)
    winner = min(summary, key=lambda row: row["nested_MAPE_percent"])["model"]
    booster_winner = min([r for r in summary if r["model"] in ("XGBoost", "CatBoost", "LightGBM", "GradientBoosting")],
                        key=lambda row: row["nested_MAPE_percent"])["model"]
    # Freeze selection before computing any holdout or external-test score.
    frozen = {"B1_nested_CV_selected_model": winner, "B1_nested_CV_selected_booster": booster_winner,
              "selection_uses_test": False, "models": selected}
    (args.output / "selected_models.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    pd.DataFrame(outer_predictions).to_csv(args.output / "outer_oof_predictions.csv", index=False)
    pd.DataFrame(outer_folds).to_csv(args.output / "outer_fold_results.csv", index=False)
    pd.DataFrame(outer_splits).to_csv(args.output / "outer_split_manifest.csv", index=False)
    manifest.to_csv(args.output / "split_manifest.csv", index=False)
    indexed = frame.set_index("cell_key", drop=False)
    predictions = []
    for row, (name, model) in zip(summary, models):
        for role in ("valid", "test_b2", "test_b3"):
            p = indexed.loc[manifest.loc[manifest.role.eq(role), "cell_key"]]
            pred = model.predict(p[FEATURES])
            row.update({f"{role}_{key}": value for key, value in metrics(p.cycle_life, pred).items()})
            predictions.extend({"model": name, "role": role, "cell_key": key,
                "actual_cycle_life": actual, "predicted_cycle_life": value,
                "APE_percent": abs(value - actual) / actual * 100}
                for key, actual, value in zip(p.cell_key, p.cycle_life, pred))
    result = pd.DataFrame(summary)
    result.to_csv(args.output / "model_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output / "predictions.csv", index=False)
    env = runtime_environment()
    for pkg in ("catboost", "lightgbm", "xgboost"):
        env["packages"][pkg] = importlib.metadata.version(pkg)
    after = {p: file_sha256(ROOT / p) for p in protected}
    if before != after:
        raise RuntimeError("An existing experiment artifact changed")
    (args.output / "runtime.json").write_text(json.dumps({"seconds": time.perf_counter() - started,
        "environment": env, "original_sha256": after}, indent=2), encoding="utf-8")
    print(result[["model", "nested_MAPE_percent", "fit_MAPE_percent", "overfit_gap_percent_points",
                  "valid_MAPE_percent", "test_b2_MAPE_percent", "test_b3_MAPE_percent"]].round(4).to_string(index=False))
    print(f"B1 nested CV selected: {winner}; best booster: {booster_winner}", flush=True)


if __name__ == "__main__":
    main()
