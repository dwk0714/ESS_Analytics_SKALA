"""Compare two correlation-selected additions with matched B1 grouped CV.

Selection uses absolute Pearson correlation with raw cycle_life; IR is excluded.
All final models are frozen before holdout or external-batch evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .compare_boosting import grouped_folds, make_model, parameter_candidates
from .evaluate import metrics
from .explore_feature_candidates import CANDIDATES
from .features import FEATURES, data_fingerprint
from .pipeline import ROOT, load_config, read_feature_csv
from .preprocess import file_sha256
from .splits import development_data, make_splits
from .train import runtime_environment

SEEDS = [42, 7, 2026]
MODEL_NAMES = {"ridge_existing": "Ridge_alpha1", "cat": "CatBoost",
               "voting": "Voting_equal", "stacking": "Stacking_Ridge"}


def rank_candidates(part, candidates):
    if not part.batch.eq("B1").all():
        raise ValueError("Candidate selection may only use B1 development-training cells")
    if len(part) < 3 or not part.cell_key.is_unique:
        raise ValueError("Insufficient or duplicate training cells")
    target = part.cycle_life.to_numpy(float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Training targets must be finite positive provided labels")
    records = []
    for candidate in candidates:
        if candidate.lower().startswith("ir_"):
            continue
        values = part[candidate].to_numpy(float)
        if not np.isfinite(values).all() or np.ptp(values) <= 0:
            raise ValueError(f"Candidate has missing or constant training data: {candidate}")
        correlation = float(np.corrcoef(values, target)[0, 1])
        records.append({"candidate": candidate, "pearson_cycle_life": correlation,
                        "abs_pearson_cycle_life": abs(correlation), "n_training": len(part)})
    records.sort(key=lambda row: (-row["abs_pearson_cycle_life"], row["candidate"]))
    if len(records) < 2:
        raise ValueError("At least two non-IR candidates are required")
    for rank, row in enumerate(records, 1):
        row.update(rank=rank, selected_top2=rank <= 2)
    return [row["candidate"] for row in records[:2]], records


def average_voting_cycles(log_predictions):
    return np.power(10.0, np.asarray(log_predictions, float)).mean(axis=1)


@dataclass
class FittedBundle:
    kind: str
    bases: list
    features: list[list[str]]
    selected_extra_features: list[str]
    parameters: dict
    meta: object = None

    def predict(self, frame):
        log_predictions = np.column_stack([base.regressor_.predict(frame) for base in self.bases])
        if self.kind == "voting":
            return average_voting_cycles(log_predictions)
        if self.kind == "stacking":
            return np.power(10.0, self.meta.predict(log_predictions))
        return np.power(10.0, log_predictions[:, 0])


class Experiment:
    """Caches only identical training sets; each cache value has its own audit scope."""

    def __init__(self, candidates, development_keys, parameter_count=12):
        self.candidates = [c for c in candidates if not c.lower().startswith("ir_")]
        self.allowed_keys = set(development_keys)
        self.settings = {kind: parameter_candidates(kind, parameter_count) for kind in ("cat", "xgb")}
        self.rankings, self.inner_search, self.selection_scopes, self.meta_predictions = [], [], [], []
        self.base_cache, self.rank_cache = {}, {}
        self.fit_count = 0

    def guard(self, part):
        if not part.batch.eq("B1").all() or not set(part.cell_key).issubset(self.allowed_keys):
            raise ValueError("A holdout or external cell entered model selection/fitting")

    def extras(self, part, augmented, scope):
        self.guard(part)
        if not augmented:
            return []
        keys = tuple(part.cell_key)
        if keys not in self.rank_cache:
            self.rank_cache[keys] = rank_candidates(part, self.candidates)
        selected, ranking = self.rank_cache[keys]
        digest = hashlib.sha256(";".join(keys).encode()).hexdigest()
        self.rankings.extend({"scope": scope, "training_keys_sha256": digest, **row} for row in ranking)
        self.selection_scopes.append({"scope": scope, "training_cell_keys": list(keys),
                                      "selected_extra_features": selected, "training_keys_sha256": digest})
        return selected.copy()

    def tune_booster(self, kind, part, augmented, seed, scope):
        self.guard(part)
        folds = grouped_folds(part, 3, seed)
        selections = []
        for fold, (train, valid) in enumerate(folds, 1):
            training = part.iloc[train].reset_index(drop=True)
            selections.append(FEATURES[:2] + self.extras(training, augmented, f"{scope}/inner{fold}"))
            self.selection_scopes.append({"scope": f"{scope}/inner{fold}/partition",
                                          "training_cell_keys": training.cell_key.tolist(),
                                          "validation_cell_keys": part.iloc[valid].cell_key.tolist(),
                                          "training_policies": sorted(training.policy.unique().tolist()),
                                          "validation_policies": sorted(part.iloc[valid].policy.unique().tolist())})
        y = part.cycle_life.to_numpy(float)
        best = None
        for trial, parameters in enumerate(self.settings[kind]):
            prediction = np.full(len(part), np.nan)
            for selected, (train, valid) in zip(selections, folds):
                estimator = make_model(kind, selected, parameters)
                estimator.fit(part.iloc[train], y[train])
                self.fit_count += 1
                prediction[valid] = estimator.predict(part.iloc[valid])
            score = metrics(y, prediction)["MAPE_percent"]
            self.inner_search.append({"scope": scope, "model_kind": kind,
                                      "variant": "plus_top2" if augmented else "baseline",
                                      "trial": trial, "parameters": json.dumps(parameters, sort_keys=True),
                                      "inner_MAPE_percent": score,
                                      "features_by_inner_fold": json.dumps(selections), "n_training": len(part)})
            ordering = (score, trial)
            if best is None or ordering < best[0]:
                best = (ordering, parameters.copy())
        return best[1]

    def base(self, kind, part, augmented, seed, scope):
        self.guard(part)
        key = (kind, augmented, seed, tuple(part.cell_key))
        if key in self.base_cache:
            return self.base_cache[key]
        extras = self.extras(part, augmented, f"{scope}/rank")
        if kind == "ols_delta":
            selected, parameters = FEATURES[:1] + extras, {}
        elif kind == "ridge_existing":
            selected, parameters = FEATURES[:2] + extras, {"alpha": 1.0}
        else:
            selected = FEATURES[:2] + extras
            parameters = self.tune_booster(kind, part, augmented, seed, f"{scope}/tune")
        estimator = make_model(kind, selected, parameters).fit(part, part.cycle_life.to_numpy(float))
        self.fit_count += 1
        result = FittedBundle(kind, [estimator], [selected], extras, parameters)
        self.base_cache[key] = result
        return result

    def fit(self, kind, part, augmented, seed, scope):
        self.guard(part)
        if kind not in ("voting", "stacking"):
            return self.base(kind, part, augmented, seed, scope)
        kinds = ("ols_delta", "ridge_existing", "xgb")
        bases = [self.base(base, part, augmented, seed, f"{scope}/{base}") for base in kinds]
        result = FittedBundle(kind, [base.bases[0] for base in bases], [base.features[0] for base in bases],
                              bases[0].selected_extra_features, {"xgb": bases[2].parameters, "meta_alpha": 1.0 if kind == "stacking" else None})
        if kind == "stacking":
            meta_folds = grouped_folds(part, 3, seed)
            oof_log = np.full((len(part), 3), np.nan)
            for fold, (train, valid) in enumerate(meta_folds, 1):
                sub = part.iloc[train].reset_index(drop=True)
                sub_bases = [self.base(base, sub, augmented, seed, f"{scope}/meta{fold}/{base}") for base in kinds]
                for column, base in enumerate(sub_bases):
                    oof_log[valid, column] = base.bases[0].regressor_.predict(part.iloc[valid])
                self.selection_scopes.append({"scope": f"{scope}/meta{fold}/partition",
                                              "training_cell_keys": sub.cell_key.tolist(),
                                              "validation_cell_keys": part.iloc[valid].cell_key.tolist(),
                                              "training_policies": sorted(sub.policy.unique().tolist()),
                                              "validation_policies": sorted(part.iloc[valid].policy.unique().tolist())})
                self.meta_predictions.extend({"scope": scope, "meta_fold": fold,
                                               "variant": "plus_top2" if augmented else "baseline",
                                               "cell_key": part.iloc[index].cell_key,
                                               "actual_cycle_life": part.iloc[index].cycle_life,
                                               "OLS_predicted_log10": oof_log[index, 0], "Ridge_predicted_log10": oof_log[index, 1],
                                               "XGB_predicted_log10": oof_log[index, 2],
                                               "selected_extra_features": ";".join(sub_bases[0].selected_extra_features)} for index in valid)
            if not np.isfinite(oof_log).all():
                raise ValueError("Stacking meta training requires exactly one finite grouped-OOF base prediction per cell")
            result.meta = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="svd"))
            result.meta.fit(oof_log, np.log10(part.cycle_life.to_numpy(float)))
            self.fit_count += 1
        return result


def protected_hashes(output):
    paths = []
    for directory in ("config", "models", "data/processed", "results"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and not p.is_relative_to(output))
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(paths)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/runs/top2_feature_comparison_20261002")
    parser.add_argument("--parameter-count", type=int, default=12)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Existing experimental results are preserved; use a new output folder")
    started = time.perf_counter()
    config = load_config()
    original = read_feature_csv(ROOT / "data/processed/cell_features.csv", config)
    manifest = pd.read_csv(ROOT / "results/split_manifest.csv")
    pd.testing.assert_frame_equal(manifest, make_splits(original, config))
    pool = [candidate for candidate in CANDIDATES if not candidate.lower().startswith("ir_")]
    candidate_file = ROOT / "results/runs/feature_candidates_20261002/features.csv"
    candidates = pd.read_csv(candidate_file, float_precision="round_trip")
    if len(candidates) != len(original) or set(candidates.cell_key) != set(original.cell_key):
        raise ValueError("Candidate rows disagree with the approved feature table")
    if not candidates.feature_latest_cycle.eq(100).all():
        raise ValueError("Candidate predictors must be available by cycle100")
    frame = original.drop(columns=pool, errors="ignore").merge(
        candidates[["cell_key"] + pool], on="cell_key", validate="one_to_one")
    development = development_data(frame, manifest)
    if len(development) != 35 or not development.batch.eq("B1").all():
        raise ValueError("Expected the original fixed B1 training35 cells")
    for role, count in (("valid", 11), ("test_b2", 39), ("test_b3", 44)):
        if int(manifest.role.eq(role).sum()) != count:
            raise ValueError(f"Unexpected evaluation count for {role}")
    before = protected_hashes(output)
    output.mkdir(parents=True)
    experiment = Experiment(pool, development.cell_key, args.parameter_count)
    full_extras, full_rank = rank_candidates(development, pool)
    pd.DataFrame(full_rank).to_csv(output / "full_B1_train_candidate_ranking.csv", index=False)
    variants = [("OLS_delta_only", "ols_delta", False)] + [
        (f"{name}_{'plus_top2' if augmented else 'baseline'}", kind, augmented)
        for kind, name in MODEL_NAMES.items() for augmented in (False, True)]
    summaries, predictions, folds, split_rows = [], [], [], []
    for name, kind, augmented in variants:
        repeat_scores, fit_scores = [], []
        for repeat, seed in enumerate(SEEDS, 1):
            prediction = np.full(len(development), np.nan)
            for fold, (train, valid) in enumerate(grouped_folds(development, 5, seed), 1):
                sub = development.iloc[train].reset_index(drop=True)
                model = experiment.fit(kind, sub, augmented, seed, f"{name}/repeat{repeat}/outer{fold}")
                prediction[valid] = model.predict(development.iloc[valid])
                fitted = metrics(sub.cycle_life, model.predict(sub))["MAPE_percent"]
                fit_scores.append((len(train), fitted))
                outer = metrics(development.iloc[valid].cycle_life, prediction[valid])
                folds.append({"model": name, "model_kind": kind, "variant": "plus_top2" if augmented else "baseline",
                              "repeat": repeat, "seed": seed, "fold": fold,
                              "n_train": len(train), "n_valid": len(valid), "fit_MAPE_percent": fitted,
                              **{f"outer_{key}": value for key, value in outer.items()},
                              "selected_extra_features": ";".join(model.selected_extra_features),
                              "base_features": json.dumps(model.features), "selected_parameters": json.dumps(model.parameters, sort_keys=True)})
                for role, indices in (("train", train), ("valid", valid)):
                    split_rows.extend({"model": name, "repeat": repeat, "seed": seed, "fold": fold, "role": role,
                                       "cell_key": development.iloc[index].cell_key, "policy": development.iloc[index].policy} for index in indices)
            score = metrics(development.cycle_life, prediction)
            repeat_scores.append(score["MAPE_percent"])
            predictions.extend({"model": name, "repeat": repeat, "seed": seed, "cell_key": key,
                                "actual_cycle_life": actual, "predicted_cycle_life": predicted,
                                "APE_percent": abs(predicted - actual) / actual * 100}
                               for key, actual, predicted in zip(development.cell_key, development.cycle_life, prediction))
        fitted_mean = sum(n * score for n, score in fit_scores) / sum(n for n, _ in fit_scores)
        summaries.append({"model": name, "model_kind": kind, "variant": "plus_top2" if augmented else "baseline",
                          "n_train": len(development), "nested_MAPE_percent": float(np.mean(repeat_scores)),
                          "repeat_MAPE_sd": float(np.std(repeat_scores, ddof=1)),
                          "repeat_MAPE_values": json.dumps(repeat_scores), "fit_MAPE_percent": fitted_mean,
                          "overfit_gap_percent_points": float(np.mean(repeat_scores)) - fitted_mean})
        pd.DataFrame(summaries).to_csv(output / "nested_cv_comparison.csv", index=False)
        print(f"{name}: repeated outer CV MAPE={np.mean(repeat_scores):.4f}%", flush=True)
    # Frozen choices and saved models precede all external evaluation.
    models, selected = [], []
    for row, (name, kind, augmented) in zip(summaries, variants):
        fitted = experiment.fit(kind, development, augmented, 42, f"final_B1_train/{name}")
        if augmented and fitted.selected_extra_features != full_extras:
            raise ValueError("Final selection disagrees with the full B1 development ranking")
        path = output / f"{name}.joblib"
        joblib.dump(fitted, path)
        row.update(selected_extra_features=";".join(fitted.selected_extra_features),
                   features=json.dumps(fitted.features), parameters=json.dumps(fitted.parameters, sort_keys=True))
        record = {"model": name, "extra_features": fitted.selected_extra_features, "base_features": fitted.features,
                  "parameters": fitted.parameters, "train_cell_keys": development.cell_key.tolist(),
                  "model_sha256": file_sha256(path), "nested_MAPE_percent": row["nested_MAPE_percent"]}
        if kind == "stacking":
            record["meta_standardized_coefficients_log10"] = fitted.meta.named_steps["ridge"].coef_.tolist()
        selected.append(record)
        models.append(fitted)
    frozen = {"selection_scope": "Fixed B1 train35 only; raw cycle_life Pearson ranking;IR excluded",
              "global_full_B1_train_extra_features": full_extras, "models": selected,
              "B1_nested_CV_selected_research_model": min(summaries, key=lambda row: row["nested_MAPE_percent"])["model"],
              "external_targets_used_for_selection": False, "default_core_model_changed": False}
    (output / "selected_models.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    pd.DataFrame(predictions).to_csv(output / "outer_oof_predictions.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "outer_fold_results.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "outer_split_manifest.csv", index=False)
    pd.DataFrame(experiment.rankings).to_csv(output / "selection_rankings_training_only.csv", index=False)
    pd.DataFrame(experiment.inner_search).to_csv(output / "inner_search.csv", index=False)
    pd.DataFrame(experiment.meta_predictions).to_csv(output / "stacking_meta_oof_predictions.csv", index=False)
    (output / "selection_scope_audit.json").write_text(json.dumps(experiment.selection_scopes, indent=2), encoding="utf-8")
    manifest.to_csv(output / "split_manifest.csv", index=False)
    indexed, external = frame.set_index("cell_key", drop=False), []
    for row, fitted in zip(summaries, models):
        for role in ("valid", "test_b2", "test_b3"):
            part = indexed.loc[manifest.loc[manifest.role.eq(role), "cell_key"]]
            predicted = fitted.predict(part)
            row[f"{role}_n"] = len(part)
            row.update({f"{role}_{key}": value for key, value in metrics(part.cycle_life, predicted).items()})
            external.extend({"model": row["model"], "role": role, "cell_key": key,
                             "actual_cycle_life": actual, "predicted_cycle_life": pred,
                             "APE_percent": abs(pred - actual) / actual * 100}
                            for key, actual, pred in zip(part.cell_key, part.cycle_life, predicted))
    comparison = pd.DataFrame(summaries)
    paired = []
    for kind, name in MODEL_NAMES.items():
        base = comparison.loc[comparison.model.eq(f"{name}_baseline")].iloc[0]
        augmented = comparison.loc[comparison.model.eq(f"{name}_plus_top2")].iloc[0]
        changes = np.array(json.loads(augmented.repeat_MAPE_values)) - np.array(json.loads(base.repeat_MAPE_values))
        paired.append({"model_kind": kind, "model": name,
                       "delta_nested_MAPE_pp": augmented.nested_MAPE_percent - base.nested_MAPE_percent,
                       "delta_repeat_MAPE_values_pp": json.dumps(changes.tolist()),
                       "repeat_improvement_count": int((changes < 0).sum()),
                       "delta_holdout_MAPE_pp": augmented.valid_MAPE_percent - base.valid_MAPE_percent,
                       "delta_B2_MAPE_pp": augmented.test_b2_MAPE_percent - base.test_b2_MAPE_percent,
                       "delta_B3_MAPE_pp": augmented.test_b3_MAPE_percent - base.test_b3_MAPE_percent,
                       "all_three_repeats_and_holdout_improve": bool((changes < 0).all() and augmented.valid_MAPE_percent < base.valid_MAPE_percent)})
    comparison.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame(external).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_improvements.csv", index=False)
    outer_selection = pd.DataFrame(folds).query("model == 'Ridge_alpha1_plus_top2'")
    frequency = outer_selection.selected_extra_features.str.split(";").explode().value_counts().rename_axis("candidate").reset_index(name="selected_outer_folds")
    frequency["n_outer_folds"] = 15
    frequency.to_csv(output / "outer_top2_selection_stability.csv", index=False)
    after = protected_hashes(output)
    if before != after:
        raise RuntimeError("An existing core/previous experimental artifact changed")
    environment = runtime_environment()
    environment["packages"].update({name: importlib.metadata.version(name) for name in ("catboost", "xgboost")})
    protocol = {"status": "research_comparison_only_no_final_model_replacement", "data_fingerprint": data_fingerprint(original),
                "candidate_input_sha256": file_sha256(candidate_file), "candidate_pool": pool, "candidate_count_after_IR_exclusion": len(pool),
                "ranking": "absolute Pearson correlation with actual raw cycle_life within each training partition;not actual cycle index or log10 life",
                "global_extra_features": full_extras, "target_transform": "log10", "outer_seeds": SEEDS,
                "outer_folds": 5, "inner_folds": 3, "parameter_candidates_per_booster": args.parameter_count,
                "base_feature_sets": {"OLS": FEATURES[:1], "Ridge": FEATURES[:2], "XGB": FEATURES[:2], "CatBoost": FEATURES[:2]},
                "feature_addition": "Append the same training-partition top2 to every base estimator in the augmented model",
                "ridge_alpha": 1.0, "stack_meta_alpha": 1.0, "voting": "equal arithmetic average in cycle-life units",
                "stacking": "group OOF log10 predictions;each meta training subset independently selects features and tunes XGB using its own3foldCV;full training models refit for inference",
                "all_selection_and_scaling_training_only": True, "external_targets_used_for_selection": False,
                "n_development": 35, "n_holdout": 11, "n_B2": 39, "n_B3": 44,
                "core_model_and_reports_changed": False, "actual_fit_count": experiment.fit_count,
                "elapsed_seconds": time.perf_counter() - started,
                "environment": environment, "preserved_original_sha256": after,
                "limitations": ["Matched new baseline fixes XGB and CatBoost to two core features;previous search over1/2/3 feature sets has different scores",
                    "Strict meta-OOF XGB retuning differs from the previous stacking implementation",
                    "35development cells;repeated folds overlap and SD is descriptive,not an independent confidence interval",
                    "Two globally selected added features are strongly collinear;high marginal correlation does not establish incremental benefit",
                    "Provided labels and existing sensor/curve QC flags are unchanged",
                    "B2/B3 were previously exposed in EDA/experiments;external results are reported after freezing and do not select models",
                    "Final model replacement requires separate user confirmation"]}
    (output / "experiment.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(comparison[["model", "nested_MAPE_percent", "valid_MAPE_percent", "test_b2_MAPE_percent", "test_b3_MAPE_percent"]].round(4).to_string(index=False))
    print(pd.DataFrame(paired).round(4).to_string(index=False))


if __name__ == "__main__":
    # Use the importable class path in joblib files even when invoked with -m.
    from .compare_top_correlations import main as imported_main
    imported_main()
