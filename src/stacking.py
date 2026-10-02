"""Protocol-grouped feature selection, XGBoost tuning and OOF Stacking.

Only B1 development-training cells may enter selection or fitting.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import metrics
from .features import FEATURES

def inverse_log10(values):
    return np.power(10.0, values)

def grouped_folds(part, n_splits, seed):
    folds = list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                 .split(part, groups=part.policy))
    covered = np.zeros(len(part), int)
    for train, valid in folds:
        if set(part.iloc[train].policy) & set(part.iloc[valid].policy):
            raise ValueError("Charging protocol overlaps training and validation")
        covered[valid] += 1
    if not np.all(covered == 1):
        raise ValueError("Expected one validation prediction per cell/repeat")
    return folds

def parameter_candidates(kind, count=12):
    if kind != "xgb":
        raise ValueError("The final Stacking uses XGBoost only")
    grid = {"max_depth": [1, 2, 3], "learning_rate": [0.03, 0.07],
            "n_estimators": [100, 250], "min_child_weight": [3, 5],
            "reg_lambda": [1.0, 10.0, 30.0]}
    return list(ParameterSampler(grid, n_iter=count, random_state=42))

def make_model(kind, selected, parameters):
    if kind == "ols_delta":
        estimator = LinearRegression()
    elif kind == "ridge_existing":
        estimator = Ridge(alpha=parameters.get("alpha", 1.0), solver="svd")
    elif kind == "xgb":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(objective="reg:squarederror", tree_method="hist",
            device="cpu", n_jobs=1, random_state=42, subsample=1.0,
            colsample_bytree=1.0, verbosity=0, **parameters)
    else:
        raise ValueError(kind)
    regression = Pipeline([("preprocess", ColumnTransformer(
        [("selected", StandardScaler(), selected)], remainder="drop")),
        ("regressor", estimator)])
    return TransformedTargetRegressor(regressor=regression, func=np.log10,
                                      inverse_func=inverse_log10)

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


class StackingTrainer:
    """Caches only identical training sets; each cache value has its own audit scope."""

    def __init__(self, candidates, development_keys, parameter_count=12):
        self.candidates = [c for c in candidates if not c.lower().startswith("ir_")]
        self.allowed_keys = set(development_keys)
        self.settings = {kind: parameter_candidates(kind, parameter_count) for kind in ("xgb",)}
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


