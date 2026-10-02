"""Early-only feature research; this module does not change the approved pipeline.

Run from the repository: python -m src.explore_feature_candidates
Only B1 development-train labels enter correlations, fitting, or evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import metrics
from .features import FEATURES, delta_q_log_variance, rms_current, relative_capacity_mad
from .preprocess import file_sha256, read_early_cells, reference_at, vector_prefix
from .train import inverse_log10, runtime_environment

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
SEEDS = [42, 7, 2026]
CANDIDATES = {
    "capacity_q100_q10_change_relative": {
        "formula": "QDischarge(actual100)/QDischarge(actual10)-1; both finite, 0<Q<=1.65Ah",
        "unit": "fraction", "latest_cycle": 100,
        "reason": "Signed retained-capacity change, independent of variance sign.",
        "limitation": "Two endpoints are noisier than a window; activation can yield a positive change."},
    "capacity_theilsen_10_100_relative": {
        "formula": "median over all i<j of (Qd[j]-Qd[i])/(actual_cycle[j]-actual_cycle[i]), cycles10..100, divided by median valid Qd2..10; >=50 valid points and >=5 reference points",
        "unit": "fraction/cycle", "latest_cycle": 100,
        "reason": "Robust signed early capacity drift on actual cycle coordinates.",
        "limitation": "Includes activation and slow drift; positive means early recovery, not degradation."},
    "capacity_theilsen_60_100_relative": {
        "formula": "Theil-Sen slope on valid Qd actual60..100 / median valid Qd2..10; >=30 slope points and >=5 reference points",
        "unit": "fraction/cycle", "latest_cycle": 100,
        "reason": "Later early-life drift can distinguish fading from initial conditioning.",
        "limitation": "Only41 observations; may still contain activation; fixed window chosen before label analysis."},
    "dq100_10_q05_ah": {
        "formula": "0.05 quantile of Qdlin100(V)-Qdlin10(V) on the common1000-point voltage grid; numpy linear quantile",
        "unit": "Ah", "latest_cycle": 100,
        "reason": "Robust lower-tail curve displacement without the fragile minimum or absolute-value sign loss.",
        "limitation": "Likely overlaps existing log variance; quantile cannot localize mechanism."},
    "dq100_10_mean_v2p0_2p4_ah": {
        "formula": "signed arithmetic mean of Qdlin100-Qdlin10 for common voltage coordinates2.0<=V<=2.4",
        "unit": "Ah", "latest_cycle": 100,
        "reason": "Localizes capacity displacement in the low-voltage discharge region.",
        "limitation": "Voltage band is chemistry/protocol dependent and fixed before label analysis."},
    "dq100_10_mean_ah": {
        "formula": "signed arithmetic mean of Qdlin100-Qdlin10 on the common1000-point2.0..3.5V grid; this uniform-grid point mean approximates normalized area",
        "unit": "Ah", "latest_cycle": 100,
        "reason": "Measures whole-curve displacement/normalized area, while existing variance measures shape spread.",
        "limitation": "Likely collinear with other curve descriptors; no electrode-mechanism claim."},
    "tavg_theilsen_10_100_degC_per_cycle": {
        "formula": "Theil-Sen slope of positive finite summary Tavg on actual cycles10..100; >=50 valid points",
        "unit": "degC/cycle", "latest_cycle": 100,
        "reason": "Observed early thermal drift can reflect changing heat generation under the test schedule.",
        "limitation": "Ambient, sensors and protocol can dominate; bad/nonpositive values are flagged, no batch correction."},
    "ir_change_91_100_vs_2_10_relative": {
        "formula": "median positive finite IR91..100 / median positive finite IR2..10 -1; >=5 valid values in each window",
        "unit": "fraction", "latest_cycle": 100,
        "reason": "Early relative resistance evolution with a per-cell robust baseline.",
        "limitation": "SOC/temperature/acquisition dependent; exactly repeated positive measurements yield0 and are flagged."},
    "tavg_median_2_100_degC": {
        "formula": "median positive finite summary Tavg on actual2..100, >=50 valid points",
        "unit": "degC", "latest_cycle": 100,
        "reason": "Robust early thermal exposure complements drift and observed capacity changes.",
        "limitation": "Temperature depends on environment/sensor/protocol; no absolute-temperature damage law is inferred."},
    "chargetime_median_2_100_minutes": {
        "formula": "median positive finite summary chargetime on actual2..100, >=50 valid points; no future-based outlier cutoff",
        "unit": "minutes", "latest_cycle": 100,
        "reason": "Robust observed charge duration describes time spent under the charging schedule.",
        "limitation": "May encode protocol and rests, overlaps RMS current; acquisition phases differ across batches."},
}


def validate_cycle_window(cycle):
    cycle = np.asarray(cycle, dtype=float)
    if not np.array_equal(cycle, np.arange(1, 101)):
        raise ValueError("Expected actual cycles1..100 only; future or missing coordinates are forbidden")


def robust_slope(cycle, values, min_n):
    valid = np.isfinite(cycle) & np.isfinite(values)
    if valid.sum() < min_n or np.unique(cycle[valid]).size != valid.sum():
        return np.nan
    return float(theilslopes(values[valid], cycle[valid])[0])


def relative_endpoint_change(q10, q100):
    if not all(np.isfinite(q) and 0 < q <= 1.65 for q in (q10, q100)):
        return np.nan
    return float(q100 / q10 - 1)


def window_median(cycle, values, lo, hi, min_n, max_value=np.inf):
    keep = (cycle >= lo) & (cycle <= hi) & np.isfinite(values) & (values > 0) & (values <= max_value)
    return float(np.median(values[keep])) if keep.sum() >= min_n else np.nan


def extract(raw_dir, config, original):
    rows, qc_rows, reads = [], [], []
    indexed = original.set_index("cell_key")
    for batch, filename in config["batches"].items():
        with h5py.File(raw_dir / filename, "r") as handle:
            for cell in read_early_cells(raw_dir, {batch: filename}):
                cy = cell.cycle
                validate_cycle_window(cy)
                summary = handle[reference_at(handle["batch"]["summary"], cell.cell_index)]
                extra = {name: vector_prefix(summary[name]) for name in ("IR", "Tavg", "Tmax", "chargetime")}
                if any(len(v) != 100 for v in extra.values()):
                    raise ValueError(f"{cell.cell_key}: summary coordinates are misaligned")
                qvalid = np.isfinite(cell.qd) & (cell.qd > 0) & (cell.qd <= 1.65)
                q = np.where(qvalid, cell.qd, np.nan)
                qref = window_median(cy, q, 2, 10, 5, 1.65)
                if not np.isfinite(qref):
                    raise ValueError(f"{cell.cell_key}: reference capacity is unavailable")
                dq = cell.q100 - cell.q10
                # Reconstruct all three approved features before using their saved values.
                reconstructed = [delta_q_log_variance(cell), rms_current(cell.current10, cell.time10)[0],
                                 relative_capacity_mad(cell)[0]]
                np.testing.assert_allclose(reconstructed, indexed.loc[cell.cell_key, FEATURES].to_numpy(float), rtol=1e-10, atol=1e-12)
                early = (cy >= 10) & (cy <= 100)
                late = (cy >= 60) & (cy <= 100)
                thermal = np.where(np.isfinite(extra["Tavg"]) & (extra["Tavg"] > 0), extra["Tavg"], np.nan)
                irref = window_median(cy, extra["IR"], 2, 10, 5)
                irend = window_median(cy, extra["IR"], 91, 100, 5)
                features = {
                    "capacity_q100_q10_change_relative": relative_endpoint_change(q[9], q[99]),
                    "capacity_theilsen_10_100_relative": robust_slope(cy[early], q[early], 50) / qref,
                    "capacity_theilsen_60_100_relative": robust_slope(cy[late], q[late], 30) / qref,
                    "dq100_10_q05_ah": float(np.quantile(dq, .05)),
                    "dq100_10_mean_v2p0_2p4_ah": float(dq[(cell.voltage >= 2) & (cell.voltage <= 2.4)].mean()),
                    "dq100_10_mean_ah": float(dq.mean()),
                    "tavg_theilsen_10_100_degC_per_cycle": robust_slope(cy[early], thermal[early], 50),
                    "ir_change_91_100_vs_2_10_relative": float(irend / irref - 1) if np.isfinite(irend) and np.isfinite(irref) else np.nan,
                    "tavg_median_2_100_degC": window_median(cy, extra["Tavg"], 2, 100, 50),
                    "chargetime_median_2_100_minutes": window_median(cy, extra["chargetime"], 2, 100, 50),
                }
                rows.append({"cell_key": cell.cell_key, "batch": batch, "policy": cell.policy,
                             "feature_latest_cycle": 100, **dict(zip(FEATURES, reconstructed)), **features})
                raw_summary_to_curve_ratio = float(cell.qd[99] / np.ptp(cell.q100))
                qc = {"cell_key": cell.cell_key, "batch": batch,
                      "qd_invalid_2_100_n": int((~qvalid[1:]).sum()), "q100_summary_ah": float(cell.qd[99]),
                      "q100_curve_range_ah": float(np.ptp(cell.q100)),
                      "q100_summary_to_curve_ratio": raw_summary_to_curve_ratio,
                      "discharge_summary_curve_mismatch": bool(raw_summary_to_curve_ratio > 1.5),
                      "capacity_slope_10_100_positive": bool(features["capacity_theilsen_10_100_relative"] > 0),
                      "capacity_slope_60_100_positive": bool(features["capacity_theilsen_60_100_relative"] > 0)}
                for name, v in extra.items():
                    v = v[1:]
                    valid = np.isfinite(v) & (v > 0)
                    positive = v[valid]
                    qc[f"{name.lower()}_valid_2_100_n"] = int(valid.sum())
                    qc[f"{name.lower()}_invalid_2_100_n"] = int((~valid).sum())
                    qc[f"{name.lower()}_constant_positive_flag"] = bool(len(positive) > 1 and np.ptp(positive) <= 1e-12)
                    qc[f"{name.lower()}_all_missing_flag"] = bool(not len(positive))
                    qc[f"{name.lower()}_raw_min"] = float(np.nanmin(v))
                    qc[f"{name.lower()}_raw_max"] = float(np.nanmax(v))
                qc["temperature_above_100degC_flag"] = bool((extra["Tavg"][1:] > 100).any() or (extra["Tmax"][1:] > 100).any())
                qc_rows.append(qc)
                reads.append({"cell_key": cell.cell_key, "summary_actual_cycles": "1..100",
                              "summary_fields": "cycle,QDischarge,IR,Tavg,Tmax,chargetime",
                              "curve_actual_cycles": "10,100", "current_time_actual_cycles": "10",
                              "largest_feature_cycle": 100, "whole_life_arrays_loaded": False})
    features, qc = pd.DataFrame(rows), pd.DataFrame(qc_rows)
    if not features.cell_key.is_unique or set(features.cell_key) != set(original.cell_key):
        raise ValueError("Feature cell keys disagree with the original table")
    return features, qc, reads


def guard_fold(groups, train, valid):
    if np.intersect1d(train, valid).size or set(groups[train]) & set(groups[valid]):
        raise ValueError("Protocol group leakage between training and validation")


def model(alpha):
    regression = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                           ("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha, solver="svd"))])
    return TransformedTargetRegressor(regressor=regression, func=np.log10, inverse_func=inverse_log10)


def cv_predictions(x, y, groups, folds, alpha):
    pred, coverage = np.full(len(y), np.nan), np.zeros(len(y), int)
    for train, valid in folds:
        guard_fold(groups, train, valid)
        estimator = model(alpha).fit(x.iloc[train], y[train])
        pred[valid] = estimator.predict(x.iloc[valid])
        coverage[valid] += 1
    if not np.all(coverage == 1):
        raise ValueError("Each development cell must be evaluated exactly once per repeat")
    return pred


def correlations(development):
    rows = []
    original_x = development[FEATURES]
    standardized = StandardScaler().fit_transform(original_x)
    target = np.log10(development.cycle_life.to_numpy(float))
    target_residual = target - LinearRegression().fit(standardized, target).predict(standardized)
    for candidate in CANDIDATES:
        x = development[candidate].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("B1 train candidate has missing values: explicit exploration policy required")
        regress = LinearRegression().fit(standardized, x)
        residual = x - regress.predict(standardized)
        r2 = regress.score(standardized, x)
        record = {"candidate": candidate, "n_B1_train": len(x),
                  "B1_train_pearson_life": development[candidate].corr(development.cycle_life),
                  "B1_train_pearson_log10_life": np.corrcoef(x, target)[0, 1],
                  "B1_train_spearman_life": development[candidate].corr(development.cycle_life, method="spearman"),
                  "B1_train_partial_corr_log10_life_after_three_features": np.corrcoef(residual, target_residual)[0, 1],
                  "B1_train_R2_explained_by_three_features": r2,
                  "B1_train_VIF_given_three_features": 1 / max(1 - r2, 1e-15)}
        for feature in FEATURES:
            record[f"pearson_with_{feature}"] = development[candidate].corr(development[feature])
            record[f"spearman_with_{feature}"] = development[candidate].corr(development[feature], method="spearman")
        rows.append(record)
    return pd.DataFrame(rows)


def run_cv(development, output):
    groups, y = development.policy.to_numpy(), development.cycle_life.to_numpy(float)
    variants = {f"baseline{n}": FEATURES[:n] for n in (2, 3)}
    variants.update({f"baseline{n}+{candidate}": FEATURES[:n] + [candidate]
                     for n in (2, 3) for candidate in CANDIDATES})
    fixed_folds = list(GroupKFold(5).split(development, groups=groups))
    fixed = []
    for name, selected in variants.items():
        pred = cv_predictions(development[selected], y, groups, fixed_folds, alpha=1)
        fixed.append({"variant": name, "features": ";".join(selected), "alpha": 1, **metrics(y, pred)})
    fixed = pd.DataFrame(fixed)
    for n in (2, 3):
        base = fixed.loc[fixed.variant.eq(f"baseline{n}"), "MAPE_percent"].item()
        mask = fixed.variant.str.startswith(f"baseline{n}")
        fixed.loc[mask, "delta_MAPE_vs_baseline_pp"] = fixed.loc[mask, "MAPE_percent"] - base
    fixed.to_csv(output / "fixed_alpha1_group_cv.csv", index=False)
    predictions, fold_rows, inner_rows, split_rows = [], [], [], []
    for repeat, seed in enumerate(SEEDS, 1):
        outer = list(GroupKFold(5, shuffle=True, random_state=seed).split(development, groups=groups))
        for fold, (train, valid) in enumerate(outer, 1):
            guard_fold(groups, train, valid)
            dx, dy, dg = development.iloc[train].reset_index(drop=True), y[train], groups[train]
            inner = list(GroupKFold(3).split(dx, groups=dg))
            choices = {}
            for name, selected in variants.items():
                candidates = []
                for alpha in ALPHAS:
                    inner_pred = cv_predictions(dx[selected], dy, dg, inner, alpha)
                    score = metrics(dy, inner_pred)["MAPE_percent"]
                    candidates.append((score, -alpha, alpha))
                    inner_rows.append({"repeat": repeat, "seed": seed, "outer_fold": fold,
                                       "variant": name, "alpha": alpha, "inner_MAPE_percent": score})
                best_score, _, alpha = min(candidates)
                choices[name] = (best_score, len(selected), -alpha, name, alpha)
                estimator = model(alpha).fit(dx[selected], dy)
                pred = estimator.predict(development.iloc[valid][selected])
                fold_rows.append({"repeat": repeat, "seed": seed, "outer_fold": fold, "variant": name,
                                  "alpha_selected_in_inner_cv": alpha, "n_train": len(train), "n_valid": len(valid), **metrics(y[valid], pred)})
                predictions.extend({"repeat": repeat, "seed": seed, "outer_fold": fold, "variant": name,
                                    "cell_key": development.iloc[i].cell_key, "actual_cycle_life": y[i],
                                    "predicted_cycle_life": p, "selected_variant": name,
                                    "alpha_selected_in_inner_cv": alpha} for i, p in zip(valid, pred))
            # Candidate selection itself is also confined to each outer training fold.
            for n in (2, 3):
                chosen = min(choice for name, choice in choices.items() if name.startswith(f"baseline{n}"))
                _, _, _, selected_name, alpha = chosen
                source = [p for p in predictions if p["repeat"] == repeat and p["outer_fold"] == fold and p["variant"] == selected_name]
                selected_predictions = np.array([p["predicted_cycle_life"] for p in source])
                fold_rows.append({"repeat": repeat, "seed": seed, "outer_fold": fold,
                                  "variant": f"baseline{n}_inner_select_candidate", "selected_variant": selected_name,
                                  "alpha_selected_in_inner_cv": alpha, "n_train": len(train), "n_valid": len(valid), **metrics(y[valid], selected_predictions)})
                predictions.extend({**p, "variant": f"baseline{n}_inner_select_candidate"} for p in source)
            for role, indices in (("train", train), ("valid", valid)):
                split_rows.extend({"repeat": repeat, "seed": seed, "outer_fold": fold, "role": role,
                                   "cell_key": development.iloc[i].cell_key, "policy": groups[i]} for i in indices)
        print(f"Candidate research: nested repeat{repeat}/{len(SEEDS)} complete", flush=True)
    pred_frame = pd.DataFrame(predictions)
    summaries = []
    for name, part in pred_frame.groupby("variant", sort=False):
        for repeat, repeated in part.groupby("repeat"):
            if len(repeated) != 35 or not repeated.cell_key.is_unique:
                raise ValueError("Unexpected nested OOF coverage or sample count")
            summaries.append({"variant": name, "repeat": repeat, **metrics(repeated.actual_cycle_life, repeated.predicted_cycle_life)})
    by_repeat = pd.DataFrame(summaries)
    base = by_repeat.loc[by_repeat.variant.isin(["baseline2", "baseline3"])].set_index(["variant", "repeat"])
    by_repeat["delta_MAPE_vs_baseline_pp"] = [r.MAPE_percent - base.loc[(r.variant.split("+")[0].split("_inner")[0], r.repeat), "MAPE_percent"] for r in by_repeat.itertuples()]
    summary = by_repeat.groupby("variant", sort=False).agg(MAPE_mean_percent=("MAPE_percent", "mean"), MAPE_repeat_sd=("MAPE_percent", "std"),
                           delta_MAPE_mean_pp=("delta_MAPE_vs_baseline_pp", "mean"), delta_MAPE_repeat_sd=("delta_MAPE_vs_baseline_pp", "std"),
                           MAE_mean_cycles=("MAE_cycles", "mean"), RMSE_mean_cycles=("RMSE_cycles", "mean")).reset_index()
    pred_frame.to_csv(output / "nested_oof_predictions_B1_train_only.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "nested_outer_fold_results.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(output / "nested_inner_search.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "nested_split_manifest.csv", index=False)
    by_repeat.to_csv(output / "nested_repeat_results.csv", index=False)
    summary.to_csv(output / "nested_summary.csv", index=False)
    return fixed, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT.parent / "archive")
    parser.add_argument("--output", type=Path, default=ROOT / "results/runs/feature_candidates_20261002")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "features.csv").exists():
        raise FileExistsError("Existing candidate research outputs are preserved; choose a new output directory")
    started = time.perf_counter()
    config = json.loads((ROOT / "config/model.json").read_text())
    protected = [ROOT / p for p in ("config/model.json", "data/processed/cell_features.csv", "results/split_manifest.csv", "results/cv_split_manifest.csv", "models/metadata.json", "models/ridge.joblib", "results/model_performance.csv", "results/predictions.csv")]
    before = {str(p.relative_to(ROOT)): file_sha256(p) for p in protected if p.exists()}
    raw_stat = {b: {"filename": f, "bytes": (args.raw_dir / f).stat().st_size, "mtime_ns": (args.raw_dir / f).stat().st_mtime_ns}
                for b, f in config["batches"].items()}
    original = pd.read_csv(ROOT / "data/processed/cell_features.csv", float_precision="round_trip")
    manifest = pd.read_csv(ROOT / "results/split_manifest.csv")
    train_manifest = manifest.loc[manifest.role.eq("train")].copy()
    if len(train_manifest) != 35 or not train_manifest.batch.eq("B1").all() or not train_manifest.cell_key.is_unique:
        raise ValueError("Only fixed B1 development-train35 may enter this experiment")
    heldout = manifest.loc[manifest.role.eq("valid")]
    if set(train_manifest.policy) & set(heldout.policy):
        raise ValueError("Fixed holdout protocols overlap development")
    features, qc, read_audit = extract(args.raw_dir, config, original)
    features.to_csv(args.output / "features.csv", index=False)  # No target column, including B2/B3.
    qc.to_csv(args.output / "candidate_qc.csv", index=False)
    (args.output / "formulas.json").write_text(json.dumps(CANDIDATES, indent=2))
    (args.output / "read_window_audit.json").write_text(json.dumps(read_audit, indent=2))
    availability = []
    for batch, part in features.groupby("batch"):
        for name in CANDIDATES:
            values = part[name]
            availability.append({"batch": batch, "candidate": name, "n_total": len(part), "n_finite": int(np.isfinite(values).sum()),
                                 "n_missing": int(values.isna().sum()), "n_positive": int((values > 0).sum()),
                                 "median": values.median(), "q25": values.quantile(.25), "q75": values.quantile(.75), "min": values.min(), "max": values.max()})
    pd.DataFrame(availability).to_csv(args.output / "availability_distribution_no_external_targets.csv", index=False)
    development = train_manifest[["cell_key", "policy"]].merge(features.drop(columns="policy"), on="cell_key", validate="one_to_one")
    development = development.merge(original.loc[original.cell_key.isin(train_manifest.cell_key), ["cell_key", "cycle_life"]], on="cell_key", validate="one_to_one")
    if len(development) != 35 or not development.batch.eq("B1").all():
        raise ValueError("Development guard failed after feature extraction")
    train_manifest.to_csv(args.output / "development_manifest.csv", index=False)
    corr = correlations(development)
    corr.to_csv(args.output / "B1_train_correlations_and_multicollinearity.csv", index=False)
    development[FEATURES + list(CANDIDATES)].corr().to_csv(args.output / "B1_train_feature_correlation_matrix.csv")
    fixed, nested = run_cv(development, args.output)
    rank = nested.loc[nested.variant.str.startswith("baseline2+")].copy()
    rank["candidate"] = rank.variant.str.removeprefix("baseline2+")
    rank = rank.merge(corr, on="candidate").merge(fixed.loc[fixed.variant.str.startswith("baseline2+"), ["variant", "MAPE_percent", "delta_MAPE_vs_baseline_pp"]], on="variant")
    rank.sort_values("delta_MAPE_mean_pp").to_csv(args.output / "candidate_evidence_ranking.csv", index=False)
    after = {str(p.relative_to(ROOT)): file_sha256(p) for p in protected if p.exists()}
    if after != before:
        raise RuntimeError("An approved artifact changed during the research run")
    for batch, raw in raw_stat.items():
        stat = (args.raw_dir / raw["filename"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (raw["bytes"], raw["mtime_ns"]):
            raise RuntimeError("Raw source metadata changed during read-only extraction")
    metadata = {"status": "exploratory_candidates_not_adopted", "scope": "B1 fixed development-train35 only for labels/correlation/CV; holdout/B2/B3 predictions never computed",
                "candidate_count": len(CANDIDATES), "max_feature_cycle": 100, "target": "unchanged provided cycle_life", "target_transform": "log10",
                "outer_cv": "Repeated shuffled GroupKFold by original policy;3 repeats x5 folds", "seeds": SEEDS,
                "inner_cv": "3-fold GroupKFold by original policy within each outer training fold", "alpha_grid": ALPHAS,
                "preprocessing": "per-cell physical early-only extraction; all imputation, scaling and alpha/candidate selection fitted inside appropriate CV training folds",
                "fixed_reference": "original deterministic5-fold GroupKFold; Ridge alpha1; baseline2 andbaseline3 add-one",
                "candidate_selection_ranking": "Individual add-one outer-CV results are exploratory rankings; choosing the best outer-CV candidate reuses development evidence. Inner-select-candidate rows estimate the complete selection process.",
                "source_read_only_metadata": raw_stat, "preserved_original_artifact_sha256": after,
                "B2_B3_labels_used": False, "n_development": 35, "n_development_policy_groups": development.policy.nunique(),
                "elapsed_seconds": time.perf_counter() - started, "environment": runtime_environment(),
                "limitations": ["Small training sample; repeated-CV standard deviations are dependent, not confidence intervals", "Source label-ending rules and unfinished B1 records remain unchanged", "B2/B3 prior EDA exposure prevents treating future external results as fully pristine", "No candidate has established cross-batch generalization", "Constant/missing IR and anomalous temperature acquisition are flagged; no external label is repaired"]}
    (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2))
    print(rank.sort_values("delta_MAPE_mean_pp")[["candidate", "MAPE_mean_percent", "delta_MAPE_mean_pp", "MAPE_percent", "B1_train_VIF_given_three_features"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
