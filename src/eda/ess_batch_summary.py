#!/usr/bin/env python3
"""Selective, cell-level EDA of three explicitly selected ESS batches; no model fitting.

Run from the project root:
  python -m src.pipeline eda --raw-dir ../archive

Raw summary values are retained. QC masks are explicit and never infer missing
life from observation length. Late/full-curve features are descriptive, not
early-life predictors. See methodology.json for all thresholds and definitions.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ess_batch_mpl"))
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BATCH_FILES = {
    "B1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "B2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "B3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
EOL = 0.88
NEAR_EOL = 0.89  # diagnostic last-record tolerance; does not prove EOL completion
QD_HIGH_QC = 1.65  # 1.5 times nominal; flag, not a physical exclusion rule
END_JUMP_QC = 0.10
CHARGE_TIME_HIGH_QC = 60.0
COLORS = {"B1": "#406b8c", "B2": "#c67543", "B3": "#52896f"}
EARLY_FEATURES = [
    "qd_early_mean", "qd_early_std", "qd_early_median", "qd2", "qd100",
    "qd_delta_100_2", "qd_early_slope", "ir_early_mean", "ir_change_100_2",
    "ir_early_slope", "ir_zero_fraction_early", "tavg_early_mean",
    "ir_missing_fraction_early",
    "tmax_early_mean", "chargetime_early_mean",
    "chargetime_early_median", "chargetime_early_mean_qc",
    "qd_early_mean_qc", "qd_early_std_qc", "qd_early_slope_qc",
    "qd2_qc", "qd100_qc", "qd_delta_100_2_qc",
]
POLICY_FEATURES = ["c1", "switch_soc_pct", "c2", "c2_active", "stage2_soc_span_pct", "c_max", "c_eff_0_80"]
CORR_FEATURES = EARLY_FEATURES + POLICY_FEATURES


def finite_mean(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def slope(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 3 or np.ptp(x) == 0:
        return np.nan
    return float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))


def at_cycle(cycle, values, target):
    vals = values[np.isclose(cycle, target)]
    vals = vals[np.isfinite(vals)]
    return float(vals[-1]) if len(vals) else np.nan


def decode_text(dataset):
    value = dataset[()]
    if value.dtype.kind in "ui":
        return "".join(chr(int(c)) for c in value.ravel() if c)
    return str(value.item()) if value.size == 1 else str(value)


def parse_policy(text):
    canonical = re.sub(r"-newstructure$", "", text)
    result = {"policy_canonical": canonical, "newstructure_suffix": text.endswith("-newstructure"),
              "policy_type": "variable" if "varcharge" in text.lower() else "unparsed"}
    if "slowcycle" in text.lower():
        result["policy_type"] = "slow_cycle_tagged"
    result.update({k: np.nan for k in POLICY_FEATURES + ["ideal_cc_time_0_80_min", "variable_tag_c"]})
    match = re.fullmatch(r"([\d.]+)C\(([\d.]+)%\)-([\d.]+)C", canonical)
    if match:
        c1, switch, c2 = map(float, match.groups())
        result.update(policy_type="constant_two_step", c1=c1, switch_soc_pct=switch,
                      c2=c2, c2_active=c2 if switch < 80 else np.nan,
                      stage2_soc_span_pct=max(0, 80 - switch), c_max=max(c1, c2))
        if c1 > 0 and c2 > 0 and 0 <= switch <= 80:
            hours = switch / 100 / c1 + (0.8 - switch / 100) / c2
            result.update(c_eff_0_80=0.8 / hours, ideal_cc_time_0_80_min=60 * hours)
    elif result["policy_type"] == "variable":
        tag = re.search(r"VarCharge-([\d.]+)C", text, re.I)
        if tag:
            result["variable_tag_c"] = float(tag.group(1))
    return result


def extract_cell(f, batch, idx, filename):
    b = f["batch"]
    refs = {k: b[k][idx, 0] for k in ["summary", "cycles", "cycle_life", "policy_readable"]}
    summary = f[refs["summary"]]
    raw = {k: np.asarray(summary[k][()], float).ravel() for k in summary.keys()}
    lengths = {k: len(v) for k, v in raw.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{batch}c{idx}: inconsistent summary lengths {lengths}")
    cycle, qd_raw = raw["cycle"], raw["QDischarge"]
    life = float(f[refs["cycle_life"]][()].ravel()[0])
    life = life if np.isfinite(life) and life > 0 else np.nan
    policy = decode_text(f[refs["policy_readable"]])
    # Inspect reference-array shapes, never cycle-level data or Qdlin values.
    cycles_group = f[refs["cycles"]]
    cycle_counts = {k: int(np.prod(v.shape)) for k, v in cycles_group.items()}
    n_cycles = cycle_counts.get("I", max(cycle_counts.values()))
    placeholder = bool(len(cycle) and all(raw[k][0] == 0 for k in
                        ["QDischarge", "QCharge", "IR", "Tavg", "Tmax", "Tmin", "chargetime"]))
    usable = np.isfinite(cycle)
    if placeholder:
        usable[0] = False
    qd = np.where(usable & np.isfinite(qd_raw) & (qd_raw > 0), qd_raw, np.nan)
    ir_raw = raw["IR"]
    ir = np.where(usable & np.isfinite(ir_raw) & (ir_raw > 0), ir_raw, np.nan)
    early = usable & (cycle >= 2) & (cycle <= 100)
    observed_last = float(np.nanmax(cycle))
    eligible = bool(observed_last >= 100 and (not np.isfinite(life) or life >= 100))
    tail = float(qd_raw[-1])
    last_valid_idx = np.flatnonzero(np.isfinite(qd))
    qd_last_valid = float(qd[last_valid_idx[-1]]) if len(last_valid_idx) else np.nan
    prev = qd[-6:-1]
    prev = prev[np.isfinite(prev)]
    endpoint_invalid = not np.isfinite(tail) or tail <= 0 or tail > QD_HIGH_QC
    endpoint_jump = bool(np.isfinite(tail) and len(prev) and abs(tail - np.median(prev)) > END_JUMP_QC)
    if endpoint_invalid:
        endpoint_status = "abnormal_endpoint"
    elif tail <= EOL:
        endpoint_status = "at_or_below_eol"
    elif tail <= NEAR_EOL:
        endpoint_status = "near_eol_above"
    else:
        endpoint_status = "above_eol"
    high = usable & np.isfinite(qd_raw) & (qd_raw > QD_HIGH_QC)
    positive_qd = np.isfinite(qd)
    below = positive_qd & (qd <= EOL)
    clean_descriptive = np.where(high, np.nan, qd)
    late50 = usable & (cycle > observed_last - 50)
    early_n = int(early.sum())
    row = {
        "batch": batch, "filename": filename, "batch_date": filename[:10],
        "cell_index": idx, "cell_key": f"{batch}c{idx}", "cycle_life": life,
        "life_label_missing": not np.isfinite(life), "policy": policy,
        "source_collection_issue_index_candidate": batch == "B3" and idx == 37,
        "n_summary_rows": len(cycle), "n_cycles_records": n_cycles,
        "cycles_fields_count_consistent": len(set(cycle_counts.values())) == 1,
        "summary_cycles_count_match": len(cycle) == n_cycles,
        "summary_cycle_first": float(cycle[0]), "summary_cycle_last": observed_last,
        "summary_cycle_step_not_one_n": int((np.diff(cycle) != 1).sum()),
        "first_all_zero_placeholder": placeholder,
        "n_observed_nonplaceholder": int(usable.sum()), "observed_reaches_100": observed_last >= 100,
        "label_ends_before_100": np.isfinite(life) and life < 100,
        "early100_eligible": eligible, "early_window_rows_observed": early_n,
        "label_minus_last_cycle": life - observed_last if np.isfinite(life) else np.nan,
        "qd_endpoint_raw": tail, "qd_last_positive": qd_last_valid,
        "endpoint_status": endpoint_status, "endpoint_jump_gt_0_10": endpoint_jump,
        "qc_eol_compatible": bool(not endpoint_invalid and not endpoint_jump and tail <= NEAR_EOL),
        "qc_qd_high_n": int(high.sum()), "qc_qd_high_early_n": int((high & early).sum()),
        "qc_qd_nonpositive_n": int((usable & np.isfinite(qd_raw) & (qd_raw <= 0)).sum()),
        "qc_qd_nonfinite_n": int((usable & ~np.isfinite(qd_raw)).sum()),
        "qd_cycle100_high_qc": bool(np.isfinite(at_cycle(cycle, qd, 100)) and at_cycle(cycle, qd, 100) > QD_HIGH_QC),
        "qc_qd_valid_total": int(np.isfinite(clean_descriptive).sum()),
        "qd_below_eol_n": int(below.sum()),
        "qd_first_below_eol_cycle": float(cycle[np.flatnonzero(below)[0]]) if below.any() else np.nan,
        "eol_threshold_crossing_observed": bool(below.any()),
        "label_matches_first_observed_crossing": bool(below.any() and np.isfinite(life) and life == cycle[np.flatnonzero(below)[0]]),
        "life_label_beyond_observation": bool(np.isfinite(life) and life > observed_last),
        "qd_full_mean_raw_positive": finite_mean(qd),
        "qd_full_slope_raw_positive": slope(cycle, qd),
        "qd_full_slope_qc": slope(cycle, clean_descriptive),
        "qd_late50_slope_qc": slope(cycle[late50], clean_descriptive[late50]),
        "n_qd_full_slope_qc": int(np.isfinite(clean_descriptive).sum()),
        "n_qd_late50_slope_qc": int(np.isfinite(clean_descriptive[late50]).sum()),
        "ir_zero_fraction_full": float((usable & np.isfinite(ir_raw) & (ir_raw == 0)).sum() / max(1, usable.sum())),
        "ir_missing_fraction_full": float((usable & (~np.isfinite(ir_raw) | (ir_raw <= 0))).sum() / max(1, usable.sum())),
        "charge_time_gt60_full_n": int((usable & (raw["chargetime"] > CHARGE_TIME_HIGH_QC)).sum()),
        "charge_time_gt60_early_n": int((early & (raw["chargetime"] > CHARGE_TIME_HIGH_QC)).sum()),
    }
    row.update(parse_policy(policy))
    row["qc_eol_compatible_no_qd_high"] = row["qc_eol_compatible"] and not high.any()
    # All 100-cycle feature values are missing for observations/labels ending before 100.
    row.update({name: np.nan for name in EARLY_FEATURES})
    qvals = qd[early]
    qvals = qvals[np.isfinite(qvals)]
    irvals = ir[early]
    row["n_qd_early_valid"] = len(qvals) if eligible else 0
    row["n_ir_early_valid"] = int(np.isfinite(irvals).sum()) if eligible else 0
    row["n_qd_early_qc_valid"] = int(np.isfinite(clean_descriptive[early]).sum()) if eligible else 0
    if eligible:
        row.update(qd_early_mean=finite_mean(qvals),
                   qd_early_std=float(np.std(qvals, ddof=1)) if len(qvals) > 1 else np.nan,
                   qd_early_median=float(np.median(qvals)) if len(qvals) else np.nan,
                   qd2=at_cycle(cycle, qd, 2), qd100=at_cycle(cycle, qd, 100),
                   qd_early_slope=slope(cycle[early], qd[early]),
                   ir_early_mean=finite_mean(irvals),
                   ir_change_100_2=at_cycle(cycle, ir, 100) - at_cycle(cycle, ir, 2),
                   ir_early_slope=slope(cycle[early], irvals),
                   ir_missing_fraction_early=float((~np.isfinite(ir_raw[early]) | (ir_raw[early] <= 0)).sum() / max(1, early_n)),
                   ir_zero_fraction_early=float((np.isfinite(ir_raw[early]) & (ir_raw[early] == 0)).sum() / max(1, early_n)))
        row["qd_delta_100_2"] = row["qd100"] - row["qd2"]
        qc_vals = clean_descriptive[early]
        qc_vals = qc_vals[np.isfinite(qc_vals)]
        row.update(qd_early_mean_qc=finite_mean(qc_vals),
                   qd_early_std_qc=float(np.std(qc_vals, ddof=1)) if len(qc_vals) > 1 else np.nan,
                   qd_early_slope_qc=slope(cycle[early], clean_descriptive[early]),
                   qd2_qc=at_cycle(cycle, clean_descriptive, 2), qd100_qc=at_cycle(cycle, clean_descriptive, 100))
        row["qd_delta_100_2_qc"] = row["qd100_qc"] - row["qd2_qc"]
    for raw_name, feature in [("Tavg", "tavg_early_mean"), ("Tmax", "tmax_early_mean"),
                              ("chargetime", "chargetime_early_mean")]:
        # Nonpositive temperatures/time in this 30C experiment are flagged missing.
        val = raw[raw_name][early]
        valid = np.isfinite(val) & (val > 0)
        row["n_" + feature + "_valid"] = int(valid.sum()) if eligible else 0
        if eligible:
            row[feature] = finite_mean(val[valid])
            if raw_name == "chargetime":
                row["chargetime_early_median"] = float(np.median(val[valid])) if valid.any() else np.nan
                qc_valid = valid & (val <= CHARGE_TIME_HIGH_QC)
                row["chargetime_early_mean_qc"] = finite_mean(val[qc_valid])
                row["n_chargetime_early_qc_valid"] = int(qc_valid.sum())
    row.setdefault("n_chargetime_early_qc_valid", 0)
    return row, {"cycle": cycle, "qd": qd, "qd_raw": qd_raw, "ir": ir, "raw": raw,
                 "usable": usable, "life": life, "key": row["cell_key"], "policy": policy}


def batch_statistics(features):
    rows = []
    for batch in ["ALL", *BATCH_FILES]:
        d = features if batch == "ALL" else features[features.batch == batch]
        life = d.cycle_life.dropna()
        q1, q3 = life.quantile([.25, .75]) if len(life) else (np.nan, np.nan)
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        row = {"batch": batch, "n_cells": len(d), "n_life_valid": len(life), "n_life_missing": len(d) - len(life),
               "life_mean": life.mean(), "life_median": life.median(), "life_std": life.std(),
               "life_min": life.min(), "life_q1": q1, "life_q3": q3, "life_max": life.max(),
               "life_skew": life.skew() if len(life) >= 3 else np.nan,
               "iqr_lower": low, "iqr_upper": high,
               "n_iqr_outlier_low": int((life < low).sum()), "n_iqr_outlier_high": int((life > high).sum()),
               "n_short_lt500": int((life < 500).sum()), "n_mid_500_1000": int(life.between(500, 1000).sum()),
               "n_long_gt1000": int((life > 1000).sum()),
               "short_fraction_labeled": float((life < 500).mean()) if len(life) else np.nan,
               "long_fraction_labeled": float((life > 1000).mean()) if len(life) else np.nan,
               "n_eol_compatible": int(d.qc_eol_compatible.sum()),
               "n_eol_compatible_labeled": int((d.qc_eol_compatible & d.cycle_life.notna()).sum()),
               "n_early100_eligible": int(d.early100_eligible.sum()),
               "n_observation_under100": int((~d.observed_reaches_100).sum()),
               "n_label_ends_before100": int(d.label_ends_before_100.sum()),
               "n_placeholder": int(d.first_all_zero_placeholder.sum()),
               "n_qd_high_cells": int((d.qc_qd_high_n > 0).sum()),
               "n_ir_all_missing": int((d.ir_missing_fraction_full == 1).sum()),
               "n_unique_policies": d.policy.nunique(), "n_canonical_policies": d.policy_canonical.nunique()}
        for status in ["at_or_below_eol", "near_eol_above", "above_eol", "abnormal_endpoint"]:
            row["n_endpoint_" + status] = int((d.endpoint_status == status).sum())
        for feat in ["summary_cycle_last", "qd_endpoint_raw", "qd_early_mean", "qd_early_std",
                     "ir_early_mean", "tavg_early_mean", "tmax_early_mean", "chargetime_early_mean",
                     "qd_early_slope", "qd_late50_slope_qc"]:
            row[feat + "_median"] = d[feat].median()
            row[feat + "_n"] = int(d[feat].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_tables(features):
    rows = []
    subsets = {"all_labeled": features.cycle_life.notna(),
               "eol_compatible": features.cycle_life.notna() & features.qc_eol_compatible,
               "eol_compatible_no_qd_high": features.cycle_life.notna() & features.qc_eol_compatible_no_qd_high,
               "all_labeled_excl_source_issue": features.cycle_life.notna() & ~features.source_collection_issue_index_candidate,
               "eol_compatible_excl_source_issue": features.cycle_life.notna() & features.qc_eol_compatible & ~features.source_collection_issue_index_candidate}
    for subset, selection in subsets.items():
        for batch in ["ALL", *BATCH_FILES]:
            d = features[selection] if batch == "ALL" else features[selection & (features.batch == batch)]
            for feat in CORR_FEATURES:
                paired = d[["batch", feat, "cycle_life"]].dropna().copy()
                for method in ["pearson", "spearman"]:
                    n = len(paired)
                    valid = n >= 3 and paired[feat].nunique() > 1 and paired.cycle_life.nunique() > 1
                    r, p = (np.nan, np.nan)
                    if valid:
                        fn = stats.pearsonr if method == "pearson" else stats.spearmanr
                        r, p = fn(paired[feat], paired.cycle_life)
                    rows.append({"subset": subset, "scope": batch, "adjustment": "none", "method": method,
                                 "feature": feat, "n": n, "n_batches": paired.batch.nunique(),
                                 "coefficient": r, "p_unadjusted_exploratory": p})
                if batch == "ALL":
                    # Center within each batch using only the valid x/y pairs.
                    groups = paired.groupby("batch")
                    x = paired[feat] - groups[feat].transform("mean")
                    y = paired.cycle_life - groups.cycle_life.transform("mean")
                    r, p = (np.nan, np.nan)
                    if len(paired) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                        r = float(stats.pearsonr(x, y).statistic)
                        # Centering estimates batch intercepts; naive Pearson n-2
                        # p is inapplicable. Only the descriptive coefficient is saved.
                        p = np.nan
                    rows.append({"subset": subset, "scope": "ALL", "adjustment": "within_batch_mean_centered",
                                 "method": "pearson", "feature": feat, "n": len(paired),
                                 "n_batches": paired.batch.nunique(), "coefficient": r,
                                 "p_unadjusted_exploratory": p})
                    stratum = paired["batch"] + "|newstructure=" + features.loc[paired.index, "newstructure_suffix"].astype(str)
                    x = paired[feat] - paired[feat].groupby(stratum).transform("mean")
                    y = paired.cycle_life - paired.cycle_life.groupby(stratum).transform("mean")
                    r = float(stats.pearsonr(x, y).statistic) if len(paired) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan
                    rows.append({"subset": subset, "scope": "ALL", "adjustment": "within_batch_structure_centered",
                                 "method": "pearson", "feature": feat, "n": len(paired),
                                 "n_batches": paired.batch.nunique(), "coefficient": r,
                                 "p_unadjusted_exploratory": np.nan})
    return pd.DataFrame(rows)


def feature_associations(features, x_features, y_features, subsets):
    """Pairwise feature associations, including n and within-batch centering."""
    rows = []
    for subset, selection in subsets.items():
        for scope in ["ALL", *BATCH_FILES]:
            d = features[selection] if scope == "ALL" else features[selection & (features.batch == scope)]
            for xname in x_features:
                for yname in y_features:
                    if xname == yname:
                        continue
                    p = d[["batch", xname, yname]].dropna()
                    for method in ["pearson", "spearman"]:
                        r = np.nan
                        if len(p) >= 3 and p[xname].nunique() > 1 and p[yname].nunique() > 1:
                            fn = stats.pearsonr if method == "pearson" else stats.spearmanr
                            r = float(fn(p[xname], p[yname]).statistic)
                        rows.append({"subset": subset, "scope": scope, "adjustment": "none",
                                     "method": method, "feature_x": xname, "feature_y": yname,
                                     "n": len(p), "coefficient": r})
                    if scope == "ALL":
                        x = p[xname] - p.groupby("batch")[xname].transform("mean")
                        y = p[yname] - p.groupby("batch")[yname].transform("mean")
                        r = float(stats.pearsonr(x, y).statistic) if len(p) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan
                        rows.append({"subset": subset, "scope": scope, "adjustment": "within_batch_mean_centered",
                                     "method": "pearson", "feature_x": xname, "feature_y": yname,
                                     "n": len(p), "coefficient": r})
    return pd.DataFrame(rows)


def multicollinearity_outputs(features, out):
    # Descriptive subset declarations distinguish 139 observations from 129 labeled targets.
    core = ["qd_early_mean", "qd_early_std", "qd2", "qd100", "qd_delta_100_2", "qd_early_slope",
            "ir_early_mean", "ir_change_100_2", "ir_early_slope", "tavg_early_mean", "tmax_early_mean",
            "chargetime_early_mean"]
    subsets = {"all_observed": pd.Series(True, index=features.index),
               "all_labeled": features.cycle_life.notna(),
               "eol_compatible": features.cycle_life.notna() & features.qc_eol_compatible,
               "all_labeled_excl_source_issue": features.cycle_life.notna() & ~features.source_collection_issue_index_candidate}
    result = feature_associations(features, core, core, subsets)
    result = result[result.feature_x < result.feature_y]
    result.to_csv(out / "feature_correlation_tables.csv", index=False)
    result[(result.method == "pearson") & (result.coefficient.abs() > .85)].to_csv(out / "high_feature_correlations.csv", index=False)
    extended = feature_associations(features, EARLY_FEATURES, EARLY_FEATURES, subsets)
    extended = extended[extended.feature_x < extended.feature_y]
    extended.to_csv(out / "feature_correlation_tables_all_candidates.csv", index=False)
    extended[(extended.method == "pearson") & (extended.coefficient.abs() > .85)].to_csv(out / "high_feature_correlations_all_candidates.csv", index=False)
    labeled = features[features.cycle_life.notna()]
    matrix = labeled[core].corr()
    matrix.to_csv(out / "feature_correlation_matrix_pearson.csv")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(core)), core, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(core)), core, fontsize=8)
    for i, xname in enumerate(core):
        for j, yname in enumerate(core):
            r = matrix.iloc[i, j]
            n = labeled[[xname, yname]].notna().all(axis=1).sum()
            ax.text(j, i, f"{r:.2f}\n{n}", fontsize=7, ha="center", va="center", color="white" if abs(r) > .55 else "black")
    ax.set_title("Early-summary feature Pearson matrix: coefficient / pairwise n (labeled subset)")
    fig.colorbar(im, ax=ax, shrink=.7)
    fig.tight_layout()
    fig.savefig(out / "09_feature_multicollinearity.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    vif_rows = []
    bases = {"full_with_algebraic_redundancy": core,
             "without_qd2_qd100": [f for f in core if f not in ["qd2", "qd100"]],
             "robust_without_qd2_qd100": [
                 {"qd_early_mean": "qd_early_mean_qc", "qd_early_std": "qd_early_std_qc",
                  "qd_delta_100_2": "qd_delta_100_2_qc", "qd_early_slope": "qd_early_slope_qc",
                  "chargetime_early_mean": "chargetime_early_median"}.get(f, f)
                 for f in core if f not in ["qd2", "qd100"]]}
    for subset in ["all_labeled", "eol_compatible"]:
        for basis, names in bases.items():
            data = features.loc[subsets[subset], names].dropna()
            nonconstant = data.std() > 0
            data = data.loc[:, nonconstant]
            z = (data - data.mean()) / data.std(ddof=0)
            for feat in data:
                y, x = z[feat].to_numpy(), z.drop(columns=feat).to_numpy()
                fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
                r2 = 1 - np.sum((y - fitted) ** 2) / np.sum(y ** 2)
                vif = np.inf if 1 - r2 < 1e-12 else 1 / (1 - r2)
                vif_rows.append({"subset": subset, "basis": basis, "feature": feat, "n_complete_cases": len(data),
                                 "basis_n_features": len(data.columns), "basis_rank": np.linalg.matrix_rank(z.to_numpy()), "vif": vif})
    pd.DataFrame(vif_rows).to_csv(out / "vif_sensitivity.csv", index=False)


def policy_statistics(features):
    rows = []
    for scope in ["ALL", *BATCH_FILES]:
        d = features if scope == "ALL" else features[features.batch == scope]
        for policy, sub in d.groupby("policy", sort=True):
            row = {"scope": scope, "policy": policy, "policy_canonical": sub.policy_canonical.iloc[0],
                   "newstructure_suffix": bool(sub.newstructure_suffix.iloc[0]),
                   "policy_type": sub.policy_type.iloc[0], "batches": ",".join(sorted(sub.batch.unique())),
                   "raw_policy_variants": " | ".join(sorted(sub.policy.unique())), "n_cells": len(sub),
                   "n_life_valid": int(sub.cycle_life.notna().sum()), "life_mean": sub.cycle_life.mean(),
                   "life_std": sub.cycle_life.std(), "life_median": sub.cycle_life.median(),
                   "life_min": sub.cycle_life.min(), "life_max": sub.cycle_life.max(),
                   "n_eol_compatible_labeled": int((sub.qc_eol_compatible & sub.cycle_life.notna()).sum())}
            row.update({k: sub[k].iloc[0] for k in POLICY_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def save_plots(features, curves, corr, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=170, bbox_inches="tight")
        plt.close(fig)
    edges = np.arange(150, 2401, 100)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, batch in zip(axes, BATCH_FILES):
        d = features[features.batch == batch]
        ax.hist(d.cycle_life.dropna(), bins=edges, color=COLORS[batch], edgecolor="white")
        ax.axvline(500, color="#bb5544", linestyle="--", linewidth=1)
        ax.axvline(1000, color="#558866", linestyle="--", linewidth=1)
        ax.set(xlim=(150, 2300), xlabel="Stored cycle-life label", ylabel="Cells",
               title=f"{batch}: {d.cycle_life.notna().sum()} labeled / {len(d)} cells")
        if d.cycle_life.isna().all():
            ax.text(.5, .5, "No finite life labels\nObservation length is not substituted", ha="center", transform=ax.transAxes)
    save(fig, "01_life_histograms.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    valid_batches = [b for b in BATCH_FILES if features.loc[features.batch == b, "cycle_life"].notna().any()]
    axes[0].boxplot([features.loc[features.batch == b, "cycle_life"].dropna() for b in valid_batches], tick_labels=valid_batches)
    axes[0].set(ylabel="Stored life label", title="Life distribution by batch")
    for b in BATCH_FILES:
        d = features[features.batch == b]
        axes[1].scatter(d.chargetime_early_mean, d.cycle_life, color=COLORS[b], label=f"{b} (n={d.cycle_life.notna().sum()})", alpha=.7, s=25)
    axes[1].set(xlabel="Early 2-100 mean charge time (min)", ylabel="Stored life label", title="Pooled pattern can mix batch shifts")
    axes[1].legend(frameon=False)
    save(fig, "02_batch_life_charge_time.png")

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for r, b in enumerate(BATCH_FILES):
        for c in curves[b]:
            axes[r, 0].plot(c["cycle"], c["qd"], lw=.65, alpha=.45, color=COLORS[b])
            axes[r, 1].plot(c["cycle"], c["qd"], lw=.65, alpha=.45, color=COLORS[b])
        for ax in axes[r]:
            ax.axhline(EOL, color="#a64c3c", linestyle="--", lw=1)
            ax.set(xlabel="Actual summary cycle", ylabel="QD (Ah)")
        axes[r, 0].set_title(f"{b}: full observed range, raw positive QD")
        axes[r, 1].set(title=f"{b}: early 2-100 display zoom", xlim=(2, 100), ylim=(.8, 1.2))
    save(fig, "03_qd_full_and_early.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for b in BATCH_FILES:
        d = features[features.batch == b]
        axes[0].scatter(d.qd_early_slope * 1000, d.cycle_life, color=COLORS[b], label=b, alpha=.7, s=25)
        axes[1].scatter(d.qd_early_slope * 1000, d.qd_late50_slope_qc * 1000, color=COLORS[b], label=b, alpha=.7, s=25)
    axes[0].set(xlabel="Early QD slope (mAh / cycle)", ylabel="Stored life label", title="Early capacity change")
    axes[1].set(xlabel="Early QD slope (mAh / cycle)", ylabel="Last-50 QD slope (mAh / cycle)", title="Late slope is descriptive future information")
    axes[0].legend(frameon=False)
    save(fig, "04_degradation_slopes.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for b in BATCH_FILES:
        d = features[(features.batch == b) & (features.policy_type == "constant_two_step")]
        for ax, feat, label in zip(axes, ["c1", "switch_soc_pct", "c_eff_0_80"],
                                  ["First-stage C-rate", "SOC switch (%)", "Ideal effective C-rate 0-80%"]):
            ax.scatter(d[feat], d.cycle_life, color=COLORS[b], label=b, alpha=.7, s=25)
            ax.set(xlabel=label, ylabel="Stored life label")
    axes[0].legend(frameon=False)
    fig.suptitle("Policy associations; batch and newstructure differences are not controlled")
    save(fig, "05_policy_parameters.png")

    fixed = features[features.policy_type == "constant_two_step"]
    policy = fixed.groupby(["batch", "policy"]).cycle_life.agg(["mean", "std", "count"])
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    for ax, b in zip(axes, valid_batches):
        d = policy.loc[b].sort_values("mean")
        d = d[d["count"] > 0]
        ax.barh(np.arange(len(d)), d["mean"], xerr=d["std"].fillna(0), color=COLORS[b], alpha=.8)
        ax.set_yticks(np.arange(len(d)), [f"{p} (n={n})" for p, n in zip(d.index, d["count"])], fontsize=7)
        ax.set(xlabel="Stored life: mean +/- 1 sample SD (not CI)", title=b)
    save(fig, "06_policy_life_with_n.png")

    heat_features = ["qd_early_mean", "qd_early_std", "qd_delta_100_2", "qd_early_slope",
                     "ir_early_mean", "ir_change_100_2", "ir_early_slope",
                     "tavg_early_mean", "tmax_early_mean", "chargetime_early_mean", "c_eff_0_80"]
    columns = [("ALL", "all_labeled", "none"), ("B1", "all_labeled", "none"),
               ("B2", "all_labeled", "none"), ("B3", "all_labeled", "none"),
               ("ALL", "all_labeled", "within_batch_mean_centered"), ("ALL", "eol_compatible", "none")]
    vals = np.full((len(heat_features), len(columns)), np.nan)
    ns = np.zeros(vals.shape, int)
    for j, (scope, subset, adjustment) in enumerate(columns):
        d = corr[(corr.scope == scope) & (corr.subset == subset) & (corr.adjustment == adjustment) & (corr.method == "pearson")].set_index("feature")
        for i, feat in enumerate(heat_features):
            vals[i, j], ns[i, j] = d.loc[feat, "coefficient"], d.loc[feat, "n"]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(np.ma.masked_invalid(vals), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(columns)), ["Pooled", "B1", "B2", "B3", "Batch-centered", "EOL-compatible"])
    ax.set_yticks(range(len(heat_features)), heat_features)
    for i in range(len(heat_features)):
        for j in range(len(columns)):
            label = f"{vals[i,j]:.2f}\nn={ns[i,j]}" if np.isfinite(vals[i,j]) else f"NA\nn={ns[i,j]}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if abs(vals[i,j]) > .55 else "black")
    ax.set_title("Pearson association with stored life; pairwise n shown")
    fig.colorbar(im, ax=ax, shrink=.75)
    save(fig, "07_correlation_comparison.png")

    shift_features = ["qd_early_mean", "ir_early_mean", "tavg_early_mean", "chargetime_early_mean"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, feat in zip(axes, shift_features):
        ds = [features.loc[features.batch == b, feat].dropna() for b in BATCH_FILES]
        valid = [(b, d) for b, d in zip(BATCH_FILES, ds) if len(d)]
        ax.boxplot([d for b, d in valid], tick_labels=[f"{b}\nn={len(d)}" for b, d in valid])
        ax.set(title=feat)
    fig.suptitle("Early-feature batch shifts; file-local cells are not confirmed unique physical cells")
    save(fig, "08_early_feature_batch_shifts.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for b in BATCH_FILES:
        d = features[features.batch == b]
        for ax, feat in zip(axes, ["chargetime_early_mean", "chargetime_early_median"]):
            ax.scatter(d[feat], d.cycle_life, color=COLORS[b], alpha=.75, label=b, s=25)
    axes[0].set(xlabel="Raw-positive early mean time (min)", ylabel="Stored life label", title="Mean sensitive to long-time observations")
    axes[1].set(xlabel="Early median time (min)", ylabel="Stored life label", title="Robust comparison; batch/structure still confound")
    axes[0].legend(frameon=False)
    save(fig, "10_charge_time_sensitivity.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for b in BATCH_FILES:
        d = features[(features.batch == b) & (features.policy_type == "constant_two_step")]
        for r, feat in enumerate(["c_eff_0_80", "c2_active"]):
            for col, response in enumerate(["qd_early_slope_qc", "qd_late50_slope_qc"]):
                axes[r, col].scatter(d[feat], 1000 * d[response], color=COLORS[b], alpha=.7, s=25, label=b)
                axes[r, col].set(xlabel="Ideal effective C-rate 0-80%" if r == 0 else "C2 only when switch<80%",
                                 ylabel="Early QD slope (mAh/cycle)" if col == 0 else "Last-50 QD slope (mAh/cycle)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Policy and degradation slopes: associations, not causal estimates")
    save(fig, "11_policy_slope_associations.png")


def run(data_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, curves, manifest = [], {}, []
    for batch, filename in BATCH_FILES.items():
        path = data_dir / filename
        curves[batch] = []
        with h5py.File(path, "r") as f:
            n = len(f["batch/cycle_life"])
            for idx in range(n):
                row, curve = extract_cell(f, batch, idx, filename)
                rows.append(row)
                curves[batch].append(curve)
        manifest.append({"batch": batch, "file": filename, "bytes": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns, "n_cells": n})
        print(f"Loaded {batch}: {n} cells, summary values and cycles shapes only", flush=True)
    features = pd.DataFrame(rows)
    features.to_csv(out / "features.csv", index=False)
    batch_stats = batch_statistics(features)
    batch_stats.to_csv(out / "batch_stats.csv", index=False)
    corr = correlation_tables(features)
    corr.to_csv(out / "correlation_tables.csv", index=False)
    policy_statistics(features).to_csv(out / "policy_stats.csv", index=False)
    curve_subsets = {"all_fixed_observed": features.policy_type == "constant_two_step",
                     "eol_compatible_fixed": (features.policy_type == "constant_two_step") & features.qc_eol_compatible,
                     "all_fixed_excl_source_issue": (features.policy_type == "constant_two_step") & ~features.source_collection_issue_index_candidate,
                     "eol_fixed_excl_source_issue": (features.policy_type == "constant_two_step") & features.qc_eol_compatible & ~features.source_collection_issue_index_candidate}
    feature_associations(features, POLICY_FEATURES,
                         ["qd_early_slope", "qd_early_slope_qc", "qd_late50_slope_qc", "qd_full_slope_qc"],
                         curve_subsets).to_csv(out / "policy_slope_correlations.csv", index=False)
    multicollinearity_outputs(features, out)
    coverage = []
    for batch in ["ALL", *BATCH_FILES]:
        d = features if batch == "ALL" else features[features.batch == batch]
        for feat in CORR_FEATURES + ["qd_late50_slope_qc"]:
            coverage.append({"batch": batch, "feature": feat, "n_cells": len(d),
                             "n_feature_valid": int(d[feat].notna().sum()),
                             "n_feature_and_life_valid": int((d[feat].notna() & d.cycle_life.notna()).sum())})
    pd.DataFrame(coverage).to_csv(out / "feature_coverage.csv", index=False)
    structure_rows = []
    for (b, structure), d in features.groupby(["batch", "newstructure_suffix"]):
        labeled_d = d[d.cycle_life.notna()]
        row = {"batch": b, "newstructure_suffix": structure, "n_cells": len(d), "n_life_valid": len(labeled_d),
               "life_mean": labeled_d.cycle_life.mean(), "life_median": labeled_d.cycle_life.median(),
               "charge_median_feature_mean": labeled_d.chargetime_early_median.mean()}
        for feat in ["chargetime_early_mean", "chargetime_early_median", "chargetime_early_mean_qc", "c_eff_0_80"]:
            paired = labeled_d[[feat, "cycle_life"]].dropna()
            row[feat + "_n"] = len(paired)
            row[feat + "_pearson"] = paired[feat].corr(paired.cycle_life) if len(paired) >= 3 and paired[feat].nunique() > 1 else np.nan
            row[feat + "_spearman"] = paired[feat].corr(paired.cycle_life, method="spearman") if len(paired) >= 3 and paired[feat].nunique() > 1 else np.nan
        structure_rows.append(row)
    pd.DataFrame(structure_rows).to_csv(out / "batch_structure_stats.csv", index=False)
    overlaps = []
    for i, a in enumerate(BATCH_FILES):
        for b in list(BATCH_FILES)[i+1:]:
            sa = set(features.loc[features.batch == a, "policy_canonical"])
            sb = set(features.loc[features.batch == b, "policy_canonical"])
            common = sorted(sa & sb)
            overlaps.append({"batch_a": a, "batch_b": b, "n_policies_a": len(sa), "n_policies_b": len(sb),
                             "n_overlap": len(common), "jaccard": len(common) / len(sa | sb),
                             "shared_policy_text": " | ".join(common),
                             "caution": "Text overlap only; suffix stripped, cell structure/conditions not assumed identical"})
    pd.DataFrame(overlaps).to_csv(out / "policy_overlap.csv", index=False)
    labeled = features[features.cycle_life.notna()].copy()
    labeled["pooled_iqr_outlier"] = False
    bstats = batch_stats.set_index("batch")
    lo, hi = bstats.loc["ALL", ["iqr_lower", "iqr_upper"]]
    labeled["pooled_iqr_outlier"] = (labeled.cycle_life < lo) | (labeled.cycle_life > hi)
    labeled["within_batch_iqr_outlier"] = [life < bstats.loc[b, "iqr_lower"] or life > bstats.loc[b, "iqr_upper"]
                                           for b, life in zip(labeled.batch, labeled.cycle_life)]
    labeled[labeled.pooled_iqr_outlier | labeled.within_batch_iqr_outlier | (labeled.cycle_life < 500)].sort_values("cycle_life").to_csv(out / "short_and_outlier_cells.csv", index=False)
    metadata = {
        "batch_mapping": BATCH_FILES, "sources": manifest,
        "nominal_capacity_ah": 1.1, "eol_ah": EOL,
        "endpoint_near_eol_upper_ah": NEAR_EOL,
        "endpoint_near_eol_reason": "0.01 Ah last-record QC tolerance; does not prove actual EOL or censoring",
        "qd_high_qc_ah": QD_HIGH_QC, "endpoint_jump_qc_ah": END_JUMP_QC,
        "charge_time_high_qc_min": CHARGE_TIME_HIGH_QC,
        "charge_time_qc": "Positive raw mean retained, median and mean screening >60min supplied as sensitivity; high values do not prove measurement error.",
        "source_collection_issue_index_candidate": "B3c37: official same-date file raw MATLAB batch3(38) removed upfront for a collection issue; channel_id is serialized uint32 MATLAB string, actual channel decoding not verified. Index-based exclusion sensitivity only. Later shifted MATLAB indices not applied.",
        "source_collection_issue_url": "https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m",
        "physical_identity": "File-local cell_key only. Serialized barcode/channel numeric arrays are not physical identifiers; cross-file deduplication unverified.",
        "qd_masking": "No deletion below EOL. Full/late diagnostic QC slopes mask only QD>1.65 and nonpositive/nonfinite QD. Raw-positive early features include high-QD observations with flags.",
        "placeholder": "Only first all-zero QD/QC/IR/temperature/charge-time row is placeholder; other zero IR treated as missing.",
        "early_window": "Actual summary cycles 2..100; all 100-cycle features missing when observed max<100 or finite life label<100. Missing life is never imputed. A separate varcharge file is explicitly excluded by user.",
        "centering": "Pairwise-valid x/y batch mean subtraction, then Pearson; additionally batch+newstructure string flag centering. These are descriptive, not causal adjustments. Adjusted p-values intentionally missing.",
        "sensitivity": "EOL-compatible endpoint <=0.89, positive finite <=1.65, no endpoint jump>0.10; stricter subset also no QD>1.65 anywhere. These are descriptive QC subsets, not verified EOL labels.",
        "policy": "Strip -newstructure for text overlap only. Constant C1(Q1)-C2 0..80% ideal effective C=0.8/(s/C1+(0.8-s)/C2). Variable policy tag never assigned as fixed C1/C2.",
        "policy_groups": "Policy statistics and plots preserve raw policy/newstructure variants; canonical string used for overlap only. C2-active correlations omit switch=80%, where second stage has zero SOC span. Slowcycle tags are separate and not assumed constant two-stage protocols.",
        "late_features": "Full-curve and last-50-cycle slopes use future observations and must not be early-life model features.",
        "stats": "Sample std and pandas bias-corrected skew; pairwise correlations n>=3, nonconstant pairs only; p values exploratory, unadjusted.",
        "multicollinearity": "Feature Pearson/Spearman pairwise n; |r|>0.85 list. VIF is descriptive complete-case sensitivity, with/without exact relation qd_delta=qd100-qd2; no predictive model trained.",
        "raw_changed": False, "models_fitted": False,
    }
    (out / "methodology.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    save_plots(features, curves, corr, out)
    print(batch_stats[["batch", "n_cells", "n_life_valid", "life_mean", "life_min", "life_max", "n_short_lt500", "n_long_gt1000", "n_eol_compatible_labeled"]].to_string(index=False), flush=True)
    return features, batch_stats, corr


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "archive")
    parser.add_argument("--out-dir", type=Path, default=root / "results/eda/summary")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
