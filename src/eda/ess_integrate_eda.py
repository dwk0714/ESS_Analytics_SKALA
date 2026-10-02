"""Join summary, voltage and measured-current EDA; inspect feature redundancy."""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/eda/integrated"
OUT.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(ROOT / "results/eda/summary/features.csv")
s.loc[~np.isfinite(s.cycle_life) | (s.cycle_life <= 0), "cycle_life"] = np.nan
d = pd.read_csv(ROOT / "results/eda/delta/delta_features.csv")
assert s.cell_key.is_unique and d.cell_uid.is_unique
assert set(s.cell_key) == set(d.cell_uid)
delta_cols = [c for c in d if c.startswith("dq")]
x = s.merge(d[["cell_uid"] + delta_cols], left_on="cell_key", right_on="cell_uid", validate="one_to_one").copy()
current = pd.read_csv(ROOT / "results/eda/current/current_features.csv")
current_cols = [c for c in current if c.startswith("cycle10_")]
assert current.cell_key.is_unique and set(current.cell_key) == set(x.cell_key)
x = x.merge(current[["cell_key"] + current_cols], on="cell_key", validate="one_to_one")
assert set(x.batch) == {"B1", "B2", "B3"}
x["documented_collection_issue"] = x.cell_key.eq("B3c37")
# Effective CC rate in the 60-80% SOC interval, an idealized policy descriptor.
switch = x.switch_soc_pct / 100
p1 = np.maximum(np.minimum(switch, .8) - .6, 0)
p2 = .2 - p1
x["c_eff_soc60_80"] = .2 / (p1 / x.c1 + p2 / x.c2)
x.to_csv(OUT / "cell_features_with_qc.csv", index=False)

features = ["dq100_10_log_variance", "dq100_10_abs_min_log", "dq100_10_mean",
            "dq100_10_min", "dq5_2_log_variance", "qd2", "qd100",
            "qd_early_mean", "qd_early_std", "qd_delta_100_2", "qd_early_slope",
            "ir_early_mean", "ir_change_100_2", "ir_early_slope",
            "tavg_early_mean", "tmax_early_mean", "chargetime_early_mean",
            "chargetime_early_median", "chargetime_early_mean_qc", "qd_early_std_qc", "qd_early_slope_qc",
            "c1", "c2", "switch_soc_pct", "c_max", "c_eff_0_80", "c_eff_soc60_80",
            "cycle10_mean_c_timeweighted", "cycle10_rms_c_timeweighted", "cycle10_variance_c2_timeweighted",
            "cycle10_active_charge_duration_min", "cycle10_time_fraction_above_5c"]
descriptions = []
for batch, group in [("ALL", x)] + list(x.groupby("batch")):
    for feat in features:
        v = group[feat].replace([np.inf, -np.inf], np.nan).dropna()
        stats = v.describe(percentiles=[.25, .5, .75]).to_dict()
        descriptions.append({"batch": batch, "feature": feat, "n_cells": len(group),
                             "n_valid": len(v), "n_missing": len(group) - len(v), **stats})
pd.DataFrame(descriptions).to_csv(OUT / "feature_distributions.csv", index=False)

correlations = []
for subset in ["all_labeled", "eol_compatible", "eol_compatible_no_qd_high", "exclude_documented_collection_issue"]:
    g = x.loc[x.cycle_life.notna()].copy()
    if subset in ["eol_compatible", "eol_compatible_no_qd_high"]:
        g = g.loc[g["qc_" + subset]]
    elif subset == "exclude_documented_collection_issue":
        g = g.loc[~g.documented_collection_issue]
    for batch, group in [("ALL", g)] + list(g.groupby("batch")):
        for feat in features:
            v = group[["batch", feat, "cycle_life"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(v) < 3 or v[feat].nunique() < 2:
                continue
            for adjustment in ["none", "within_batch_mean_centered"] if batch == "ALL" else ["none"]:
                data = v[[feat, "cycle_life"]].copy()
                if adjustment != "none":
                    data -= v.groupby("batch")[[feat, "cycle_life"]].transform("mean")
                pr = pearsonr(data[feat], data.cycle_life)
                sr = spearmanr(data[feat], data.cycle_life)
                correlations.append({"subset": subset, "batch": batch, "adjustment": adjustment,
                                     "feature": feat, "n": len(data), "pearson_r": float(pr.statistic),
                                     "spearman_rho": float(sr.statistic), "pearson_p_exploratory": float(pr.pvalue)})
pd.DataFrame(correlations).to_csv(OUT / "feature_life_correlations.csv", index=False)

labeled = x.loc[x.cycle_life.notna()]
pairs = []
for i, a in enumerate(features):
    for b in features[i + 1:]:
        vals = labeled[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) >= 3 and vals[a].nunique() > 1 and vals[b].nunique() > 1:
            r = float(pearsonr(vals[a], vals[b]).statistic)
            pairs.append({"feature_a": a, "feature_b": b, "n": len(vals), "pearson_r": r,
                          "high_redundancy_abs_r085": abs(r) >= .85})
pd.DataFrame(pairs).to_csv(OUT / "feature_pair_correlations.csv", index=False)

# VIF auxiliary regressions use features only, never the lifetime target.
vif_sets = {
    "redundant_candidates": ["dq100_10_log_variance", "dq100_10_abs_min_log", "qd2", "qd100",
                              "qd_early_mean", "qd_delta_100_2", "tavg_early_mean", "tmax_early_mean",
                              "chargetime_early_mean", "c1", "c2", "switch_soc_pct", "c_eff_0_80"],
    "compact_candidates": ["dq100_10_log_variance", "qd2", "qd_delta_100_2", "tavg_early_mean",
                            "chargetime_early_mean", "c1", "c2", "switch_soc_pct"],
    "compact_robust_candidates": ["dq100_10_log_variance", "qd2_qc", "qd_delta_100_2_qc", "tavg_early_mean",
                                   "chargetime_early_median", "c1", "c2", "switch_soc_pct"],
    "measured_current_redundant_candidates": ["dq100_10_log_variance", "qd2_qc", "qd_delta_100_2_qc",
        "tavg_early_mean", "chargetime_early_median", "cycle10_mean_c_timeweighted", "cycle10_rms_c_timeweighted",
        "cycle10_variance_c2_timeweighted", "cycle10_active_charge_duration_min", "cycle10_time_fraction_above_5c"],
    "measured_current_compact_candidates": ["dq100_10_log_variance", "qd2_qc", "qd_delta_100_2_qc",
        "tavg_early_mean", "chargetime_early_median", "cycle10_rms_c_timeweighted", "cycle10_time_fraction_above_5c"],
}
vifs = []
for name, cols in vif_sets.items():
    vals = labeled[cols].replace([np.inf, -np.inf], np.nan).dropna()
    z = (vals - vals.mean()) / vals.std(ddof=0)
    for feat in cols:
        target = z[feat].to_numpy()
        design = np.column_stack([np.ones(len(z)), z.drop(columns=feat).to_numpy()])
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
        residual_variance = float(np.sum((target - design @ beta) ** 2) / np.sum(target ** 2))
        vifs.append({"candidate_set": name, "feature": feat, "n_complete_cases": len(z),
                     "vif": 1 / residual_variance if residual_variance > 1e-12 else np.inf,
                     "design_rank": int(np.linalg.matrix_rank(design))})
pd.DataFrame(vifs).to_csv(OUT / "vif_diagnostics.csv", index=False)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
display_feats = ["dq100_10_log_variance", "dq100_10_abs_min_log", "qd2", "qd_delta_100_2",
                 "ir_early_mean", "tavg_early_mean", "tmax_early_mean", "chargetime_early_median",
                 "c_max", "switch_soc_pct"]
correlation_cols = display_feats + ["cycle_life"]
mat = labeled[correlation_cols].corr()
fig, ax = plt.subplots(figsize=(13, 10))
im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
labels = display_feats + ["life_cycle"]
ax.set_xticks(range(len(correlation_cols)), labels, rotation=45, ha="right")
ax.set_yticks(range(len(correlation_cols)), labels)
for i in range(len(mat)):
    for j in range(len(mat)):
        value = mat.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(value) > .6 else "black")
target_index = len(correlation_cols) - 1
ax.axvline(target_index - .5, color="black", linewidth=1.5)
ax.axhline(target_index - .5, color="black", linewidth=1.5)
ax.get_xticklabels()[-1].set_color("darkgreen")
ax.get_yticklabels()[-1].set_color("darkgreen")
fig.colorbar(im, ax=ax, fraction=.04, label="Pearson r; pairwise complete observations")
ax.set_title("Early feature redundancy and relationship with life_cycle\n"
             "The life_cycle row/column is for target correlation only; VIF remains feature-only.")
fig.tight_layout()
fig.savefig(OUT / "feature_redundancy.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
box_features = ["dq100_10_log_variance", "qd2", "qd_delta_100_2", "ir_early_mean", "tavg_early_mean", "chargetime_early_median"]
for ax, feat in zip(axes.ravel(), box_features):
    values = [x.loc[x.batch == b, feat].dropna().to_numpy() for b in ["B1", "B2", "B3"]]
    ax.boxplot(values, tick_labels=["B1", "B2", "B3"], showfliers=True)
    ax.set_title(feat)
    ax.grid(axis="y", alpha=.15)
fig.suptitle("Early feature distributions differ by batch")
fig.tight_layout()
fig.savefig(OUT / "core_feature_distributions.png", dpi=180)
plt.close(fig)

manifest = {"cell_rows": len(x), "labeled_rows": int(x.cycle_life.notna().sum()),
            "early_prediction_candidates": vif_sets["compact_robust_candidates"],
            "measured_current_comparison_candidates": vif_sets["measured_current_compact_candidates"],
            "measured_current_window": "Cycle 10 alone, time-weighted active positive-current intervals before first discharge; never an early-100 average.",
            "c_eff_soc60_80": "0.2/(portion_SOC60_80_at_C1/C1+portion_at_C2/C2), ideal CC descriptor",
            "future_diagnostics_excluded_from_prediction": ["knee", "qd_late50_slope_qc", "qd_full_slope_qc", "qd_endpoint_raw", "n_summary_rows"],
            "vif_note": "Exploratory feature-only regressions, pairwise correlation and complete-case VIF have different n; exact QD100=QD2+delta is intentional redundancy diagnostic.",
            "centered_statistics_note": "Mean-centered Spearman is not partial Spearman. Pearson p values use scipy's ordinary degrees of freedom, not batch-adjusted inference; no significance conclusions.",
            "lifetime_model_trained": False, "physical_identity": "Unresolved MATLAB-object barcode, no cross-batch cell joining"}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("Joined", len(x), "cell rows,", int(x.cycle_life.notna().sum()), "finite lifetime labels")
print(pd.DataFrame(correlations).query("subset=='all_labeled' and batch=='ALL' and adjustment=='none'").sort_values("pearson_r").to_string(index=False))
print("Saved to", OUT)
