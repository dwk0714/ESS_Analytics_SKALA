"""Descriptive, read-only source analysis. No estimator is fitted or selected.

All writes are confined to this script's directory. Actual lifetime groups are
used only for retrospective diagnostics, never to construct predictors/weights.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from src.preprocess import read_early_cells

CFG = json.loads((ROOT / "config/model.json").read_text())
SOURCE_PATHS = [ROOT / "src/preprocess.py", ROOT / "src/features.py",
                ROOT / "data/processed/cell_features.csv", ROOT / "results/split_manifest.csv",
                ROOT / "results/runs/model_comparison_20261002/provided_labels/predictions.csv",
                ROOT / "results/runs/model_comparison_20261002/provided_labels/cv_predictions.csv"]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before_sha = {str(p.relative_to(ROOT)): sha(p) for p in SOURCE_PATHS}


def life_group(values):
    return np.select([values < 500, values > 1000], ["short_<500", "long_>1000"],
                     default="mid_500_1000")


def dispersion(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(n=0)
    q1, q3 = np.quantile(a, [0.25, 0.75])
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return dict(n=len(a), mean=float(np.mean(a)), median=med,
                min=float(np.min(a)), max=float(np.max(a)),
                sd=float(np.std(a, ddof=1)) if len(a) > 1 else np.nan,
                variance=float(np.var(a, ddof=1)) if len(a) > 1 else np.nan,
                mad=mad, iqr=float(q3 - q1))


f = pd.read_csv(ROOT / "data/processed/cell_features.csv")
split = pd.read_csv(ROOT / "results/split_manifest.csv")
f = f.merge(split[["cell_key", "role"]], on="cell_key", how="left", validate="one_to_one")
f["role"] = f.role.fillna("unlabeled")
f["life_group"] = life_group(f.cycle_life)
f.loc[f.cycle_life.isna(), "life_group"] = "unlabeled"
labeled = f.dropna(subset=["cycle_life"]).copy()

# Lifetime dispersion reflects bounds imposed by the group definitions.
life_rows = []
for scope, base, cols in [("batch", labeled, ["batch", "life_group"]),
                         ("role", labeled, ["batch", "role", "life_group"]),
                         ("all_batches", labeled, ["life_group"])]:
    for key, group in base.groupby(cols, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        life_rows.append({"scope": scope, **dict(zip(cols, key)),
                          **dispersion(group.cycle_life),
                          **{f"log10_{k}": v for k, v in dispersion(np.log10(group.cycle_life)).items()}})
life_stats = pd.DataFrame(life_rows)
life_stats.to_csv(OUT / "lifetime_dispersion.csv", index=False)

# Extract only cycles 1..100 and curves at cycles 10 and 100.
early_rows, qd_arrays, dq_arrays = [], {}, {}
for cell in read_early_cells(ROOT.parent / "archive", CFG["batches"]):
    valid = np.isfinite(cell.qd) & (cell.qd > 0) & (cell.qd <= CFG["quality"]["qd_max_ah"])
    keep = valid & (cell.cycle >= 2) & (cell.cycle <= 100)
    q = cell.qd[keep]
    cycle = cell.cycle[keep]
    ref = np.median(cell.qd[valid & (cell.cycle >= 2) & (cell.cycle <= 10)])
    med = np.median(q)
    slope, intercept = np.polyfit(cycle, q, 1)
    residual = q - (slope * cycle + intercept)
    dq = cell.q100 - cell.q10
    # No target is used in the following feature computations.
    early_rows.append({"cell_key": cell.cell_key, "qd_valid_n": len(q),
        "qd_early_mean_ah": float(np.mean(q)), "q_ref_ah_recomputed": ref,
        "qd_early_variance_ah2": float(np.var(q, ddof=1)),
        "qd_early_mad_ah": float(np.median(np.abs(q - med))),
        "qd_early_mad_relative_recomputed": float(np.median(np.abs(q - med)) / ref),
        "qd_residual_variance_ah2": float(np.var(residual, ddof=1)),
        "qd_residual_mad_relative": float(np.median(np.abs(residual - np.median(residual))) / ref),
        "dq_variance_ah2": float(np.var(dq, ddof=1)),
        "dq_mad_ah": float(np.median(np.abs(dq - np.median(dq))))})
    qd_curve = np.full(99, np.nan)
    qd_curve[cycle.astype(int) - 2] = q
    qd_arrays[cell.cell_key] = qd_curve
    dq_arrays[cell.cell_key] = dq
early = pd.DataFrame(early_rows).merge(f, on="cell_key", validate="one_to_one")
early.to_csv(OUT / "cell_early_variances.csv", index=False)
assert np.allclose(early.dq_variance_ah2, 10 ** early.dq100_10_log_variance)
assert np.allclose(early.qd_early_mad_relative_recomputed, early.capacity_mad_relative)

signals = ["qd_early_variance_ah2", "qd_early_mad_ah", "capacity_mad_relative",
           "qd_residual_variance_ah2", "qd_residual_mad_relative", "dq_variance_ah2",
           "dq100_10_log_variance", "dq_mad_ah", "qd_early_mean_ah", "q_ref_ah_recomputed"]
signal_rows, between_rows = [], []
for scope, base, cols in [("batch", early.dropna(subset=["cycle_life"]), ["batch", "life_group"]),
                         ("all_batches", early.dropna(subset=["cycle_life"]), ["life_group"])]:
    for key, group in base.groupby(cols, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        meta = {"scope": scope, **dict(zip(cols, key))}
        for signal in signals:
            signal_rows.append({**meta, "signal": signal, **dispersion(group[signal])})
        qa = np.stack([qd_arrays[k] for k in group.cell_key])
        da = np.stack([dq_arrays[k] for k in group.cell_key])
        between_rows.append({**meta, "n_cells": len(group),
            "qd_median_within_cell_variance_ah2": float(group.qd_early_variance_ah2.median()),
            "qd_median_pointwise_between_cell_variance_ah2": float(np.nanmedian(np.nanvar(qa, axis=0, ddof=1))),
            "qd_variance_of_cell_early_means_ah2": float(group.qd_early_mean_ah.var(ddof=1)),
            "dq_median_within_curve_variance_ah2": float(group.dq_variance_ah2.median()),
            "dq_median_pointwise_between_cell_variance_ah2": float(np.median(np.var(da, axis=0, ddof=1)))})
signal_stats = pd.DataFrame(signal_rows)
signal_stats.to_csv(OUT / "early_signal_dispersion.csv", index=False)
pd.DataFrame(between_rows).to_csv(OUT / "within_vs_between_cell_variance.csv", index=False)

# Diagnose saved OOF and frozen evaluation predictions. Never refit.
pred_frames = []
for model_group in ["", "ensembles/"]:
    base = ROOT / f"results/runs/model_comparison_20261002/{model_group}provided_labels"
    cv = pd.read_csv(base / "cv_predictions.csv")
    cv["role"] = "train_oof"
    cv["model_family"] = model_group or "individual"
    pred = pd.read_csv(base / "predictions.csv")
    pred["model_family"] = model_group or "individual"
    pred_frames.extend([cv, pred])
pred = pd.concat(pred_frames, ignore_index=True)
pred = pred.merge(f[["cell_key", "batch"]], on="cell_key", validate="many_to_one")
pred["life_group"] = life_group(pred.actual_cycle_life)
pred["residual_cycles"] = pred.predicted_cycle_life - pred.actual_cycle_life
pred["residual_log10"] = np.log10(pred.predicted_cycle_life / pred.actual_cycle_life)
pred["residual_relative_percent"] = 100 * pred.residual_cycles / pred.actual_cycle_life
pred["APE_percent_recomputed"] = np.abs(pred.residual_relative_percent)
pred.to_csv(OUT / "saved_prediction_residuals.csv", index=False)
error_rows = []
for key, group in pred.groupby(["model", "role", "batch", "life_group"], observed=True):
    meta = dict(zip(["model", "role", "batch", "life_group"], key))
    for signal in ["residual_cycles", "residual_log10", "residual_relative_percent"]:
        error_rows.append({**meta, "signal": signal, **dispersion(group[signal]),
                          "MAPE_percent": group.APE_percent_recomputed.mean()})
errors = pd.DataFrame(error_rows)
errors.to_csv(OUT / "error_dispersion.csv", index=False)

# Equal-variance tests are exploratory and neither causal nor selection evidence.
tests = []
for scope, frame in [("all_batches_descriptive_confounded", early.dropna(subset=["cycle_life"])),
                     ("B2_only_tiny_long_group", early[early.batch.eq("B2")].dropna(subset=["cycle_life"]))]:
    a = frame[frame.life_group.eq("short_<500")]
    b = frame[frame.life_group.eq("long_>1000")]
    for signal in ["cycle_life", "dq100_10_log_variance", "capacity_mad_relative"]:
        st, p = stats.levene(a[signal], b[signal], center="median")
        tests.append({"scope": scope, "signal": signal, "n_short": len(a), "n_long": len(b),
                      "statistic": st, "pvalue": p, "interpretation": "descriptive; not causal; no multiple-test adjustment"})
ridge_cv = pred[pred.model.eq("Ridge_existing") & pred.role.eq("train_oof")]
for signal in ["residual_cycles", "residual_log10", "residual_relative_percent"]:
    a = ridge_cv[ridge_cv.life_group.eq("mid_500_1000")][signal]
    b = ridge_cv[ridge_cv.life_group.eq("long_>1000")][signal]
    st, p = stats.levene(a, b, center="median")
    rho, p_rho = stats.spearmanr(ridge_cv.actual_cycle_life, np.abs(ridge_cv[signal] - ridge_cv[signal].median()))
    tests.append({"scope": "B1_train35_Ridge_OOF_mid_vs_long", "signal": signal,
                  "n_mid": len(a), "n_long": len(b), "statistic": st, "pvalue": p,
                  "spearman_life_abs_centered_residual": rho, "spearman_pvalue": p_rho,
                  "interpretation": "OOF folds reuse training; selected CV is optimistic; low-power descriptive diagnostic"})
pd.DataFrame(tests).to_csv(OUT / "exploratory_variance_tests.csv", index=False)

# Show lifetime truncation, cell-level signal information, and saved OOF residual scale.
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
colors = {"B1": "#2c6686", "B2": "#bf7148", "B3": "#588657"}
for batch, g in labeled.groupby("batch"):
    axes[0, 0].hist(g.cycle_life, bins=np.arange(350, 2050, 100), histtype="step", linewidth=2, label=f"{batch}, n={len(g)}", color=colors[batch])
    sg = early[early.batch.eq(batch)].dropna(subset=["cycle_life"])
    axes[0, 1].scatter(sg.cycle_life, sg.dq100_10_log_variance, s=25, alpha=.8, color=colors[batch], label=batch)
    axes[0, 2].scatter(sg.cycle_life, sg.capacity_mad_relative, s=25, alpha=.8, color=colors[batch], label=batch)
axes[0, 0].set(title="Lifetime support differs across batches", xlabel="Stored cycle life", ylabel="Cell count")
axes[0, 0].legend(frameon=False)
axes[0, 1].set(title="Within-cell Delta Q variance is already a feature", xlabel="Stored cycle life", ylabel="log10 variance over voltage")
axes[0, 2].set(title="Early capacity MAD: robust signal dispersion", xlabel="Stored cycle life", ylabel="MAD / reference capacity", yscale="log")
for ax in axes[0]:
    ax.axvline(500, linestyle="--", color="gray", linewidth=1)
    ax.axvline(1000, linestyle="--", color="gray", linewidth=1)
for ax, signal, name in zip(axes[1], ["residual_cycles", "residual_log10", "residual_relative_percent"], ["Cycle residual", "log10 residual", "Relative residual (%)"]):
    ax.scatter(ridge_cv.actual_cycle_life, ridge_cv[signal], color=colors["B1"], s=35)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.axvline(1000, linestyle="--", color="gray", linewidth=1)
    ax.set(title=f"Ridge saved B1 OOF, n={len(ridge_cv)}", xlabel="Stored cycle life", ylabel=name)
fig.suptitle("Retrospective diagnostics only: no life-based inputs, weights, or new training", fontsize=13)
fig.savefig(OUT / "variance_diagnostics.png", dpi=170)
plt.close(fig)

after_sha = {str(p.relative_to(ROOT)): sha(p) for p in SOURCE_PATHS}
verification = {"source_files_unchanged": before_sha == after_sha,
    "source_sha256": after_sha, "raw_mode": "HDF5 read-only, cycles <=100",
    "n_all_cells": len(f), "n_labeled": len(labeled), "n_train_oof_per_model": 35,
    "n_short_in_B1_train": int(((f.role == "train") & (f.cycle_life < 500)).sum()),
    "delta_variance_agrees_with_existing_feature": bool(np.allclose(early.dq_variance_ah2, 10 ** early.dq100_10_log_variance)),
    "capacity_mad_agrees_with_existing_feature": bool(np.allclose(early.qd_early_mad_relative_recomputed, early.capacity_mad_relative)),
    "n_fits": 0, "inputs_selected": False, "models_changed": False,
    "group_cutoffs": {"short": "stored cycle_life <500", "mid": "500<=stored cycle_life<=1000", "long": "stored cycle_life >1000"},
    "limitations": ["Life groups are retrospective target-based diagnostics only.",
       "Between-cell and within-cell variation are not equivalent to conditional model error variance.",
       "Qd variance includes trend and potential spikes, not independent noise replicates.",
       "Delta-Q variance across correlated voltage points summarizes curve shape, not measurement-noise variance.",
       "B2 has only 3 long-lived labels; pooled groups confound batch with lifetime.",
       "Groups truncate the target distribution; larger raw long-life variance alone does not prove heteroscedasticity.",
       "Stored labels are unchanged and include the documented unfinished/near-threshold label limitations.",
       "OOF diagnostics use selected model CV predictions and are not nested evaluation."]}
assert verification["source_files_unchanged"]
(OUT / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False))
print("Verification:", json.dumps({k: v for k, v in verification.items() if k != "source_sha256"}, ensure_ascii=False))
print("\nLifetime dispersion:\n", life_stats[life_stats.scope.eq("batch")].to_string(index=False))
print("\nWithin vs between:\n", pd.DataFrame(between_rows).to_string(index=False))
print("\nRidge residual dispersion:\n", errors[errors.model.eq("Ridge_existing")].to_string(index=False))
print("\nExploratory tests:\n", pd.DataFrame(tests).to_string(index=False))
