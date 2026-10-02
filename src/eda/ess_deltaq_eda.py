"""Selective, reproducible early-cycle Q(V) EDA for three selected archives.

Run from the repository root with python -m src.pipeline eda --raw-dir ../archive.
No raw archives or the scratch notebook are changed; no model is trained.
"""
import os
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[2]
BATCH_FILES = ["2017-05-12_batchdata_updated_struct_errorcorrect.mat",
               "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
               "2018-04-12_batchdata_updated_struct_errorcorrect.mat"]
OUT = ROOT / "results/eda/delta"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def read_vector(f, ref):
    return np.asarray(f[ref][()]).ravel()


def corr_record(d, feature, target, batch, subset):
    v = d[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    result = {"batch": batch, "subset": subset, "feature": feature,
              "target": target, "n": len(v)}
    if len(v) >= 3 and v[feature].nunique() > 1 and v[target].nunique() > 1:
        pr = pearsonr(v[feature], v[target])
        sr = spearmanr(v[feature], v[target])
        result.update(pearson_r=float(pr.statistic), pearson_p=float(pr.pvalue),
                      spearman_rho=float(sr.statistic), spearman_p=float(sr.pvalue))
    return result


rows, curves, qc = [], {}, []
common_grid = None
for bi, filename in enumerate(BATCH_FILES, 1):
    path = Path(os.environ["ESS_RAW_DIR"]) / filename
    batch_name = f"B{bi}"
    with h5py.File(path, "r") as f:
        b = f["batch"]
        for ci in range(b["cycle_life"].size):
            uid = f"{batch_name}c{ci}"
            life = float(read_vector(f, b["cycle_life"][()].ravel()[ci])[0])
            policy = "".join(chr(int(x)) for x in read_vector(f, b["policy_readable"][()].ravel()[ci]))
            s = f[b["summary"][()].ravel()[ci]]
            cycle_ids = np.asarray(s["cycle"][()]).ravel()
            qd = np.asarray(s["QDischarge"][()]).ravel()
            c = f[b["cycles"][()].ravel()[ci]]
            refs = c["Qdlin"][()].ravel()
            voltage = read_vector(f, b["Vdlin"][()].ravel()[ci])
            assert len(refs) == len(cycle_ids), f"{uid}: cycle alignment mismatch"
            assert len(np.unique(cycle_ids)) == len(cycle_ids), f"{uid}: repeated cycle IDs"
            assert np.all(np.diff(voltage) < 0), f"{uid}: unexpected voltage order"
            if common_grid is None:
                common_grid = voltage
            assert np.array_equal(common_grid, voltage), f"{uid}: grid differs"
            positive = qd[np.isfinite(qd) & (qd > 0)]
            row = {"cell_uid": uid, "batch": batch_name, "source_file": path.name,
                   "cell_id": ci, "cycle_life": life, "policy": policy,
                   "observed_cycles": len(cycle_ids),
                   "last_QD": float(positive[-1]) if len(positive) else np.nan,
                   "observed_below_088": bool(np.any(positive <= .88)),
                   "terminal_near_or_below_089": bool(len(positive) and positive[-1] <= .89),
                   "eligible_life100": bool(np.isfinite(life) and life > 100)}
            chosen = {}
            for num in [2, 5, 10, 20, 100]:
                matches = np.flatnonzero(cycle_ids == num)
                if len(matches) == 1:
                    arr = read_vector(f, refs[matches[0]])
                    if len(arr) == len(voltage) and np.isfinite(arr).all() and np.ptp(arr) > .1:
                        chosen[num] = arr
                    else:
                        qc.append({"cell_uid": uid, "cycle": num,
                                   "issue": "Qdlin invalid/nonfinite/placeholder"})
            for later, earlier in [(5, 2), (20, 10), (100, 10)]:
                prefix = f"dq{later}_{earlier}"
                if later in chosen and earlier in chosen:
                    delta = chosen[later] - chosen[earlier]
                    variance = float(np.var(delta, ddof=1))
                    row[prefix + "_variance"] = variance
                    row[prefix + "_log_variance"] = np.log10(variance) if variance > 0 else np.nan
                    row[prefix + "_min"] = float(delta.min())
                    row[prefix + "_mean"] = float(delta.mean())
                    row[prefix + "_abs_min_log"] = np.log10(abs(delta.min())) if delta.min() != 0 else np.nan
                    row[prefix + "_endpoint"] = float(delta[-1])
                    interior = delta[(voltage >= 2.5) & (voltage <= 3.3)]
                    ivar = float(np.var(interior, ddof=1))
                    row[prefix + "_interior_log_variance"] = np.log10(ivar) if ivar > 0 else np.nan
                    if later == 100:
                        curves[uid] = {"voltage": voltage, "delta": delta,
                                       "q10": chosen[10], "q100": chosen[100]}
                else:
                    qc.append({"cell_uid": uid, "cycle": later,
                               "issue": f"{prefix} unavailable"})
            rows.append(row)

d = pd.DataFrame(rows)
d.loc[~np.isfinite(d.cycle_life) | (d.cycle_life <= 0), "cycle_life"] = np.nan
d["log10_cycle_life"] = np.where(d.cycle_life > 0, np.log10(d.cycle_life), np.nan)
d.to_csv(OUT / "delta_features.csv", index=False)
identity = d[["cell_uid"]].copy()
identity["physical_identity_status"] = "unresolved_matlab_object_barcode_not_decoded"
identity.to_csv(OUT / "physical_identity_audit.csv", index=False)
pd.DataFrame(qc, columns=["cell_uid", "cycle", "issue"]).to_csv(OUT / "delta_quality.csv", index=False)
pd.DataFrame({"voltage_V": common_grid}).to_csv(OUT / "voltage_grid.csv", index=False)
features = [c for c in d if c.startswith("dq") and ("log_variance" in c or c.endswith("_min") or c.endswith("_endpoint") or c.endswith("_mean") or c.endswith("_abs_min_log"))]
records = []
for subset in ["all_labeled_100_eligible", "terminal_near_or_below_089"]:
    usable = d.loc[d.eligible_life100].copy()
    if subset == "terminal_near_or_below_089":
        usable = usable.loc[usable.terminal_near_or_below_089]
    for batch_name, group in [("pooled", usable)] + list(usable.groupby("batch")):
        for feat in features:
            for target in ["cycle_life", "log10_cycle_life"]:
                records.append(corr_record(group, feat, target, batch_name, subset))
    for feat in features:
        for target in ["cycle_life", "log10_cycle_life"]:
            matched = usable[["batch", feat, target]].dropna().copy()
            # Re-center after pairwise complete-case selection, including target.
            for col in [feat, target]:
                matched[col] -= matched.groupby("batch")[col].transform("mean")
            records.append(corr_record(matched, feat, target, "within_batch_centered", subset))
corr = pd.DataFrame(records)
corr.to_csv(OUT / "delta_correlations.csv", index=False)

stats = d.groupby("batch").agg(
    cells=("cell_uid", "size"), labeled=("cycle_life", "count"),
    available_delta100=("dq100_10_log_variance", "count"),
    logvar_median=("dq100_10_log_variance", "median"),
    logvar_min=("dq100_10_log_variance", "min"),
    logvar_max=("dq100_10_log_variance", "max"),
    endpoint_median=("dq100_10_endpoint", "median"))
stats.to_csv(OUT / "delta_batch_stats.csv")
group_rows = []
for scope, group in [("ALL", d)] + list(d.groupby("batch")):
    labeled = group.loc[np.isfinite(group.cycle_life)]
    masks = {"short_lt500": labeled.cycle_life < 500,
             "middle_500_1000": labeled.cycle_life.between(500, 1000),
             "long_gt1000": labeled.cycle_life > 1000}
    for name, mask in masks.items():
        a = labeled.loc[mask]
        group_rows.append({"scope": scope, "life_group": name, "n": len(a),
                           "log_variance_median": a.dq100_10_log_variance.median(),
                           "delta_min_median": a.dq100_10_min.median(),
                           "delta_mean_median": a.dq100_10_mean.median()})
pd.DataFrame(group_rows).to_csv(OUT / "delta_life_group_stats.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), sharex=True)
panels = [("ALL", d.loc[np.isfinite(d.cycle_life)], "fixed_thresholds")]
panels += [(b, g.loc[np.isfinite(g.cycle_life)], "quartiles") for b, g in d.groupby("batch")]
for ax, (scope, group, mode) in zip(axes, panels):
    if mode == "fixed_thresholds":
        selections = [("Short <500", group.loc[group.cycle_life < 500], "#b55f43"),
                      ("Long >1000", group.loc[group.cycle_life > 1000], "#377e9b")]
    else:
        q1, q3 = group.cycle_life.quantile([.25, .75])
        selections = [("Lower life quartile", group.loc[group.cycle_life <= q1], "#b55f43"),
                      ("Upper life quartile", group.loc[group.cycle_life >= q3], "#377e9b")]
    for label, selected, color in selections:
        matrix = np.stack([curves[uid]["delta"] for uid in selected.cell_uid if uid in curves])
        lower, median, upper = np.quantile(matrix, [.25, .5, .75], axis=0)
        ax.plot(common_grid, median, color=color, label=f"{label} (n={len(matrix)})")
        ax.fill_between(common_grid, lower, upper, color=color, alpha=.15)
    ax.axhline(0, color="#999999", lw=.6)
    ax.set(title=f"{scope}: {'EDA thresholds' if mode == 'fixed_thresholds' else 'within-batch quartiles'}",
           xlabel="Discharge voltage (V)", ylabel="Delta Q (Ah)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.15)
fig.suptitle("Median and IQR of Delta Q: pooled groups and within-batch comparison")
fig.tight_layout()
fig.savefig(OUT / "delta_life_group_shapes.png", dpi=180)
plt.close(fig)

finite_life = d.cycle_life[np.isfinite(d.cycle_life)]
norm = Normalize(finite_life.min(), finite_life.max())
cmap = plt.get_cmap("viridis")
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True)
for ax, (batch_name, group) in zip(axes.ravel(), d.groupby("batch")):
    for _, row in group.iterrows():
        if row.cell_uid not in curves:
            continue
        curve = curves[row.cell_uid]
        color = cmap(norm(row.cycle_life)) if np.isfinite(row.cycle_life) else "#929292"
        style = "--" if not np.isfinite(row.cycle_life) else "-"
        ax.plot(curve["voltage"], curve["delta"], color=color, alpha=.8, lw=1, ls=style,
                label=None)
    ax.axhline(0, color="#909090", lw=.6)
    ax.set(title=f"{batch_name}: Q100(V) - Q10(V), n={len(group)}; labeled={group.cycle_life.count()}",
           xlabel="Discharge voltage (V)", ylabel="Delta Q (Ah)")
    ax.grid(alpha=.15)
    if group.cycle_life.isna().any():
        ax.plot([], [], color="#929292", ls="--", label="Missing life label")
        ax.legend(fontsize=8)
fig.subplots_adjust(right=.88, hspace=.3, wspace=.25)
cax = fig.add_axes([.9, .18, .015, .6])
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Stored cycle-life label (cycles)")
fig.suptitle("Early-cycle voltage curve changes on the verified common 3.5-2.0 V grid", y=.98)
fig.savefig(OUT / "delta_q_curves.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
colors = {"B1": "#277da8", "B2": "#cd7645", "B3": "#628645"}
for ax, batch_name in zip(axes, ["B1", "B2", "B3"]):
    sub = d.loc[(d.batch == batch_name) & d.eligible_life100].dropna(subset=["dq100_10_log_variance"])
    near = sub.terminal_near_or_below_089
    ax.scatter(sub.loc[near, "dq100_10_log_variance"], sub.loc[near, "cycle_life"],
               s=38, color=colors[batch_name], alpha=.85, label="Terminal QD <= 0.89 Ah")
    ax.scatter(sub.loc[~near, "dq100_10_log_variance"], sub.loc[~near, "cycle_life"],
               s=50, marker="x", color="#444444", label="Terminal QD > 0.89 Ah")
    r = corr_record(sub, "dq100_10_log_variance", "cycle_life", batch_name, "all")
    ax.set(title=f"{batch_name} (n={len(sub)}), Pearson r={r.get('pearson_r', np.nan):.2f}",
           xlabel="log10 Var[Q100(V)-Q10(V)] (Ah^2)", ylabel="Stored cycle-life label")
    ax.grid(alpha=.15)
axes[0].legend(fontsize=8)
fig.suptitle("Delta-Q shape variation and life: assess within each batch")
fig.tight_layout()
fig.savefig(OUT / "delta_feature_vs_life.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, (later, earlier) in zip(axes, [(5, 2), (20, 10), (100, 10)]):
    feat = f"dq{later}_{earlier}_log_variance"
    for batch_name, color in colors.items():
        sub = d.loc[(d.batch == batch_name) & d.eligible_life100].dropna(subset=[feat])
        ax.scatter(sub[feat], sub.cycle_life, c=color, s=27, alpha=.75, label=batch_name)
    ax.set(title=f"Q{later}(V)-Q{earlier}(V)", xlabel="log10 Var(delta Q) (Ah^2)", ylabel="Stored cycle-life label")
    ax.grid(alpha=.15)
axes[0].legend()
fig.suptitle("Distinct observation horizons; 100-cycle features cannot be used at cycle 5")
fig.tight_layout()
fig.savefig(OUT / "delta_observation_windows.png", dpi=180)
plt.close(fig)

# Q(V) overlays for lower-/higher-life representatives, selected descriptively.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, batch_name in zip(axes, ["B1", "B2", "B3"]):
    sub = d.loc[(d.batch == batch_name) & d.eligible_life100 & d.cell_uid.isin(curves)]
    selected = pd.concat([sub.nsmallest(1, "cycle_life"), sub.nlargest(1, "cycle_life")])
    for color, (_, row) in zip(["#b85844", "#327c9a"], selected.iterrows()):
        curve = curves[row.cell_uid]
        ax.plot(curve["voltage"], curve["q10"], color=color, ls="--", lw=1,
                label=f"{row.cell_uid}: {row.cycle_life:.0f} cycles, Q10")
        ax.plot(curve["voltage"], curve["q100"], color=color, lw=1.2,
                label=f"{row.cell_uid}: Q100")
    ax.set(title=batch_name, xlabel="Discharge voltage (V)", ylabel="Discharge capacity Q(V) (Ah)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.15)
fig.suptitle("Early Q(V) curves often overlap; subtraction exposes shape changes")
fig.tight_layout()
fig.savefig(OUT / "qv_representatives.png", dpi=180)
plt.close(fig)

details = {"voltage_grid": {"points": len(common_grid), "min_V": float(common_grid.min()),
                          "max_V": float(common_grid.max()), "equal_for_all_cells": True},
           "delta_definition": "Qdlin(cycle 100) - Qdlin(cycle 10)",
           "variance_ddof": 1, "life_imputation": False,
           "analysis_unit": "one row per cell", "selected_files": BATCH_FILES,
           "qc_note": "Terminal QD <=0.89 is a sensitivity proxy, not certified EOL or censoring classification."}
(OUT / "method.json").write_text(json.dumps(details, indent=2))
print(stats.to_string())
sel = corr.loc[(corr.feature == "dq100_10_log_variance") & (corr.target == "cycle_life")]
print(sel.to_string(index=False))
print("Cells with absent/invalid Delta100:", d.loc[d.dq100_10_log_variance.isna(), ["cell_uid", "cycle_life", "observed_cycles"]].to_dict("records"))
print("Saved to", OUT)
