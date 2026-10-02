# Three-batch ESS summary EDA implementation plan

**Goal:** Reproduce questions 1, 2, 4 and 5 using selective HDF5 summary loading for the three explicitly selected local files, without altering raw data or fitting a predictive model.

**Architecture:** One Python entry point extracts one record per file-local cell, retains full summary curves for plots, and exports batch, policy and correlation tables. Missing labels, incomplete 100-cycle windows and endpoint anomalies remain explicit flags. Korean findings connect observations to interpretations and their limits.

**Tech stack:** Existing `.venv` Python, h5py, NumPy, pandas, SciPy, Matplotlib.

**Spec:** The user correction supersedes the initial four-file request: B1=2017-05-12, B2=2018-02-20, B3=2018-04-12. The separate 2018-04-03 varcharge file is excluded and is not deleted. Missing life is never imputed; newstructure comparability is not assumed.

## Constraints and review focus

- Read `summary` values and `cycles` field shapes only; do not load cycle-level Qdlin.
- Use actual summary cycle coordinates, and record summary/cycles count mismatches.
- Remove a first all-zero placeholder from descriptive curves; preserve original row counts.
- Treat nonpositive IR as missing; retain zero/invalid fractions. Do not delete QD merely for crossing 0.88 Ah.
- Mask 100-cycle features when the label ends before 100 or the observations do not reach 100.
- Use 0.88 Ah as nominal EOL; endpoint compatibility is a QC flag, not confirmed censoring or adjudicated complete life.
- Compare pooled, within-batch and endpoint-compatible correlations with valid pair n; preserve variable policies separately.

## Tasks

- [x] Extract raw mappings, policies, summary/cycles lengths, quality flags and cell features in `analysis/ess_batch_summary.py`.
- [x] Export features, batch statistics, policy statistics, policy overlap, feature coverage and correlations to `results/eda/summary/`.
- [x] Generate scientific PNG figures for life distributions, QD curves, degradation slopes, policy relations, batch shifts and correlation comparisons.
- [x] Independently validate cell counts, missingness, thresholds, sampled features and correlation n against raw HDF5.
- [x] Write `findings.md` with quantitative evidence, interpretation reasons, candidate features and limitations.

No Git repository is present, so branch/worktree/commit operations are inapplicable. The notebook and raw files are kept unchanged.
