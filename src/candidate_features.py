"""Target-free candidate measurements available by actual cycle100.

These formulas reproduce the approved early-feature research. IR is excluded
from the ranking pool; a missing optional candidate is not silently repaired.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import theilslopes

from .preprocess import EarlyCell

ADDED_FEATURES = ["dq100_10_q05_ah", "dq100_10_mean_ah"]
RANKING_CANDIDATES = [
    "capacity_q100_q10_change_relative",
    "capacity_theilsen_10_100_relative",
    "capacity_theilsen_60_100_relative",
    "dq100_10_q05_ah",
    "dq100_10_mean_v2p0_2p4_ah",
    "dq100_10_mean_ah",
    "tavg_theilsen_10_100_degC_per_cycle",
    "tavg_median_2_100_degC",
    "chargetime_median_2_100_minutes",
]


def _window_median(cycle, values, lower, upper, min_valid):
    keep = ((cycle >= lower) & (cycle <= upper)
            & np.isfinite(values) & (values > 0))
    return float(np.median(values[keep])) if int(keep.sum()) >= min_valid else np.nan


def _robust_slope(cycle, values, min_valid):
    valid = np.isfinite(values) & np.isfinite(cycle)
    if int(valid.sum()) < min_valid:
        return np.nan
    return float(theilslopes(values[valid], cycle[valid])[0])


def extract_ranking_candidates(cell: EarlyCell, quality: dict) -> dict[str, float]:
    """Compute nine features without consulting cell.cycle_life or later data."""
    cycle = np.asarray(cell.cycle, float)
    if not np.array_equal(cycle, np.arange(1, 101)):
        raise ValueError(f"{cell.cell_key}: 후보 변수는 실제 cycle1..100만 사용합니다.")
    if cell.tavg is None or cell.chargetime is None:
        raise ValueError(f"{cell.cell_key}: 초기 Tavg/chargetime prefix가 필요합니다.")
    qd, tavg, chargetime = (np.asarray(values, float) for values in (cell.qd, cell.tavg, cell.chargetime))
    if any(values.shape != cycle.shape for values in (qd, tavg, chargetime)):
        raise ValueError(f"{cell.cell_key}: 후보 summary 값과 실제 사이클 번호 길이가 다릅니다.")
    max_qd = quality["qd_max_ah"]
    q = np.where(np.isfinite(qd) & (qd > 0) & (qd <= max_qd), qd, np.nan)
    qref = _window_median(cycle, q, 2, 10, quality["reference_min_valid"])
    if not np.isfinite(qref):
        raise ValueError(f"{cell.cell_key}: 후보 변수 기준 용량의 유효 관측이 부족합니다.")
    early = (cycle >= 10) & (cycle <= 100)
    late = (cycle >= 60) & (cycle <= 100)
    thermal = np.where(np.isfinite(tavg) & (tavg > 0), tavg, np.nan)
    delta = cell.q100 - cell.q10
    endpoint = float(q[99] / q[9] - 1) if np.isfinite(q[9]) and np.isfinite(q[99]) else np.nan
    return {
        "capacity_q100_q10_change_relative": endpoint,
        "capacity_theilsen_10_100_relative": _robust_slope(cycle[early], q[early], 50) / qref,
        "capacity_theilsen_60_100_relative": _robust_slope(cycle[late], q[late], 30) / qref,
        "dq100_10_q05_ah": float(np.quantile(delta, 0.05, method="linear")),
        "dq100_10_mean_v2p0_2p4_ah": float(delta[(cell.voltage >= 2) & (cell.voltage <= 2.4)].mean()),
        "dq100_10_mean_ah": float(delta.mean()),
        "tavg_theilsen_10_100_degC_per_cycle": _robust_slope(cycle[early], thermal[early], 50),
        "tavg_median_2_100_degC": _window_median(cycle, tavg, 2, 100, 50),
        "chargetime_median_2_100_minutes": _window_median(cycle, chargetime, 2, 100, 50),
    }
