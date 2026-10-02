"""Legacy predictors and approved early-only stacking candidates; no model fit."""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .preprocess import EarlyCell, file_sha256, read_early_cells
from .candidate_features import ADDED_FEATURES, RANKING_CANDIDATES, extract_ranking_candidates

FEATURES = ["dq100_10_log_variance", "cycle10_rms_c_timeweighted", "capacity_mad_relative"]
STACKING_FEATURES = FEATURES[:2] + ADDED_FEATURES
KEY_COLUMNS = ["cell_key", "batch", "policy", "cycle_life"]


def delta_q_log_variance(cell: EarlyCell, min_ptp: float = 0.1) -> float:
    for curve in (cell.q10, cell.q100):
        if curve.shape != cell.voltage.shape or not np.isfinite(curve).all() or np.ptp(curve) <= min_ptp:
            raise ValueError(f"{cell.cell_key}: Q10/Q100 곡선의 길이·유한성·범위를 확인하세요.")
    variance = float(np.var(cell.q100 - cell.q10, ddof=1))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError(f"{cell.cell_key}: ΔQ 분산이 양수가 아닙니다. epsilon을 추가하지 않습니다.")
    return float(np.log10(variance))


def rms_current(current: np.ndarray, time: np.ndarray) -> tuple[float, dict]:
    current, time = np.asarray(current, float), np.asarray(time, float)
    if len(current) != len(time) or len(current) < 3:
        raise ValueError("cycle10 전류와 시간 길이가 맞지 않거나 너무 짧습니다.")
    discharge = np.flatnonzero(np.isfinite(current) & (current < -0.5))
    if not len(discharge):
        raise ValueError("cycle10에서 첫 방전 시작점을 찾지 못했습니다.")
    end = int(discharge[0])
    a, b = current[:max(0, end - 1)], current[1:end]
    left, right = time[:max(0, end - 1)], time[1:end]
    dt = right - left
    finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(left) & np.isfinite(right)
    active = finite & (a > 0.1) & (b > 0.1) & (dt > 0)
    if int(active.sum()) < 2:
        raise ValueError("cycle10에 유효 충전 구간이2개 미만입니다.")
    av, bv, duration = a[active], b[active], dt[active]
    second = np.dot(duration, (av * av + av * bv + bv * bv) / 3) / duration.sum()
    qc = {
        "rms_active_intervals": int(active.sum()),
        "rms_active_minutes": float(duration.sum()),
        "rms_nonfinite_intervals": int((~finite).sum()),
        "rms_nonpositive_dt_intervals": int((np.isfinite(dt) & (dt <= 0)).sum()),
    }
    return float(np.sqrt(second)), qc


def relative_capacity_mad(cell: EarlyCell, max_qd: float = 1.65, min_reference: int = 5) -> tuple[float, dict]:
    valid = np.isfinite(cell.qd) & (cell.qd > 0) & (cell.qd <= max_qd)
    values = cell.qd[valid & (cell.cycle >= 2) & (cell.cycle <= 100)]
    reference = cell.qd[valid & (cell.cycle >= 2) & (cell.cycle <= 10)]
    if len(values) < 1 or len(reference) < min_reference:
        raise ValueError(f"{cell.cell_key}: MAD 또는 기준 용량의 유효 관측이 부족합니다.")
    q_ref = float(np.median(reference))
    mad = float(np.median(np.abs(values - np.median(values))) / q_ref)
    return mad, {"mad_qd_valid_n": int(len(values)), "q_ref_ah": q_ref,
                 "q_ref_valid_n": int(len(reference))}


def feature_row(cell: EarlyCell, quality: dict) -> dict:
    dq = delta_q_log_variance(cell, quality["curve_min_ptp_ah"])
    try:
        rms, rms_qc = rms_current(cell.current10, cell.time10)
    except ValueError as exc:
        raise ValueError(f"{cell.cell_key}: {exc}") from exc
    mad, mad_qc = relative_capacity_mad(cell, quality["qd_max_ah"], quality["reference_min_valid"])
    candidates = extract_ranking_candidates(cell, quality)
    return {"cell_key": cell.cell_key, "batch": cell.batch, "cell_index": cell.cell_index,
            "policy": cell.policy, "cycle_life": cell.cycle_life,
            FEATURES[0]: dq, FEATURES[1]: rms, FEATURES[2]: mad,
            "feature_latest_cycle": 100, "feature_origin": "raw_early_observations", **rms_qc, **mad_qc,
            **candidates}


def check_feature_table(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    required = KEY_COLUMNS + FEATURES
    if config.get("model_type") == "stacking":
        required += RANKING_CANDIDATES
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"특징 표에 필요한 열이 없습니다: {sorted(missing)}")
    if frame.empty or frame.cell_key.isna().any() or not frame.cell_key.is_unique:
        raise ValueError("셀 ID가 비어 있거나 중복됩니다.")
    if frame.policy.isna().any() or frame.policy.astype(str).str.strip().eq("").any():
        raise ValueError("충전 프로토콜이 비어 있습니다.")
    if not {"B1", "B2"}.issubset(set(frame.batch)) or not set(frame.batch).issubset(config["batches"]):
        raise ValueError("B1/B2가 필요하며 지정하지 않은 배치는 포함할 수 없습니다.")
    finite_features = FEATURES + [name for name in ADDED_FEATURES if name in frame.columns]
    if not np.isfinite(frame[finite_features].to_numpy(dtype=float)).all():
        raise ValueError("최종 입력 또는 기존 특징에 결측/무한대가 있습니다. 기록을 임의로 버리거나 대치하지 않습니다.")
    for batch, group in frame.groupby("batch"):
        if len(group) != config["expected_records"][batch]:
            raise ValueError(f"{batch}: 셀 개수가 예상값과 다릅니다: {len(group)}")
        expected_keys = {f"{batch}c{i}" for i in range(config["expected_records"][batch])}
        if set(group.cell_key) != expected_keys:
            raise ValueError(f"{batch}: 파일 내 셀 ID 목록이 다릅니다.")
        labeled = np.isfinite(group.cycle_life) & (group.cycle_life > 0)
        if int(labeled.sum()) != config["expected_labeled"][batch]:
            raise ValueError(f"{batch}: 유효 수명 개수가 예상값과 다릅니다: {labeled.sum()}")
    return frame


def extract_features(raw_dir: Path, config: dict, batches: list[str] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    chosen = batches or list(config["batches"])
    if not {"B1", "B2"}.issubset(chosen) or not set(chosen).issubset(config["batches"]):
        raise ValueError("특징 추출 배치는 B1 B2 또는 B1 B2 B3로 지정하세요.")
    files = {batch: config["batches"][batch] for batch in chosen}
    frame = pd.DataFrame(feature_row(cell, config["quality"]) for cell in read_early_cells(raw_dir, files))
    check_feature_table(frame, config)
    sources = [{"batch": batch, "filename": name, "path": str((Path(raw_dir) / name).resolve()),
                "bytes": (Path(raw_dir) / name).stat().st_size,
                "sha256": file_sha256(Path(raw_dir) / name)} for batch, name in files.items()]
    return frame, sources


def data_fingerprint(frame: pd.DataFrame) -> str:
    additional = [name for name in RANKING_CANDIDATES if name in frame.columns]
    ordered = frame.sort_values("cell_key")[KEY_COLUMNS + FEATURES + additional].reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).values.tobytes()).hexdigest()
