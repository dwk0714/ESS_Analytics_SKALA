"""Read the existing MATLAB v7.3 files without loading whole-life waveforms."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


@dataclass
class EarlyCell:
    cell_key: str
    batch: str
    cell_index: int
    policy: str
    cycle_life: float  # The stored target, never a feature/QC decision.
    cycle: np.ndarray
    qd: np.ndarray
    voltage: np.ndarray
    q10: np.ndarray
    q100: np.ndarray
    current10: np.ndarray
    time10: np.ndarray
    tavg: np.ndarray | None = None
    chargetime: np.ndarray | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_prefix(dataset: h5py.Dataset, limit: int = 100) -> np.ndarray:
    """Transfer only the first limit entries, not a full summary vector."""
    if dataset.ndim == 1:
        values = dataset[:limit]
    elif dataset.ndim == 2 and dataset.shape[0] == 1:
        values = dataset[0, :limit]
    elif dataset.ndim == 2 and dataset.shape[1] == 1:
        values = dataset[:limit, 0]
    else:
        raise ValueError(f"Unsupported vector shape: {dataset.name} {dataset.shape}")
    return np.asarray(values, dtype=float).reshape(-1)


def reference_at(dataset: h5py.Dataset, index: int):
    if dataset.ndim == 1:
        return dataset[index]
    if dataset.ndim == 2 and dataset.shape[1] == 1:
        return dataset[index, 0]
    if dataset.ndim == 2 and dataset.shape[0] == 1:
        return dataset[0, index]
    raise ValueError(f"Unsupported reference shape: {dataset.name} {dataset.shape}")


def read_reference_vector(handle: h5py.File, ref) -> np.ndarray:
    return np.asarray(handle[ref][()], dtype=float).reshape(-1)


def read_early_cells(raw_dir: Path, batches: dict[str, str]) -> Iterator[EarlyCell]:
    """Use cycle1..100 Qd/Tavg/chargetime, Q10/Q100 and only cycle-10 I/t.

    Stored cycle_life and policy are read separately as target/split metadata.
    A future capacity, knee or observation endpoint is never used here.
    """
    common_voltage = None
    for batch, filename in batches.items():
        path = Path(raw_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(f"원본 파일이 없습니다: {path}. data/README.md를 확인하세요.")
        with h5py.File(path, "r") as handle:
            if "batch" not in handle:
                raise ValueError(f"batch 데이터가 없는 파일입니다: {path.name}")
            raw = handle["batch"]
            required = ("summary", "cycles", "cycle_life", "policy_readable", "Vdlin")
            if any(name not in raw for name in required):
                raise ValueError(f"원본 필드가 부족합니다: {path.name}")
            counts = {name: raw[name].size for name in required}
            if len(set(counts.values())) != 1:
                raise ValueError(f"셀별 필드 개수가 다릅니다: {path.name} {counts}")
            for index in range(raw["summary"].size):
                key = f"{batch}c{index}"
                summary = handle[reference_at(raw["summary"], index)]
                cycle = vector_prefix(summary["cycle"])
                if (len(cycle) != 100 or not np.isfinite(cycle).all()
                        or not np.array_equal(cycle, np.arange(1, 101))):
                    raise ValueError(f"{key}: 실제 cycle1~100 번호를 확인해야 합니다.")
                positions = {number: int(np.flatnonzero(cycle == number)[0]) for number in (10, 100)}
                qd = vector_prefix(summary["QDischarge"])
                if len(qd) != len(cycle):
                    raise ValueError(f"{key}: 초기 사이클 번호와 Qd 길이가 다릅니다.")
                if any(name not in summary for name in ("Tavg", "chargetime")):
                    raise ValueError(f"{key}: 초기 Tavg/chargetime 필드가 없습니다.")
                tavg, chargetime = (vector_prefix(summary[name]) for name in ("Tavg", "chargetime"))
                if any(len(values) != len(cycle) for values in (tavg, chargetime)):
                    raise ValueError(f"{key}: 초기 사이클 번호와 Tavg/chargetime 길이가 다릅니다.")
                cycles = handle[reference_at(raw["cycles"], index)]
                if any(cycles[name].size < 100 for name in ("Qdlin", "I", "t")):
                    raise ValueError(f"{key}: 필요한 사이클 파형이 없습니다.")
                voltage = read_reference_vector(handle, reference_at(raw["Vdlin"], index))
                if len(voltage) != 1000 or not np.isfinite(voltage).all() or not (np.diff(voltage) < 0).all():
                    raise ValueError(f"{key}: 전압축이 예상한 내림차순1000점이 아닙니다.")
                if common_voltage is None:
                    common_voltage = voltage.copy()
                if not np.array_equal(voltage, common_voltage):
                    raise ValueError(f"{key}: 다른 셀과 전압 좌표가 다릅니다. 임의 보간하지 않습니다.")
                life_raw = read_reference_vector(handle, reference_at(raw["cycle_life"], index))
                if life_raw.size != 1:
                    raise ValueError(f"{key}: 셀당 수명 정답은 한 값이어야 합니다.")
                life = float(life_raw[0])
                life = life if np.isfinite(life) and life > 0 else np.nan
                text = handle[reference_at(raw["policy_readable"], index)][()]
                policy = "".join(chr(int(code)) for code in text.reshape(-1) if code)
                yield EarlyCell(
                    key, batch, index, policy, life, cycle, qd, voltage,
                    read_reference_vector(handle, reference_at(cycles["Qdlin"], positions[10])),
                    read_reference_vector(handle, reference_at(cycles["Qdlin"], positions[100])),
                    read_reference_vector(handle, reference_at(cycles["I"], positions[10])),
                    read_reference_vector(handle, reference_at(cycles["t"], positions[10])),
                    tavg=tavg, chargetime=chargetime,
                )
