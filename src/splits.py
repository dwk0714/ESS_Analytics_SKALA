"""Split B1 by its original charging-policy strings, not random cycle rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .features import check_feature_table


def labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    mask = np.isfinite(frame.cycle_life) & (frame.cycle_life > 0)
    return frame.loc[mask].copy().reset_index(drop=True)


def make_splits(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    check_feature_table(frame, config)
    labeled = labeled_rows(frame)
    b1 = labeled.loc[labeled.batch.eq("B1")].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=config["holdout_group_fraction"],
                                 random_state=config["random_state"])
    training, holdout = next(splitter.split(b1, groups=b1.policy))
    train_policies, holdout_policies = set(b1.iloc[training].policy), set(b1.iloc[holdout].policy)
    if train_policies & holdout_policies:
        raise ValueError("학습과 holdout에 같은 충전 프로토콜이 포함됐습니다.")
    if len(training) != config["expected_development_rows"] or len(holdout) != config["expected_holdout_rows"]:
        raise ValueError("B1 학습35개/holdout11개 분할과 다릅니다.")
    if holdout_policies != set(config["holdout_policies"]):
        raise ValueError("고정한 holdout 충전 프로토콜 목록과 다릅니다.")
    roles = {key: "train" for key in b1.iloc[training].cell_key}
    roles.update({key: "valid" for key in b1.iloc[holdout].cell_key})
    manifest = labeled[["cell_key", "batch", "policy"]].copy()
    manifest["role"] = [roles[key] if batch == "B1" else f"test_{batch.lower()}"
                        for key, batch in zip(manifest.cell_key, manifest.batch)]
    return manifest


def development_data(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    keys = manifest.loc[manifest.role.eq("train"), "cell_key"]
    return frame.set_index("cell_key", drop=False).loc[keys].reset_index(drop=True)


def group_folds(development: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=n_splits)
    folds = list(splitter.split(development, groups=development.policy))
    coverage = np.zeros(len(development), dtype=int)
    for train, valid in folds:
        if set(development.iloc[train].policy) & set(development.iloc[valid].policy):
            raise ValueError("CV 학습과 평가에 같은 충전 프로토콜이 포함됐습니다.")
        coverage[valid] += 1
    if not np.all(coverage == 1):
        raise ValueError("각 학습 셀은 CV 평가에 정확히 한 번 포함돼야 합니다.")
    return folds

