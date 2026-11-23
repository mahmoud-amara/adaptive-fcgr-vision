"""Group-aware dataset splitting to prevent data and homology leakage.

Ensures that:
1. No group_id is shared across train and test folds.
2. Label balance is maintained across folds where possible.
3. Strict invariant checks are enforced and verifiable.
"""

from typing import Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold


def assign_group_folds(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    stratified: bool = True,
) -> pd.DataFrame:
    """Assign cross-validation fold index to each record in the manifest.

    Parameters
    ----------
    manifest : pd.DataFrame
        Dataset manifest containing 'label' and 'group_id' columns.
    n_splits : int
        Number of cross-validation folds.
    random_state : int
        Random seed for fold assignment.
    stratified : bool
        Whether to balance labels across group folds (using StratifiedGroupKFold).

    Returns
    -------
    pd.DataFrame
        Updated manifest with a new 'fold' column (0 to n_splits - 1).
    """
    df = manifest.copy()
    if "group_id" not in df.columns:
        raise ValueError("Manifest must contain 'group_id' column for group-aware splitting.")
    if "label" not in df.columns:
        raise ValueError("Manifest must contain 'label' column for splitting.")

    X = np.zeros(len(df))
    y = df["label"].values
    groups = df["group_id"].values

    if stratified and len(np.unique(y)) > 1:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    else:
        splitter = GroupKFold(n_splits=n_splits)

    df["fold"] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups)):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx

    # Verify split invariants
    validate_splits(df)

    return df


def validate_splits(manifest: pd.DataFrame) -> None:
    """Verify that no group_id appears in more than one fold.

    Raises
    ------
    ValueError
        If group leakage is detected across folds.
    """
    if "fold" not in manifest.columns or "group_id" not in manifest.columns:
        raise ValueError("Manifest missing 'fold' or 'group_id' column.")

    group_folds = manifest.groupby("group_id")["fold"].nunique()
    leaked_groups = group_folds[group_folds > 1]
    if len(leaked_groups) > 0:
        raise ValueError(
            f"Group leakage detected! {len(leaked_groups)} groups appear in multiple folds: "
            f"{leaked_groups.index.tolist()[:5]}"
        )


def get_fold_train_val_indices(
    manifest: pd.DataFrame,
    fold_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get train and validation row indices for a specific fold."""
    if "fold" not in manifest.columns:
        raise ValueError("Manifest does not contain 'fold' column.")

    val_mask = manifest["fold"] == fold_idx
    train_mask = manifest["fold"] != fold_idx

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    return train_indices, val_indices
