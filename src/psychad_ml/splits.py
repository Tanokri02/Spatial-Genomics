"""Donor-level cross-validation folds shared by every experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_folds(donors: pd.DataFrame, n_splits: int, seeds: list[int], stratify_by_cohort: bool = True) -> pd.DataFrame:
    """Return DataFrame[seed, donor_id, fold]. `donors` needs index donor_id and columns task_label, cohort.

    One row per donor, so a plain stratified K-fold on donors guarantees that no donor appears in both train and
    test. Stratification is on label x cohort where every stratum has >= n_splits donors; strata that are too
    small fall back to label only.
    """
    counts = donors["task_label"].value_counts()
    if (counts < n_splits).any():
        raise ValueError(f"Classes with fewer than n_splits={n_splits} donors: {counts[counts < n_splits].to_dict()}")
    key = donors["task_label"].astype(str)
    if stratify_by_cohort:
        full = key + "|" + donors["cohort"].astype(str)
        ok = full.map(full.value_counts()) >= n_splits
        key = full.where(ok, key)
    rows = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold = np.empty(len(donors), dtype=int)
        for f, (_, te) in enumerate(skf.split(np.zeros(len(donors)), key.values)):
            fold[te] = f
        rows.append(pd.DataFrame({"seed": seed, "donor_id": donors.index.values, "fold": fold}))
    return pd.concat(rows, ignore_index=True)


def cell_leaky_folds(y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Deliberately leaky cell-level folds (cells of the same donor land in train and test)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.empty(len(y), dtype=int)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[te] = f
    return fold


def check_no_donor_overlap(train_donors, test_donors) -> None:
    inter = set(train_donors) & set(test_donors)
    if inter:
        raise AssertionError(f"donor leakage: {len(inter)} donors in both train and test")


def load_or_make_folds(path, donors: pd.DataFrame, cfg_cv: dict) -> pd.DataFrame:
    from pathlib import Path

    path = Path(path)
    if path.exists():
        folds = pd.read_csv(path)
        if set(folds["donor_id"]) == set(donors.index):
            return folds
    folds = make_folds(donors, cfg_cv["n_splits"], cfg_cv["seeds"], cfg_cv.get("stratify_by_cohort", True))
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)
    return folds
