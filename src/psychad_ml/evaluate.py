"""Metrics, donor-level aggregation and donor-bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "log_loss")


def prob_cols(classes: list[str]) -> list[str]:
    return [f"p_{c}" for c in classes]


def aggregate_donor(proba: np.ndarray, donor_ids: np.ndarray, method: str = "mean_prob") -> pd.DataFrame:
    """Cell-level class probabilities -> donor-level probabilities (rows sum to 1)."""
    df = pd.DataFrame(proba)
    if method == "mean_logprob":
        df = np.log(np.clip(df, 1e-6, 1.0))
    elif method == "vote":
        df = pd.get_dummies(df.values.argmax(1)).reindex(columns=range(proba.shape[1]), fill_value=0).astype(float)
    g = df.groupby(np.asarray(donor_ids), sort=True).mean()
    v = g.values
    if method == "mean_logprob":
        v = np.exp(v - v.max(1, keepdims=True))
    v = v / v.sum(1, keepdims=True)
    return pd.DataFrame(v, index=g.index)


def classification_metrics(y: np.ndarray, proba: np.ndarray, n_classes: int) -> dict:
    """y: integer labels, proba: (n, n_classes). AUROC is OvR-macro over classes present in y."""
    pred = proba.argmax(1)
    out = {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0)),
    }
    present = [k for k in range(n_classes) if (y == k).any() and (y != k).any()]
    if n_classes == 2 and len(present) == 2:
        out["macro_auroc"] = float(roc_auc_score(y, proba[:, 1]))
    elif present:
        out["macro_auroc"] = float(np.mean([roc_auc_score((y == k).astype(int), proba[:, k]) for k in present]))
    else:
        out["macro_auroc"] = float("nan")
    try:
        out["log_loss"] = float(log_loss(y, np.clip(proba, 1e-6, 1), labels=list(range(n_classes))))
    except ValueError:
        out["log_loss"] = float("nan")
    for k in range(n_classes):
        m = y == k
        out[f"recall_{k}"] = float((pred[m] == k).mean()) if m.any() else float("nan")
    return out


def preds_to_arrays(preds: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y = preds["y_true"].map({c: i for i, c in enumerate(classes)}).values.astype(int)
    return y, preds[prob_cols(classes)].values


def donor_metrics(preds: pd.DataFrame, classes: list[str]) -> dict:
    y, p = preds_to_arrays(preds, classes)
    m = classification_metrics(y, p, len(classes))
    for k, c in enumerate(classes):
        m[f"recall_{c}"] = m.pop(f"recall_{k}")
    return m


def bootstrap_ci(preds: pd.DataFrame, classes: list[str], reps: int = 1000, seed: int = 0,
                 metrics: tuple[str, ...] = ("macro_f1", "balanced_accuracy", "macro_auroc")) -> dict:
    """Percentile CI by resampling donors (one row per donor in `preds`)."""
    y, p = preds_to_arrays(preds, classes)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {m: [] for m in metrics}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        r = classification_metrics(y[idx], p[idx], len(classes))
        for m in metrics:
            draws[m].append(r[m])
    return {m: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) if v else (np.nan, np.nan)
            for m, v in draws.items()}


def confusion_df(preds: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    y, p = preds_to_arrays(preds, classes)
    cm = confusion_matrix(y, p.argmax(1), labels=list(range(len(classes))))
    return pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])


def summarize_preds(preds: pd.DataFrame, classes: list[str], reps: int = 1000, seed: int = 0) -> pd.DataFrame:
    """One row per (model, level). `preds` has donor-level rows with columns
    model, seed, fold, donor_id, y_true, p_<class>.

    Per-seed metrics pool out-of-fold donors; we report mean/sd over seeds, and a donor-bootstrap CI
    computed on the seed-averaged probabilities.
    """
    rows = []
    for model, d in preds.groupby("model"):
        per_seed = [donor_metrics(s, classes) for _, s in d.groupby("seed")]
        ps = pd.DataFrame(per_seed)
        avg = (d.groupby(["donor_id", "y_true"], as_index=False)[prob_cols(classes)].mean())
        ci = bootstrap_ci(avg, classes, reps=reps, seed=seed)
        row = {"model": model, "n_donors": avg.shape[0], "n_seeds": len(per_seed)}
        for m in ps.columns:
            row[m] = ps[m].mean()
            if m in ("macro_f1", "balanced_accuracy", "macro_auroc", "accuracy"):
                row[m + "_sd"] = ps[m].std(ddof=0)
        for m, (lo, hi) in ci.items():
            row[m + "_ci_lo"], row[m + "_ci_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def cell_level_metrics(y: np.ndarray, proba: np.ndarray, n_classes: int) -> dict:
    return {f"cell_{k}": v for k, v in classification_metrics(y, proba, n_classes).items()
            if not k.startswith("recall")}
