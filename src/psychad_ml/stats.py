"""Paired comparisons of saved predictions (reads results only; trains nothing). `psychad compare`

Two models scored on the same donors are not independent, so "do their CIs overlap?" is the wrong test. Here every
bootstrap replicate resamples donors ONCE and scores both models on that same sample; the distribution of the
difference gives a CI and a one-sided p-value (P[delta <= 0]). With many models, Holm's step-down correction is
applied within each (task, metric).

Output: results/<run>/aim1/<task>/paired_vs_baseline.csv  (each model vs the best baseline: cov_lr / cov_rf / prior)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import classification_metrics, prob_cols

BASELINES = ("prior", "cov_lr", "cov_rf")
METRICS = ("macro_f1", "balanced_accuracy", "macro_auroc")


def seed_averaged(preds: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """One row per (model, donor): probabilities averaged over CV seeds."""
    return preds.groupby(["model", "donor_id", "y_true"], as_index=False)[prob_cols(classes)].mean()


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, classes: list[str], reps: int = 2000, seed: int = 0,
                     metrics=METRICS) -> dict:
    """delta = metric(a) - metric(b) on the donors both have. a, b: one row per donor (donor_id, y_true, p_*)."""
    m = a.merge(b, on=["donor_id", "y_true"], suffixes=("_a", "_b"))
    cmap = {c: i for i, c in enumerate(classes)}
    y = m["y_true"].map(cmap).values.astype(int)
    pa = m[[f"{c}_a" for c in prob_cols(classes)]].values
    pb = m[[f"{c}_b" for c in prob_cols(classes)]].values
    K = len(classes)
    obs_a, obs_b = classification_metrics(y, pa, K), classification_metrics(y, pb, K)
    rng = np.random.default_rng(seed)
    d = {k: [] for k in metrics}
    n = len(y)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        ra, rb = classification_metrics(y[idx], pa[idx], K), classification_metrics(y[idx], pb[idx], K)
        for k in metrics:
            d[k].append(ra[k] - rb[k])
    out = {"n_donors": n}
    for k in metrics:
        v = np.asarray(d[k])
        out[f"{k}_a"], out[f"{k}_b"] = obs_a[k], obs_b[k]
        out[f"{k}_delta"] = obs_a[k] - obs_b[k]
        out[f"{k}_delta_ci_lo"], out[f"{k}_delta_ci_hi"] = np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)
        out[f"{k}_p"] = float((1 + np.sum(v <= 0)) / (1 + len(v)))     # one-sided: is a better than b?
    return out


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values (family = the given series)."""
    p = p.astype(float)
    order = p.sort_values().index
    m = len(p)
    adj, running = pd.Series(index=p.index, dtype=float), 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return adj


def compare_to_baseline(task_dir: Path, classes: list[str] | None = None, reps: int = 2000, seed: int = 0,
                        baseline: str | None = None) -> pd.DataFrame | None:
    """Every non-baseline model in <task_dir>/preds vs the best baseline (by seed-averaged macro-F1)."""
    from .experiments.cv import read_preds

    preds = read_preds(task_dir)
    if preds.empty:
        return None
    classes = classes or [c[2:] for c in preds.columns if c.startswith("p_")]
    avg = seed_averaged(preds, classes)
    groups = dict(tuple(avg.groupby("model")))
    bases = [b for b in BASELINES if b in groups]
    if not bases:
        return None
    if baseline is None:
        cmap = {c: i for i, c in enumerate(classes)}

        def f1(g):
            return classification_metrics(g["y_true"].map(cmap).values, g[prob_cols(classes)].values, len(classes))["macro_f1"]
        baseline = max(bases, key=lambda b: f1(groups[b]))
    rows = []
    for name, g in sorted(groups.items()):
        if name in BASELINES:
            continue
        r = paired_bootstrap(g.drop(columns="model"), groups[baseline].drop(columns="model"), classes, reps, seed)
        rows.append({"model": name, "baseline": baseline, **r})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    for k in METRICS:
        res[f"{k}_p_holm"] = holm(res[f"{k}_p"]).values
    res.to_csv(task_dir / "paired_vs_baseline.csv", index=False)
    return res
