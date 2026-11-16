"""Rare-class decision rule: per-class log-probability offsets tuned on TRAINING donors only.

With 29 PD / 33 DLB donors the argmax of a calibrated model almost never picks PD or DLB, so their F1 is 0 and the
macro-F1 is capped. Adding an offset b_k to log p_k before the argmax trades some AD/Control accuracy for rare-class
recall. b is chosen to maximise macro-F1 on predictions for donors the fitted model did not train on (inner CV on the
training donors, or the neural nets' validation donors), never on the outer test fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

GRID = np.round(np.linspace(-3.0, 3.0, 25), 3)


def apply_bias(P: np.ndarray, b: np.ndarray) -> np.ndarray:
    """softmax(log P + b): rows still sum to 1; argmax follows the shifted scores."""
    L = np.log(np.clip(P, 1e-9, 1.0)) + b
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def _score(P, y, b, K):
    return f1_score(y, (np.log(np.clip(P, 1e-9, 1.0)) + b).argmax(1), average="macro", labels=list(range(K)),
                    zero_division=0)


def tune_bias(P: np.ndarray, y: np.ndarray, rounds: int = 3, grid=GRID) -> np.ndarray:
    """Coordinate ascent on macro-F1 over per-class offsets (class 0 fixed at 0). Ties -> smaller |offset|."""
    K = P.shape[1]
    y = np.asarray(y).astype(int)
    b = np.zeros(K)
    best = _score(P, y, b, K)
    for _ in range(rounds):
        changed = False
        for k in range(1, K):
            cands = sorted(grid, key=abs)
            for v in cands:
                bb = b.copy()
                bb[k] = v
                s = _score(P, y, bb, K)
                if s > best + 1e-9:
                    best, b, changed = s, bb, True
        if not changed:
            break
    return b


def inner_oof_proba(make_model, X: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> np.ndarray | None:
    """Out-of-fold probabilities on the training donors (stratified inner folds). None if a class is too small."""
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y).astype(int)
    counts = np.bincount(y)
    n_folds = min(n_folds, int(counts[counts > 0].min()))
    if n_folds < 2:
        return None
    K = int(y.max()) + 1
    P = np.zeros((len(y), K))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        P[te, : p.shape[1]] = p
    return P
