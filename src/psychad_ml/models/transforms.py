"""Cell-level input transforms, fitted on the TRAINING cells of a fold only and then applied to its test cells.

  {type: pca, n_components: 64, fit_cells: 60000}  z-score genes (clip 10), then PCA. Fitted on a random subset of
                                                    the training cells; turns 5,000 sparse genes into a small dense input
  {type: genes, n: 1000}                            top-n genes of the HVG-ranked panel (column prefix)
  {type: standardize}                               z-score each column (for dense inputs such as the scVI latent)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _dense(X) -> np.ndarray:
    return (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32, copy=False)


class InputTransform:
    def __init__(self, spec: dict | None, seed: int = 0):
        self.spec = dict(spec or {})
        self.type = self.spec.get("type", "none")
        self.seed = seed

    def fit(self, X):
        t = self.type
        if t in ("none", "genes"):
            return self
        rng = np.random.default_rng(self.seed)
        n_fit = min(self.spec.get("fit_cells", 60000), X.shape[0])
        idx = np.sort(rng.choice(X.shape[0], n_fit, replace=False))
        D = _dense(X[idx])
        self.mu = D.mean(0)
        self.sd = D.std(0)
        self.sd[self.sd < 1e-3] = 1e-3
        if t == "standardize":
            return self
        if t != "pca":
            raise ValueError(f"unknown input transform {t}")
        D = np.clip((D - self.mu) / self.sd, -10, 10)
        from sklearn.decomposition import PCA

        k = min(self.spec.get("n_components", 64), D.shape[1], D.shape[0] - 1)
        self.pca = PCA(k, svd_solver="randomized", random_state=self.seed).fit(D)
        return self

    def transform(self, X, batch: int = 20000):
        t = self.type
        if t == "none":
            return X
        if t == "genes":
            return X[:, : self.spec["n"]]
        out = []
        for i in range(0, X.shape[0], batch):
            D = (_dense(X[i:i + batch]) - self.mu) / self.sd
            if t == "pca":
                D = self.pca.transform(np.clip(D, -10, 10))
            out.append(D.astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, self.n_out), np.float32)

    @property
    def n_out(self) -> int | None:
        return self.pca.n_components_ if self.type == "pca" else None
