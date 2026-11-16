"""scikit-learn / XGBoost wrappers with a common fit / predict_proba interface.

Used for both cell-level models (sparse CSR input) and donor-level models (dense pseudobulk input).

Extra options for dense (donor-level) logistic regression, all fitted inside the training fold:
  select_k: 3000        keep the k features with the highest ANOVA F on the training donors
  block_pca: 20         PCA with this many components per feature block ('EN|gene' -> block EN), then LR
  l1_ratios: [..]       elastic net (saga), l1_ratio chosen with C by the inner CV (needs Cs)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BlockPCA(BaseEstimator, TransformerMixin):
    """PCA per column block; blocks with <= n_components columns pass through unchanged."""

    def __init__(self, blocks=None, n_components: int = 20, seed: int = 0):
        self.blocks, self.n_components, self.seed = blocks, n_components, seed

    def fit(self, X, y=None):
        from sklearn.decomposition import PCA

        self.parts_ = []
        for b in np.unique(self.blocks):
            cols = np.where(self.blocks == b)[0]
            k = min(self.n_components, len(cols), X.shape[0] - 1)
            pca = PCA(k, svd_solver="randomized", random_state=self.seed).fit(X[:, cols]) if len(cols) > self.n_components else None
            self.parts_.append((cols, pca))
        return self

    def transform(self, X):
        return np.hstack([p.transform(X[:, c]) if p is not None else X[:, c] for c, p in self.parts_])


class SkModel:
    def __init__(self, family: str, params: dict, n_classes: int, seed: int = 0, device: str = "cpu",
                 n_jobs: int = -1):
        self.family, self.params, self.n_classes = family, dict(params), n_classes
        self.seed, self.device, self.n_jobs = seed, device, n_jobs
        self.est = None
        self.feature_names = None

    def set_feature_names(self, names) -> None:
        self.feature_names = np.asarray(names).astype(str)

    def _blocks(self, n_features: int) -> np.ndarray:
        if self.feature_names is None or len(self.feature_names) != n_features:
            raise ValueError("block_pca needs feature names ('block|feature'); call set_feature_names first")
        return np.array([n.split("|", 1)[0] for n in self.feature_names])

    def _build(self, dense: bool, n_features: int = 0):
        p = dict(self.params)
        if self.family == "logreg":
            penalty = p.pop("penalty", "l2")
            Cs = p.pop("Cs", None)
            l1_ratios = p.pop("l1_ratios", None)
            select_k = p.pop("select_k", None)
            block_pca = p.pop("block_pca", None)
            max_iter = p.pop("max_iter", 500)
            pre = [("scaler", StandardScaler())] if dense else []
            if dense and select_k and select_k < n_features:
                pre.append(("select", SelectKBest(f_classif, k=select_k)))
            if dense and block_pca:
                if select_k:
                    raise ValueError("use select_k or block_pca, not both")
                pre += [("bpca", BlockPCA(self._blocks(n_features), block_pca, self.seed)), ("scaler2", StandardScaler())]
            if Cs is not None:   # inner 3-fold CV over C (use for small donor-level data; too slow for millions of cells)
                kw = dict(Cs=list(Cs), cv=3, scoring="balanced_accuracy", max_iter=max_iter, n_jobs=1)
                if l1_ratios:
                    est = LogisticRegressionCV(l1_ratios=list(l1_ratios), solver="saga", random_state=self.seed,
                                               use_legacy_attributes=False, **kw)
                else:
                    est = LogisticRegressionCV(solver="lbfgs", **kw)
                return Pipeline([*pre, ("clf", est)]) if dense else est
            kw = dict(C=p.pop("C", 1.0), max_iter=max_iter)
            if penalty == "l2":
                est = LogisticRegression(solver="lbfgs", **kw)
            elif penalty == "l1":
                est = LogisticRegression(l1_ratio=1.0, solver="saga", **kw)
            else:  # elasticnet
                est = LogisticRegression(l1_ratio=p.pop("l1_ratio", 0.5), solver="saga", **kw)
            return Pipeline([*pre, ("clf", est)]) if dense else est
        if self.family == "rf":
            return RandomForestClassifier(n_jobs=self.n_jobs, random_state=self.seed, **p)
        if self.family == "xgb":
            from xgboost import XGBClassifier

            dev = "cuda" if self.device == "cuda" else "cpu"
            return XGBClassifier(tree_method="hist", device=dev, n_jobs=max(0, self.n_jobs),
                                 random_state=self.seed, eval_metric="mlogloss", **p)
        raise ValueError(f"unknown family {self.family}")

    def fit(self, X, y, w=None, groups=None):
        self.est = self._build(dense=not sp.issparse(X), n_features=X.shape[1])
        if self.family == "logreg" and not sp.issparse(X):
            self.est.fit(X, y, clf__sample_weight=w)
        else:
            self.est.fit(X, y, sample_weight=w)
        self.classes_ = np.asarray(self.est.classes_ if hasattr(self.est, "classes_") else self.est[-1].classes_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self.est.predict_proba(X)
        if p.shape[1] != self.n_classes:  # a class was missing from training
            full = np.zeros((p.shape[0], self.n_classes), dtype=p.dtype)
            full[:, self.classes_] = p
            p = full
        return p

    def coef(self) -> np.ndarray | None:
        """(n_classes, n_features) linear coefficients for logreg, else None (in the model's input space)."""
        est = self.est[-1] if hasattr(self.est, "steps") else self.est
        return getattr(est, "coef_", None)
