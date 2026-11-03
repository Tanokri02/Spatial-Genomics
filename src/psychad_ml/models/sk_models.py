"""scikit-learn / XGBoost wrappers with a common fit / predict_proba interface.

Used for both cell-level models (sparse CSR input) and donor-level models (dense pseudobulk input).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SkModel:
    def __init__(self, family: str, params: dict, n_classes: int, seed: int = 0, device: str = "cpu",
                 n_jobs: int = -1):
        self.family, self.params, self.n_classes = family, dict(params), n_classes
        self.seed, self.device, self.n_jobs = seed, device, n_jobs
        self.est = None

    def _build(self, dense: bool):
        p = dict(self.params)
        if self.family == "logreg":
            penalty = p.pop("penalty", "l2")
            Cs = p.pop("Cs", None)
            if Cs is not None:   # inner 3-fold CV over C (use for small donor-level data; too slow for millions of cells)
                est = LogisticRegressionCV(Cs=list(Cs), cv=3, scoring="balanced_accuracy", solver="lbfgs",
                                           max_iter=p.pop("max_iter", 500), n_jobs=1)
                return Pipeline([("scaler", StandardScaler()), ("clf", est)]) if dense else est
            kw = dict(C=p.pop("C", 1.0), max_iter=p.pop("max_iter", 500))
            if penalty == "l2":
                est = LogisticRegression(solver="lbfgs", **kw)
            elif penalty == "l1":
                est = LogisticRegression(penalty="l1", solver="saga", **kw)
            else:  # elasticnet
                est = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=p.pop("l1_ratio", 0.5), **kw)
            return Pipeline([("scaler", StandardScaler()), ("clf", est)]) if dense else est
        if self.family == "rf":
            return RandomForestClassifier(n_jobs=self.n_jobs, random_state=self.seed, **p)
        if self.family == "xgb":
            from xgboost import XGBClassifier

            dev = "cuda" if self.device == "cuda" else "cpu"
            return XGBClassifier(tree_method="hist", device=dev, n_jobs=max(0, self.n_jobs),
                                 random_state=self.seed, eval_metric="mlogloss", **p)
        raise ValueError(f"unknown family {self.family}")

    def fit(self, X, y, w=None, groups=None):
        self.est = self._build(dense=not sp.issparse(X))
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
        """(n_classes, n_features) linear coefficients for logreg, else None."""
        est = self.est[-1] if hasattr(self.est, "steps") else self.est
        return getattr(est, "coef_", None)
