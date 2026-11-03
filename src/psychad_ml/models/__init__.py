"""Model factory. A model spec comes from cfg['models'][name]."""
from __future__ import annotations

from .sk_models import SkModel
from .weights import make_sample_weights

__all__ = ["SkModel", "build_model", "make_sample_weights"]


def build_model(spec: dict, n_classes: int, seed: int = 0, device: str = "cpu", n_jobs: int = -1):
    fam = spec["family"]
    if fam in ("logreg", "rf", "xgb"):
        return SkModel(fam, spec["params"], n_classes, seed=seed, device=device, n_jobs=n_jobs)
    if fam == "mlp":
        from .torch_models import MLPModel

        return MLPModel(spec["params"], n_classes, seed=seed, device=device)
    if fam == "mil":
        from .torch_models import MILModel

        return MILModel(spec["params"], n_classes, seed=seed, device=device)
    raise ValueError(f"unknown model family: {fam}")
