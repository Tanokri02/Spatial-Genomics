"""Config loading (default.yaml + optional profile/override) and path helpers."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge `upd` into a copy of `base` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(profile: str | None = None, config: str | os.PathLike | None = None) -> dict:
    """Load configs/default.yaml, then configs/<profile>.yaml and/or an explicit override file."""
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    overrides = []
    if profile and profile != "default":
        overrides.append(CONFIG_DIR / f"{profile}.yaml")
    if config:
        overrides.append(Path(config))
    for p in overrides:
        with open(p) as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    cfg["_profile"] = profile
    return cfg


@dataclass(frozen=True)
class Paths:
    root: Path
    derived: Path      # data/derived/<run_name>
    raw: Path          # data/raw (optional local h5ad downloads)
    results: Path      # results/<run_name>

    def shard(self, cell_class: str) -> Path:
        return self.derived / f"cells_{cell_class}.h5ad"

    @property
    def donors(self) -> Path:
        return self.derived / "donors.csv"

    @property
    def genes(self) -> Path:
        return self.derived / "gene_panel.csv"

    @property
    def pseudobulk(self) -> Path:
        return self.derived / "pseudobulk.h5ad"

    @property
    def splits(self) -> Path:
        return self.results / "splits"

    def out(self, *parts: str) -> Path:
        p = self.results.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def get_paths(cfg: dict) -> Paths:
    def resolve(p: str) -> Path:
        p = Path(os.environ.get("PSYCHAD_" + "DATA_DIR", p)) if p == cfg["paths"]["data_dir"] else Path(p)
        return p if p.is_absolute() else REPO_ROOT / p

    data = resolve(cfg["paths"]["data_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir = results_dir if results_dir.is_absolute() else REPO_ROOT / results_dir
    run = cfg["run_name"]
    paths = Paths(root=REPO_ROOT, derived=data / "derived" / run, raw=data / "raw", results=results_dir / run)
    paths.derived.mkdir(parents=True, exist_ok=True)
    paths.results.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
