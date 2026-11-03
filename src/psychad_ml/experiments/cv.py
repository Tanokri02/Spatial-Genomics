"""Shared cross-validation runner: one (model, seed, fold) at a time, resumable, donor-level outputs.

All experiments (Aim 1 benchmark, controls, Aim 2, scaling) go through `run_cv`, so they share folds, class
weighting, donor aggregation and on-disk formats:

    <out_dir>/preds/<tag>__s<seed>_f<fold>.parquet   donor-level probabilities for the held-out donors
    <out_dir>/preds/<tag>__s<seed>_f<fold>.json      n_params, timings, cell-level metrics
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths, resolve_device
from ..evaluate import aggregate_donor, cell_level_metrics, prob_cols
from ..features import CellData, cap_per_group, donor_labels, pseudobulk
from ..models import build_model, make_sample_weights
from ..splits import check_no_donor_overlap


@dataclass
class Ctx:
    cfg: dict
    paths: Paths
    device: str = "cpu"
    n_jobs: int = -1
    verbose: bool = True

    @classmethod
    def create(cls, cfg: dict, paths: Paths) -> Ctx:
        return cls(cfg, paths, resolve_device(cfg["compute"]["device"]), cfg["compute"]["n_jobs"])

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pb(cd: CellData, features: str) -> pd.DataFrame:
    """Pseudobulk features, cached on the CellData object (they do not depend on labels or folds)."""
    cache = cd.__dict__.setdefault("_pb_cache", {})
    if features not in cache:
        cache[features] = pseudobulk(cd, by_class=(features == "pseudobulk_by_class"))
    return cache[features]


def _class_onehot(cd: CellData) -> np.ndarray:
    return pd.get_dummies(pd.Categorical(cd.obs["cell_class"], categories=cd.cell_classes)).values.astype(np.float32)


def _donor_df(ids, proba, y_names: dict, classes: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(np.asarray(proba), columns=prob_cols(classes))
    df.insert(0, "donor_id", list(ids))
    df.insert(1, "y_true", [y_names[d] for d in ids])
    return df


def fit_predict_fold(ctx: Ctx, cd: CellData, spec: dict, train_donors: list, test_donors: list, seed: int,
                     train_cap: int | None, save_cell_metrics: bool = True) -> tuple[pd.DataFrame, dict]:
    """Train on train_donors, predict test_donors. Returns (donor-level predictions, meta)."""
    check_no_donor_overlap(train_donors, test_donors)
    classes, K = cd.classes, len(cd.classes)
    dlab = donor_labels(cd)
    y_names = {d: classes[i] for d, i in dlab.items()}
    kind = spec["kind"]
    t0 = time.time()
    meta: dict = {"kind": kind}
    model = build_model(spec, K, seed=seed, device=ctx.device, n_jobs=ctx.n_jobs)

    if kind == "donor":
        pb = _pb(cd, spec["features"])
        tr = [d for d in train_donors if d in pb.index]
        te = [d for d in test_donors if d in pb.index]
        model.fit(pb.loc[tr].values, dlab.loc[tr].values)
        proba = model.predict_proba(pb.loc[te].values)
        meta["n_features"] = pb.shape[1]
        meta["n_train_donors"] = len(tr)
        ids = te
    else:
        in_tr = cd.obs["donor_id"].isin(train_donors).values
        tr_idx = np.where(in_tr)[0]
        sub = cd.obs.iloc[tr_idx]
        tr_idx = tr_idx[cap_per_group(sub, train_cap, seed)]
        te_idx = np.where(cd.obs["donor_id"].isin(test_donors).values)[0]
        g_tr = cd.obs["donor_id"].values[tr_idx]
        y_tr = cd.obs["y"].values[tr_idx]
        meta["n_train_cells"], meta["n_features"] = len(tr_idx), int(cd.X.shape[1])
        if kind == "cell":
            model.fit(cd.X[tr_idx], y_tr, make_sample_weights(y_tr, g_tr), g_tr)
            cell_p = model.predict_proba(cd.X[te_idx])
            donor_p = aggregate_donor(cell_p, cd.obs["donor_id"].values[te_idx], ctx.cfg["cv"]["donor_aggregation"])
            proba, ids = donor_p.values, list(donor_p.index)
            if save_cell_metrics:
                meta.update(cell_level_metrics(cd.obs["y"].values[te_idx], cell_p, K))
        elif kind == "bag":
            extra = _class_onehot(cd)
            tr_labels = dlab.loc[sorted(set(g_tr))]
            model.fit(cd.X[tr_idx], g_tr, tr_labels, extra=extra[tr_idx])
            donor_p = model.predict_donor_proba(cd.X[te_idx], cd.obs["donor_id"].values[te_idx], extra=extra[te_idx])
            proba, ids = donor_p.values, list(donor_p.index)
        else:
            raise ValueError(kind)
    meta["n_params"] = int(getattr(model, "n_params", 0) or (
        np.size(model.coef()) if hasattr(model, "coef") and model.coef() is not None else 0))
    meta["fit_predict_seconds"] = round(time.time() - t0, 1)
    return _donor_df(ids, proba, y_names, classes), meta


def run_cv(ctx: Ctx, cd: CellData, folds: pd.DataFrame, model_name: str, out_dir: Path, tag: str | None = None,
           spec: dict | None = None, seeds: list[int] | None = None, train_cap: int | None = "cfg",
           save_cell_metrics: bool = True) -> pd.DataFrame:
    """Run (all seeds x all folds) for one model. Returns the concatenated donor-level predictions."""
    spec = spec or ctx.cfg["models"][model_name]
    tag = tag or model_name
    train_cap = ctx.cfg["cv"]["train_cells_per_donor_class"] if train_cap == "cfg" else train_cap
    pdir = Path(out_dir) / "preds"
    pdir.mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds is not None else sorted(folds["seed"].unique())
    donors_here = set(cd.donors.index)
    outs = []
    for seed in seeds:
        f_s = folds[(folds["seed"] == seed) & folds["donor_id"].isin(donors_here)]
        for fold in sorted(f_s["fold"].unique()):
            path = pdir / f"{tag}__s{seed}_f{fold}.parquet"
            if path.exists():
                outs.append(pd.read_parquet(path))
                continue
            te = list(f_s.loc[f_s["fold"] == fold, "donor_id"])
            tr = list(f_s.loc[f_s["fold"] != fold, "donor_id"])
            ctx.log(f"{tag} seed={seed} fold={fold}  train_donors={len(tr)} test_donors={len(te)}")
            df, meta = fit_predict_fold(ctx, cd, spec, tr, te, seed, train_cap, save_cell_metrics)
            df.insert(0, "fold", fold)
            df.insert(0, "seed", seed)
            df.insert(0, "model", tag)
            df.to_parquet(path)
            path.with_suffix(".json").write_text(json.dumps(meta))
            outs.append(df)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


def read_preds(out_dir: Path, tags: list[str] | None = None) -> pd.DataFrame:
    files = sorted((Path(out_dir) / "preds").glob("*.parquet"))
    if tags is not None:
        files = [f for f in files if f.name.rsplit("__s", 1)[0] in tags]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def read_meta(out_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((Path(out_dir) / "preds").glob("*.json")):
        tag, rest = f.stem.rsplit("__s", 1)
        seed, fold = rest.split("_f")
        rows.append({"model": tag, "seed": int(seed), "fold": int(fold), **json.loads(f.read_text())})
    return pd.DataFrame(rows)
