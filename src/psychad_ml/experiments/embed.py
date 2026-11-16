"""`psychad embed`: label-free scVI latent for every stored cell -> data/derived/<run>/scvi/latent_<class>.npz

Why: the 5,000-gene input lets a network memorise donors (each donor has its own batch signature) instead of learning
disease. scVI compresses each nucleus to ~30 numbers that model counts with library size and (optionally) cohort as
nuisance factors, so models trained on the latent see less technical variation.

Design / leakage notes
  * scVI never sees labels. It is trained once on a donor-balanced subsample of ALL kept donors (train and test donors
    of every fold), like the label-free HVG panel. This is the standard transductive use of an unsupervised embedding;
    the alternative (one scVI per fold, 15x the cost) is listed as a limitation.
  * batch_key defaults to `cohort` (MSSM vs RADC). NEVER the donor: that would remove donor-level biology, which
    includes the disease. Set embed.scvi.batch_key: null to keep cohort effects in the latent.
  * Memory: the shards hold ~1e9 non-zeros, so scVI trains on <= train_cells_per_donor_class cells per donor x class
    and then encodes every cell shard by shard in chunks.
"""
from __future__ import annotations

import json
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..features import cap_per_group, latent_path, read_donors
from .cv import Ctx


def _accelerator(pref: str) -> str:
    if pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "gpu"
    return "cpu"   # scvi-tools' MPS support is incomplete (sparse ops / some distributions); CPU is the safe default


def run_scvi_embed(ctx: Ctx, force: bool = False) -> dict:
    try:
        import scvi
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install scvi-tools (see environment.yml)") from e

    cfg, paths = ctx.cfg, ctx.paths
    e = cfg["embed"]["scvi"]
    out_dir = paths.derived / "scvi"
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = cfg["cell_classes_use"]
    if not force and all(latent_path(paths, cc).exists() for cc in classes):
        ctx.log(f"scVI latent cached in {out_dir} (use --force to recompute)")
        return json.loads((out_dir / "embed_meta.json").read_text()) if (out_dir / "embed_meta.json").exists() else {}
    scvi.settings.seed = cfg["seed"]
    donors = read_donors(paths)
    keep = set(donors.index[donors["keep"]])
    batch_key = e.get("batch_key")

    # 1. donor-balanced training subsample (raw counts), all classes together
    t0 = time.time()
    Xs, obss = [], []
    for cc in classes:
        a = ad.read_h5ad(paths.shard(cc), backed="r")
        o = a.obs.copy()
        for c in ("donor_id", "cell_class", "cell_type", "cohort"):
            o[c] = o[c].astype(str)
        ok = np.where(o["donor_id"].isin(keep).values)[0]
        pos = ok[cap_per_group(o.iloc[ok], e["train_cells_per_donor_class"], cfg["seed"])]
        Xs.append(sp.csr_matrix(a.X[np.sort(pos)]))
        obss.append(o.iloc[np.sort(pos)])
        var = a.var.copy()
        a.file.close()
    train = ad.AnnData(X=sp.vstack(Xs, format="csr").astype(np.float32), obs=pd.concat(obss), var=var)
    train.obs_names_make_unique()
    ctx.log(f"scVI training set: {train.n_obs:,} cells x {train.n_vars} genes ({time.time() - t0:.0f}s to load)")

    scvi.model.SCVI.setup_anndata(train, batch_key=batch_key)
    model = scvi.model.SCVI(train, n_latent=e["n_latent"], n_hidden=e["n_hidden"], n_layers=e["n_layers"],
                            dropout_rate=e.get("dropout_rate", 0.1), gene_likelihood=e.get("gene_likelihood", "nb"))
    t0 = time.time()
    model.train(max_epochs=e["max_epochs"], batch_size=e["batch_size"], early_stopping=True,
                early_stopping_patience=e.get("early_stopping_patience", 5), train_size=0.9,
                accelerator=_accelerator(e.get("accelerator", "auto")), enable_progress_bar=False)
    hist = model.history
    n_ep = int(len(hist["elbo_train"])) if "elbo_train" in hist else -1
    ctx.log(f"scVI trained: {n_ep} epochs in {(time.time() - t0) / 60:.1f} min")
    model.save(str(out_dir / "model"), overwrite=True, save_anndata=False)

    # 2. encode every stored cell, shard by shard, in chunks (posterior mean of the latent)
    for cc in classes:
        a = ad.read_h5ad(paths.shard(cc), backed="r")
        o = a.obs.copy()
        Z = np.zeros((a.n_obs, e["n_latent"]), np.float32)
        step = e.get("encode_chunk", 50000)
        for i in range(0, a.n_obs, step):
            chunk = ad.AnnData(X=sp.csr_matrix(a.X[i:i + step]).astype(np.float32), obs=o.iloc[i:i + step].copy(),
                               var=var)
            for c in ("cohort",):
                chunk.obs[c] = chunk.obs[c].astype(str)
            Z[i:i + step] = model.get_latent_representation(chunk, batch_size=4096)
        a.file.close()
        np.savez(latent_path(paths, cc), Z=Z, obs_names=np.asarray(o.index.astype(str).values, dtype="U"))  # no pickle
        ctx.log(f"latent {cc}: {Z.shape}")
    meta = {"n_train_cells": int(train.n_obs), "epochs": n_ep, "batch_key": batch_key,
            "elbo_validation_last": float(hist["elbo_validation"].iloc[-1, 0]) if "elbo_validation" in hist else None,
            **{k: e[k] for k in ("n_latent", "n_hidden", "n_layers", "train_cells_per_donor_class")}}
    (out_dir / "embed_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
