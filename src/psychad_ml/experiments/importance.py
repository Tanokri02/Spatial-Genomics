"""Which genes drive the predictions, and how stable are they?

  * Logistic-regression coefficients per (cell class, disease): mean, sd, sign consistency and top-k frequency
    across donor folds/seeds. Only genes that are stable across folds are reported as 'stable'.
  * Optional XGBoost SHAP (mean |SHAP|) on held-out cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import cap_per_group
from ..models import build_model, make_sample_weights
from .benchmark import prepare
from .cv import Ctx


def run_lr_importance(ctx: Ctx, task: str | None = None, cell_classes: list[str] | None = None) -> pd.DataFrame:
    task = task or ctx.cfg["default_task"]
    cd, folds = prepare(ctx, task)
    spec = ctx.cfg["models"][ctx.cfg["importance"]["model"]]
    assert spec["family"] == "logreg", "importance.model must be a logistic regression"
    top_k = ctx.cfg["importance"]["top_k"]
    cap = ctx.cfg["cv"]["train_cells_per_donor_class"]
    K = len(cd.classes)
    genes = cd.genes
    rows = []
    for cc in (cell_classes or cd.cell_classes + ["ALL"]):
        sub = cd if cc == "ALL" else cd.subset((cd.obs["cell_class"] == cc).values)
        coefs = []
        for seed in sorted(folds["seed"].unique()):
            fs = folds[folds["seed"] == seed].set_index("donor_id")["fold"]
            for fold in sorted(fs.unique()):
                tr_d = set(fs.index[fs != fold])
                idx = np.where(sub.obs["donor_id"].isin(tr_d).values)[0]
                idx = idx[cap_per_group(sub.obs.iloc[idx], cap, seed)]
                g, y = sub.obs["donor_id"].values[idx], sub.obs["y"].values[idx]
                m = build_model(spec, K, seed=seed, n_jobs=ctx.n_jobs)
                m.fit(sub.X[idx], y, make_sample_weights(y, g), g)
                c = m.coef()
                coefs.append(c if c.shape[0] == K else np.vstack([-c[0], c[0]]))   # binary -> 2 symmetric rows
            ctx.log(f"importance {cc} seed={seed}")
        C = np.stack(coefs)                                   # (runs, K, genes)
        for k, cls in enumerate(cd.classes):
            ck = C[:, k, :]
            topk = np.argsort(-np.abs(ck), axis=1)[:, :top_k]
            freq = np.bincount(topk.ravel(), minlength=ck.shape[1]) / ck.shape[0]
            mean = ck.mean(0)
            consistency = np.abs(np.sign(ck).mean(0))        # 1 = same sign in every run
            rows.append(pd.DataFrame({
                "cell_class": cc, "class": cls, "feature_id": genes.index, "gene": genes["feature_name"].values,
                "mean_coef": mean, "sd_coef": ck.std(0), "sign_consistency": consistency, "topk_freq": freq}))
    res = pd.concat(rows, ignore_index=True)
    res["stable"] = (res["topk_freq"] >= 0.8) & (res["sign_consistency"] >= 0.8)
    out = ctx.paths.out("importance", task)
    res.to_csv(out / "lr_importance.csv", index=False)
    print(res[res["stable"]].groupby(["cell_class", "class"]).size().unstack(fill_value=0))
    return res


def run_shap(ctx: Ctx, task: str | None = None, cell_classes: list[str] | None = None) -> pd.DataFrame:
    try:
        import shap
    except ImportError as e:
        raise ImportError("pip install shap") from e
    task = task or ctx.cfg["default_task"]
    cd, folds = prepare(ctx, task)
    spec = ctx.cfg["models"][ctx.cfg["importance"]["shap_model"]]
    assert spec["family"] == "xgb"
    K = len(cd.classes)
    n_cells = ctx.cfg["importance"]["shap_cells"]
    cap = ctx.cfg["cv"]["train_cells_per_donor_class"]
    seed = ctx.cfg["cv"]["seeds"][0]
    fs = folds[folds["seed"] == seed].set_index("donor_id")["fold"]
    rows = []
    for cc in (cell_classes or cd.cell_classes):
        sub = cd.subset((cd.obs["cell_class"] == cc).values)
        for fold in sorted(fs.unique()):
            tr = np.where(sub.obs["donor_id"].isin(set(fs.index[fs != fold])).values)[0]
            tr = tr[cap_per_group(sub.obs.iloc[tr], cap, seed)]
            te = np.where(sub.obs["donor_id"].isin(set(fs.index[fs == fold])).values)[0]
            te = np.random.default_rng(seed).choice(te, min(n_cells, len(te)), replace=False)
            g, y = sub.obs["donor_id"].values[tr], sub.obs["y"].values[tr]
            m = build_model(spec, K, seed=seed, device=ctx.device, n_jobs=ctx.n_jobs)
            m.fit(sub.X[tr], y, make_sample_weights(y, g), g)
            sv = shap.TreeExplainer(m.est).shap_values(sub.X[te].toarray())
            sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)
            if sv.ndim == 2:                                # binary: (cells, genes)
                sv = np.stack([-sv, sv], -1)
            mean_abs = np.abs(sv).mean(0)                   # (genes, K)
            for k, cls in enumerate(cd.classes):
                rows.append(pd.DataFrame({"cell_class": cc, "class": cls, "fold": fold,
                                          "gene": cd.genes["feature_name"].values, "mean_abs_shap": mean_abs[:, k]}))
            ctx.log(f"shap {cc} fold={fold}")
    res = pd.concat(rows, ignore_index=True)
    agg = res.groupby(["cell_class", "class", "gene"], as_index=False)["mean_abs_shap"].agg(["mean", "std"]).reset_index()
    agg.to_csv(ctx.paths.out("importance", task) / "shap_importance.csv", index=False)
    return agg
