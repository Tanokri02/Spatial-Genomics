"""Aim 2: is the disease signal cell-type specific?

Protocol (fairness first):
  * Matched donors: only donors with >= m cells in EVERY cell class, so every class sees the same donors/folds.
  * Matched cells: exactly m cells per donor per class (m = aim2.matched_cells_per_donor).
  * Same donor folds as Aim 1; class-balanced donor-equal weights; donor-level metrics + bootstrap CIs.
  * Downsampling curve: macro-F1 vs cells/donor per class, to show differences are not just data volume.
  * Transfer matrix: train X-vs-Control, test Y-vs-Control, per class (shared vs disease-specific signal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..evaluate import aggregate_donor, bootstrap_ci, donor_metrics
from ..features import CellData, cap_per_group, load_cells
from ..models import build_model, make_sample_weights
from ..splits import load_or_make_folds
from .benchmark import summarize
from .cv import Ctx, run_cv


def matched_data(ctx: Ctx, task: str, m: int | None = None):
    """Load cells for `task`, keep donors with >= m cells in every class, cap to exactly m per donor x class."""
    m = m or ctx.cfg["aim2"]["matched_cells_per_donor"]
    cd = load_cells(ctx.cfg, ctx.paths, task=task, seed=ctx.cfg["seed"])
    n = cd.obs.groupby(["donor_id", "cell_class"]).size().unstack(fill_value=0).reindex(columns=cd.cell_classes, fill_value=0)
    ok = n.index[(n >= m).all(axis=1)]
    cd = cd.subset(cd.obs["donor_id"].isin(ok).values)
    cd = cd.subset(cap_per_group(cd.obs, m, ctx.cfg["seed"]))
    folds = load_or_make_folds(ctx.paths.splits / f"folds_{task}.csv", cd.donors, ctx.cfg["cv"])
    ctx.log(f"matched: {len(ok)} donors with >= {m} cells in all of {cd.cell_classes}; "
            f"donors/class={cd.donors['task_label'].value_counts().to_dict()}")
    return cd, folds


def class_subset(cd: CellData, cell_class: str) -> CellData:
    return cd.subset((cd.obs["cell_class"] == cell_class).values)


def run_celltype(ctx: Ctx, task: str | None = None, models: list[str] | None = None, curve: bool = True) -> pd.DataFrame:
    task = task or ctx.cfg["default_task"]
    models = models or ctx.cfg["aim2"]["models"]
    cd, folds = matched_data(ctx, task)
    out = ctx.paths.out("aim2", task)
    for cc in cd.cell_classes:
        sub = class_subset(cd, cc)
        for mname in models:
            run_cv(ctx, sub, folds, mname, out, tag=f"{cc}__{mname}", train_cap=None)
    # pooled reference: all classes together, same donors and same total cells per donor class
    for mname in models:
        run_cv(ctx, cd, folds, mname, out, tag=f"ALL__{mname}", train_cap=None)
    tbl = summarize(ctx, task, subdir="aim2")
    tbl[["cell_class", "base_model"]] = tbl["model"].str.split("__", n=1, expand=True)
    tbl.to_csv(out / "summary.csv", index=False)
    if curve:
        run_downsample_curve(ctx, task, cd, folds)
    return tbl


def run_downsample_curve(ctx: Ctx, task: str, cd: CellData, folds: pd.DataFrame) -> pd.DataFrame:
    """macro-F1 vs cells/donor for each class (cheap model, first seed)."""
    out = ctx.paths.out("aim2", task)
    mname = ctx.cfg["aim2"]["transfer_model"]
    rows = []
    for n in ctx.cfg["aim2"]["downsample_curve"]:
        sub_all = cd.subset(cap_per_group(cd.obs, n, ctx.cfg["seed"]))
        for cc in cd.cell_classes + ["ALL"]:
            sub = sub_all if cc == "ALL" else class_subset(sub_all, cc)
            preds = run_cv(ctx, sub, folds, mname, out / "curve", tag=f"{cc}__{mname}__n{n}", train_cap=None,
                           seeds=[ctx.cfg["cv"]["seeds"][0]], save_cell_metrics=False)
            avg = preds.groupby(["donor_id", "y_true"], as_index=False).mean(numeric_only=True)
            m = donor_metrics(avg, cd.classes)
            rows.append({"cell_class": cc, "cells_per_donor": n, "macro_f1": m["macro_f1"],
                         "balanced_accuracy": m["balanced_accuracy"], "macro_auroc": m["macro_auroc"]})
    res = pd.DataFrame(rows)
    res.to_csv(out / "downsample_curve.csv", index=False)
    print(res.pivot(index="cells_per_donor", columns="cell_class", values="macro_f1").round(3))
    return res


# -------------------------------------------------------------------------------------------------------
def run_transfer(ctx: Ctx) -> pd.DataFrame:
    """Train 'X vs Control' on training donors, test on held-out 'Y vs Control' donors, per cell class.

    Uses the four_class labels so every disease is available. The diagonal (X == Y) is ordinary donor-held-out CV.
    Held-out donors of disease Y and controls come from the same fold, never from the training donors.
    """
    cfg = ctx.cfg
    cd, folds = matched_data(ctx, "four_class")
    diseases = [c for c in cd.classes if c != "Control"]
    spec = cfg["models"][cfg["aim2"]["transfer_model"]]
    lab = cd.donors["task_label"]
    out = ctx.paths.out("aim2", "transfer")
    rows, preds_all = [], []
    for cc in cd.cell_classes:
        sub = class_subset(cd, cc)
        d_ids = sub.obs["donor_id"].values
        donor_of_cell = sub.obs["donor_id"].map(lab).values
        for seed in sorted(folds["seed"].unique()):
            fs = folds[folds["seed"] == seed].set_index("donor_id")["fold"]
            for fold in sorted(fs.unique()):
                te_all = set(fs.index[fs == fold]) & set(lab.index)
                tr_all = set(fs.index[fs != fold]) & set(lab.index)
                for X in diseases:
                    tr_d = [d for d in tr_all if lab[d] in ("Control", X)]
                    tr = np.where(np.isin(d_ids, tr_d))[0]
                    y = (donor_of_cell[tr] == X).astype(int)
                    if y.sum() == 0 or y.sum() == len(y):
                        continue
                    g = d_ids[tr]
                    model = build_model(spec, 2, seed=seed, device=ctx.device, n_jobs=ctx.n_jobs)
                    model.fit(sub.X[tr], y, make_sample_weights(y, g), g)
                    for Y in diseases:
                        te_d = [d for d in te_all if lab[d] in ("Control", Y)]
                        te = np.where(np.isin(d_ids, te_d))[0]
                        if len(te) == 0:
                            continue
                        dp = aggregate_donor(model.predict_proba(sub.X[te]), d_ids[te], cfg["cv"]["donor_aggregation"])
                        preds_all.append(pd.DataFrame({
                            "cell_class": cc, "train_on": X, "test_on": Y, "seed": seed, "fold": fold,
                            "donor_id": dp.index, "label": [lab[d] for d in dp.index], "p_disease": dp[1].values}))
            ctx.log(f"transfer {cc} seed={seed} done")
    P = pd.concat(preds_all, ignore_index=True)
    P.to_parquet(out / "transfer_preds.parquet")
    for (cc, X, Y), d in P.groupby(["cell_class", "train_on", "test_on"]):
        d = d.groupby(["donor_id", "label"], as_index=False)["p_disease"].mean()   # average over seeds
        yb = (d["label"] == Y).astype(int).values
        if yb.sum() < 2 or (1 - yb).sum() < 2:
            continue
        auc = roc_auc_score(yb, d["p_disease"].values)
        frame = pd.DataFrame({"y_true": np.where(yb == 1, "D", "C"), "p_C": 1 - d["p_disease"].values,
                              "p_D": d["p_disease"].values})
        ci = bootstrap_ci(frame, ["C", "D"], reps=cfg["cv"]["bootstrap_reps"], metrics=("macro_auroc",))["macro_auroc"]
        rows.append({"cell_class": cc, "train_on": X, "test_on": Y, "auroc": auc, "ci_lo": ci[0], "ci_hi": ci[1],
                     "n_disease_donors": int(yb.sum()), "n_control_donors": int((1 - yb).sum())})
    res = pd.DataFrame(rows)
    res.to_csv(out / "transfer_auroc.csv", index=False)
    for cc, d in res.groupby("cell_class"):
        print(f"\n{cc}: AUROC, train on (row) -> test on (col)")
        print(d.pivot(index="train_on", columns="test_on", values="auroc").round(2))
    return res
