"""Aim 1: benchmark classifiers with donor-aware splits, then summarise (donor-level metrics + bootstrap CIs)."""
from __future__ import annotations

import pandas as pd

from ..evaluate import confusion_df, prob_cols, summarize_preds
from ..features import load_cells
from ..splits import load_or_make_folds
from .cv import Ctx, read_meta, read_preds, run_cv


def prepare(ctx: Ctx, task: str, **load_kw):
    """Load all cells + donor folds for a task. Folds are cached so every experiment shares them."""
    load_kw.setdefault("cells_per_donor_class", ctx.cfg["cv"].get("load_cells_per_donor_class"))
    cd = load_cells(ctx.cfg, ctx.paths, task=task, seed=ctx.cfg["seed"], **load_kw)
    folds = load_or_make_folds(ctx.paths.splits / f"folds_{task}.csv", cd.donors, ctx.cfg["cv"])
    ctx.log(f"task={task}: {cd.n_cells:,} cells, {len(cd.donors)} donors, classes={cd.classes}, "
            f"donors/class={cd.donors['task_label'].value_counts().to_dict()}")
    return cd, folds


def run_benchmark(ctx: Ctx, task: str | None = None, models: list[str] | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    task = task or ctx.cfg["default_task"]
    models = models or ctx.cfg["benchmark_models"]
    cd, folds = prepare(ctx, task)
    out = ctx.paths.out("aim1", task)
    for m in models:
        run_cv(ctx, cd, folds, m, out, seeds=seeds)
    return summarize(ctx, task)


def summarize(ctx: Ctx, task: str, subdir: str = "aim1", tags: list[str] | None = None) -> pd.DataFrame:
    """Write summary.csv (one row per model), confusion matrices and per-class recalls."""
    out = ctx.paths.out(subdir, task)
    preds = read_preds(out, tags)
    if preds.empty:
        raise FileNotFoundError(f"no predictions in {out}/preds - run the benchmark first")
    classes = [c[2:] for c in prob_cols_from(preds)]
    tbl = summarize_preds(preds, classes, reps=ctx.cfg["cv"]["bootstrap_reps"], seed=ctx.cfg["seed"])
    meta = read_meta(out)
    if not meta.empty:
        spec = {"n_params": ("n_params", "max")}
        if "fit_predict_seconds" in meta:
            spec["fit_predict_seconds"] = ("fit_predict_seconds", "sum")
        spec.update({c: (c, "mean") for c in meta.columns if c.startswith("cell_")})
        tbl = tbl.merge(meta.groupby("model").agg(**spec).reset_index(), on="model", how="left")
    tbl.to_csv(out / "summary.csv", index=False)
    for model, d in preds.groupby("model"):
        avg = d.groupby(["donor_id", "y_true"], as_index=False)[prob_cols(classes)].mean()
        confusion_df(avg, classes).to_csv(out / f"confusion_{model}.csv")
    print(tbl[["model", "n_donors", "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi", "balanced_accuracy",
               "macro_auroc"]].round(3).to_string(index=False))
    return tbl


def prob_cols_from(preds: pd.DataFrame) -> list[str]:
    return [c for c in preds.columns if c.startswith("p_")]
