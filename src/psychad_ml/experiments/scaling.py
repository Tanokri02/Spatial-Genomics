"""'Optimise accuracy first, then scale down': one-factor sweeps around the full-size model.

  arch   : hidden layer sizes (parameters)   -> macro-F1 vs n_params
  genes  : top-k genes of the HVG-ranked panel
  cells  : training cells per donor x class

Output scaling.csv + `scaled_down_choice`: the cheapest setting within `tolerance` of the best macro-F1.
"""
from __future__ import annotations

import copy

import pandas as pd

from ..evaluate import donor_metrics
from .benchmark import prepare
from .cv import Ctx, read_meta, run_cv


def _score(preds: pd.DataFrame, classes: list[str]) -> dict:
    ps = [donor_metrics(d, classes) for _, d in preds.groupby("seed")]
    df = pd.DataFrame(ps)
    return {"macro_f1": df["macro_f1"].mean(), "macro_f1_sd": df["macro_f1"].std(ddof=0),
            "balanced_accuracy": df["balanced_accuracy"].mean(), "macro_auroc": df["macro_auroc"].mean()}


def run_scaling(ctx: Ctx, task: str | None = None, model_name: str | None = None, sweeps=("arch", "genes", "cells")) -> pd.DataFrame:
    task = task or ctx.cfg["default_task"]
    model_name = model_name or ctx.cfg["scaling"]["model"]
    sc = ctx.cfg["scaling"]
    base = ctx.cfg["models"][model_name]
    cd, folds = prepare(ctx, task)
    out = ctx.paths.out("scaling", task, model_name)
    seeds = [ctx.cfg["cv"]["seeds"][0]]
    full_cap = ctx.cfg["cv"]["train_cells_per_donor_class"]
    rows = []

    def one(sweep, setting, spec, data, cap):
        tag = f"{sweep}__{setting}"
        preds = run_cv(ctx, data, folds, model_name, out, tag=tag, spec=spec, seeds=seeds, train_cap=cap,
                       save_cell_metrics=False)
        meta = read_meta(out)
        meta = meta[meta["model"] == tag]
        rows.append({"sweep": sweep, "setting": setting, "n_params": int(meta["n_params"].max()),
                     "n_genes": data.X.shape[1], "train_cells_per_donor_class": cap,
                     "seconds": float(meta["fit_predict_seconds"].sum()), **_score(preds, cd.classes)})

    if "arch" in sweeps and "hidden" in base["params"]:
        for h in sc["hidden_grid"]:
            spec = copy.deepcopy(base)
            spec["params"]["hidden"] = list(h)
            one("arch", "x".join(map(str, h)), spec, cd, full_cap)
    if "genes" in sweeps:
        for k in sc["gene_grid"]:
            one("genes", str(k), base, cd.top_genes(k), full_cap)
    if "cells" in sweeps:
        for n in sc["cells_per_donor_grid"]:
            one("cells", str(n), base, cd, n)
    res = pd.DataFrame(rows)
    res["cost"] = res["n_params"].clip(lower=1) * res["train_cells_per_donor_class"]
    best = res["macro_f1"].max()
    ok = res[res["macro_f1"] >= best - sc["tolerance"]].sort_values("cost")
    res["scaled_down_choice"] = res.index == ok.index[0]
    res.to_csv(out / "scaling.csv", index=False)
    print(res.round(3).to_string(index=False))
    print(f"\nbest macro-F1={best:.3f}; cheapest within {sc['tolerance']}: {ok.iloc[0][['sweep', 'setting']].tolist()}")
    return res
