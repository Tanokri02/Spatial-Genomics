"""Controls that make the Aim 1 numbers believable.

  covariate : age/sex/cohort-only models (+ majority-prior baseline). The cell models must beat these.
  leaky     : random *cell* split (cells of one donor in both train and test) vs the donor-aware split.
  permutation: donor-label permutation null for a fast donor-level model.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..evaluate import aggregate_donor, cell_level_metrics, donor_metrics, prob_cols
from ..features import cap_per_group, covariate_matrix, donor_labels, pseudobulk
from ..models import build_model, make_sample_weights
from ..splits import cell_leaky_folds, make_folds
from .benchmark import prepare, summarize
from .cv import Ctx, _donor_df, read_preds


# ---------------------------------------------------------------------------------------------------
def run_covariate_baselines(ctx: Ctx, task: str | None = None) -> pd.DataFrame:
    """cov_lr / cov_rf on [age, sex, cohort]; prior = predicts training class frequencies (majority class)."""
    task = task or ctx.cfg["default_task"]
    cd, folds = prepare(ctx, task)
    out = ctx.paths.out("aim1", task)
    pdir = out / "preds"
    pdir.mkdir(exist_ok=True)
    cov = covariate_matrix(cd.donors, ctx.cfg["controls"]["covariate_features"])
    dlab = donor_labels(cd)
    classes, K = cd.classes, len(cd.classes)
    y_names = {d: classes[i] for d, i in dlab.items()}
    specs = {"cov_lr": {"family": "logreg", "params": {"C": 1.0, "max_iter": 500}},
             "cov_rf": {"family": "rf", "params": {"n_estimators": 300, "min_samples_leaf": 3}}}
    for seed in sorted(folds["seed"].unique()):
        f_s = folds[folds["seed"] == seed].set_index("donor_id")["fold"]
        for fold in sorted(f_s.unique()):
            te = list(f_s.index[f_s == fold])
            tr = list(f_s.index[f_s != fold])
            for name in ("cov_lr", "cov_rf", "prior"):
                path = pdir / f"{name}__s{seed}_f{fold}.parquet"
                if path.exists():
                    continue
                if name == "prior":
                    freq = np.bincount(dlab.loc[tr].values, minlength=K) / len(tr)
                    proba = np.tile(freq, (len(te), 1))
                else:
                    m = build_model(specs[name], K, seed=seed, n_jobs=ctx.n_jobs)
                    w = make_sample_weights(dlab.loc[tr].values, np.array(tr))
                    m.fit(cov.loc[tr].values.astype(float), dlab.loc[tr].values, w)
                    proba = m.predict_proba(cov.loc[te].values.astype(float))
                df = _donor_df(te, proba, y_names, classes)
                df.insert(0, "fold", fold), df.insert(0, "seed", seed), df.insert(0, "model", name)
                df.to_parquet(path)
                path.with_suffix(".json").write_text(json.dumps({"kind": "covariate", "n_params": 0}))
    return summarize(ctx, task)


# ---------------------------------------------------------------------------------------------------
def run_leaky_comparison(ctx: Ctx, task: str | None = None, model_name: str | None = None) -> pd.DataFrame:
    """Same model, same train-cell budget; random cell folds (leaky) vs donor folds (honest)."""
    task = task or ctx.cfg["default_task"]
    model_name = model_name or ctx.cfg["controls"]["leaky_model"]
    cd, folds = prepare(ctx, task)
    out = ctx.paths.out("controls", task)
    spec = ctx.cfg["models"][model_name]
    K = len(cd.classes)
    cap = ctx.cfg["cv"]["train_cells_per_donor_class"]
    rows = []
    for seed in sorted(folds["seed"].unique()):
        cell_fold = cell_leaky_folds(cd.obs["y"].values, ctx.cfg["cv"]["n_splits"], seed)
        P = np.zeros((cd.n_cells, K))
        for f in range(ctx.cfg["cv"]["n_splits"]):
            tr = np.where(cell_fold != f)[0]
            tr = tr[cap_per_group(cd.obs.iloc[tr], cap, seed)]
            te = np.where(cell_fold == f)[0]
            g, y = cd.obs["donor_id"].values[tr], cd.obs["y"].values[tr]
            m = build_model(spec, K, seed=seed, device=ctx.device, n_jobs=ctx.n_jobs)
            m.fit(cd.X[tr], y, make_sample_weights(y, g), g)
            P[te] = m.predict_proba(cd.X[te])
            ctx.log(f"leaky {model_name} seed={seed} fold={f}")
        y_all = cd.obs["y"].values
        r = {"seed": seed, "split": "leaky_cell_split", **cell_level_metrics(y_all, P, K)}
        dp = aggregate_donor(P, cd.obs["donor_id"].values, ctx.cfg["cv"]["donor_aggregation"])
        dd = pd.DataFrame({"y_true": cd.donors.loc[dp.index, "task_label"].values})
        dd[prob_cols(cd.classes)] = dp.values
        r.update({f"donor_{k}": v for k, v in donor_metrics(dd, cd.classes).items() if not k.startswith("recall")})
        rows.append(r)
    leaky = pd.DataFrame(rows)

    # honest donor-split counterpart (cell-level metrics are stored per fold in the Aim 1 json files)
    from .cv import read_meta, run_cv
    aim1 = ctx.paths.out("aim1", task)
    run_cv(ctx, cd, folds, model_name, aim1)
    meta = read_meta(aim1)
    meta = meta[meta["model"] == model_name]
    honest = meta.groupby("seed")[[c for c in meta.columns if c.startswith("cell_")]].mean().reset_index()
    honest["split"] = "donor_split"
    preds = read_preds(aim1, [model_name])
    for seed, d in preds.groupby("seed"):
        m = donor_metrics(d, cd.classes)
        for k, v in m.items():
            if not k.startswith("recall"):
                honest.loc[honest["seed"] == seed, f"donor_{k}"] = v
    res = pd.concat([leaky, honest], ignore_index=True)
    res.insert(0, "model", model_name)
    res.to_csv(out / "leaky_vs_donor_split.csv", index=False)
    print(res.groupby("split")[["cell_macro_f1", "cell_balanced_accuracy", "donor_macro_f1", "donor_balanced_accuracy"]].mean().round(3))
    return res


# ---------------------------------------------------------------------------------------------------
def run_permutation(ctx: Ctx, task: str | None = None, model_name: str | None = None,
                    reps: int | None = None, within_cohort: bool = True) -> pd.DataFrame:
    """Donor-label permutation null for a donor-level model.

    within_cohort=True shuffles labels inside each cohort, so the null keeps any label~cohort association;
    beating it means signal beyond cohort. Folds are re-drawn for every permutation.
    """
    task = task or ctx.cfg["default_task"]
    model_name = model_name or ctx.cfg["controls"]["permutation_model"]
    reps = reps or ctx.cfg["controls"]["permutation_reps"]
    cd, folds = prepare(ctx, task)
    spec = ctx.cfg["models"][model_name]
    assert spec["kind"] == "donor", "permutation test is implemented for donor-level models"
    pb = pseudobulk(cd, by_class=(spec["features"] == "pseudobulk_by_class"))
    K = len(cd.classes)
    rng = np.random.default_rng(ctx.cfg["seed"])
    base = cd.donors.copy()
    out_rows = []
    for r in range(reps + 1):  # r == 0 is the observed (unpermuted) run with the same pipeline
        d = base.copy()
        if r > 0:
            if within_cohort:
                d["task_label"] = d.groupby("cohort")["task_label"].transform(lambda s: rng.permutation(s.values))
            else:
                d["task_label"] = rng.permutation(d["task_label"].values)
        f = make_folds(d, ctx.cfg["cv"]["n_splits"], [r], ctx.cfg["cv"].get("stratify_by_cohort", True))
        f = f.set_index("donor_id")["fold"]
        y = d["task_label"].map({c: i for i, c in enumerate(cd.classes)}).astype(int)
        P = pd.DataFrame(index=d.index, columns=range(K), dtype=float)
        for fold in sorted(f.unique()):
            te, tr = list(f.index[f == fold]), list(f.index[f != fold])
            m = build_model(spec, K, seed=r, n_jobs=ctx.n_jobs)
            m.fit(pb.loc[tr].values, y.loc[tr].values)
            P.loc[te] = m.predict_proba(pb.loc[te].values)
        dd = pd.DataFrame({"y_true": [cd.classes[i] for i in y.values]})
        dd[prob_cols(cd.classes)] = P.values
        m = donor_metrics(dd, cd.classes)
        out_rows.append({"rep": r, "observed": r == 0, "macro_f1": m["macro_f1"],
                         "balanced_accuracy": m["balanced_accuracy"], "macro_auroc": m["macro_auroc"]})
        if r % 10 == 0:
            ctx.log(f"permutation {r}/{reps}")
    res = pd.DataFrame(out_rows)
    null = res[~res["observed"]]
    obs = res[res["observed"]].iloc[0]
    summary = {m: {"observed": float(obs[m]), "null_mean": float(null[m].mean()), "null_95pct": float(null[m].quantile(0.95)),
                   "p_value": float((1 + (null[m] >= obs[m]).sum()) / (1 + len(null)))}
               for m in ("macro_f1", "balanced_accuracy", "macro_auroc")}
    out = ctx.paths.out("controls", task)
    res.to_csv(out / f"permutation_{model_name}.csv", index=False)
    (out / f"permutation_{model_name}_summary.json").write_text(json.dumps(
        {"model": model_name, "within_cohort": within_cohort, "reps": reps, **summary}, indent=2))
    print(json.dumps(summary, indent=2))
    return res
