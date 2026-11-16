"""`psychad cohort-transfer`: train on one cohort, test on the other (an external-validation check).

Cross-validation mixes MSSM and RADC donors in every fold. If a model has learnt disease biology (and not something
specific to one cohort's processing), it should still separate cases from controls in the cohort it never saw.
Only tasks whose classes exist in both cohorts make sense: ad_vs_control (MSSM 315/191, RADC 41/26). PD and DLB are
(almost) MSSM-only, so they cannot be checked this way.

Output: results/<run>/cohort_transfer/<task>/{preds/, cohort_transfer.csv}. Metrics: AUROC (threshold-free, the
primary number since class balance differs between cohorts), balanced accuracy, macro-F1, bootstrap CIs over test
donors.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..evaluate import bootstrap_ci, donor_metrics
from ..features import covariate_matrix, donor_labels
from ..models import build_model, make_sample_weights
from .benchmark import prepare, representation
from .cv import Ctx, _donor_df, fit_predict_fold


def run_cohort_transfer(ctx: Ctx, task: str = "ad_vs_control", models: list[str] | None = None) -> pd.DataFrame:
    cfg = ctx.cfg
    models = models or cfg["cohort_transfer"]["models"]
    out = ctx.paths.out("cohort_transfer", task)
    (out / "preds").mkdir(exist_ok=True)
    rows = []
    by_rep: dict[str, list[str]] = {}
    for m in models:
        by_rep.setdefault(representation(cfg["models"][m]) if m in cfg["models"] else "genes", []).append(m)
    for rep, ms in by_rep.items():
        cd, _ = prepare(ctx, task, representation=rep)
        coh = cd.donors["cohort"]
        cohorts = sorted(coh.unique())
        for tr_c in cohorts:
            for te_c in cohorts:
                if tr_c == te_c:
                    continue
                tr, te = list(coh.index[coh == tr_c]), list(coh.index[coh == te_c])
                for m in ms:
                    path = out / "preds" / f"{m}__{tr_c}_to_{te_c}.parquet"
                    if path.exists():
                        df = pd.read_parquet(path)
                    else:
                        ctx.log(f"cohort transfer {task}: {m} train {tr_c} ({len(tr)}) -> test {te_c} ({len(te)})")
                        if m == "cov_lr":   # age + sex only (cohort is constant within the training set)
                            df = _cov_lr(ctx, cd, tr, te)
                        else:
                            df, meta, adj = fit_predict_fold(ctx, cd, cfg["models"][m], tr, te, cfg["seed"],
                                                             cfg["cv"]["train_cells_per_donor_class"], False)
                            if adj is not None:
                                adj.to_parquet(out / "preds" / f"{m}+bias__{tr_c}_to_{te_c}.parquet")
                            path.with_suffix(".json").write_text(json.dumps(meta))
                        df.to_parquet(path)
                    rows.append(_score(df, cd.classes, m, tr_c, te_c, len(tr), cfg))
                    bp = out / "preds" / f"{m}+bias__{tr_c}_to_{te_c}.parquet"
                    if bp.exists():
                        rows.append(_score(pd.read_parquet(bp), cd.classes, f"{m}+bias", tr_c, te_c, len(tr), cfg))
    res = pd.DataFrame(rows)
    res.to_csv(out / "cohort_transfer.csv", index=False)
    print(res.round(3).to_string(index=False))
    return res


def _cov_lr(ctx, cd, tr, te) -> pd.DataFrame:
    cov = covariate_matrix(cd.donors, ["age", "sex"])
    dlab = donor_labels(cd)
    K = len(cd.classes)
    m = build_model({"family": "logreg", "params": {"C": 1.0, "max_iter": 500}}, K, seed=ctx.cfg["seed"])
    m.fit(cov.loc[tr].values.astype(float), dlab.loc[tr].values, make_sample_weights(dlab.loc[tr].values, np.array(tr)))
    y_names = {d: cd.classes[i] for d, i in dlab.items()}
    return _donor_df(te, m.predict_proba(cov.loc[te].values.astype(float)), y_names, cd.classes)


def _score(df, classes, model, tr_c, te_c, n_train, cfg) -> dict:
    m = donor_metrics(df, classes)
    ci = bootstrap_ci(df, classes, reps=cfg["cv"]["bootstrap_reps"], seed=cfg["seed"])
    return {"model": model, "train_cohort": tr_c, "test_cohort": te_c, "n_train_donors": n_train,
            "n_test_donors": len(df), "auroc": m["macro_auroc"], "auroc_ci_lo": ci["macro_auroc"][0],
            "auroc_ci_hi": ci["macro_auroc"][1], "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"]}
