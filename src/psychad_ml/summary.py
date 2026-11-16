"""`psychad summary`: read every result that exists and write one readable report.

Output: results/<run>/RESULTS_SUMMARY.md (also printed). Safe to run at any time, on partial results: sections whose
inputs are missing say so instead of failing. Nothing here trains or changes a result.

Sections: run inventory (what finished / what is missing) -> Aim 1 tables with a verdict against the covariate and
prior baselines -> controls (leaky split, permutation null) -> Aim 2 (per cell class, downsampling, transfer)
-> scaling -> importance / DE.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASELINES = ("prior", "cov_lr", "cov_rf")


def _md(df: pd.DataFrame, floatfmt: int = 3) -> str:
    """Tiny markdown table (avoids the optional `tabulate` dependency)."""
    if df is None or df.empty:
        return "_(empty)_"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.astype(str).values]
    return "\n".join([head, sep, *rows])


def _read(p: Path):
    return pd.read_csv(p) if p.exists() else None


def _derived(cfg: dict) -> Path:
    from .config import get_paths

    return get_paths(cfg).derived


def _missing(what: str, hint: str) -> str:
    return f"_Not available: {what}. Run: `{hint}`_\n"


# ------------------------------------------------------------------------------------------------ inventory
def _inventory(cfg: dict, res: Path, tasks: list[str]) -> str:
    n_exp = cfg["cv"]["n_splits"] * len(cfg["cv"]["seeds"])
    rows = []
    for t in tasks:
        pdir = res / "aim1" / t / "preds"
        have = {}
        if pdir.exists():
            for f in pdir.glob("*.parquet"):
                have[f.name.rsplit("__s", 1)[0]] = have.get(f.name.rsplit("__s", 1)[0], 0) + 1
        r2 = [m for g in (cfg.get("round2_models") or {}).values() for m in g]
        listed = cfg["benchmark_models"] + ["cell_mlp_large", *BASELINES] + r2
        on_disk = sorted(m for m in have if m not in listed and not m.endswith("+bias"))
        for m in listed + on_disk:
            n = have.get(m, 0)
            if n or (t == cfg["default_task"] and m not in r2):
                ns = cfg["cv"]["n_splits"]
                status = ("done" if n >= n_exp or (m == "prior" and n)
                          else f"done ({n // ns} seed(s) only)" if n and n % ns == 0
                          else "partial" if n else "MISSING")
                rows.append({"task": t, "model": m, "folds_done": f"{n}/{n_exp}", "status": status})
    inv = pd.DataFrame(rows)
    extra = []
    for label, path in [("controls: leaky vs donor split", res / "controls" / cfg["default_task"] / "leaky_vs_donor_split.csv"),
                        ("controls: permutation null", res / "controls" / cfg["default_task"] / f"permutation_{cfg['controls']['permutation_model']}.csv"),
                        ("Aim 2: per-cell-class summary", res / "aim2" / cfg["default_task"] / "summary.csv"),
                        ("Aim 2: downsampling curve", res / "aim2" / cfg["default_task"] / "downsample_curve.csv"),
                        ("Aim 2: transfer", res / "aim2" / "transfer" / "transfer_auroc.csv"),
                        ("importance (LR)", res / "importance" / cfg["default_task"] / "lr_importance.csv"),
                        ("importance (SHAP)", res / "importance" / cfg["default_task"] / "shap_importance.csv"),
                        ("DE", res / "de" / "four_class" / "de_results.csv"),
                        ("DE vs model overlap", res / "de" / "four_class" / "de_vs_model_overlap.csv"),
                        ("scVI latent (psychad embed)", _derived(cfg) / "scvi" / "embed_meta.json"),
                        ("cohort transfer", res / "cohort_transfer" / "ad_vs_control" / "cohort_transfer.csv")]:
        extra.append({"output": label, "status": "done" if path.exists() else "MISSING"})
    sc = sorted((res / "scaling" / cfg["default_task"]).glob("*/scaling.csv")) if (res / "scaling").exists() else []
    extra.append({"output": "scaling sweeps", "status": f"done ({', '.join(p.parent.name for p in sc)})" if sc else "MISSING"})
    return ("Expected folds per model = n_splits x n_seeds = %d.\n\n%s\n\n%s\n" % (n_exp, _md(inv), _md(pd.DataFrame(extra))))


# ------------------------------------------------------------------------------------------------ aim 1
def _aim1(cfg: dict, res: Path, task: str) -> str:
    s = _read(res / "aim1" / task / "summary.csv")
    if s is None:
        return _missing(f"Aim 1 summary for {task}", f"psychad benchmark --task {task}")
    cols = [c for c in ["model", "n_donors", "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi", "balanced_accuracy",
                        "macro_auroc", "n_params", "fit_predict_seconds"] if c in s.columns]
    s = s.sort_values("macro_f1", ascending=False)
    out = [_md(s[cols])]
    base = s[s["model"].isin(BASELINES)]
    models = s[~s["model"].isin(BASELINES)]
    if base.empty:
        out.append("\n**No baselines yet**: run `psychad controls --which covariate` before reading these numbers.")
    elif not models.empty:
        b = base.sort_values("macro_f1", ascending=False).iloc[0]
        out.append(f"\nBest baseline: **{b['model']}** macro-F1 = {b['macro_f1']:.3f} "
                   f"(CI {b['macro_f1_ci_lo']:.3f}-{b['macro_f1_ci_hi']:.3f}).\n")
        pv = _read(res / "aim1" / task / "paired_vs_baseline.csv")
        if pv is not None and not pv.empty:
            out.append(f"**Paired test vs {pv['baseline'].iloc[0]}** (same donors resampled for both models; one-sided p, "
                       "Holm-corrected over the models of this task). A model beats the baseline when the whole delta CI "
                       "is above 0 and p_holm < 0.05.\n")
            v = pv[["model", "macro_f1_delta", "macro_f1_delta_ci_lo", "macro_f1_delta_ci_hi", "macro_f1_p_holm",
                    "macro_auroc_delta", "macro_auroc_delta_ci_lo", "macro_auroc_delta_ci_hi", "macro_auroc_p_holm"]].copy()
            v["verdict"] = v.apply(_verdict, axis=1)
            out.append(_md(v.sort_values("macro_f1_delta", ascending=False)))
        else:
            v = models[["model", "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi"]].copy()
            v["delta_vs_baseline"] = v["macro_f1"] - b["macro_f1"]
            v["verdict"] = "run `psychad compare` for a paired test"
            out.append(_md(v[["model", "macro_f1", "delta_vs_baseline", "verdict"]]))
    # per-class recall of the best model
    best = models.iloc[0] if not models.empty else None
    rec = [c for c in s.columns if c.startswith("recall_")]
    if best is not None and rec:
        out.append(f"\nPer-class recall, best model (**{best['model']}**): " +
                   ", ".join(f"{c[7:]} = {best[c]:.2f}" for c in rec))
        out.append("\nPer-class recall (fraction of each class's donors predicted correctly), all models:\n\n" +
                   _md(s[["model", *rec]].rename(columns={c: c[7:] for c in rec}), 2))
    return "\n".join(out) + "\n"


def _verdict(r) -> str:
    f1 = r["macro_f1_delta_ci_lo"] > 0 and r["macro_f1_p_holm"] < 0.05
    au = r["macro_auroc_delta_ci_lo"] > 0 and r["macro_auroc_p_holm"] < 0.05
    if f1 and au:
        return "beats baseline (F1 and AUROC)"
    if f1 or au:
        return "beats baseline on " + ("macro-F1" if f1 else "AUROC") + " only"
    return "not shown to beat baseline"


def _training_diagnostics(res: Path, task: str) -> str:
    """Neural nets: where early stopping picked the checkpoint. best = first evaluation means it never improved."""
    from .experiments.cv import read_meta

    pdir = res / "aim1" / task
    if not (pdir / "preds").exists():
        return ""
    meta = read_meta(pdir)
    if meta.empty or "best_val_loss" not in meta:
        return ""
    nn = meta[meta["best_val_loss"].notna()]
    if nn.empty:
        return ""
    agg = {"folds": ("fold", "size"), "best_epoch_mean": ("best_epoch", "mean"), "epochs_run_mean": ("epochs_run", "mean"),
           "best_val_loss": ("best_val_loss", "mean")}
    for c in ("best_val_auroc", "best_step_mean", "steps_run_mean", "n_ensemble"):
        if c in nn:
            agg[c] = (c, "mean")
    t = nn.groupby("model").agg(**agg).reset_index()
    return ("**Neural-net training diagnostics** (means over folds). If the best checkpoint is the first evaluation "
            "(best_epoch_mean ~1 for per-epoch evaluation), the network never improved on held-out donors. "
            "best_val_auroc is on validation donors (part of the training set), not on the test fold.\n\n" + _md(t) + "\n")


# ------------------------------------------------------------------------------------------------ controls
def _controls(cfg: dict, res: Path, task: str) -> str:
    out = []
    lk = _read(res / "controls" / task / "leaky_vs_donor_split.csv")
    if lk is None:
        out.append(_missing("leaky-split comparison", f"psychad controls --task {task} --which leaky"))
    else:
        cols = [c for c in ["cell_macro_f1", "cell_balanced_accuracy", "donor_macro_f1", "donor_balanced_accuracy"] if c in lk.columns]
        g = lk.groupby("split")[cols].mean().reset_index()
        out.append("**Leaky (random cell split) vs honest (donor split)**: a large gap = what donor leakage would have "
                   "faked.\n\n" + _md(g))
    for f in sorted((res / "controls" / task).glob("permutation_*_summary.json")):
        j = json.loads(f.read_text())
        rows = [{"metric": m, "observed": j[m]["observed"], "null_mean": j[m]["null_mean"],
                 "null_95pct": j[m]["null_95pct"], "p_value": j[m]["p_value"]} for m in ("macro_f1", "balanced_accuracy", "macro_auroc")]
        out.append(f"\n**Label-permutation null** ({j['model']}, {j['reps']} reps, within-cohort={j['within_cohort']}): "
                   "observed must exceed null_95pct.\n\n" + _md(pd.DataFrame(rows), 4))
    if not out:
        out.append(_missing("controls", f"psychad controls --task {task}"))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ aim 2
def _aim2(cfg: dict, res: Path, task: str) -> str:
    out = []
    s = _read(res / "aim2" / task / "summary.csv")
    if s is None:
        out.append(_missing("per-cell-class results", f"psychad celltype --task {task}"))
    else:
        for metric in ("macro_f1", "macro_auroc"):
            if metric in s.columns:
                s2 = s.copy()
                if f"{metric}_ci_lo" in s2:
                    s2["cell"] = s2.apply(lambda r, m=metric: f"{r[m]:.3f} [{r[m + '_ci_lo']:.2f}-{r[m + '_ci_hi']:.2f}]",
                                          axis=1)
                else:
                    s2["cell"] = s2[metric].map(lambda v: f"{v:.3f}")
                p = s2.pivot(index="cell_class", columns="base_model", values="cell")
                out.append(f"**{metric}** [95% donor-bootstrap CI] by cell class (rows) and model (cols); ALL = pooled "
                           "reference. Overlapping CIs = no evidence that one cell class carries more signal.\n\n" +
                           _md(p.reset_index()))
    d = _read(res / "aim2" / task / "downsample_curve.csv")
    if d is not None:
        out.append("\n**Downsampling** (macro-F1 vs cells/donor): if class ranking is stable, differences are not just data volume.\n\n" +
                   _md(d.pivot(index="cells_per_donor", columns="cell_class", values="macro_f1").reset_index()))
    t = _read(res / "aim2" / "transfer" / "transfer_auroc.csv")
    if t is None:
        out.append(_missing("transfer matrix", "psychad transfer"))
    else:
        t = t.copy()
        if "ci_lo" in t:
            t["cell"] = t.apply(lambda r: f"{r['auroc']:.2f} [{r['ci_lo']:.2f}-{r['ci_hi']:.2f}]", axis=1)
        else:
            t["cell"] = t["auroc"].map(lambda v: f"{v:.2f}")
        out.append("\n**Transfer AUROC [95% CI]** per cell class: train on row (disease vs Control) -> test on column "
                   "(disease vs Control). A CI that includes 0.5 = no evidence of signal.\n")
        for cc, g in t.groupby("cell_class"):
            out.append(f"\n*{cc}*\n\n" + _md(g.pivot(index="train_on", columns="test_on", values="cell").reset_index()))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ scaling / biology
def _scaling(cfg: dict, res: Path, task: str) -> str:
    files = sorted((res / "scaling" / task).glob("*/scaling.csv")) if (res / "scaling" / task).exists() else []
    if not files:
        return _missing("scaling sweeps", f"psychad scaling --task {task}")
    out = []
    for f in files:
        d = pd.read_csv(f)
        ch = d[d["scaled_down_choice"]].iloc[0]
        out.append(f"**{f.parent.name}**: best macro-F1 = {d['macro_f1'].max():.3f}; cheapest within tolerance = "
                   f"`{ch['sweep']}={ch['setting']}` ({int(ch['n_params']):,} params, macro-F1 {ch['macro_f1']:.3f}).\n\n" +
                   _md(d[["sweep", "setting", "n_params", "macro_f1", "seconds"]]))
    return "\n".join(out) + "\n"


def _biology(cfg: dict, res: Path, task: str) -> str:
    out = []
    imp = _read(res / "importance" / task / "lr_importance.csv")
    if imp is None:
        out.append(_missing("LR importance", f"psychad importance --task {task}"))
    else:
        st = imp[imp["stable"]]
        out.append("**Stable genes per (cell class x disease class)** (top-k in >=80% of runs, same sign):\n\n" +
                   _md(st.groupby(["cell_class", "class"]).size().unstack(fill_value=0).reset_index()))
        for (cc, cls), g in st.groupby(["cell_class", "class"]):
            if cls != "Control" and cc in ("ALL", "Micro", "EN", "Astro"):
                top = g.reindex(g["mean_coef"].abs().sort_values(ascending=False).index).head(10)
                out.append(f"- {cc} / {cls}: " + ", ".join(top["gene"]))
    de = _read(res / "de" / "four_class" / "de_results.csv")
    if de is None:
        out.append("\n" + _missing("DE", "psychad de"))
    elif not de.empty:
        n = de.assign(sig=de["padj"] < cfg["de"]["padj"]).groupby(["cell_class", "disease"])["sig"].sum().unstack(fill_value=0)
        out.append(f"\n**DE genes at padj < {cfg['de']['padj']}** (pseudobulk, age/sex/cohort adjusted)\n\n" + _md(n.reset_index()))
    ov = _read(res / "de" / "four_class" / "de_vs_model_overlap.csv")
    if ov is not None:
        out.append("\n**Overlap of top model genes with top DE genes**\n\n" + _md(ov))
    return "\n".join(out) + "\n"


def _cohort_transfer(cfg: dict, res: Path) -> str:
    files = sorted((res / "cohort_transfer").glob("*/cohort_transfer.csv")) if (res / "cohort_transfer").exists() else []
    if not files:
        return _missing("cohort transfer", "psychad cohort-transfer --task ad_vs_control")
    out = []
    for f in files:
        d = pd.read_csv(f)
        out.append(f"**{f.parent.name}**: trained on one cohort, tested on the other (no donor or cohort overlap). "
                   "cov_lr uses age + sex only.\n\n" +
                   _md(d[["model", "train_cohort", "test_cohort", "n_train_donors", "n_test_donors", "auroc",
                          "auroc_ci_lo", "auroc_ci_hi", "balanced_accuracy", "macro_f1"]]))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ driver
def write_summary(cfg: dict, paths, tasks: list[str] | None = None) -> Path:
    res = Path(paths.results)
    if not tasks:
        order = list(cfg["tasks"])
        have = [d.name for d in (res / "aim1").iterdir() if d.is_dir()] if (res / "aim1").exists() else []
        tasks = sorted(have, key=lambda t: order.index(t) if t in order else len(order))
    tasks = [t for t in tasks if (res / "aim1" / t).exists()] or [cfg["default_task"]]
    parts = [f"# Results summary: run `{cfg['run_name']}`\n",
             "Generated by `psychad summary` from files in `results/%s/`. Metrics are donor-level (out-of-fold donors, "
             "probabilities averaged over cells and seeds); CIs are bootstrap over donors.\n" % cfg["run_name"],
             "## 0. What finished\n", _inventory(cfg, res, tasks)]
    for i, t in enumerate(tasks):
        parts += [f"## 1.{i + 1} Aim 1: {t}\n", _aim1(cfg, res, t), _training_diagnostics(res, t),
                  f"### Controls ({t})\n", _controls(cfg, res, t)]
    parts += ["## 1.9 External check: train on one cohort, test on the other\n", _cohort_transfer(cfg, res)]
    parts += [f"## 2. Aim 2: is the signal cell-type specific? ({cfg['default_task']})\n", _aim2(cfg, res, cfg["default_task"]),
              "## 3. Scaling (accuracy vs size)\n", _scaling(cfg, res, cfg["default_task"]),
              "## 4. Which genes? (importance, DE)\n", _biology(cfg, res, cfg["default_task"]),
              "## How to read this\n",
              ("- A model is only interesting if it **beats the covariate/prior baseline** (paired test) and the "
               "**permutation null**.\n"
              "- `<model>+bias` = the same model with the rare-class decision rule (per-class offsets tuned on training "
              "donors only). It trades AD/Control accuracy for PD/DLB recall: compare macro-F1 AND per-class recall; AUROC "
              "is (almost) unchanged by design.\n"
              "- Point estimates are computed on seed-averaged probabilities (the same predictions the CI resamples).\n"
              "- If the leaky-split score is far above the donor-split score, cell-level metrics are inflated by leakage; "
              "trust the donor-level numbers.\n"
              "- PD (~29) and DLB (~33) donors: wide CIs are expected. If a CI includes chance, say so and lean on `lbd_group` "
              "or one-vs-control tasks.\n"
               "- Cohort confound: DLB is 100% MSSM. Compare against the within-cohort permutation null.\n")]
    out = res / "RESULTS_SUMMARY.md"
    out.write_text("\n".join(parts))
    return out
