"""`psychad <command>`: one entry point for the whole pipeline. Run `psychad --help` for the list.

Typical order:  inspect -> build -> eda -> splits -> benchmark -> controls -> celltype -> transfer
                -> importance -> de -> scaling -> report
Optional:       embed (scVI latent for the *_scvi models) -> cohort-transfer (train one cohort, test the other)
                -> compare (paired tests vs baseline; also run by `summary`)
"""
from __future__ import annotations

import argparse
import sys

from .config import get_paths, load_config


def _ctx(cfg, models=None):
    """Run context. `models` = the model names this command will train. On macOS, if none of them is a neural net,
    torch is never imported (it ships its own OpenMP runtime, which conflicts with sklearn/XGBoost's in the same
    process: segfaults/deadlocks). models=None keeps the default behaviour (device auto-detected via torch)."""
    from .experiments.cv import Ctx

    if models is not None and sys.platform == "darwin":
        fams = {cfg["models"].get(m, {}).get("family") for m in models}
        if not fams & {"mlp", "mil"}:
            return Ctx(cfg, get_paths(cfg), "cpu", cfg["compute"]["n_jobs"])
    return Ctx.create(cfg, get_paths(cfg))


def cmd_inspect(a, cfg):
    from .data.inspect_data import run_inspect

    run_inspect(cfg, get_paths(cfg), all_cohorts=not a.configured_cohorts_only, refresh=a.refresh)


def cmd_build(a, cfg):
    from .data.build import run_build

    run_build(cfg, get_paths(cfg), stages=a.stages, force=a.force, keep_parts=a.keep_parts)


def cmd_eda(a, cfg):
    import pandas as pd

    from .viz import fig_eda

    paths = get_paths(cfg)
    for p in fig_eda(pd.read_csv(paths.donors, index_col="donor_id"), paths.results / "figures"):
        print(p)


def cmd_splits(a, cfg):
    from .experiments.benchmark import prepare

    ctx = _ctx(cfg, [])
    for t in (a.task or [cfg["default_task"]]):
        prepare(ctx, t)


def _runtime_groups(cfg, models: list[str]) -> list[list[str]]:
    """Split models so XGBoost and torch never run in the same process.

    On macOS, once XGBoost's OpenMP thread pool has run, any later multi-threaded torch CPU op in the same process
    can deadlock forever (two OpenMP runtimes). Observed: `torch.stack` stuck at a libomp join barrier after a
    cell_xgb fit. Separate processes avoid it entirely.
    """
    fam = lambda m: cfg["models"].get(m, {}).get("family")  # noqa: E731
    xgb = [m for m in models if fam(m) == "xgb"]
    nn = [m for m in models if fam(m) in ("mlp", "mil")]
    rest = [m for m in models if m not in xgb and m not in nn]
    if xgb and nn:
        return [g for g in (rest + xgb, nn) if g]
    return [models]


def _rerun_in_groups(a, groups: list[list[str]]) -> None:
    import subprocess

    base = [sys.executable, "-m", "psychad_ml"]
    if a.profile:
        base += ["--profile", a.profile]
    if a.config:
        base += ["--config", a.config]
    extra = []
    if getattr(a, "task", None):
        extra += ["--task", *a.task]
    if getattr(a, "no_curve", False):
        extra += ["--no-curve"]
    if getattr(a, "seeds", None):
        extra += ["--seeds", *map(str, a.seeds)]
    for g in groups:
        print(f"[{a.cmd}] XGBoost and torch models are run in separate processes: {g}", flush=True)
        r = subprocess.run([*base, a.cmd, *extra, "--models", *g])
        if r.returncode != 0:
            raise SystemExit(f"{a.cmd} --models {' '.join(g)} failed with exit code {r.returncode}")


def cmd_benchmark(a, cfg):
    from .experiments.benchmark import run_benchmark

    models = a.models or cfg["benchmark_models"]
    groups = _runtime_groups(cfg, models)
    if len(groups) > 1:
        return _rerun_in_groups(a, groups)
    ctx = _ctx(cfg, models)
    for t in (a.task or [cfg["default_task"]]):
        run_benchmark(ctx, t, a.models, a.seeds)


def cmd_controls(a, cfg):
    from .experiments import controls as C

    ctx = _ctx(cfg, [cfg["controls"]["leaky_model"], cfg["controls"]["permutation_model"]])
    task = a.task[0] if a.task else cfg["default_task"]
    which = a.which or ["covariate", "leaky", "permutation"]
    if "covariate" in which:
        C.run_covariate_baselines(ctx, task)
    if "leaky" in which:
        C.run_leaky_comparison(ctx, task, seeds=a.seeds)
    if "permutation" in which:
        C.run_permutation(ctx, task, reps=a.reps, within_cohort=not a.global_permutation)


def cmd_celltype(a, cfg):
    from .experiments.celltype import run_celltype

    groups = _runtime_groups(cfg, a.models or cfg["aim2"]["models"])
    if len(groups) > 1:
        return _rerun_in_groups(a, groups)
    ctx = _ctx(cfg, a.models or cfg["aim2"]["models"])
    for t in (a.task or [cfg["default_task"]]):
        run_celltype(ctx, t, a.models, curve=not a.no_curve)


def cmd_transfer(a, cfg):
    from .experiments.celltype import run_transfer

    run_transfer(_ctx(cfg, [cfg["aim2"]["transfer_model"]]))


def cmd_importance(a, cfg):
    from .experiments import importance as I

    ctx = _ctx(cfg, [cfg["importance"]["model"]] + ([cfg["importance"]["shap_model"]] if a.shap else []))
    task = a.task[0] if a.task else cfg["default_task"]
    I.run_lr_importance(ctx, task, seeds=a.seeds)
    if a.shap:
        I.run_shap(ctx, task)


def cmd_de(a, cfg):
    from .experiments import de

    ctx = _ctx(cfg, [])
    de.run_de(ctx, "four_class", enrich=not a.no_enrichment)
    try:
        de.compare_de_to_model(ctx, "four_class")
    except FileNotFoundError:
        print("run `psychad importance --task four_class` to get the DE-vs-model overlap table")


def cmd_scaling(a, cfg):
    from .experiments.scaling import run_scaling

    run_scaling(_ctx(cfg, [a.model or cfg["scaling"]["model"]]), a.task[0] if a.task else None, a.model,
                tuple(a.sweeps) if a.sweeps else None)


def cmd_embed(a, cfg):
    from .experiments.embed import run_scvi_embed

    run_scvi_embed(_ctx(cfg), force=a.force)


def cmd_cohort_transfer(a, cfg):
    from .experiments.cohort import run_cohort_transfer

    models = a.models or cfg["cohort_transfer"]["models"]
    groups = _runtime_groups(cfg, models)
    if len(groups) > 1:
        return _rerun_in_groups(a, groups)
    ctx = _ctx(cfg, models)
    for t in (a.task or cfg["cohort_transfer"]["tasks"]):
        run_cohort_transfer(ctx, t, models)


def cmd_compare(a, cfg):
    from .stats import compare_to_baseline

    paths = get_paths(cfg)
    aim1 = paths.results / "aim1"
    tasks = a.task or (sorted(d.name for d in aim1.iterdir() if d.is_dir()) if aim1.exists() else [])
    for t in tasks:
        r = compare_to_baseline(aim1 / t, reps=cfg["cv"]["bootstrap_reps"], seed=cfg["seed"])
        if r is None or r.empty:
            print(f"[compare] {t}: no baseline or no models yet")
            continue
        cols = ["model", "baseline", "macro_f1_delta", "macro_f1_delta_ci_lo", "macro_f1_delta_ci_hi", "macro_f1_p_holm",
                "macro_auroc_delta", "macro_auroc_p_holm"]
        print(f"\n== {t}: paired donor bootstrap vs baseline ==")
        print(r[cols].round(3).to_string(index=False))


def cmd_report(a, cfg):
    from .viz import make_all_figures

    paths = get_paths(cfg)
    for t in (a.task or [cfg["default_task"]]):
        for p in make_all_figures(paths, t):
            print(p)


def cmd_summary(a, cfg):
    from .experiments.benchmark import summarize
    from .experiments.cv import Ctx
    from .summary import write_summary

    paths = get_paths(cfg)
    # refresh summary.csv + confusion matrices from the SAVED predictions (reads only; never trains anything)
    aim1 = paths.results / "aim1"
    tasks = a.task or (sorted(d.name for d in aim1.iterdir() if d.is_dir()) if aim1.exists() else [])
    ctx = Ctx(cfg, paths, "cpu", cfg["compute"]["n_jobs"])
    for t in tasks:
        if (aim1 / t / "preds").exists():
            try:
                summarize(ctx, t)
            except Exception as e:  # noqa: BLE001
                print(f"[summary] could not refresh aim1/{t}: {e}")
    try:  # paired tests vs the baseline (reads saved predictions only)
        from .stats import compare_to_baseline

        for t in tasks:
            if (aim1 / t / "preds").exists():
                compare_to_baseline(aim1 / t, reps=cfg["cv"]["bootstrap_reps"], seed=cfg["seed"])
    except Exception as e:  # noqa: BLE001
        print(f"[summary] paired comparison failed: {e}")
    p = write_summary(cfg, paths, a.task)
    print(p.read_text())
    print(f"\n[written to {p}]")


def main(argv=None):
    p = argparse.ArgumentParser(prog="psychad", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", help="config profile, e.g. dev (merges configs/<profile>.yaml)")
    p.add_argument("--config", help="extra YAML override file")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        s = sub.add_parser(name, help=help_)
        s.set_defaults(fn=fn)
        return s

    s = add("inspect", cmd_inspect, "audit cohorts, labels, ages and cell types from metadata only")
    s.add_argument("--refresh", action="store_true"), s.add_argument("--configured-cohorts-only", action="store_true")
    s = add("build", cmd_build, "extract + subsample + HVG -> data/derived/<run>/ (resumable)")
    s.add_argument("--stages", nargs="+", choices=["select", "genes", "fetch", "assemble", "pseudobulk", "composition"])
    s.add_argument("--force", action="store_true", help="recompute cached selection/gene panel for the given stages")
    s.add_argument("--keep-parts", action="store_true")
    add("eda", cmd_eda, "EDA figures of the derived donor table")
    add("splits", cmd_splits, "create/cache donor folds")
    s = add("benchmark", cmd_benchmark, "Aim 1: model benchmark with donor-aware CV")
    s.add_argument("--models", nargs="+"), s.add_argument("--seeds", nargs="+", type=int)
    s = add("controls", cmd_controls, "covariate baselines, leaky-split comparison, permutation null")
    s.add_argument("--which", nargs="+", choices=["covariate", "leaky", "permutation"])
    s.add_argument("--reps", type=int), s.add_argument("--global-permutation", action="store_true")
    s.add_argument("--seeds", nargs="+", type=int, help="limit the leaky-split comparison to these CV seeds (faster)")
    s = add("celltype", cmd_celltype, "Aim 2: per-cell-class models on matched donors/cells")
    s.add_argument("--models", nargs="+"), s.add_argument("--no-curve", action="store_true")
    add("transfer", cmd_transfer, "Aim 2: train-on-X / test-on-Y transfer matrix per cell class")
    s = add("importance", cmd_importance, "stable LR coefficients (+ optional SHAP)")
    s.add_argument("--shap", action="store_true")
    s.add_argument("--seeds", nargs="+", type=int, help="limit LR-importance to these CV seeds (faster)")
    s = add("de", cmd_de, "pseudobulk DE (PyDESeq2) + enrichment")
    s.add_argument("--no-enrichment", action="store_true")
    s = add("scaling", cmd_scaling, "accuracy-vs-size sweeps; picks the cheapest near-best config")
    s.add_argument("--model")
    s.add_argument("--sweeps", nargs="+", help="arch genes pcs cells (default: scaling.grids.<model>.sweeps, else arch genes cells)")
    s = add("embed", cmd_embed, "label-free scVI latent of every stored cell (for the *_scvi models)")
    s.add_argument("--force", action="store_true")
    s = add("cohort-transfer", cmd_cohort_transfer, "train on one cohort, test on the other (external check)")
    s.add_argument("--models", nargs="+")
    add("compare", cmd_compare, "paired donor-bootstrap tests of every model vs the best baseline (reads results only)")
    add("report", cmd_report, "render all figures from existing results")
    add("summary", cmd_summary, "write results/<run>/RESULTS_SUMMARY.md (tables + verdicts vs baselines) from existing results")

    for name, sp in sub.choices.items():
        if name in ("benchmark", "splits", "controls", "celltype", "importance", "scaling", "report", "summary",
                    "cohort-transfer", "compare"):
            sp.add_argument("--task", nargs="+", help="task name(s) from config `tasks`")

    real_cli = argv is None          # invoked as a command (not from tests / Python code)
    a = p.parse_args(argv)
    cfg = load_config(a.profile, a.config)
    try:
        a.fn(a, cfg)
    except BaseException:  # noqa: BLE001
        if not real_cli:
            raise
        import traceback

        traceback.print_exc()
        hard_exit(1)
    if real_cli:
        hard_exit(0)


def hard_exit(code: int = 0) -> None:
    """Flush and exit without interpreter teardown.

    On macOS, once both torch and xgboost have run in one process, normal shutdown can hang forever inside libomp
    (two OpenMP runtimes). That would stall a sequential overnight run after a step has already finished, so the CLI
    ends with os._exit. All outputs are written (and closed) before this point.
    """
    import os

    try:  # os._exit skips joblib's cleanup, which would leave idle loky worker processes (~200 MB each) orphaned
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception:  # noqa: BLE001
        pass
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:  # noqa: BLE001
            pass
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main())
