"""`psychad <command>`: one entry point for the whole pipeline. Run `psychad --help` for the list.

Typical order:  inspect -> build -> eda -> splits -> benchmark -> controls -> celltype -> transfer
                -> importance -> de -> scaling -> report
"""
from __future__ import annotations

import argparse
import sys

from .config import get_paths, load_config


def _ctx(cfg):
    from .experiments.cv import Ctx

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

    ctx = _ctx(cfg)
    for t in (a.task or [cfg["default_task"]]):
        prepare(ctx, t)


def cmd_benchmark(a, cfg):
    from .experiments.benchmark import run_benchmark

    ctx = _ctx(cfg)
    for t in (a.task or [cfg["default_task"]]):
        run_benchmark(ctx, t, a.models, a.seeds)


def cmd_controls(a, cfg):
    from .experiments import controls as C

    ctx = _ctx(cfg)
    task = a.task[0] if a.task else cfg["default_task"]
    which = a.which or ["covariate", "leaky", "permutation"]
    if "covariate" in which:
        C.run_covariate_baselines(ctx, task)
    if "leaky" in which:
        C.run_leaky_comparison(ctx, task)
    if "permutation" in which:
        C.run_permutation(ctx, task, reps=a.reps, within_cohort=not a.global_permutation)


def cmd_celltype(a, cfg):
    from .experiments.celltype import run_celltype

    ctx = _ctx(cfg)
    for t in (a.task or [cfg["default_task"]]):
        run_celltype(ctx, t, a.models, curve=not a.no_curve)


def cmd_transfer(a, cfg):
    from .experiments.celltype import run_transfer

    run_transfer(_ctx(cfg))


def cmd_importance(a, cfg):
    from .experiments import importance as I

    ctx = _ctx(cfg)
    task = a.task[0] if a.task else cfg["default_task"]
    I.run_lr_importance(ctx, task)
    if a.shap:
        I.run_shap(ctx, task)


def cmd_de(a, cfg):
    from .experiments import de

    ctx = _ctx(cfg)
    de.run_de(ctx, "four_class", enrich=not a.no_enrichment)
    try:
        de.compare_de_to_model(ctx, "four_class")
    except FileNotFoundError:
        print("run `psychad importance --task four_class` to get the DE-vs-model overlap table")


def cmd_scaling(a, cfg):
    from .experiments.scaling import run_scaling

    run_scaling(_ctx(cfg), a.task[0] if a.task else None, a.model, tuple(a.sweeps))


def cmd_report(a, cfg):
    from .viz import make_all_figures

    paths = get_paths(cfg)
    for t in (a.task or [cfg["default_task"]]):
        for p in make_all_figures(paths, t):
            print(p)


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
    s.add_argument("--stages", nargs="+", choices=["select", "genes", "fetch", "assemble", "pseudobulk"])
    s.add_argument("--force", action="store_true", help="recompute cached selection/gene panel for the given stages")
    s.add_argument("--keep-parts", action="store_true")
    add("eda", cmd_eda, "EDA figures of the derived donor table")
    add("splits", cmd_splits, "create/cache donor folds")
    s = add("benchmark", cmd_benchmark, "Aim 1: model benchmark with donor-aware CV")
    s.add_argument("--models", nargs="+"), s.add_argument("--seeds", nargs="+", type=int)
    s = add("controls", cmd_controls, "covariate baselines, leaky-split comparison, permutation null")
    s.add_argument("--which", nargs="+", choices=["covariate", "leaky", "permutation"])
    s.add_argument("--reps", type=int), s.add_argument("--global-permutation", action="store_true")
    s = add("celltype", cmd_celltype, "Aim 2: per-cell-class models on matched donors/cells")
    s.add_argument("--models", nargs="+"), s.add_argument("--no-curve", action="store_true")
    add("transfer", cmd_transfer, "Aim 2: train-on-X / test-on-Y transfer matrix per cell class")
    s = add("importance", cmd_importance, "stable LR coefficients (+ optional SHAP)")
    s.add_argument("--shap", action="store_true")
    s = add("de", cmd_de, "pseudobulk DE (PyDESeq2) + enrichment")
    s.add_argument("--no-enrichment", action="store_true")
    s = add("scaling", cmd_scaling, "accuracy-vs-size sweeps; picks the cheapest near-best config")
    s.add_argument("--model"), s.add_argument("--sweeps", nargs="+", default=["arch", "genes", "cells"])
    add("report", cmd_report, "render all figures from existing results")

    for name, sp in sub.choices.items():
        if name in ("benchmark", "splits", "controls", "celltype", "importance", "scaling", "report"):
            sp.add_argument("--task", nargs="+", help="task name(s) from config `tasks`")

    a = p.parse_args(argv)
    cfg = load_config(a.profile, a.config)
    a.fn(a, cfg)


if __name__ == "__main__":
    sys.exit(main())
