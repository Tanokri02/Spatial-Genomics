#!/usr/bin/env python
"""End-to-end run of every pipeline stage on synthetic data (offline, ~1-3 min). `make smoke`.

Checks that the code runs and that planted signal is recovered; it says nothing about real performance.
"""
import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from psychad_ml.config import get_paths
from psychad_ml.data.build import run_build
from psychad_ml.data.inspect_data import run_inspect
from psychad_ml.data.synthetic import make_smoke_config
from psychad_ml.experiments import celltype, controls, de, importance, scaling
from psychad_ml.experiments.benchmark import run_benchmark
from psychad_ml.experiments.cv import Ctx
from psychad_ml.viz import make_all_figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="stages to skip: shap de scaling")
    a = ap.parse_args()
    w = Path(a.workdir or tempfile.mkdtemp(prefix="psychad_smoke_"))
    shutil.rmtree(w, ignore_errors=True)
    t0 = time.time()
    cfg = make_smoke_config(w)
    paths = get_paths(cfg)
    ctx = Ctx.create(cfg, paths)
    ctx.log(f"workdir {w}  device={ctx.device}")

    run_inspect(cfg, paths)
    run_build(cfg, paths)
    for task in ("four_class", "lbd_group"):
        run_benchmark(ctx, task)
    controls.run_covariate_baselines(ctx, "four_class")
    controls.run_leaky_comparison(ctx, "four_class")
    controls.run_permutation(ctx, "four_class")
    celltype.run_celltype(ctx, "four_class", models=["cell_lr"])
    celltype.run_transfer(ctx)
    importance.run_lr_importance(ctx, "four_class")
    if "shap" not in a.skip:
        importance.run_shap(ctx, "four_class", cell_classes=["Micro"])
    if "de" not in a.skip:
        de.run_de(ctx, "four_class", enrich=False)
        de.compare_de_to_model(ctx, "four_class")
    if "scaling" not in a.skip:
        scaling.run_scaling(ctx, "four_class")
    figs = make_all_figures(paths, "four_class")

    s = pd.read_csv(paths.results / "aim1" / "four_class" / "summary.csv").set_index("model")
    ctx.log(f"{len(figs)} figures; total {time.time() - t0:.0f}s")
    best = s["macro_f1"].drop(["prior", "cov_lr", "cov_rf"], errors="ignore")
    assert best.max() > s.loc["prior", "macro_f1"] + 0.1, "no model beats the prior: planted signal not recovered"
    print("\nSMOKE TEST OK. best model:", best.idxmax(), round(best.max(), 3))
    if not a.keep and a.workdir is None:
        shutil.rmtree(w, ignore_errors=True)


if __name__ == "__main__":
    main()
