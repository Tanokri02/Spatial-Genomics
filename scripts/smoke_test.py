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

try:  # the smoke test deliberately runs XGBoost and torch in ONE process on tiny data; on macOS XGBoost must be
    import xgboost  # noqa: F401   # imported before torch or it segfaults. Real runs never mix them (see cli._runtime_groups).
except Exception:  # noqa: BLE001
    pass

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
    ap.add_argument("--skip", nargs="*", default=[], help="stages to skip: shap de scaling scvi")
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
    # round 2: improved pseudobulk models, small nets, scVI latent, cohort transfer, paired tests
    r2 = cfg["round2_models"]
    run_benchmark(ctx, "four_class", models=r2["pb"] + r2["nn"])
    if "scvi" not in a.skip:
        # separate process, exactly like the real pipeline (`psychad embed`): this process has already run XGBoost,
        # and scVI's multi-threaded torch ops then deadlock on macOS (two OpenMP runtimes; see README)
        import subprocess

        import yaml

        cfg_file = w / "smoke_config.yaml"
        cfg_file.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))
        base = [sys.executable, "-m", "psychad_ml", "--config", str(cfg_file)]
        for args in (["embed"], ["benchmark", "--task", "four_class", "--models", *r2["scvi"]]):
            r = subprocess.run([*base, *args], check=False)
            assert r.returncode == 0, f"psychad {' '.join(args)} failed on synthetic data"
    if "scaling" not in a.skip:
        scaling.run_scaling(ctx, "four_class", "mlp_small")
    controls.run_covariate_baselines(ctx, "ad_vs_control")
    from psychad_ml.experiments.cohort import run_cohort_transfer
    from psychad_ml.stats import compare_to_baseline

    ct = run_cohort_transfer(ctx, "ad_vs_control", ["cov_lr", "pb_lr", "pb_class_comp_lr"])
    assert set(ct["train_cohort"]) == {"MSSM", "RADC"}, "cohort transfer did not run both directions"
    pv = compare_to_baseline(paths.results / "aim1" / "four_class", reps=50)
    assert pv is not None and "macro_f1_p_holm" in pv, "paired comparison missing"
    figs = make_all_figures(paths, "four_class")
    from psychad_ml.summary import write_summary
    md = write_summary(cfg, paths).read_text()
    assert "Aim 1: four_class" in md and "verdict" in md, "RESULTS_SUMMARY.md missing expected sections"
    assert "Paired test vs" in md and "External check" in md and "training diagnostics" in md, "round-2 sections missing"

    s = pd.read_csv(paths.results / "aim1" / "four_class" / "summary.csv").set_index("model")
    ctx.log(f"{len(figs)} figures; total {time.time() - t0:.0f}s")
    best = s["macro_f1"].drop(["prior", "cov_lr", "cov_rf"], errors="ignore")
    assert best.max() > s.loc["prior", "macro_f1"] + 0.1, "no model beats the prior: planted signal not recovered"
    print("\nSMOKE TEST OK. best model:", best.idxmax(), round(best.max(), 3))
    if not a.keep and a.workdir is None:
        shutil.rmtree(w, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        from psychad_ml.cli import hard_exit

        hard_exit(1)
    from psychad_ml.cli import hard_exit   # avoids the macOS torch+xgboost OpenMP hang at interpreter exit

    hard_exit(0)
