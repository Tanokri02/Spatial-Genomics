#!/usr/bin/env python
"""Sanity-check finished results BEFORE interpreting them.  `python scripts/check_results.py [--profile dev]`

For every saved Aim 1 prediction set it verifies:
  * all (seed, fold) files exist (n_splits x n_seeds)
  * within each seed, every donor is predicted exactly once (so the folds partition the donors: no donor is both
    trained on and tested on, and none is silently missing)
  * probabilities are finite, in [0, 1] and sum to 1
  * the donor set and labels agree across models (same donors are being compared)
and that figures / RESULTS_SUMMARY.md exist. Prints PASS/WARN/FAIL lines; exit code 1 if any FAIL.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from psychad_ml.cli import hard_exit
from psychad_ml.config import get_paths, load_config

fails, warns = 0, 0


def report(level: str, msg: str):
    global fails, warns
    fails += level == "FAIL"
    warns += level == "WARN"
    print(f"[{level}] {msg}")


def check_task(res: Path, task: str, n_exp: int, aim: str = "aim1", n_splits: int = 5):
    pdir = res / aim / task / "preds"
    if not pdir.exists():
        report("WARN", f"{aim}/{task}: no predictions yet")
        return
    files = sorted(pdir.glob("*.parquet"))
    by_model: dict[str, list[Path]] = {}
    for f in files:
        by_model.setdefault(f.name.rsplit("__s", 1)[0], []).append(f)
    ref_donors = None
    for model, fs in sorted(by_model.items()):
        d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        pc = [c for c in d.columns if c.startswith("p_")]
        tag = f"{aim}/{task}/{model}"
        is_curve = "__n" in model or model.startswith(("ALL__", "cov", "prior")) or "__" in model
        if len(fs) < n_exp and len(fs) % n_splits == 0 and len(fs) > 0:
            report("WARN", f"{tag}: only {len(fs) // n_splits} of {n_exp // n_splits} seeds (complete 5-fold CV, fewer repeats; "
                           "intended for the slow models)")
        if len(fs) < n_exp and len(fs) % n_splits != 0 and not (model == "prior"):
            report("WARN" if is_curve else "FAIL", f"{tag}: {len(fs)}/{n_exp} fold files (incomplete)")
        elif "seed" in d:
            # every donor once per seed
            dup = d.groupby(["seed", "donor_id"]).size()
            if (dup > 1).any():
                report("FAIL", f"{tag}: {int((dup > 1).sum())} donor(s) predicted more than once in a seed")
            per_seed = d.groupby("seed")["donor_id"].nunique()
            if per_seed.nunique() != 1:
                report("FAIL", f"{tag}: seeds cover different donor counts {per_seed.to_dict()}")
        P = d[pc].values.astype(float)
        if not np.isfinite(P).all():
            report("FAIL", f"{tag}: NaN/inf probabilities")
        elif (P < -1e-6).any() or (P > 1 + 1e-6).any():
            report("FAIL", f"{tag}: probabilities outside [0, 1]")
        elif not np.allclose(P.sum(1), 1, atol=1e-3):
            report("FAIL", f"{tag}: probabilities do not sum to 1 (max dev {abs(P.sum(1) - 1).max():.3g})")
        else:
            report("PASS", f"{tag}: {len(fs)} folds, {d['donor_id'].nunique()} donors, probabilities valid")
        if aim == "aim1" and not is_curve:
            ids = set(d["donor_id"])
            ref_donors = ids if ref_donors is None else ref_donors
            if ids != ref_donors:
                report("WARN", f"{tag}: donor set differs from other models ({len(ids)} vs {len(ref_donors)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile")
    ap.add_argument("--config", help="extra YAML override file (same as `psychad --config`)")
    a = ap.parse_args()
    cfg = load_config(a.profile, a.config)
    paths = get_paths(cfg)
    res = paths.results
    n_exp = cfg["cv"]["n_splits"] * len(cfg["cv"]["seeds"])
    print(f"results dir: {res}   expected folds/model: {n_exp}")
    tasks = sorted(p.name for p in (res / "aim1").iterdir() if p.is_dir()) if (res / "aim1").exists() else []
    if not tasks:
        report("FAIL", "no results/aim1/<task> directories: nothing has been trained yet")
    for t in tasks:
        check_task(res, t, n_exp, n_splits=cfg["cv"]["n_splits"])
    # donor folds: partition + no leakage by construction
    for f in sorted((res / "splits").glob("folds_*.csv")) if (res / "splits").exists() else []:
        s = pd.read_csv(f)
        ok = (s.groupby(["seed", "donor_id"]).size() == 1).all()
        report("PASS" if ok else "FAIL", f"splits/{f.name}: each donor in exactly one fold per seed" if ok
               else f"splits/{f.name}: a donor appears in several folds in one seed")
    for label, p in [("RESULTS_SUMMARY.md", res / "RESULTS_SUMMARY.md"), ("figures/", res / "figures")]:
        report("PASS" if p.exists() else "WARN", f"{label} {'exists' if p.exists() else 'missing: run scripts/analyze.sh'}")
    print(f"\n{fails} FAIL, {warns} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    code = main()
    hard_exit(code)
