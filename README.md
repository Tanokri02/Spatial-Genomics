# Learning disease-specific representations from PsychAD snRNA-seq

Donor-aware classification of **Control / AD / PD / DLB** from single-nucleus RNA-seq of human prefrontal cortex
(Mount Sinai PsychAD, via CELLxGENE), and a test of whether the signal is **cell-type specific**.
See `Project Proposal.docx` (aims), `PLAN.md` (original plan) and `docs/` (verified data facts, 3-week work plan).

> **The unit of inference is the donor, not the cell.** There are millions of nuclei but only 29 PD and 33 DLB
> donors. Every split is by donor, every headline metric is donor-level with bootstrap CIs over donors.

## Status

All scripts are written and tested **on synthetic data** (28 unit tests + an end-to-end smoke test of every stage,
~40 s). The real-data path is verified up to metadata (`psychad inspect` ran on the live Census) and expression
reads (contiguous windows from Census returned raw counts at ~750 cells/s). **The full real build and all model
training have not been run yet.** GPU/MPS code paths for the torch models are untested (CPU only).

## Quick start

```bash
conda env create -f environment.yml && conda activate psychad      # or: pip install -e ".[all]"
make test                  # unit tests
make smoke                 # every stage end-to-end on synthetic data, offline
psychad --profile dev inspect      # audit cohorts/labels/cell types from the live CELLxGENE Census (metadata only)
psychad build                      # ~2 h; resumable; writes data/derived/full/
bash scripts/run_all.sh            # the whole pipeline in order (PSYCHAD_PROFILE=dev for a fast pass)
```

Every command is `psychad [--profile dev] <command>`; `psychad --help` lists them. Outputs go to
`results/<run_name>/` (CSV/parquet/PNG); nothing in `data/` or `results/` is tracked by git.

## Pipeline

| Command | What it does | Output |
|---|---|---|
| `inspect` | label/cohort/age/cell-type audit from metadata only | `results/<run>/inspect/*.csv` |
| `build` | donor table + filters, donor-balanced cell selection (≤N cells/donor/class), streaming label-free HVG panel, expression fetch, per-class shards, pseudobulk | `data/derived/<run>/` |
| `eda` | donors by label×cohort, age, cells per donor | `figures/eda_*.png` |
| `splits` | cached stratified donor folds, shared by everything | `splits/folds_<task>.csv` |
| `benchmark` | **Aim 1**: LR, RF, XGBoost, MLP (+large), pseudobulk LR/XGB, attention-MIL | `aim1/<task>/summary.csv`, confusion matrices |
| `controls` | covariate-only + prior baselines, **leaky cell split vs donor split**, label-permutation null | `aim1/` (baselines), `controls/` |
| `celltype` | **Aim 2**: per-cell-class models on *matched* donors and cells; pooled reference; downsampling curve | `aim2/<task>/` |
| `transfer` | train X-vs-Control, test Y-vs-Control, per cell class | `aim2/transfer/transfer_auroc.csv` |
| `importance` | LR coefficient stability across folds (+ `--shap` for XGBoost) | `importance/<task>/` |
| `de` | pseudobulk DE (PyDESeq2, age/sex/cohort adjusted) + Enrichr + overlap with model genes | `de/four_class/` |
| `scaling` | accuracy vs size: architecture, gene count, cells/donor; picks cheapest near-best | `scaling/<task>/<model>/` |
| `report` | renders every figure for which results exist | `figures/` |

Resumability: `build` caches each stage and each read window; every CV run caches each (model, seed, fold).
Interrupted jobs (Colab timeout, SLURM wall-time) resume by re-running the same command.

## Suggested run order (CPU first, GPU only if it is worth it)

`bash scripts/run_all.sh` runs it in stages; each stage is resumable.

1. **Data (CPU, ~2 h):** `inspect` (read the tables) -> `build` -> `eda` -> `splits`.
2. **CPU models + controls:** LR, pseudobulk LR/XGB, RF, XGBoost, covariate/prior baselines, leaky split,
   permutation, Aim 2 with LR, transfer, importance, DE. `bash scripts/run_all.sh`
3. **Checkpoint:** do the models beat the covariate baseline and the permutation null? If not, stop and discuss.
4. **GPU stage:** `cell_mlp`, `mil_attn`. `bash scripts/run_all.sh gpu` (time one `--profile dev` run first).
5. **Optional:** `cell_mlp_large`, `scaling` sweeps, SHAP. `bash scripts/run_all.sh extra`

Hyperparameters are fixed in `configs/default.yaml` (no tuning on test folds); say so in the report.

## Design decisions (and why)

- **Cohorts**: MSSM + RADC only (default). `Aging` is controls-only and `HBCC` has no neurodegeneration cases, so
  including them lets a model learn the cohort instead of the disease (Aging would add only 10 new control donors, since
  most of its donors are already in MSSM; see `docs/DATA_NOTES.md`). Edit `cohorts_use` to change.
- **Labels**: CELLxGENE `disease` is a multi-valued string. Donors with ≥2 of AD/PD/DLB, any other diagnosis, or
  unspecified "dementia" are dropped (`donors.csv` keeps every donor with `exclude_reason`). "dementia || Parkinson
  disease" counts as PD. **Age ≥ 60** only; otherwise AD-vs-Control becomes old-vs-young.
- **Tasks** (`tasks:` in the config): `four_class`, `lbd_group` (PD+DLB merged), and `*_vs_control`. With ~30 donors
  for PD and DLB the four-class result will have wide CIs; the grouped and one-vs-control tasks are the fallback.
- **Cell classes**: EN, IN, Astro, Micro, Oligo, OPC. Vascular and immune types are too sparse per donor.
- **Honest evaluation**: donor-level folds stratified by label×cohort; class- and donor-balanced sample weights;
  training cells capped per donor×class (test cells never capped); donor probabilities = mean of cell probabilities.
- **Controls the cell models must beat**: age/sex/cohort-only models, majority prior, label permutation (within
  cohort by default), and a deliberately leaky random-cell split to show the inflation.
- **Accuracy first, then scale down** (no up-front size cap): `cell_mlp_large` and `mil_attn` set the ceiling;
  `psychad scaling` then sweeps width/depth, gene count and cells/donor and reports the smallest config within
  `scaling.tolerance` of the best.
- **Leakage hygiene**: HVG panel is chosen without labels from a donor-sampled subset (documented shortcut vs
  per-fold selection); everything supervised happens inside folds; MLP/MIL early stopping uses held-out *training donors*.

## Layout

```
configs/      default.yaml (all decisions), dev.yaml (10x smaller, fast iteration)
src/psychad_ml/
  config.py labels.py features.py splits.py evaluate.py viz.py cli.py
  data/         sources.py (Census | h5ad)  select.py  hvg.py  build.py  inspect_data.py  synthetic.py
  models/       sk_models.py (LR/RF/XGB)  torch_models.py (MLP, attention-MIL)  weights.py
  experiments/  cv.py (shared runner)  benchmark.py  controls.py  celltype.py  importance.py  de.py  scaling.py
scripts/      smoke_test.py  run_all.sh  slurm/job.sbatch
tests/        28 unit/integration tests on synthetic data
docs/         DATA_NOTES.md (verified facts, caveats)  WORKPLAN.md (3 weeks, 2 people)
```

## Compute

Local Mac: `inspect`, `build`, `eda`, LR/RF/pseudobulk. GPU (Colab, cluster, cloud): `cell_mlp*`, `mil_attn`,
XGBoost (`device: cuda` is automatic), SHAP. `scripts/slurm/job.sbatch` wraps any command for a SLURM cluster;
parallelise by giving each job a different `--models` list.
