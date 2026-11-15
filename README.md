# Learning disease-specific representations from PsychAD snRNA-seq

Donor-aware classification of **Control / AD / PD / DLB** from single-nucleus RNA-seq of human prefrontal cortex
(Mount Sinai PsychAD, via CELLxGENE), and a test of whether the signal is **cell-type specific**.
See `Project Proposal.docx` (aims), `PLAN.md` (original plan) and `docs/` (verified data facts, 3-week work plan).

> **The unit of inference is the donor, not the cell.** There are millions of nuclei but only 29 PD and 33 DLB
> donors. Every split is by donor, every headline metric is donor-level with bootstrap CIs over donors.

## Status

All scripts are written and tested **on synthetic data** (28 unit tests + an end-to-end smoke test of every stage,
~40 s). The real-data path is verified up to metadata (`psychad inspect` ran on the live Census) and expression
reads (contiguous windows from Census returned raw counts at ~750 cells/s). **The full real build has been run**
(Census 2025-11-08; ~5.3 h on a Mac; see [Data](#data-dataderivedfull) below). **No models have been trained yet.**
GPU/MPS code paths for the torch models are untested (CPU only).

## Quick start

```bash
conda env create -f environment.yml && conda activate psychad      # or: pip install -e ".[all]"
make test                  # unit tests
make smoke                 # every stage end-to-end on synthetic data, offline
psychad --profile dev inspect      # audit cohorts/labels/cell types from the live CELLxGENE Census (metadata only)
psychad build                      # ~2-5 h (5.3 h for the full run); resumable; writes data/derived/full/
bash scripts/run_all.sh            # the whole pipeline in order (PSYCHAD_PROFILE=dev for a fast pass)
```

Every command is `psychad [--profile dev] <command>`; `psychad --help` lists them. Outputs go to
`results/<run_name>/` (CSV/parquet/PNG); nothing in `data/` or `results/` is tracked by git.

## Data (`data/derived/full/`)

Output of `psychad build` (seed 0, CELLxGENE Census 2025-11-08, cohorts MSSM + RADC). Not tracked by git (1.8 GB);
share it out-of-band or rebuild with `psychad build`.

| File | Contents |
|---|---|
| `cells_{EN,IN,Astro,Micro,Oligo,OPC}.h5ad` | single-cell **raw integer counts** (float32), 5,000 genes; obs = donor_id, cell_class, cell_type, cohort, total_counts. No label column: join to `donors.csv` on `donor_id` |
| `pseudobulk.h5ad` | 3,731 donor x cell-class profiles of summed counts; obs = donor_id, cell_class, n_cells, label, age, sex, cohort |
| `donors.csv` | all 1,194 donors with label, age, sex, cohort, `keep`, `exclude_reason`, per-class cell counts |
| `selection.parquet` | the 899,307 selected cells (donor, class, cell type, cohort, total counts) |
| `gene_panel.csv` | 5,000 label-free HVGs (protein-coding; MT/ribo/MALAT1/NEAT1/sex-linked genes removed) |
| `build_config.json` | exact config used |

Raw counts are stored; normalization (`target_sum=10000`, log1p) is done downstream.

### Data statistics

Census holds 4,834,135 cells from 1,194 donors in MSSM + RADC. After donor filters, **635 donors** and
**899,307 cells** remain (cap: 300 cells per donor per cell class; ages top-coded at 90).

**Donors by label** (kept donors only)

| Label | Donors | MSSM | RADC | Female / Male | Mean age (range) | Cells (all classes) |
|---|---|---|---|---|---|---|
| AD | 356 | 315 | 41 | 227 / 129 | 83.5 (60-90) | 498,370 |
| Control | 217 | 191 | 26 | 117 / 100 | 78.6 (60-90) | 313,538 |
| DLB | 33 | 33 | 0 | 16 / 17 | 80.1 (64-90) | 47,645 |
| PD | 29 | 26 | 3 | 15 / 14 | 81.2 (65-90) | 40,751 |

**Cells by class**

| Class | Cells | Donors with cells |
|---|---|---|
| Oligo | 181,331 | 631 |
| Astro | 166,872 | 630 |
| EN | 163,717 | 616 |
| IN | 159,236 | 616 |
| OPC | 117,276 | 626 |
| Micro | 110,875 | 612 |

**Pseudobulk profiles** (3,731 total; median 300 cells per profile, min 20): AD 2,087, Control 1,280, DLB 194, PD 170.

**Excluded donors (559 of 1,194):** other diagnosis 336, age < 60 73, comorbid AD+DLB 72, unspecified dementia 60,
too few cells 9, comorbid AD+PD 7, AD+DLB+PD 2.

**Caveats for modeling**
- DLB (33) and PD (29) are small next to AD (356): use donor-level stratified splits and macro metrics.
- Cohort confound: DLB is 100% MSSM and PD ~90% MSSM, so cohort can masquerade as disease signal.
- Age/sex confound: Controls are younger than AD (78.6 vs 83.5) and AD is skewed female. Age and sex are in the
  pseudobulk `obs` for use as covariates.
- 81 AD+DLB/PD comorbid donors are dropped, so DLB and PD labels are pure cases only.

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

1. **Data (CPU, ~2-5 h; `build` is done for `full`):** `inspect` (read the tables) -> `build` -> `eda` -> `splits`.
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
