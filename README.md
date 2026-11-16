# Learning disease-specific representations from PsychAD snRNA-seq

Donor-aware classification of **Control / AD / PD / DLB** from single-nucleus RNA-seq of human prefrontal cortex
(Mount Sinai PsychAD, via CELLxGENE), and a test of whether the signal is **cell-type specific**.
See `Project Proposal.docx` (aims), `PLAN.md` (original plan) and `docs/` (verified data facts, 3-week work plan).

> **The unit of inference is the donor, not the cell.** There are millions of nuclei but only 29 PD and 33 DLB
> donors. Every split is by donor, every headline metric is donor-level with bootstrap CIs over donors.

## Status

All scripts are written and tested **on synthetic data** (37 unit tests + an end-to-end smoke test of every stage). The real-data path is verified up to metadata (`psychad inspect` ran on the live Census) and expression
reads (contiguous windows from Census returned raw counts at ~750 cells/s). **The full real build has been run**
(Census 2025-11-08; ~5.3 h on a Mac; see [Data](#data-dataderivedfull) below). Round 1 was trained overnight on
15-16 Nov (see `results/full/RESULTS_SUMMARY.md`); round 2 (below) is implemented and tested on synthetic data, not yet
trained on the real data.
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

## Round 2: changes after the first results (16 Nov 2026)

Round 1 (overnight 15-16 Nov) showed three things, and round 2 responds to each. No setting below was chosen by looking
at test-fold scores; the round-1 models are unchanged, and their cached results are still valid (verified: the old and
new code give bit-identical predictions for the round-1 configs).

| Round-1 finding | Response (model names in `configs/default.yaml`) |
|---|---|
| `cell_mlp` and `mil_attn` picked their **first** checkpoint in every fold, with validation loss at chance: 2.7M parameters vs ~430 training donors, so the nets memorised donors | **Small nets** on a 64-component PCA input fitted inside each fold (`mlp_small`, ~9k params); constant LR (the one-cycle warm-up meant "epoch 1" was nearly untrained); validation every 100 steps; early stopping on **donor-level AUROC**; **5-member ensembles** (different seeds and validation donors). **Cell-class-aware MIL** (`mil_small`: attention within each cell class, class embeddings concatenated). **scVI latent** as input (`psychad embed`, then `mlp_scvi`, `mil_scvi`, `pb_scvi_lr`) |
| Accuracy is flat from 25 to 50 cells per donor: the donor count, not cell count, is the bottleneck | Donor-level features: true **cell-type composition** from the metadata of all 2.6M nuclei (`build --stages composition`; `pb_comp_lr`, `pb_class_comp_lr`), **subtype pseudobulk** (`pb_subtype_lr`), a **gene filter** that drops rarely detected genes (`pb_class_lr_filt`), **PCA per cell class** (`pb_class_pca_lr`), **elastic net** (`pb_class_enet`) |
| The best models never predict PD/DLB (recall 0) | **Rare-class decision rule** (`decision: {tune: true}`): per-class offsets tuned on training donors only (inner CV, or the nets' validation donors). Each such model also writes `<model>+bias` predictions, so the summary shows both |
| "CI above the baseline's point" is a weak test; the point estimate could lie outside its own CI | **Paired donor-bootstrap** test vs the best baseline, Holm-corrected (`psychad compare`, run by `summary`); point estimates now computed on the same seed-averaged predictions as the CI. CIs added to the Aim 2 and transfer tables |
| No test set outside cross-validation | **Cohort transfer** (`psychad cohort-transfer`): train on MSSM, test on RADC and vice versa (AD vs Control only; PD/DLB are MSSM-only) |

Leakage notes for round 2: PCA/standardisation are fitted on each fold's training cells; composition and the gene filter
are label-free; scVI is label-free and fitted once on all donors (transductive, like the HVG panel; per-fold scVI
would cost 15x). The decision rule never sees the test fold.

## Pipeline

| Command | What it does | Output |
|---|---|---|
| `inspect` | label/cohort/age/cell-type audit from metadata only | `results/<run>/inspect/*.csv` |
| `build` | donor table + filters, donor-balanced cell selection (≤N cells/donor/class), streaming label-free HVG panel, expression fetch, per-class shards, pseudobulk, true cell-type composition (`--stages composition`, metadata only) | `data/derived/<run>/` |
| `eda` | donors by label×cohort, age, cells per donor | `figures/eda_*.png` |
| `splits` | cached stratified donor folds, shared by everything | `splits/folds_<task>.csv` |
| `benchmark` | **Aim 1**: LR, RF, XGBoost, MLP (+large), pseudobulk LR/XGB, attention-MIL | `aim1/<task>/summary.csv`, confusion matrices |
| `controls` | covariate-only + prior baselines, **leaky cell split vs donor split**, label-permutation null | `aim1/` (baselines), `controls/` |
| `celltype` | **Aim 2**: per-cell-class models on *matched* donors and cells; pooled reference; downsampling curve | `aim2/<task>/` |
| `transfer` | train X-vs-Control, test Y-vs-Control, per cell class | `aim2/transfer/transfer_auroc.csv` |
| `importance` | LR coefficient stability across folds (+ `--shap` for XGBoost; `--seeds 0` to speed up) | `importance/<task>/` |
| `de` | pseudobulk DE (PyDESeq2, age/sex/cohort adjusted) + Enrichr + overlap with model genes | `de/four_class/` |
| `scaling` | accuracy vs size: architecture, gene count, cells/donor; picks cheapest near-best | `scaling/<task>/<model>/` |
| `embed` | label-free scVI latent of every stored cell (trained on a donor-balanced subsample; batch = cohort) | `data/derived/<run>/scvi/` |
| `cohort-transfer` | train on one cohort, test on the other (AD vs Control) | `cohort_transfer/<task>/` |
| `compare` | paired donor-bootstrap test of every model vs the best baseline, Holm-corrected | `aim1/<task>/paired_vs_baseline.csv` |
| `report` | renders every figure for which results exist | `figures/<task>/` |
| `summary` | consolidated tables + verdicts vs baselines (reads saved predictions, trains nothing) | `RESULTS_SUMMARY.md` |

Resumability: `build` caches each stage and each read window; every CV run caches each (model, seed, fold).
Interrupted jobs (Colab timeout, SLURM wall-time) resume by re-running the same command.

## Train everything unattended, then analyse

```bash
conda activate psychad                       # env needs llvm-openmp (macOS XGBoost); see environment.yml
caffeinate -ims bash scripts/overnight.sh    # all models, controls, Aim 2, importance, DE. Plug in the charger, lid open
bash scripts/analyze.sh --open               # afterwards: figures + RESULTS_SUMMARY.md + sanity checks (trains nothing)
```

`scripts/overnight.sh` runs every step sequentially, most valuable and cheapest first, long optional steps last. A failing
step is retried once, then skipped; the night continues. Everything is cached per (model, seed, fold), so re-running
the script resumes. Logs: `logs/overnight_<time>/<step>.log` and `status.tsv` (`tail -f logs/overnight_*/status.tsv`).
`SKIP="extras big"` / `ONLY="core"` select steps; `PSYCHAD_CONFIG=configs/default.yaml` runs the exact defaults.
Round-2 steps (groups `r2_*`) run first; then the round-1 steps, which only train what is still missing (mainly the slow
cell-level models). Steps whose output already exists (permutation null, transfer, importance, DE, leaky split, scVI
embedding) are skipped unless `RERUN_DONE=1`. `cell_mlp_large` and the round-1 net scaling sweeps are in group
`legacy_nn`, skipped by default (`SKIP=""` to include them).

**Measured cost on an M4 / 16 GB** (one fold, ~287k training cells; a full benchmark is 15 folds = 5 folds x 3 seeds):

| Model | per fold | 15 folds | Note |
|---|---|---|---|
| pb_lr / pb_class_lr / pb_class_xgb | 1 / 0.6 / 1.2 min | ~20 min together | donor-level (pseudobulk) |
| cell_lr | 6 min | 90 min | **seed 0 only overnight (~30 min)** |
| cell_mlp (2.7 M params) | 1.1 min | 17 min | MPS GPU |
| mil_attn (2.8 M) | 1.3 min | 19 min | MPS GPU |
| cell_mlp_large (12.9 M) | 1.9 min | 29 min | MPS GPU |
| cell_xgb | 17.5 min | 4.4 h | **seed 0 only overnight (5 folds, ~90 min)** |
| cell_rf (300 trees) | >40 min | >10 h | **seed 0 only, 100 trees (`configs/overnight.yaml`)** |

Expect roughly 9-11 h for everything; the core table (pseudobulk, cell_xgb, cell_lr, nets), the permutation null, Aim 2 and DE come first (~6-7 h). Steps with a
tighter time cap (XGBoost, RF, scaling) stop at 3-4 h and keep their finished folds.

`scripts/analyze.sh` is safe on partial results. It renders figures (`results/<run>/figures/<task>/`), writes
`results/<run>/RESULTS_SUMMARY.md` (tables, verdicts vs the covariate/prior baselines, controls, Aim 2, scaling, genes)
and runs `scripts/check_results.py` (each donor predicted once per seed, probabilities valid, folds partition donors).

**macOS OpenMP rule (important).** torch, scikit-learn and xgboost each ship their own OpenMP runtime. XGBoost and torch
in one process can segfault or deadlock (observed on the real data: a hang inside `torch.stack` after a `cell_xgb` fit,
and a segfault when torch was imported first). The code therefore never imports torch or xgboost at package import,
skips torch entirely for commands without a neural net, splits mixed `--models` lists into separate processes, and
ends the CLI with `os._exit` (normal shutdown can also hang). Do not undo these; run XGBoost and torch models in separate
commands if you call the library from your own scripts.

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
  config.py labels.py features.py splits.py evaluate.py stats.py summary.py viz.py cli.py
  data/         sources.py (Census | h5ad)  select.py  hvg.py  build.py  inspect_data.py  synthetic.py
  models/       sk_models.py (LR/RF/XGB)  torch_models.py (MLP, attention-MIL)  weights.py
                decision.py (rare-class decision rule)  transforms.py (in-fold PCA / standardize)
  experiments/  cv.py (shared runner)  benchmark.py  controls.py  celltype.py  importance.py  de.py  scaling.py
                embed.py (scVI)  cohort.py (cohort transfer)
scripts/      smoke_test.py  run_all.sh  slurm/job.sbatch
tests/        28 unit/integration tests on synthetic data
docs/         DATA_NOTES.md (verified facts, caveats)  WORKPLAN.md (3 weeks, 2 people)
```

## Compute

Local Mac: `inspect`, `build`, `eda`, LR/RF/pseudobulk. GPU (Colab, cluster, cloud): `cell_mlp*`, `mil_attn`,
XGBoost (`device: cuda` is automatic), SHAP. `scripts/slurm/job.sbatch` wraps any command for a SLURM cluster;
parallelise by giving each job a different `--models` list.
