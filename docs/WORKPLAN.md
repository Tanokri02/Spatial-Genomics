# 3-week work plan (2 people)

Scripts are written; the work is now running, checking and interpreting. Person A = data and classical models,
Person B = deep models and biology. Both own the report. Dates are relative (Day 1 = start of compute).

## Week 1: data and a trustworthy pipeline

| Day | Person A | Person B |
|---|---|---|
| 1 | Run `psychad inspect`, read the tables together, confirm cohorts/labels/age rule. Install env, `make test`, `make smoke`. | Same setup; get GPU access running (Colab / cluster / cloud), run `make smoke` there. |
| 1-2 | `psychad build` (≈2 h, resumable). Check `donors.csv` and the build log. | `psychad --profile dev` is not available until the build exists: meanwhile read `models/torch_models.py`, decide MLP/MIL size grids. |
| 2-3 | `eda`, `splits` for all tasks. Look at the EDA figures for confounds. | Dev-profile runs of `cell_mlp`, `mil_attn` (GPU). Fix anything torch-specific (device, memory). |
| 4-5 | `benchmark` for `cell_lr`, `pb_lr`, `pb_class_lr`, `pb_class_xgb`; `controls covariate`. | Same for `cell_mlp`, `cell_mlp_large`, `mil_attn`. |
| 6-7 | **Checkpoint:** do any models beat the covariate baseline and the permutation null? If not, discuss (merge PD+DLB, add Aging controls, drop cohort-confounded donors) before spending more compute. | |

## Week 2: Aim 1 complete

- A: `cell_rf`, `cell_xgb`; `controls leaky`, `controls permutation` (donor-level model, cheap).
- B: `scaling` for `cell_mlp` and `mil_attn`; pick the scaled-down config; repeat the best model with 3 seeds.
- Both: run `benchmark` for `lbd_group` and `*_vs_control`; `psychad report`; start the Methods section.
- Output: Aim 1 table with CIs, confusion matrices, leaky-vs-donor figure, permutation figure.

## Week 3: Aim 2, biology, writing

- A: `celltype`, `transfer`.
- B: `importance` (+ `--shap`), `de`, overlap table, enrichment.
- Both (last 3 days): figures, interpretation (which cell class carries the signal; shared vs specific), limitations,
  report and slides. Freeze all results by day 18; leave days 19-21 for writing only.

## Stop/go rules

- If PD/DLB results are noise (CI includes chance), say so and present `lbd_group` / one-vs-control as the primary result.
- If the covariate baseline matches the cell models, the honest conclusion is "confounded"; do not tune until it disappears.
- Cut order if time runs short: SHAP, enrichment, MIL, `cell_rf`, three seeds → one. Never cut the controls.
