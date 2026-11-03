# Data notes (verified against the live CELLxGENE Census, Nov 2026)

Reproduce every number here with `psychad inspect` (metadata only, a few minutes). Census build: `2025-11-08`.

## Where the data is

PsychAD ("Population-scale cross-disorder atlas of the human prefrontal cortex at single-cell resolution") is
**four CELLxGENE datasets**, 1,494 donors and 7,652,614 nuclei in total, all DLPFC, 10x 3' v3, nuclei:

| Cohort | Nuclei | Donors | Contents |
|---|---|---|---|
| MSSM | 4.14 M | 1,042 | AD / PD / DLB / controls / psychiatric / other |
| RADC | 0.69 M | 152 | AD + controls |
| Aging | 1.33 M | 291 | **controls only** (no cases); 113 of these donors also have nuclei in MSSM, 178 in HBCC |
| HBCC | 1.49 M | 300 | psychiatric (SCZ, BD) + young controls (178 shared with Aging) |

Dataset IDs are in `configs/default.yaml`.

**Donors overlap across datasets**: the per-cohort donor counts sum to 1,785 but there are only 1,494 unique donors,
because 291 donors have nuclei in two datasets (113 Aging+MSSM, 178 Aging+HBCC). Nuclei are different, donor is the same,
so always split and count by `donor_id`, never by dataset.

## What the Census does and does not contain

- Contains: `donor_id`, `disease` (donor-level, multi-valued, `" || "` separated), `cell_type` (ontology names),
  `development_stage` (age string; "80 year-old and over" is censored), `sex`, `raw_sum` (library size), raw counts.
- **Does not contain**: author columns such as post-mortem interval, library/pool/batch, ancestry (all
  `unknown`), cell subclass beyond the ontology names, comorbidity flags. `cohort` is the only batch proxy.
  If these are needed, download the cohort h5ad (`source.kind: h5ad`, one file at a time; MSSM is large) and map
  columns in `source.obs_columns`.
- Reads must be **contiguous key windows**. A 3,000-cell scattered read did not finish in 30 minutes; a 10,000-cell
  contiguous window took 13 s (~750 cells/s). Expect ~2 h for a full pass over MSSM+RADC.

## Donor counts after our filters

Single-diagnosis donors in all cohorts (before the age filter): Control 433, AD 367, DLB 34, PD 29.
After `min_age: 60`, **MSSM + RADC only**: AD 356, Control 217, DLB 33, PD 29. (Adding Aging: Control 227, i.e. +10 donors only.)

- PD and DLB have ~30 donors each: ~6 test donors per class per fold. Expect wide CIs; report them. The `lbd_group`
  (PD+DLB = 62) and one-vs-control tasks are the backup.
- Excluded for comorbidity: 81 donors (e.g. 74 AD+DLB). Excluded for "other diagnosis": 490 (mostly psychiatric).
  Unspecified "dementia" alone: 60.
- Aim 2 needs donors with ≥ m cells in *every* class. Microglia are limiting: m=50 keeps 184 Control / 286 AD /
  22 PD / 24 DLB; m=100 would keep only 16 PD / 20 DLB. Default m=50.

## Confounds to keep in mind

- **Cohort vs diagnosis.** Cases (AD/PD/DLB) exist only in MSSM and RADC; PD/DLB are almost entirely MSSM. Folds are
  stratified on label×cohort and `cohort` is in the covariate baseline. If the cell models do not clearly beat that
  baseline, the signal is cohort/age/sex.
- **Why Aging is excluded.** It has no cases. Because 113 of its donors are also in MSSM, including it adds only 10 new
  control donors (217 -> 227) but gives 68 existing controls nuclei from a second dataset (255k extra nuclei) that no case
  donor has, and the 10 new donors would come purely from a controls-only dataset. Small gain, real confound.
- **Age.** Even with age ≥ 60, AD donors are older on average (≈83.5 vs 78.6 for controls; age 90 = "80 and over").
  The covariate baseline and the DE model (age-adjusted) account for it; mention it in the report.
- **Sex** differs across labels; sex-chromosome genes are removed from the gene panel.

## Known limits of this pipeline

- Hyperparameters are fixed a priori (config), not tuned by nested CV; MLP/MIL early-stop on held-out training donors,
  donor-level LR chooses C by inner CV. Do not tune on test folds.
- HVGs are label-free but chosen from one donor-sampled subset of all donors (not re-chosen per fold).
- DE and pseudobulk are restricted to the stored gene panel (5,000 HVGs) and to ≤ N cells per donor×class.
- Enrichr (`de`) needs internet; it is skipped gracefully if unavailable.
- Synthetic-data tests prove the code runs and recovers planted signal; they say nothing about real accuracy.
