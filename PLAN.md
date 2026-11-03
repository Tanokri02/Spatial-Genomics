# Project Plan: Disease-Specific Representations from PsychAD snRNA-seq

> **Status (3 Nov 2026):** the code is written and tested on synthetic data. See `README.md` (usage, design decisions),
> `docs/DATA_NOTES.md` (verified facts about the CELLxGENE data, donor counts, confounds) and `docs/WORKPLAN.md`
> (3 weeks, 2 people). Where this plan and those files disagree, the newer files win (e.g. cohorts, 3-week timeline).


Working title: *Learning Disease-Specific Representations from Single-Nucleus RNA-seq Data*
Source: `Project Proposal.docx` (Aim 1: benchmark classifiers with donor-aware splits. Aim 2: cell-type-specific signal.)

## 0. Constraints that shape every decision

| Constraint | Consequence |
|---|---|
| Large dataset (PsychAD: ~1,494 donors, ~6.3M nuclei, DLPFC) | Never load the full matrix. Stream, subsample, and store a compact derived dataset. |
| Small disk | Keep raw files out of the working copy where possible. Delete shards after processing. Store only derived data (target: < 10 GB). |
| Little compute (CPU, free Colab GPU) | Use shallow models on HVGs or PCA. Keep the MLP small. Do cross-validation on a capped cell budget. |
| Class project timeline | Define a minimum deliverable (Section 8) and treat the rest as stretch. |

**The unit of inference is the donor, not the cell.** There are millions of cells but only tens to hundreds of donors per disease. Every design choice below follows from that.

## 1. Risks to check first (Phase 0)

These are unverified and can change the whole project, so resolve them first.

1. **Data access.** The AWS registry lists PsychAD as controlled access. The Roussos lab page says snRNA-seq is available via the AD Knowledge Portal (Synapse) and CELLxGENE. Check which route works for us now and whether we need a data-use request. If approval takes weeks, start the request today.
2. **Donor counts per diagnosis.** I could not find the PD and DLB counts (the preprint's Fig. 1a / Supp. Table 1 should have them). If PD or DLB has only about 10–30 donors, we are limited to a few folds and wide confidence intervals. We would then report that openly, or merge or reframe classes. For example, we could run one-vs-control tasks only, or group Lewy body disease (PD + DLB).
3. **Comorbidity.** Many donors carry multiple diagnoses (for example AD plus DLB). Decide up front: keep only donors with a single diagnosis. This matches the preprint's "minimal comorbidity" subset of 1,160 donors.
4. **Course fit.** This is a Spatial Genomics class and PsychAD is dissociated snRNA-seq with no spatial coordinates. Ask the instructor whether that is acceptable. Section 7 gives a cheap spatial extension if it is not.

## 2. Data strategy (low disk)

**Preferred route, in order:**
1. **CELLxGENE.** Query by metadata and download subsets. Use `cellxgene_census` (TileDB-SOMA) if PsychAD is in Census, since it streams only the rows you ask for. Otherwise download the per-dataset `.h5ad`. Check whether the collection is split by cell class, which makes downloads smaller.
2. **AD Knowledge Portal / Synapse.** Use `synapseclient` to download one file at a time.

**Rules for handling data:**
- Open `.h5ad` with `anndata.read_h5ad(path, backed="r")` and process in chunks. Never call `.to_memory()` on the whole thing.
- Make one pass that does the following and writes a small derived dataset:
  1. Filter to donors with a single diagnosis among {Control, AD, PD, DLB}, and age-matched as per Section 3.
  2. Subsample up to **N cells per donor per major cell class** (start with N = 200–500). This gives donors equal weight and shrinks the data 10–50x.
  3. Keep raw counts for ~3,000–5,000 highly variable genes, plus a gene list for later enrichment.
  4. Save as sparse `float32` `.h5ad` (or parquet plus npz) with all metadata.
- Estimated derived size: ~1,000 donors x 7 classes x 300 cells = ~2M cells x 4k genes sparse, which is a few GB. Reduce N or the gene count if it is too large.
- Pseudobulk matrices (donor x cell class x gene) are tiny. Build them in the same pass and use them for baselines and for DE sanity checks.
- Keep raw downloads and derived data outside git. Record every step in a script for reproducibility.

## 3. Preprocessing and confounder control

- QC: use the dataset's existing QC. Do not redo it.
- Normalize: library-size normalize to 10k, then `log1p`. Select HVGs using only the **training donors in each fold** (or, as an acceptable shortcut, HVGs from a donor-balanced subset with a note about it).
- Metadata to retain: donor ID, diagnosis, age, sex, ancestry, PMI, batch/pool, brain bank, cell class and subclass.
- **Age.** Controls in PsychAD span ages 0–108. Restrict to adults (for example age ≥ 60) or match controls to cases by age. Otherwise "AD vs Control" becomes "old vs young".
- **Batch.** If diagnosis correlates with batch or pool, models can learn batch. Mitigate by:
  - stratifying folds by batch where possible;
  - adding a batch-only baseline (see 4);
  - optionally regressing out batch or using scVI/Harmony (stretch).

## 4. Aim 1: Benchmarking classifiers

**Task:** 4-class (Control / AD / PD / DLB). If the PD or DLB counts are low, also run one-vs-control tasks.

**Splitting:**
- `StratifiedGroupKFold` with donor as the group. Run 5 folds, repeated with different seeds.
- No donor ever appears in both train and test.
- Nested CV for hyperparameters, or a fixed small grid within the training folds.

**Models:**
| Model | Notes |
|---|---|
| Logistic regression (L2 / elastic net) | Main interpretable baseline. `scikit-learn`, runs on CPU. |
| Random forest | `scikit-learn`, `n_jobs=-1`. |
| XGBoost | `tree_method="hist"` (GPU optional). |
| MLP | PyTorch, 2 hidden layers, dropout, class-weighted loss. Fits on a free Colab GPU or CPU. |

**Baselines and controls (these make the results believable):**
- Majority class and stratified-random baselines.
- **Covariate-only baseline** (age, sex, ancestry, batch). The cell models must beat it.
- **Cell-level random split** as a deliberate "leaky" comparison. It demonstrates the inflation that donor-aware splits remove. This is a good figure for the report.
- **Label permutation** at the donor level to get a null distribution.

**Evaluation:**
- Report at both levels: per-cell and **per-donor** (aggregate by mean predicted probability, or majority vote).
- Metrics: accuracy, **macro-F1**, per-class recall, confusion matrices, plus balanced accuracy and AUROC (one-vs-rest).
- Use class weights or balanced sampling for imbalance. Add bootstrap confidence intervals over donors.

## 5. Aim 2: Cell-type-specific signal

1. Define cell classes from the dataset's annotations (excitatory neurons, inhibitory neurons, astrocytes, microglia, oligodendrocytes, OPCs, vascular). Subclasses are optional.
2. Train the best-performing Aim 1 models (and logistic regression as a fixed reference) **separately within each cell class**, using the same donor folds.
3. **Fair comparison.** Cell types have very different abundances, so match donors and cells per donor across classes. Otherwise a performance difference may only reflect data size. Check with a downsampling curve.
4. **Shared vs specific signal.** Build a train-on-X, test-on-Y transfer matrix. For example, train AD vs Control and test on PD vs Control, per cell class.
5. **Feature importance:**
   - LR coefficients (stable across folds) and XGBoost SHAP values;
   - integrated gradients for the MLP (optional);
   - keep only features that are stable across folds.
6. **Biological validation.**
   - Compare top genes with known markers (for example microglial activation genes such as *APOE*, *TREM2*, *SPP1*).
   - Run pathway enrichment (`gseapy`).
   - Run pseudobulk DE per cell class (`pydeseq2`) as an independent check on the classifier's top genes.

## 6. Compute and tooling

- **Local (Mac):** data extraction, preprocessing, pseudobulk, LR, RF. Use `scanpy`, `anndata`, `scikit-learn`, `pandas`.
- **Colab (free):** XGBoost with GPU, MLP training, SHAP. Read the derived dataset from Google Drive (check your Drive quota) or re-fetch it.
- Cache fold indices and predictions to disk so that experiments are resumable after Colab timeouts.
- Repo layout:
  ```
  Project/
    PLAN.md
    data/            # gitignored: raw + derived
    src/             # extract.py, preprocess.py, splits.py, models.py, eval.py
    notebooks/       # exploration + figures
    results/         # metrics, predictions, figures
    environment.yml
  ```

## 7. Optional extension for the spatial angle (stretch)

Only attempt after the core is done, and ask the instructor first.
- Project the top cell-type-specific gene signatures onto a public spatial dataset of human cortex (for example SEA-AD MERFISH, or 10x Visium DLPFC) and check layer or region localization.
- Or use spatial/layer-aware cell-subtype annotations to see whether disease signal concentrates in particular cortical layers.

## 8. Timeline (relative weeks, to be fit to the real deadline)

| Phase | Work | Output |
|---|---|---|
| Week 0 | Data access, donor counts per diagnosis, comorbidity filtering, instructor check on spatial scope | Go/no-go on class design |
| Week 1 | Streaming extraction, subsample, derived dataset, EDA (donor/cell counts, age/batch confounders) | `derived.h5ad`, EDA figures |
| Week 2 | Aim 1 splits, baselines, LR/RF/XGB, leaky-vs-donor comparison | Aim 1 results table |
| Week 3 | MLP, permutation and covariate controls, per-donor aggregation | Final Aim 1 figures |
| Week 4 | Aim 2 per-cell-type models, transfer matrix, matched-size controls | Aim 2 figures |
| Week 5 | Feature importance, enrichment, DE check, optional spatial extension | Biology figures |
| Final week | Report and slides | Deliverable |

**Minimum deliverable:** donor-aware Aim 1 benchmark (LR, RF/XGB, small MLP) with the leaky-split comparison, plus Aim 2 for at least 4 cell classes and top features for one or two of them. Everything else is stretch.

## 9. Open questions for you

1. What is the deadline, and how many people are on the team?
2. Is the non-spatial dataset acceptable for the class, or do we need the Section 7 extension?
3. Do you already have PsychAD access (Synapse account, data-use approval), or do we start that now?
4. Roughly how much free disk do you have? That sets N cells per donor and the gene count.

## Sources
- [Population-scale cross-disorder atlas of the human prefrontal cortex at single-cell resolution](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12726817/)
- [Single-cell atlas of transcriptomic vulnerability across multiple neurodegenerative and neuropsychiatric diseases (medRxiv)](https://www.medrxiv.org/content/10.1101/2024.10.31.24316513.full.pdf)
- [The Mount Sinai PsychAD Study (AWS Open Data registry)](https://registry.opendata.aws/mssm-psych-ad/)
- [Roussos lab data resources](https://labs.icahn.mssm.edu/roussos-lab/resources/)
