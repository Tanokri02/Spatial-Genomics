import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from psychad_ml.experiments import celltype, controls
from psychad_ml.experiments.benchmark import run_benchmark
from psychad_ml.features import cap_per_group, load_cells, normalize_log1p, pseudobulk


def test_build_outputs_and_donor_filters(smoke):
    cfg, paths, _ = smoke
    donors = pd.read_csv(paths.donors, index_col="donor_id")
    kept = donors[donors["keep"]]
    assert set(kept["label"]) == {"Control", "AD", "PD", "DLB"}
    reasons = set(donors.loc[~donors["keep"], "exclude_reason"])
    assert {"age_below_min", "unspecified_dementia", "other_diagnosis"} <= reasons
    assert any(r.startswith("comorbid") for r in reasons)               # AD+DLB donors dropped
    assert (kept["age"] >= cfg["donors"]["min_age"]).all()
    for cc in cfg["cell_classes_use"]:
        a = ad.read_h5ad(paths.shard(cc))
        assert (a.obs["cell_class"] == cc).all()
        assert a.obs.groupby("donor_id").size().max() <= cfg["build"]["cells_per_donor_class"]
        assert a.n_vars == cfg["build"]["n_hvg"] and (a.var["hvg_rank"].values == np.arange(a.n_vars)).all()
        assert a.X.min() >= 0 and (a.obs["total_counts"] >= a.X.sum(1).A1 - 1e-3).all()   # library size over ALL genes
    panel = pd.read_csv(paths.genes, index_col=0)
    assert not panel["feature_name"].str.match("^(MT-|RPL|RPS)").any()


def test_build_is_resumable(smoke):
    cfg, paths, _ = smoke
    from psychad_ml.data.build import run_build
    before = {p: p.stat().st_mtime for p in paths.derived.glob("cells_*.h5ad")}
    run_build(cfg, paths, stages=["select", "genes"])        # cached stages must not recompute
    assert before == {p: p.stat().st_mtime for p in paths.derived.glob("cells_*.h5ad")}


def test_normalize_uses_total_counts_not_panel_sum():
    c = sp.csr_matrix(np.array([[1.0, 1.0, 0.0]]))
    X = normalize_log1p(c, np.array([100.0]), 1e4)           # library size 100 although the panel sums to 2
    assert np.allclose(X.toarray()[0, :2], np.log1p(100.0))


def test_cap_per_group_and_load(smoke):
    cfg, paths, _ = smoke
    cd = load_cells(cfg, paths, task="four_class", cells_per_donor_class=7, n_genes=30)
    assert cd.X.shape[1] == 30 and cd.obs.groupby(["donor_id", "cell_class"]).size().max() <= 7
    assert set(cd.obs["donor_id"]) <= set(cd.donors.index)
    cd2 = load_cells(cfg, paths, task="ad_vs_control")
    assert cd2.classes == ["Control", "AD"] and set(cd2.donors["label"]) == {"Control", "AD"}
    pos = cap_per_group(cd.obs, 2, seed=0)
    assert len(pos) == len(set(pos))


def test_load_cells_returns_plain_string_keys(smoke):
    """Regression: categorical donor/class columns made groupby() emit zero-size groups for filtered donors."""
    cfg, paths, _ = smoke
    cd = load_cells(cfg, paths, task="ad_vs_control")        # drops PD/DLB donors that exist in the shards
    for c in ("donor_id", "cell_class", "cell_type", "cohort"):
        assert not isinstance(cd.obs[c].dtype, pd.CategoricalDtype), c
    sizes = cd.obs.groupby(["donor_id", "cell_class"]).size()
    assert (sizes > 0).all() and set(sizes.index.get_level_values(0)) == set(cd.donors.index)


def test_pseudobulk_shapes(smoke):
    cfg, paths, _ = smoke
    cd = load_cells(cfg, paths, task="four_class", n_genes=20)
    assert pseudobulk(cd, False).shape == (len(cd.donors), 20)
    assert pseudobulk(cd, True).shape == (len(cd.donors), 20 * len(cd.cell_classes))
    assert not pseudobulk(cd, True).isna().any().any()


def test_benchmark_runs_resumes_and_no_leakage(smoke):
    cfg, paths, ctx = smoke
    tbl = run_benchmark(ctx, "four_class", models=["cell_lr", "pb_lr"])
    assert {"cell_lr", "pb_lr"} <= set(tbl["model"]) and tbl["macro_f1"].between(0, 1).all()
    preds = pd.concat(pd.read_parquet(f) for f in (paths.results / "aim1" / "four_class" / "preds").glob("cell_lr__*.parquet"))
    assert not preds.duplicated(["seed", "donor_id"]).any()           # each donor predicted once per seed (held out once)
    t0 = (paths.results / "aim1" / "four_class" / "preds" / "cell_lr__s0_f0.parquet").stat().st_mtime
    run_benchmark(ctx, "four_class", models=["cell_lr"])              # resume: nothing recomputed
    assert t0 == (paths.results / "aim1" / "four_class" / "preds" / "cell_lr__s0_f0.parquet").stat().st_mtime


def test_covariate_baseline_and_matched_data(smoke):
    cfg, paths, ctx = smoke
    tbl = controls.run_covariate_baselines(ctx, "four_class")
    assert {"cov_lr", "cov_rf", "prior"} <= set(tbl["model"])
    cd, folds = celltype.matched_data(ctx, "four_class")
    n = cd.obs.groupby(["donor_id", "cell_class"]).size()
    assert (n == cfg["aim2"]["matched_cells_per_donor"]).all()          # identical cells/donor/class
    assert n.unstack().notna().all().all()                              # identical donors in every class
