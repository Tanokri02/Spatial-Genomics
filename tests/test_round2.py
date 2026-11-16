"""Round-2 additions: decision rule, input transforms, small/per-class MIL, donor features, paired tests."""
import numpy as np
import pandas as pd
import pytest

from psychad_ml.evaluate import summarize_preds
from psychad_ml.models.decision import apply_bias, inner_oof_proba, tune_bias
from psychad_ml.models.transforms import InputTransform
from psychad_ml.stats import holm, paired_bootstrap


# ------------------------------------------------------------------ decision rule
def test_tune_bias_recovers_a_never_predicted_class():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], [60, 60, 15])
    P = rng.dirichlet(np.ones(3) * 5, len(y))
    P[y == 2, 2] += 0.12                       # rare class: higher than elsewhere, but almost never the argmax
    P /= P.sum(1, keepdims=True)
    from sklearn.metrics import f1_score
    f1_before = f1_score(y, P.argmax(1), average="macro")
    b = tune_bias(P, y)
    f1_after = f1_score(y, apply_bias(P, b).argmax(1), average="macro")
    assert b[0] == 0 and f1_after > f1_before
    assert np.allclose(apply_bias(P, b).sum(1), 1)
    assert np.array_equal(apply_bias(P, np.zeros(3)).argmax(1), P.argmax(1))


def test_inner_oof_proba_only_uses_given_rows():
    from sklearn.linear_model import LogisticRegression
    X = np.random.default_rng(1).normal(size=(40, 3))
    y = np.repeat([0, 1], 20)
    P = inner_oof_proba(lambda: LogisticRegression(), X, y, 3, 0)
    assert P.shape == (40, 2) and np.allclose(P.sum(1), 1)
    assert inner_oof_proba(lambda: LogisticRegression(), X, np.r_[np.zeros(39), 1].astype(int), 3, 0) is None


# ------------------------------------------------------------------ input transforms
def test_input_pca_fits_on_given_cells_and_projects_others():
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(0.3, size=(500, 50)).astype(np.float32))
    tf = InputTransform({"type": "pca", "n_components": 6, "fit_cells": 300}, seed=0).fit(X[:400])
    Z = tf.transform(X[400:])
    assert Z.shape == (100, 6) and Z.dtype == np.float32 and np.isfinite(Z).all()
    Z2 = InputTransform({"type": "pca", "n_components": 6, "fit_cells": 300}, seed=0).fit(X[:400]).transform(X[400:])
    assert np.allclose(Z, Z2)                                    # deterministic given the seed
    st = InputTransform({"type": "standardize"}).fit(np.asarray(X[:400].todense()))
    assert np.allclose(st.transform(np.asarray(X[:400].todense())).mean(0), 0, atol=1e-4)
    assert InputTransform({"type": "genes", "n": 7}).fit(X).transform(X).shape[1] == 7


# ------------------------------------------------------------------ per-class MIL
def test_class_mil_net_handles_missing_classes():
    torch = pytest.importorskip("torch")
    from psychad_ml.models.torch_models import ClassMILNet
    net = ClassMILNet(in_dim=5, hidden=[8], attn_dim=4, n_classes_cell=3, n_out=4, dropout=0.0, input_dropout=0.0)
    x = torch.randn(2, 3, 6, 5)
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    out, a = net(x * mask[:, :, None, None], mask)
    assert out.shape == (2, 4) and torch.isfinite(out).all()
    assert torch.allclose(a.sum(-1), torch.ones(2, 3))


def test_small_nets_train_tune_and_write_bias_files(smoke):
    cfg, paths, ctx = smoke
    from psychad_ml.experiments.benchmark import run_benchmark
    run_benchmark(ctx, "four_class", models=["mlp_small", "mil_small"])
    pdir = paths.results / "aim1" / "four_class" / "preds"
    for m in ("mlp_small", "mil_small"):
        main = sorted(pdir.glob(f"{m}__s*.parquet"))
        adj = sorted(pdir.glob(f"{m}+bias__s*.parquet"))
        assert main and len(main) == len(adj)
        meta = pd.read_json(pdir / f"{m}__s0_f0.json", typ="series")
        assert meta["n_ensemble"] == 2 and meta["n_inputs"] == 8 and len(meta["decision_bias"]) == 4
        d = pd.read_parquet(adj[0])
        assert np.allclose(d.filter(like="p_").sum(1), 1)
        assert set(d["donor_id"]) == set(pd.read_parquet(main[0])["donor_id"])   # same held-out donors


# ------------------------------------------------------------------ donor features
def test_donor_features_composition_subtype_filter(smoke, tmp_path):
    cfg, paths, _ = smoke
    from psychad_ml.features import composition_features, donor_feature_matrix, expressed_gene_idx, load_cells
    cd = load_cells(cfg, paths, task="four_class")
    comp_fallback = composition_features(cd, None)
    assert len(comp_fallback) == len(cd.donors) and np.isfinite(comp_fallback.values).all()
    counts = cd.obs.groupby(["donor_id", "cell_type"]).size().unstack(fill_value=0)
    cls = cd.obs.drop_duplicates("cell_type").set_index("cell_type")["cell_class"]
    counts.columns = [f"{cls[t]}::{t}" for t in counts.columns]
    counts["Other::endothelial cell"] = 5
    f = composition_features(cd, counts)
    assert any(c.startswith("frac|") for c in f.columns) and "frac|Other" in f.columns
    idx = expressed_gene_idx(cd, 0.05)
    assert 0 < len(idx) <= cd.X.shape[1]
    X = donor_feature_matrix(cd, {"features": ["pseudobulk_by_class", "composition"], "gene_filter": {"min_frac": 0.05}},
                             counts)
    assert X.shape[0] == len(cd.donors) and X.shape[1] == len(idx) * len(cd.cell_classes) + f.shape[1]
    sub = donor_feature_matrix(cd, {"features": "pseudobulk_by_subtype", "subtype_min_cells": 2})
    assert sub.shape[0] == len(cd.donors) and not sub.isna().any().any()


def test_round2_pseudobulk_models_run(smoke):
    cfg, paths, ctx = smoke
    from psychad_ml.experiments.benchmark import run_benchmark
    ms = ["pb_class_lr_filt", "pb_comp_lr", "pb_class_comp_lr", "pb_subtype_lr", "pb_class_pca_lr", "pb_class_enet"]
    tbl = run_benchmark(ctx, "four_class", models=ms)
    assert set(ms) <= set(tbl["model"]) and {f"{m}+bias" for m in ms} <= set(tbl["model"])


# ------------------------------------------------------------------ statistics
def test_paired_bootstrap_and_holm():
    rng = np.random.default_rng(0)
    classes = ["a", "b"]
    y = np.repeat(classes, 40)
    good = pd.DataFrame({"donor_id": np.arange(80), "y_true": y, "p_a": np.where(y == "a", 0.8, 0.2)})
    good["p_b"] = 1 - good["p_a"]
    rnd = good.copy()
    rnd["p_a"] = rng.uniform(size=80)
    rnd["p_b"] = 1 - rnd["p_a"]
    r = paired_bootstrap(good, rnd, classes, reps=300)
    assert r["macro_auroc_delta"] > 0.3 and r["macro_auroc_p"] < 0.01 and r["macro_auroc_delta_ci_lo"] > 0
    same = paired_bootstrap(good, good, classes, reps=100)
    assert same["macro_f1_delta"] == 0 and same["macro_f1_p"] == 1.0
    adj = holm(pd.Series([0.01, 0.04, 0.03, 0.5]))
    assert list(adj.round(2)) == [0.04, 0.09, 0.09, 0.5]


def test_summary_point_estimate_inside_its_ci():
    rng = np.random.default_rng(3)
    rows = []
    for seed in range(3):
        for i in range(60):
            p = rng.dirichlet(np.ones(3))
            rows.append({"model": "m", "seed": seed, "fold": 0, "donor_id": i, "y_true": "abc"[i % 3],
                         "p_a": p[0], "p_b": p[1], "p_c": p[2]})
    t = summarize_preds(pd.DataFrame(rows), ["a", "b", "c"], reps=200).iloc[0]
    for m in ("macro_f1", "macro_auroc", "balanced_accuracy"):
        assert t[f"{m}_ci_lo"] - 1e-9 <= t[m] <= t[f"{m}_ci_hi"] + 1e-9
        assert f"{m}_seedmean" in t
