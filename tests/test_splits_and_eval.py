import numpy as np
import pandas as pd
import pytest

from psychad_ml.evaluate import aggregate_donor, bootstrap_ci, classification_metrics
from psychad_ml.models.weights import make_sample_weights
from psychad_ml.splits import check_no_donor_overlap, make_folds


def _donors(n=(30, 40, 12, 12)):
    rows = []
    for lab, k, coh in zip(["Control", "AD", "PD", "DLB"], n, ["A", "B", "A", "B"]):
        rows += [(f"{lab}{i}", lab, coh if i % 3 else "B") for i in range(k)]
    return pd.DataFrame(rows, columns=["donor_id", "task_label", "cohort"]).set_index("donor_id")


def test_folds_partition_donors_and_stratify():
    d = _donors()
    f = make_folds(d, 5, [0, 1])
    for seed, g in f.groupby("seed"):
        assert sorted(g["donor_id"]) == sorted(d.index)               # every donor exactly once
        per = g.merge(d, left_on="donor_id", right_index=True).groupby(["fold", "task_label"]).size().unstack()
        assert (per.min() >= 2).all()                                 # each class in each fold
    assert not f[f.seed == 0].sort_values("donor_id")["fold"].equals(f[f.seed == 1].sort_values("donor_id")["fold"])


def test_folds_reject_tiny_classes():
    d = _donors((30, 40, 3, 12))
    with pytest.raises(ValueError):
        make_folds(d, 5, [0])


def test_leakage_guard():
    check_no_donor_overlap(["a"], ["b"])
    with pytest.raises(AssertionError):
        check_no_donor_overlap(["a", "b"], ["b"])


def test_sample_weights_balance_donors_and_classes():
    y = np.array([0] * 100 + [0] * 10 + [1] * 20)
    g = np.array(["a"] * 100 + ["b"] * 10 + ["c"] * 20)
    w = make_sample_weights(y, g)
    tot = pd.Series(w).groupby(g).sum()
    assert np.isclose(tot["a"], tot["b"])                              # donors of one class weigh equally
    assert np.isclose(tot["a"] + tot["b"], tot["c"])                   # classes weigh equally
    assert np.isclose(w.mean(), 1.0)


def test_aggregate_donor_rows_sum_to_one():
    p = np.random.default_rng(0).dirichlet(np.ones(3), 50)
    d = np.repeat(["x", "y"], 25)
    for method in ("mean_prob", "mean_logprob", "vote"):
        out = aggregate_donor(p, d, method)
        assert out.shape == (2, 3) and np.allclose(out.sum(1), 1)


def test_metrics_and_bootstrap_perfect_and_chance():
    y = np.repeat([0, 1, 2], 20)
    perfect = np.eye(3)[y]
    m = classification_metrics(y, perfect, 3)
    assert m["macro_f1"] == 1.0 and m["macro_auroc"] == 1.0
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.dirichlet(np.ones(3), 60), columns=["p_a", "p_b", "p_c"])
    df["y_true"] = np.repeat(["a", "b", "c"], 20)
    ci = bootstrap_ci(df, ["a", "b", "c"], reps=100)["macro_auroc"]
    assert ci[0] < 0.5 < ci[1] + 0.15
