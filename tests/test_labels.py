import pandas as pd
import pytest

from psychad_ml.config import load_config
from psychad_ml.labels import CellClassMapper, parse_age, parse_label

LAB = load_config()["labels"]


@pytest.mark.parametrize("disease,label,reason", [
    ("normal", "Control", "ok"),
    ("Alzheimer disease", "AD", "ok"),
    ("dementia || Alzheimer disease", "AD", "ok"),
    ("dementia || Parkinson disease", "PD", "ok"),              # PD with dementia stays PD
    ("Lewy body dementia", "DLB", "ok"),
    ("dementia || Alzheimer disease || Lewy body dementia", None, "comorbid:AD+DLB"),
    ("dementia", None, "unspecified_dementia"),
    ("schizophrenia", None, "other_diagnosis"),
    ("dementia || Alzheimer disease || type 2 diabetes mellitus", None, "other_diagnosis"),
])
def test_parse_label(disease, label, reason):
    assert parse_label(disease, LAB) == (label, reason)


@pytest.mark.parametrize("stage,age", [("73-year-old stage", 73.0), ("80 year-old and over stage", 90.0),
                                       ("4-month-old stage", 0.5), ("unknown", float("nan"))])
def test_parse_age(stage, age):
    got = parse_age(stage)
    assert (got != got and age != age) or got == age


def test_cell_class_mapping_uses_real_census_names():
    cfg = load_config()
    m = CellClassMapper(cfg["cell_classes"])
    expect = {
        "L2/3 intratelencephalic projecting glutamatergic neuron": "EN",
        "L6b glutamatergic neuron of the primary motor cortex": "EN",
        "sst GABAergic cortical interneuron": "IN",
        "GABAergic neuron": "IN",
        "astrocyte": "Astro",
        "microglial cell": "Micro",
        "oligodendrocyte": "Oligo",
        "oligodendrocyte precursor cell": "OPC",       # must not be swallowed by the Oligo rule
        "endothelial cell": None, "T cell": None, "perivascular macrophage": None, "pericyte": None,
    }
    for ct, cls in expect.items():
        assert m.map_one(ct) == cls, ct
    s = m.map_series(pd.Series(list(expect)))
    assert s.tolist() == list(expect.values())
