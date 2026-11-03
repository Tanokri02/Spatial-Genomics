"""Donor labels, age parsing and cell-class mapping, all driven by the config."""
from __future__ import annotations

import re

import pandas as pd

_AGE_RE = re.compile(r"(\d+)[ -]year-old")
_INFANT_RE = re.compile(r"(\d+)-(month|week|day)-old")


def parse_age(stage: str, top_code: float = 90.0) -> float:
    """CELLxGENE development_stage -> age in years. '80 year-old and over stage' -> top_code (censored)."""
    if not isinstance(stage, str):
        return float("nan")
    if "and over" in stage:
        m = _AGE_RE.search(stage)
        return max(float(m.group(1)), top_code) if m else top_code
    m = _AGE_RE.search(stage)
    if m:
        return float(m.group(1))
    if _INFANT_RE.search(stage):
        return 0.5
    return float("nan")


def parse_label(disease: str, lab_cfg: dict) -> tuple[str | None, str]:
    """Map a multi-valued CELLxGENE disease string to (label, reason).

    label is one of lab_cfg['classes'] or None; reason explains the outcome (also for exclusions).
    """
    terms = {t.strip() for t in str(disease).split("||") if t.strip()}
    specific = {cls: set(v) for cls, v in lab_cfg["terms"].items()}
    known = set().union(*specific.values()) | set(lab_cfg["control_terms"]) | set(lab_cfg["generic_terms"])
    hits = [cls for cls, v in specific.items() if terms & v]
    if terms - known:
        return None, "other_diagnosis"
    if len(hits) > 1:
        return None, "comorbid:" + "+".join(sorted(hits))
    if len(hits) == 1:
        return hits[0], "ok"
    if terms and terms <= set(lab_cfg["control_terms"]):
        return "Control", "ok"
    if terms and terms <= set(lab_cfg["generic_terms"]):
        return None, "unspecified_dementia"
    return None, "no_label"


class CellClassMapper:
    """Ordered regex -> cell class. First match wins; unmatched cell types map to None."""

    def __init__(self, cell_classes: list[dict]):
        self.rules = [(c["name"], re.compile(c["pattern"], re.IGNORECASE)) for c in cell_classes]

    def map_one(self, cell_type: str) -> str | None:
        for name, rx in self.rules:
            if rx.search(cell_type):
                return name
        return None

    def map_series(self, s: pd.Series) -> pd.Series:
        # map on unique values only (millions of cells, few dozen cell types)
        uniq = pd.Series(s.astype(str).unique())
        lut = {u: self.map_one(u) for u in uniq}
        return s.astype(str).map(lut)


def task_labels(donors: pd.DataFrame, task_map: dict) -> pd.Series:
    """Donor label -> task label (donors absent from the map get NaN, i.e. are dropped for that task)."""
    return donors["label"].map(task_map)


def task_classes(task_map: dict, order: list[str]) -> list[str]:
    seen = []
    for k in order:
        v = task_map.get(k)
        if v is not None and v not in seen:
            seen.append(v)
    return seen
