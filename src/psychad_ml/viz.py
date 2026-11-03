"""Figures. Palette: validated categorical slots in fixed order (never cycled); gray = Control; sequential = blue.

Every function reads tables written by the experiments (CSV/parquet) and writes PNG into results/<run>/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LABEL_COLOR = {"Control": "#8a8a85", "AD": CAT[0], "PD": CAT[1], "DLB": CAT[2], "LBD": CAT[6]}
CLASS_COLOR = {c: CAT[i] for i, c in enumerate(["EN", "IN", "Astro", "Micro", "Oligo", "OPC", "ALL"])}
CLASS_COLOR["ALL"] = "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _model_color(names: list[str]) -> dict:
    return {n: CAT[i % len(CAT)] for i, n in enumerate(names)}


# ---------------------------------------------------------------------------------------------- EDA
def fig_eda(donors: pd.DataFrame, out: Path) -> list[Path]:
    _style()
    d = donors[donors["keep"]]
    paths = []
    ct = pd.crosstab(d["label"], d["cohort"])
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bottom = np.zeros(len(ct))
    for i, c in enumerate(ct.columns):
        ax.bar(ct.index, ct[c], bottom=bottom, color=CAT[i], label=c, width=0.6, edgecolor="white", linewidth=1.5)
        bottom += ct[c].values
    for x, tot in zip(range(len(ct)), bottom):
        ax.text(x, tot + 3, int(tot), ha="center", color=MUTED, fontsize=9)
    ax.set_ylabel("donors"), ax.set_title("Donors per diagnosis, by cohort"), ax.legend(title="cohort")
    ax.grid(axis="x", visible=False)
    paths.append(_save(fig, out / "eda_donors_by_label_cohort.png"))

    labels = [c for c in LABEL_COLOR if c in set(d["label"])]
    fig, axes = plt.subplots(1, len(labels), figsize=(2.3 * len(labels), 2.6), sharex=True, sharey=True)
    for ax, lab in zip(np.atleast_1d(axes), labels):
        ax.hist(d.loc[d["label"] == lab, "age"].dropna(), bins=np.arange(55, 96, 5), color=LABEL_COLOR[lab],
                edgecolor="white")
        ax.set_title(f"{lab} (n={int((d['label'] == lab).sum())})", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("donors"), fig.supxlabel("age (90 = '80 and over')", fontsize=9)
    paths.append(_save(fig, out / "eda_age_by_label.png"))

    cols = [c for c in d.columns if c.startswith("n_") and c != "n_cells_total" and c != "n_cohorts"
            and c != "n_disease_strings"]
    if cols:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.boxplot([d[c] for c in cols], tick_labels=[c[2:] for c in cols], showfliers=False,
                   medianprops={"color": INK}, boxprops={"color": MUTED}, whiskerprops={"color": MUTED})
        ax.set_ylabel("selected cells per donor"), ax.set_title("Cells per donor after capping")
        ax.grid(axis="x", visible=False)
        paths.append(_save(fig, out / "eda_cells_per_donor_by_class.png"))
    return paths


# ---------------------------------------------------------------------------------------------- Aim 1
def fig_model_comparison(summary: pd.DataFrame, out: Path, metric: str = "macro_f1", title: str = "") -> Path:
    _style()
    s = summary.sort_values(metric)
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(s) + 1.2))
    y = np.arange(len(s))
    base = s["model"].str.startswith(("cov_", "prior"))
    colors = np.where(base, "#8a8a85", CAT[0])
    lo, hi = s.get(f"{metric}_ci_lo"), s.get(f"{metric}_ci_hi")
    if lo is not None:
        ax.hlines(y, lo, hi, color=colors, linewidth=1.6)
    ax.scatter(s[metric], y, color=colors, s=42, zorder=3, edgecolor="white", linewidth=1.2)
    ax.set_yticks(y, s["model"]), ax.set_xlabel(f"{metric.replace('_', ' ')} (donor level, 95% bootstrap CI)")
    ax.set_title(title or "Model comparison (gray = covariate / prior baselines)")
    ax.grid(axis="y", visible=False)
    return _save(fig, out / f"model_comparison_{metric}.png")


def fig_confusion(cm: pd.DataFrame, name: str, out: Path) -> Path:
    _style()
    cmn = cm.div(cm.sum(1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cmn.values, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(cm.shape[1]), [c[5:] for c in cm.columns]), ax.set_yticks(range(cm.shape[0]), [i[5:] for i in cm.index])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center", color="white" if cmn.values[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted"), ax.set_ylabel("true"), ax.set_title(f"{name} (donors; row-normalised shading)", fontsize=9)
    return _save(fig, out / f"confusion_{name}.png")


def fig_leaky(df: pd.DataFrame, out: Path) -> Path:
    _style()
    cols = ["cell_macro_f1", "donor_macro_f1"]
    g = df.groupby("split")[cols].mean()
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = np.arange(len(cols))
    w = 0.34
    for i, (split, color) in enumerate([("leaky_cell_split", CAT[1]), ("donor_split", CAT[0])]):
        if split in g.index:
            ax.bar(x + (i - 0.5) * w, g.loc[split, cols], w - 0.04, color=color, label=split.replace("_", " "))
            for xi, v in zip(x + (i - 0.5) * w, g.loc[split, cols]):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x, ["per cell", "per donor"]), ax.set_ylabel("macro-F1"), ax.legend()
    ax.set_title("Leakage inflates accuracy when cells are split randomly")
    ax.grid(axis="x", visible=False)
    return _save(fig, out / "leaky_vs_donor_split.png")


def fig_permutation(res: pd.DataFrame, name: str, out: Path) -> Path:
    _style()
    null, obs = res[~res["observed"]]["macro_f1"], res[res["observed"]]["macro_f1"].iloc[0]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.hist(null, bins=20, color="#8a8a85", edgecolor="white")
    ax.axvline(obs, color=CAT[0], linewidth=2)
    ax.text(obs, ax.get_ylim()[1] * 0.92, f" observed {obs:.2f}", color=CAT[0], fontsize=9)
    ax.set_xlabel("macro-F1 under permuted donor labels"), ax.set_ylabel("permutations")
    ax.set_title(f"Permutation null: {name}")
    return _save(fig, out / f"permutation_{name}.png")


# ---------------------------------------------------------------------------------------------- Aim 2
def fig_celltype(summary: pd.DataFrame, out: Path, metric: str = "macro_f1") -> Path:
    _style()
    s = summary.copy()
    classes = [c for c in CLASS_COLOR if c in set(s["cell_class"])]
    models = list(dict.fromkeys(s["base_model"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.8 / max(len(models), 1)
    for k, m in enumerate(models):
        d = s[s["base_model"] == m].set_index("cell_class").reindex(classes)
        x = np.arange(len(classes)) + (k - (len(models) - 1) / 2) * w
        lo, hi = d.get(f"{metric}_ci_lo"), d.get(f"{metric}_ci_hi")
        ax.errorbar(x, d[metric], yerr=[d[metric] - lo, hi - d[metric]] if lo is not None else None, fmt="o",
                    color=CAT[k], capsize=0, linewidth=1.3, markersize=6, markeredgecolor="white", label=m)
    ax.set_xticks(range(len(classes)), classes), ax.set_ylabel(metric.replace("_", " ")), ax.legend(title="model")
    ax.set_title("Disease signal by cell class (matched donors and cells)")
    ax.grid(axis="x", visible=False)
    return _save(fig, out / "celltype_performance.png")


def fig_downsample(curve: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for cc, d in curve.groupby("cell_class"):
        d = d.sort_values("cells_per_donor")
        ax.plot(d["cells_per_donor"], d["macro_f1"], marker="o", markersize=5, linewidth=1.8,
                color=CLASS_COLOR.get(cc, MUTED), label=cc)
        ax.text(d["cells_per_donor"].iloc[-1] * 1.03, d["macro_f1"].iloc[-1], cc, color=INK, fontsize=8, va="center")
    ax.set_xscale("log"), ax.set_xlabel("cells per donor (log)"), ax.set_ylabel("macro-F1"), ax.legend(fontsize=8)
    ax.set_title("Is it data volume? Performance vs cells per donor")
    return _save(fig, out / "downsample_curve.png")


def fig_transfer(tr: pd.DataFrame, out: Path) -> Path:
    _style()
    classes = list(tr["cell_class"].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(2.6 * len(classes), 2.8), squeeze=False)
    for ax, cc in zip(axes[0], classes):
        m = tr[tr["cell_class"] == cc].pivot(index="train_on", columns="test_on", values="auroc")
        ax.imshow(m.values, cmap="Blues", vmin=0.5, vmax=1.0)
        ax.grid(False)
        ax.set_xticks(range(m.shape[1]), m.columns, fontsize=8), ax.set_yticks(range(m.shape[0]), m.index, fontsize=8)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.8 else INK)
        ax.set_title(cc, fontsize=10)
    fig.supxlabel("test on (disease vs Control)", fontsize=9), fig.supylabel("train on", fontsize=9)
    return _save(fig, out / "transfer_matrix.png")


# ---------------------------------------------------------------------------------------------- scaling
def fig_scaling(df: pd.DataFrame, out: Path, name: str = "") -> Path:
    _style()
    sweeps = ["arch", "genes", "cells"]
    xlabel = {"arch": "parameters (log)", "genes": "genes in panel", "cells": "training cells per donor x class"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, sw in zip(axes, sweeps):
        d = df[df["sweep"] == sw]
        if d.empty:
            ax.set_visible(False)
            continue
        x = {"arch": d["n_params"], "genes": d["n_genes"], "cells": d["train_cells_per_donor_class"]}[sw]
        order = np.argsort(x.values)
        ax.plot(x.values[order], d["macro_f1"].values[order], marker="o", color=CAT[0], linewidth=1.8, markersize=5)
        pick = d[d["scaled_down_choice"]]
        if len(pick):
            px = {"arch": pick["n_params"], "genes": pick["n_genes"], "cells": pick["train_cells_per_donor_class"]}[sw]
            ax.scatter(px, pick["macro_f1"], s=90, facecolor="none", edgecolor=CAT[1], linewidth=2, zorder=4)
        if sw == "arch":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel[sw])
    axes[0].set_ylabel("macro-F1 (donor level)")
    fig.suptitle(f"Scaling down {name} (orange ring = cheapest within tolerance of best)", fontsize=10, fontweight="bold")
    return _save(fig, out / f"scaling_{name}.png")


# ---------------------------------------------------------------------------------------------- driver
def make_all_figures(paths, task: str) -> list[Path]:
    """Render every figure for which results exist."""
    fig_dir = paths.results / "figures"
    made: list[Path] = []
    if paths.donors.exists():
        made += fig_eda(pd.read_csv(paths.donors, index_col="donor_id"), fig_dir)
    a1 = paths.results / "aim1" / task
    if (a1 / "summary.csv").exists():
        s = pd.read_csv(a1 / "summary.csv")
        made.append(fig_model_comparison(s, fig_dir, "macro_f1", f"Aim 1 ({task}): macro-F1"))
        made.append(fig_model_comparison(s, fig_dir, "balanced_accuracy", f"Aim 1 ({task}): balanced accuracy"))
        for f in sorted(a1.glob("confusion_*.csv")):
            made.append(fig_confusion(pd.read_csv(f, index_col=0), f.stem.replace("confusion_", ""), fig_dir / "confusion"))
    c = paths.results / "controls" / task
    if (c / "leaky_vs_donor_split.csv").exists():
        made.append(fig_leaky(pd.read_csv(c / "leaky_vs_donor_split.csv"), fig_dir))
    for f in sorted(c.glob("permutation_*.csv")) if c.exists() else []:
        made.append(fig_permutation(pd.read_csv(f), f.stem.replace("permutation_", ""), fig_dir))
    a2 = paths.results / "aim2" / task
    if (a2 / "summary.csv").exists():
        made.append(fig_celltype(pd.read_csv(a2 / "summary.csv"), fig_dir))
    if (a2 / "downsample_curve.csv").exists():
        made.append(fig_downsample(pd.read_csv(a2 / "downsample_curve.csv"), fig_dir))
    t = paths.results / "aim2" / "transfer" / "transfer_auroc.csv"
    if t.exists():
        made.append(fig_transfer(pd.read_csv(t), fig_dir))
    for f in sorted((paths.results / "scaling" / task).glob("*/scaling.csv")):
        made.append(fig_scaling(pd.read_csv(f), fig_dir, f.parent.name))
    return made
