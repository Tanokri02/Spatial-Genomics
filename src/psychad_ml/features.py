"""Load derived shards, normalise, cap cells, build donor-level (pseudobulk) features."""
from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import Paths
from .labels import task_classes


@dataclass
class CellData:
    X: sp.csr_matrix                 # log1p(CP10k) float32, columns = gene panel in HVG-rank order
    obs: pd.DataFrame                # index cell_id; donor_id, cell_class, cell_type, cohort, total_counts, y
    genes: pd.DataFrame              # feature_id (index), feature_name
    donors: pd.DataFrame             # index donor_id; label, task_label, age, sex, cohort, ...
    classes: list[str]               # task class names; y indexes into this
    cell_classes: list[str] = field(default_factory=list)
    counts: sp.csr_matrix | None = None   # raw counts (kept only if requested; needed for DE / pseudobulk sums)

    @property
    def n_cells(self) -> int:
        return self.X.shape[0]

    def subset(self, mask_or_idx) -> CellData:
        idx = np.where(mask_or_idx)[0] if getattr(mask_or_idx, "dtype", None) == bool else np.asarray(mask_or_idx)
        obs = self.obs.iloc[idx]
        return CellData(self.X[idx], obs, self.genes, self.donors.loc[self.donors.index.isin(obs["donor_id"])],
                        self.classes, self.cell_classes, None if self.counts is None else self.counts[idx])

    def top_genes(self, k: int) -> CellData:
        """Panel is stored in HVG-rank order so the top-k genes are a column prefix."""
        k = min(k, self.X.shape[1])
        return CellData(self.X[:, :k], self.obs, self.genes.iloc[:k], self.donors, self.classes, self.cell_classes,
                        None if self.counts is None else self.counts[:, :k])


def normalize_log1p(counts: sp.csr_matrix, total_counts: np.ndarray, target_sum: float = 1e4) -> sp.csr_matrix:
    """log1p(counts / total * target_sum). `total_counts` is the library size over ALL genes (not just the panel)."""
    X = sp.diags((target_sum / np.maximum(total_counts, 1.0)).astype(np.float32)) @ counts.astype(np.float32)
    X = sp.csr_matrix(X)
    np.log1p(X.data, out=X.data)
    return X


def cap_per_group(obs: pd.DataFrame, n: int | None, seed: int, by=("donor_id", "cell_class")) -> np.ndarray:
    """Positions of at most n random cells per group."""
    if n is None:
        return np.arange(len(obs))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(obs))
    shuffled = obs.iloc[order]
    rank = shuffled.groupby(list(by), sort=False).cumcount().values
    return np.sort(order[rank < n])


def read_donors(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.donors, index_col="donor_id")


def load_cells(cfg: dict, paths: Paths, cell_classes: list[str] | None = None, task: str | None = None,
               cells_per_donor_class: int | None = None, n_genes: int | None = None, seed: int = 0,
               keep_counts: bool = False, donors_subset: list[str] | None = None) -> CellData:
    """Load the per-class shards, join donor metadata and the task label, normalise, optionally cap cells."""
    task = task or cfg["default_task"]
    tmap = cfg["tasks"][task]
    classes = task_classes(tmap, cfg["labels"]["classes"])
    cell_classes = cell_classes or cfg["cell_classes_use"]
    donors = read_donors(paths)
    donors = donors[donors["keep"]].copy()
    donors["task_label"] = donors["label"].map(tmap)
    donors = donors[donors["task_label"].notna()]
    if donors_subset is not None:
        donors = donors.loc[donors.index.isin(donors_subset)]

    xs, obss, cs = [], [], []
    for cc in cell_classes:
        a = ad.read_h5ad(paths.shard(cc))
        a = a[a.obs["donor_id"].isin(donors.index)]
        o = a.obs.copy()
        # h5ad round-trips strings as categoricals; groupby on a categorical yields zero-size groups for absent
        # levels (e.g. filtered donors), which silently breaks per-group counts. Keep plain strings.
        for c in ("donor_id", "cell_class", "cell_type", "cohort"):
            o[c] = o[c].astype(str)
        pos = cap_per_group(o, cells_per_donor_class, seed)
        counts = sp.csr_matrix(a.X[pos] if not isinstance(a.X, np.ndarray) else sp.csr_matrix(a.X[pos]))
        o = o.iloc[pos]
        if n_genes:
            counts = counts[:, :n_genes]
        xs.append(normalize_log1p(counts, o["total_counts"].values, cfg["build"]["target_sum"]))
        if keep_counts:
            cs.append(counts)
        obss.append(o)
        genes = a.var if not n_genes else a.var.iloc[:n_genes]
    obs = pd.concat(obss)
    obs["y"] = obs["donor_id"].map(donors["task_label"]).map({c: i for i, c in enumerate(classes)}).astype(int)
    X = sp.vstack(xs, format="csr")
    counts = sp.vstack(cs, format="csr") if keep_counts else None
    donors = donors.loc[donors.index.isin(obs["donor_id"])]
    return CellData(X, obs, genes[["feature_name"]] if "feature_name" in genes else genes, donors, classes,
                    list(cell_classes), counts)


def donor_labels(cd: CellData) -> pd.Series:
    """Integer task label per donor (index donor_id)."""
    m = {c: i for i, c in enumerate(cd.classes)}
    return cd.donors["task_label"].map(m).astype(int)


def pseudobulk(cd: CellData, by_class: bool) -> pd.DataFrame:
    """Mean log-normalised expression per donor (optionally per cell class, columns 'class|gene')."""
    donors = list(cd.donors.index)
    d_idx = pd.Categorical(cd.obs["donor_id"], categories=donors).codes

    def mean_by_donor(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.where(mask)[0]
        G = sp.csr_matrix((np.ones(len(rows)), (d_idx[rows], np.arange(len(rows)))), shape=(len(donors), len(rows)))
        n = np.asarray(G.sum(1)).ravel()
        M = sp.diags(1.0 / np.maximum(n, 1)) @ (G @ cd.X[rows])
        return np.asarray(M.todense(), dtype=np.float32), n

    gene_names = cd.genes.index.astype(str).values
    if not by_class:
        M, _ = mean_by_donor(np.ones(cd.n_cells, bool))
        return pd.DataFrame(M, index=donors, columns=gene_names)
    blocks, cols = [], []
    for cc in cd.cell_classes:
        M, n = mean_by_donor((cd.obs["cell_class"] == cc).values)
        present = n > 0
        if present.any() and (~present).any():           # donors lacking this class: fill with the class mean
            M[~present] = M[present].mean(0)
        blocks.append(M)
        cols += [f"{cc}|{g}" for g in gene_names]
    return pd.DataFrame(np.hstack(blocks), index=donors, columns=cols)


def covariate_matrix(donors: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """age (z-scored later by the model pipeline), sex and cohort as 0/1 dummies."""
    parts = []
    for f in features:
        if f in ("age", "pmi"):
            parts.append(donors[[f]].astype(float).fillna(donors[f].median()))
        else:
            parts.append(pd.get_dummies(donors[f].astype(str), prefix=f, drop_first=False).astype(float))
    return pd.concat(parts, axis=1)
