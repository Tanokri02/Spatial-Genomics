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
               keep_counts: bool = False, donors_subset: list[str] | None = None,
               representation: str = "genes") -> CellData:
    """Load the per-class shards, join donor metadata and the task label, normalise, optionally cap cells.

    representation="genes": X = log1p(CP10k) of the gene panel (sparse).
    representation="scvi" : X = the scVI latent (dense float32) written by `psychad embed`; expression is not read.
    """
    if representation == "scvi":
        return _load_latent(cfg, paths, cell_classes, task, cells_per_donor_class, seed, donors_subset)
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


def _task_donors(cfg: dict, paths: Paths, task: str, donors_subset=None) -> tuple[pd.DataFrame, list[str]]:
    tmap = cfg["tasks"][task]
    classes = task_classes(tmap, cfg["labels"]["classes"])
    donors = read_donors(paths)
    donors = donors[donors["keep"]].copy()
    donors["task_label"] = donors["label"].map(tmap)
    donors = donors[donors["task_label"].notna()]
    if donors_subset is not None:
        donors = donors.loc[donors.index.isin(donors_subset)]
    return donors, classes


def latent_path(paths: Paths, cell_class: str, name: str = "scvi"):
    return paths.derived / name / f"latent_{cell_class}.npz"


def _load_latent(cfg, paths, cell_classes, task, cells_per_donor_class, seed, donors_subset) -> CellData:
    """Same cells, order and capping as load_cells, but X = scVI latent (written by `psychad embed`)."""
    task = task or cfg["default_task"]
    donors, classes = _task_donors(cfg, paths, task, donors_subset)
    cell_classes = cell_classes or cfg["cell_classes_use"]
    xs, obss = [], []
    for cc in cell_classes:
        lp = latent_path(paths, cc)
        if not lp.exists():
            raise FileNotFoundError(f"{lp} missing: run `psychad embed` first (scVI latent for {cc})")
        z = np.load(lp, allow_pickle=False)
        a = ad.read_h5ad(paths.shard(cc), backed="r")
        o = a.obs.copy()
        a.file.close()
        if len(o) != z["Z"].shape[0] or not np.array_equal(o.index.astype(str).values, z["obs_names"].astype(str)):
            raise ValueError(f"{lp} does not match {paths.shard(cc)} (re-run `psychad embed --force`)")
        keep = o["donor_id"].astype(str).isin(donors.index).values
        o, Z = o[keep], z["Z"][keep]
        for c in ("donor_id", "cell_class", "cell_type", "cohort"):
            o[c] = o[c].astype(str)
        pos = cap_per_group(o, cells_per_donor_class, seed)
        xs.append(Z[pos].astype(np.float32))
        obss.append(o.iloc[pos])
    obs = pd.concat(obss)
    obs["y"] = obs["donor_id"].map(donors["task_label"]).map({c: i for i, c in enumerate(classes)}).astype(int)
    X = np.vstack(xs)
    genes = pd.DataFrame({"feature_name": [f"scvi_{i}" for i in range(X.shape[1])]},
                         index=[f"scvi_{i}" for i in range(X.shape[1])])
    donors = donors.loc[donors.index.isin(obs["donor_id"])]
    return CellData(X, obs, genes, donors, classes, list(cell_classes), None)


def donor_labels(cd: CellData) -> pd.Series:
    """Integer task label per donor (index donor_id)."""
    m = {c: i for i, c in enumerate(cd.classes)}
    return cd.donors["task_label"].map(m).astype(int)


def _mean_by_donor(cd: CellData, mask: np.ndarray, gene_idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(n_donors x genes) mean of X over the masked cells of each donor, and the per-donor cell count."""
    donors = list(cd.donors.index)
    d_idx = pd.Categorical(cd.obs["donor_id"], categories=donors).codes
    rows = np.where(mask)[0]
    G = sp.csr_matrix((np.ones(len(rows)), (d_idx[rows], np.arange(len(rows)))), shape=(len(donors), len(rows)))
    n = np.asarray(G.sum(1)).ravel()
    X = cd.X[rows] if gene_idx is None else cd.X[rows][:, gene_idx]
    M = sp.diags(1.0 / np.maximum(n, 1)) @ (G @ X)
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    return M.astype(np.float32), n


def pseudobulk(cd: CellData, by_class: bool, gene_idx: np.ndarray | None = None) -> pd.DataFrame:
    """Mean log-normalised expression per donor (optionally per cell class, columns 'class|gene').

    Works for sparse expression and for a dense latent (scVI). `gene_idx` restricts the columns (gene filter).
    """
    donors = list(cd.donors.index)
    gene_names = cd.genes.index.astype(str).values
    if gene_idx is not None:
        gene_names = gene_names[gene_idx]
    if not by_class:
        M, _ = _mean_by_donor(cd, np.ones(cd.n_cells, bool), gene_idx)
        return pd.DataFrame(M, index=donors, columns=gene_names)
    blocks, cols = [], []
    for cc in cd.cell_classes:
        M, n = _mean_by_donor(cd, (cd.obs["cell_class"] == cc).values, gene_idx)
        present = n > 0
        if present.any() and (~present).any():           # donors lacking this class: fill with the class mean
            M[~present] = M[present].mean(0)
        blocks.append(M)
        cols += [f"{cc}|{g}" for g in gene_names]
    return pd.DataFrame(np.hstack(blocks), index=donors, columns=cols)


def pseudobulk_by_subtype(cd: CellData, gene_idx: np.ndarray | None = None, min_cells: int = 10,
                          min_donor_frac: float = 0.5) -> pd.DataFrame:
    """Pseudobulk per fine cell type (e.g. L2/3 IT, PVALB, SST ...), columns 'subtype|gene'.

    A subtype is used if >= min_donor_frac of donors have >= min_cells of it. A donor below min_cells for a used
    subtype gets that subtype's mean profile (no information either way). Label-free, so no fold logic is needed.
    """
    donors = list(cd.donors.index)
    gene_names = cd.genes.index.astype(str).values
    if gene_idx is not None:
        gene_names = gene_names[gene_idx]
    counts = cd.obs.groupby(["donor_id", "cell_type"]).size().unstack(fill_value=0).reindex(donors, fill_value=0)
    use = [t for t in counts.columns if (counts[t] >= min_cells).mean() >= min_donor_frac]
    blocks, cols = [], []
    for t in sorted(use):
        M, n = _mean_by_donor(cd, (cd.obs["cell_type"] == t).values, gene_idx)
        ok = n >= min_cells
        if ok.any() and (~ok).any():
            M[~ok] = M[ok].mean(0)
        blocks.append(M)
        short = _short_type(t)
        cols += [f"{short}|{g}" for g in gene_names]
    return pd.DataFrame(np.hstack(blocks), index=donors, columns=cols)


def _short_type(t: str) -> str:
    return (t.replace(" glutamatergic", "").replace(" GABAergic", "").replace(" cortical", "").replace(" neuron", "")
            .replace(" projecting", "").replace("intratelencephalic", "IT").replace(" ", "_"))


def expressed_gene_idx(cd: CellData, min_frac: float) -> np.ndarray:
    """Genes detected (count > 0) in >= min_frac of the cells of at least one cell class. Label-free.

    Removes genes like olfactory receptors that are almost never detected and only add noise/batch features.
    """
    if sp.issparse(cd.X):
        keep = np.zeros(cd.X.shape[1], bool)
        for cc in cd.cell_classes:
            rows = np.where((cd.obs["cell_class"] == cc).values)[0]
            if len(rows):
                frac = np.asarray((cd.X[rows] > 0).sum(0)).ravel() / len(rows)
                keep |= frac >= min_frac
        return np.where(keep)[0]
    return np.arange(cd.X.shape[1])          # dense latent: nothing to filter


def composition_features(cd: CellData, composition: pd.DataFrame | None) -> pd.DataFrame:
    """Donor-level cell-type composition, as log fractions (pseudocount 0.5 cells).

    composition: donors x cell_type counts of ALL nuclei (uncapped; `psychad build --stages composition`). Gives
      the fraction of every major class among all nuclei and of every subtype within its class.
    Without it, only within-class subtype fractions are computed from the loaded cells (the per-class cap is a random
      subsample, so these stay unbiased; class fractions do not, so they are skipped).
    """
    donors = list(cd.donors.index)
    feats = []
    if composition is not None:                     # columns '<class>::<cell_type>' (class 'Other' = not used)
        comp = composition.reindex(donors).fillna(0)
        cmap = {c: c.split("::", 1)[0] for c in comp.columns}
        cls = comp.T.groupby(cmap).sum().T                                          # donors x class (incl. Other)
        tot = comp.sum(1)
        cf = np.log((cls + 0.5).div(tot + 0.5 * cls.shape[1], axis=0))
        cf.columns = [f"frac|{c}" for c in cf.columns]
        feats.append(cf)
        sub_src = comp
    else:
        sub_src = cd.obs.groupby(["donor_id", "cell_type"]).size().unstack(fill_value=0).reindex(donors, fill_value=0)
        known = cd.obs.drop_duplicates("cell_type").set_index("cell_type")["cell_class"].to_dict()
        cmap = {t: known[t] for t in sub_src.columns}
    for cc in cd.cell_classes:
        types = [t for t in sub_src.columns if cmap[t] == cc]
        if len(types) < 2:
            continue
        sub = sub_src[types]
        f = np.log((sub + 0.5).div(sub.sum(1) + 0.5 * len(types), axis=0))
        f.columns = [f"sub|{cc}|{_short_type(t.split('::', 1)[-1])}" for t in types]
        feats.append(f)
    out = pd.concat(feats, axis=1) if feats else pd.DataFrame(index=donors)
    return out.astype(np.float32)


def donor_feature_matrix(cd: CellData, spec: dict, composition: pd.DataFrame | None = None) -> pd.DataFrame:
    """Donor x feature matrix for a donor-level model spec.

    spec["features"]: one name or a list of: pseudobulk_all | pseudobulk_by_class | pseudobulk_by_subtype | composition
    spec["gene_filter"]: optional {min_frac: 0.05, n_genes: 2000} (label-free; n_genes = HVG-rank prefix)
    Blocks are concatenated; columns are 'block|...' so block-wise PCA can find them.
    """
    names = spec["features"] if isinstance(spec["features"], list) else [spec["features"]]
    gf = spec.get("gene_filter") or {}
    cache = cd.__dict__.setdefault("_feat_cache", {})
    key_g = (gf.get("min_frac"), gf.get("n_genes"))
    if ("genes", key_g) not in cache:
        idx = np.arange(cd.X.shape[1])
        if gf.get("n_genes"):
            idx = idx[: gf["n_genes"]]
        if gf.get("min_frac"):
            idx = np.intersect1d(idx, expressed_gene_idx(cd, gf["min_frac"]))
        cache[("genes", key_g)] = None if len(idx) == cd.X.shape[1] else idx
    gidx = cache[("genes", key_g)]
    blocks = []
    for n in names:
        key = (n, key_g, spec.get("subtype_min_cells", 10))
        if key not in cache:
            if n == "pseudobulk_all":
                cache[key] = pseudobulk(cd, False, gidx)
            elif n == "pseudobulk_by_class":
                cache[key] = pseudobulk(cd, True, gidx)
            elif n == "pseudobulk_by_subtype":
                cache[key] = pseudobulk_by_subtype(cd, gidx, min_cells=spec.get("subtype_min_cells", 10))
            elif n == "composition":
                cache[key] = composition_features(cd, composition)
            else:
                raise ValueError(f"unknown donor feature block {n}")
        blocks.append(cache[key])
    return pd.concat(blocks, axis=1) if len(blocks) > 1 else blocks[0]


def read_composition(paths: Paths) -> pd.DataFrame | None:
    p = paths.derived / "composition.csv"
    return pd.read_csv(p, index_col="donor_id") if p.exists() else None


def covariate_matrix(donors: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """age (z-scored later by the model pipeline), sex and cohort as 0/1 dummies."""
    parts = []
    for f in features:
        if f in ("age", "pmi"):
            parts.append(donors[[f]].astype(float).fillna(donors[f].median()))
        else:
            parts.append(pd.get_dummies(donors[f].astype(str), prefix=f, drop_first=False).astype(float))
    return pd.concat(parts, axis=1)
