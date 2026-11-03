"""Independent check on the classifier: pseudobulk differential expression (PyDESeq2) + pathway enrichment.

Counts are summed over each donor's (capped) cells of one cell class, over the stored gene panel. Model per cell
class: ~ covariates + disease label, one contrast (disease vs Control) per disease. NOTE: DE is restricted to the
HVG panel (not all genes); re-run `psychad build` with a larger n_hvg if a genome-wide DE is needed.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from .cv import Ctx


def run_de(ctx: Ctx, task: str = "four_class", enrich: bool = True) -> pd.DataFrame:
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as e:
        raise ImportError("pip install pydeseq2 gseapy") from e
    cfg = ctx.cfg
    tmap = cfg["tasks"][task]
    pb = ad.read_h5ad(ctx.paths.pseudobulk)
    obs = pb.obs.copy()
    obs["group"] = obs["label"].astype(str).map(tmap)
    obs = obs[obs["group"].notna() & (obs["n_cells"] >= cfg["build"]["min_cells_per_donor_class"])]
    counts_all = pd.DataFrame(np.asarray(pb.X.todense()) if hasattr(pb.X, "todense") else pb.X,
                              index=pb.obs_names, columns=pb.var["feature_name"].astype(str).values)
    covs = [c for c in cfg["de"]["covariates"]]
    out = ctx.paths.out("de", task)
    all_res = []
    for cc, o in obs.groupby("cell_class"):
        for dis in [g for g in o["group"].unique() if g != "Control"]:
            m = o[o["group"].isin(["Control", dis])].copy()
            n_by = m["group"].astype(str).value_counts()
            if len(n_by) < 2 or n_by.min() < cfg["de"]["min_donors_per_group"]:
                ctx.log(f"DE skip {cc} {dis}: groups {n_by.to_dict()}")
                continue
            meta = pd.DataFrame(index=m.index)
            meta["group"] = pd.Categorical(m["group"], categories=["Control", dis])
            if "age" in covs:
                meta["age_z"] = ((m["age"] - m["age"].mean()) / (m["age"].std() or 1)).values
            if "sex" in covs:
                meta["sex"] = m["sex"].astype(str).values
            if "cohort" in covs and m["cohort"].nunique() > 1:
                meta["cohort"] = m["cohort"].astype(str).values
            terms = [c for c in meta.columns if c != "group"] + ["group"]
            cnt = counts_all.loc[m.index].round().astype(int)
            cnt = cnt.loc[:, cnt.sum(0) >= 10]
            cnt = cnt.loc[:, ~cnt.columns.duplicated()]
            dds = DeseqDataSet(counts=cnt, metadata=meta, design="~" + " + ".join(terms), quiet=True)
            dds.deseq2()
            st = DeseqStats(dds, contrast=["group", dis, "Control"], quiet=True)
            st.summary()
            r = st.results_df.reset_index().rename(columns={"index": "gene"})
            r.insert(0, "disease", dis), r.insert(0, "cell_class", cc)
            r["n_disease"], r["n_control"] = int(n_by[dis]), int(n_by["Control"])
            all_res.append(r)
            ctx.log(f"DE {cc} {dis}: {int((r['padj'] < cfg['de']['padj']).sum())} genes at padj<{cfg['de']['padj']}")
    res = pd.concat(all_res, ignore_index=True) if all_res else pd.DataFrame(columns=["cell_class", "disease", "gene"])
    res.to_csv(out / "de_results.csv", index=False)
    if enrich and len(res):
        run_enrichment(ctx, res, out)
    return res


def run_enrichment(ctx: Ctx, res: pd.DataFrame, out) -> pd.DataFrame:
    """Enrichr over-representation for up / down DE genes per (cell class, disease). Needs internet access."""
    try:
        import gseapy as gp
    except ImportError:
        ctx.log("gseapy missing; skipping enrichment")
        return pd.DataFrame()
    padj = ctx.cfg["de"]["padj"]
    background = list(res["gene"].unique())
    rows = []
    for (cc, dis), r in res.groupby(["cell_class", "disease"]):
        for direction, sel in (("up", r[(r["padj"] < padj) & (r["log2FoldChange"] > 0)]),
                               ("down", r[(r["padj"] < padj) & (r["log2FoldChange"] < 0)])):
            if len(sel) < 5:
                continue
            try:
                e = gp.enrichr(gene_list=list(sel["gene"]), gene_sets=ctx.cfg["de"]["gene_sets"],
                               background=background, outdir=None, no_plot=True).results
            except Exception as ex:  # noqa: BLE001  (network / service errors)
                ctx.log(f"enrichr failed for {cc} {dis} {direction}: {ex}")
                continue
            e.insert(0, "direction", direction), e.insert(0, "disease", dis), e.insert(0, "cell_class", cc)
            rows.append(e)
    enr = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    enr.to_csv(out / "enrichment.csv", index=False)
    return enr


def compare_de_to_model(ctx: Ctx, task: str = "four_class", top_k: int | None = None) -> pd.DataFrame:
    """Overlap between top-|coef| stable LR genes and DE genes (hypergeometric p-value), per cell class x disease."""
    from scipy.stats import hypergeom

    top_k = top_k or ctx.cfg["importance"]["top_k"]
    de = pd.read_csv(ctx.paths.results / "de" / task / "de_results.csv")
    if de.empty:
        return pd.DataFrame()
    imp = pd.read_csv(ctx.paths.results / "importance" / task / "lr_importance.csv")
    rows = []
    for (cc, dis), r in de.groupby(["cell_class", "disease"]):
        i = imp[(imp["cell_class"] == cc) & (imp["class"] == dis)]
        if i.empty:
            continue
        universe = set(r["gene"]) & set(i["gene"])
        model_top = set(i.loc[i["gene"].isin(universe)].assign(a=lambda d: d["mean_coef"].abs()).nlargest(top_k, "a")["gene"])
        de_top = set(r[r["gene"].isin(universe)].nsmallest(top_k, "pvalue")["gene"])
        ov = len(model_top & de_top)
        p = hypergeom.sf(ov - 1, len(universe), len(de_top), len(model_top))
        rows.append({"cell_class": cc, "disease": dis, "top_k": top_k, "overlap": ov, "universe": len(universe),
                     "hypergeom_p": p})
    res = pd.DataFrame(rows)
    res.to_csv(ctx.paths.out("de", task) / "de_vs_model_overlap.csv", index=False)
    return res
