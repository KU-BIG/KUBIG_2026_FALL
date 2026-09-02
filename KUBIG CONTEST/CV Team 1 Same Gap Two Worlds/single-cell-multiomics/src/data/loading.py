"""Loading & splitting for the GSE194122 BMMC processed h5ad files (PLAN.md Phase 1).

We use GEO's combined processed AnnData directly (one file per assay pair,
GEX+ADT and GEX+ATAC, each modality living in `var['feature_types']`) rather
than the competition's own train/test split, so that we control the
held-out split ourselves (see docs/HISTORY.md, 2026-08-13, decision 3).
"""
from __future__ import annotations

import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split

RAW_PATHS = {
    "cite": "data/raw/cite_BMMC_processed.h5ad",   # GEX + ADT, 90,261 cells
    "multiome": "data/raw/multiome_BMMC_processed.h5ad",  # GEX + ATAC, 69,249 cells
}


def load_bmmc(pair: str, backed: str | None = None) -> ad.AnnData:
    if pair not in RAW_PATHS:
        raise ValueError(f"pair must be one of {list(RAW_PATHS)}, got {pair!r}")
    adata = ad.read_h5ad(RAW_PATHS[pair], backed=backed)
    adata.var_names_make_unique()
    return adata


def split_modalities(adata: ad.AnnData, mod1: str, mod2: str) -> tuple[ad.AnnData, ad.AnnData]:
    """Split a combined AnnData into two single-modality AnnDatas using
    var['feature_types']. Row order (cells) is preserved and identical
    across both, so row index i is the same cell in both outputs.
    """
    types = adata.var["feature_types"]
    a1 = adata[:, types == mod1].copy()
    a2 = adata[:, types == mod2].copy()
    return a1, a2


def held_out_split(
    adata: ad.AnnData,
    test_frac: float = 0.2,
    seed: int = 0,
    stratify_col: str = "batch",
):
    """Cell-level held-out split, stratified by batch so every batch's
    proportion is preserved in both splits (PLAN.md Phase 1: "held-out cell
    기준 gap 측정", avoiding the leakage risk of measuring gap on cells the
    encoder already saw).
    """
    n = adata.shape[0]
    strata = adata.obs[stratify_col].values if stratify_col in adata.obs else None
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=strata
    )
    return train_idx, test_idx
