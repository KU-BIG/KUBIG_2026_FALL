"""Phase 1 PCA-2D embedding visualization (presentation asset, not part of
PLAN.md's numeric deliverables). Regenerates the SemArt-style "two clouds"
scatter plot for this project's modality pairs: PCA(2) of the held-out test
embeddings from gap_report()'s inputs, colored by modality (GEX vs ADT/ATAC)
instead of SemArt's (image vs visual_text vs contextual_text).

Reuses the exact preprocessing/encoder pipeline from phase1_baseline.py and
phase1_baseline_matchclot.py so the plotted embeddings match the numbers
already reported in docs/HISTORY.md (linear CCA: full data, seed 0;
MatchCLOT-arch: 1 seed only here, for speed -- numeric tables elsewhere use
the 3-seed mean).

Low-RAM-session note: anndata's backed="r" mode only backs `.X`; sparse
`.layers` (e.g. `layers["counts"]`, which every preprocessing function here
actually needs) are read fully into memory on open regardless of backed
mode. For multiome that is ~6GB just for opening the file, before any
subsetting -- fine on the 216GB machine PLAN.md's numeric results were
produced on, but it OOMs a ~14GB no-GPU session outright. `_load_multiome_
counts_subset` below bypasses anndata for multiome and reads only the
required row ranges directly from the on-disk CSR arrays via h5py, so the
plot can still be regenerated (on a cell subsample) without that 6GB floor.
cite doesn't need this (ADT is tiny; full backed load there is ~1.3GB).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

import anndata as ad
from src.data.loading import held_out_split, load_bmmc, split_modalities
from src.data.preprocessing import (
    clr_normalize_adt,
    normalize_gex,
    pca_reduce,
    select_hvgs,
    LSITransformer,
)
from src.encoders.linear_baseline import LinearCCAEncoder
from src.encoders.matchclot_arch import encode, train_modality_clip

N_TOP_GENES = 2000
GEX_PCA_N_COMPONENTS = 100
CCA_N_COMPONENTS = 32
ATAC_LSI_N_COMPONENTS = 128
TEST_FRAC = 0.2
SEED = 0
MATCHCLOT_HPARAMS = dict(n_epochs=150, embedding_dim=64, layers_dim_mod1=(512, 256), layers_dim_mod2=(512, 256))
# Visualization-only cap: full multiome (69,249 cells x 116,490 ATAC peaks)
# is what PLAN.md's numeric results use; this cap only shrinks the *plot*,
# read directly from disk in a handful of evenly-spaced contiguous chunks
# (not a uniform random sample of individual rows -- that would need
# per-row h5py reads, far slower) so it still spans the whole dataset
# rather than one arbitrary contiguous block.
MAX_CELLS_FOR_PLOT = 15000
N_CHUNKS = 5

MULTIOME_H5AD_PATH = "data/raw/multiome_BMMC_processed.h5ad"

PAIRS = [("cite", "ADT"), ("multiome", "ATAC")]


def _decode(arr) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def _read_csr_row_range(h5_group, row_start: int, row_end: int) -> sp.csr_matrix:
    indptr_full = h5_group["indptr"][row_start:row_end + 1].astype(np.int64)
    s, e = int(indptr_full[0]), int(indptr_full[-1])
    data = h5_group["data"][s:e]
    indices = h5_group["indices"][s:e]
    indptr = indptr_full - s
    n_cols = int(h5_group.attrs["shape"][1])
    return sp.csr_matrix((data, indices, indptr), shape=(row_end - row_start, n_cols))


def _load_multiome_counts_subset(n_cells: int, seed: int, n_chunks: int = N_CHUNKS):
    """Bypass anndata entirely: read var/feature_types + a handful of
    contiguous row ranges of layers/counts straight from the h5ad's HDF5
    arrays. Returns (gex_counts, atac_counts, gex_var_names) as in-memory
    scipy.sparse matrices -- never materializes the full 69,249-row layer.
    """
    import h5py

    with h5py.File(MULTIOME_H5AD_PATH, "r") as f:
        codes = f["var/feature_types"][:]
        categories = _decode(f["var/__categories/feature_types"][:])
        feature_types = categories[codes]
        var_names = _decode(f["var/_index"][:])
        gex_mask = feature_types == "GEX"
        atac_mask = feature_types == "ATAC"

        counts_grp = f["layers/counts"]
        n_total = int(counts_grp.attrs["shape"][0])
        chunk_size = n_cells // n_chunks
        starts = np.linspace(0, n_total - chunk_size, n_chunks, dtype=int)
        chunks = [_read_csr_row_range(counts_grp, int(s), int(s) + chunk_size) for s in starts]
        counts = sp.vstack(chunks).tocsc()

    gex_counts = counts[:, gex_mask].tocsr()
    atac_counts = counts[:, atac_mask].tocsr()
    return gex_counts, atac_counts, var_names[gex_mask]


def _preprocess_multiome_subset(seed: int = SEED):
    gex_counts, atac_counts, gex_names = _load_multiome_counts_subset(MAX_CELLS_FOR_PLOT, seed)
    n = gex_counts.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * TEST_FRAC))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    gex_adata = ad.AnnData(X=gex_counts, layers={"counts": gex_counts}, var=pd.DataFrame(index=gex_names))
    hvg_names = select_hvgs(gex_adata[train_idx], n_top_genes=N_TOP_GENES, seed=seed)
    gex_train = normalize_gex(gex_adata[train_idx], gene_subset=hvg_names)
    gex_test = normalize_gex(gex_adata[test_idx], gene_subset=hvg_names)

    lsi = LSITransformer(n_components=ATAC_LSI_N_COMPONENTS)
    other_train = lsi.fit_transform(atac_counts[train_idx])
    other_test = lsi.transform(atac_counts[test_idx])
    return gex_train, gex_test, other_train, other_test


def _preprocess(pair: str, other_modality: str, seed: int = SEED):
    if pair == "multiome":
        return _preprocess_multiome_subset(seed)

    adata = load_bmmc(pair)
    gex, other = split_modalities(adata, mod1="GEX", mod2=other_modality)
    train_idx, test_idx = held_out_split(adata, test_frac=TEST_FRAC, seed=seed)

    hvg_names = select_hvgs(gex[train_idx], n_top_genes=N_TOP_GENES, seed=seed)
    gex_train = normalize_gex(gex[train_idx], gene_subset=hvg_names)
    gex_test = normalize_gex(gex[test_idx], gene_subset=hvg_names)

    if other_modality == "ADT":
        other_train = clr_normalize_adt(other[train_idx])
        other_test = clr_normalize_adt(other[test_idx])
    else:
        lsi = LSITransformer(n_components=ATAC_LSI_N_COMPONENTS)
        other_train = lsi.fit_transform(other[train_idx].layers["counts"])
        other_test = lsi.transform(other[test_idx].layers["counts"])

    return gex_train, gex_test, other_train, other_test


def _linear_cca_embeddings(pair: str, other_modality: str):
    gex_train, gex_test, other_train, other_test = _preprocess(pair, other_modality)
    gex_train_pca, gex_test_pca = pca_reduce(gex_train, gex_test, n_components=GEX_PCA_N_COMPONENTS, seed=SEED)
    encoder = LinearCCAEncoder(n_components=CCA_N_COMPONENTS)
    emb_gex, emb_other = encoder.fit_transform(gex_train_pca, other_train, gex_test_pca, other_test)
    return emb_gex, emb_other


def _matchclot_arch_embeddings(pair: str, other_modality: str):
    gex_train, gex_test, other_train, other_test = _preprocess(pair, other_modality)
    model = train_modality_clip(gex_train, other_train, hparams=MATCHCLOT_HPARAMS, seed=SEED, verbose_every=0)
    emb_gex, emb_other = encode(model, gex_test, other_test)
    return emb_gex, emb_other


def _plot_panel(ax, emb_gex: np.ndarray, emb_other: np.ndarray, other_label: str, title: str):
    pooled = np.concatenate([emb_gex, emb_other], axis=0)
    coords = PCA(n_components=2, random_state=SEED).fit_transform(pooled)
    n_gex = emb_gex.shape[0]
    ax.scatter(coords[:n_gex, 0], coords[:n_gex, 1], s=4, alpha=0.35, label="GEX", color="#3b82f6")
    ax.scatter(coords[n_gex:, 0], coords[n_gex:, 1], s=4, alpha=0.35, label=other_label, color="#f97316")
    # centroids, to make the delta_gap visual explicit
    c_gex = coords[:n_gex].mean(axis=0)
    c_other = coords[n_gex:].mean(axis=0)
    ax.scatter(*c_gex, s=140, marker="X", color="#1d4ed8", edgecolor="white", linewidth=1.2, zorder=5)
    ax.scatter(*c_other, s=140, marker="X", color="#c2410c", edgecolor="white", linewidth=1.2, zorder=5)
    ax.plot([c_gex[0], c_other[0]], [c_gex[1], c_other[1]], color="black", linewidth=1, linestyle="--", zorder=4)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, markerscale=2)


def main(include_matchclot: bool = True):
    """include_matchclot=False: linear-CCA-only 1x2 figure. Use this when no
    GPU is available in the current session -- MatchCLOT-arch training is a
    torch MLP loop that is fast on the A100 this project normally runs on,
    but becomes impractically slow on CPU-only sessions."""
    ncols = 2 if include_matchclot else 1
    fig, axes = plt.subplots(2, ncols, figsize=(5.5 * ncols, 10), squeeze=False)
    for row, (pair, other_modality) in enumerate(PAIRS):
        print(f"[{pair}] linear CCA embeddings...")
        emb_gex_cca, emb_other_cca = _linear_cca_embeddings(pair, other_modality)
        _plot_panel(
            axes[row, 0], emb_gex_cca, emb_other_cca, other_modality,
            f"GEX-{other_modality} ({pair}) -- linear CCA",
        )
        if include_matchclot:
            print(f"[{pair}] MatchCLOT-arch embeddings...")
            emb_gex_mc, emb_other_mc = _matchclot_arch_embeddings(pair, other_modality)
            _plot_panel(
                axes[row, 1], emb_gex_mc, emb_other_mc, other_modality,
                f"GEX-{other_modality} ({pair}) -- MatchCLOT-arch",
            )

    fig.suptitle("Phase 1 shared-embedding space, held-out test cells (PCA to 2D)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    suffix = "" if include_matchclot else "_linear_cca_only"
    out_path = f"results/figures/phase1_pca_2d{suffix}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import sys
    import torch
    main(include_matchclot="--linear-only" not in sys.argv and torch.cuda.is_available())
