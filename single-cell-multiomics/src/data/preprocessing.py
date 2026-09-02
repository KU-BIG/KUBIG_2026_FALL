"""Per-modality preprocessing (PLAN.md Phase 1 / Phase 2 exp A quantity axis).

GEX: standard scanpy normalize_total + log1p + HVG selection (n_top_genes is
the exact knob the "quantity axis" dial-swipe in exp A turns).
ADT: CLR (centered log-ratio), the standard CITE-seq antibody normalization.
ATAC: TF-IDF + LSI, vendored from AI4SCR/MatchCLOT (BSD-3-Clause, see
external/MatchCLOT/LICENSE) since it is the exact transform MatchCLOT's own
encoder is trained on and is dependency-light (anndata + sklearn only, no
catalyst) — see docs/HISTORY.md 2026-08-13 decision 2.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import sklearn.decomposition
import sklearn.preprocessing


def pca_reduce(x_train: np.ndarray, x_test: np.ndarray, n_components: int = 100, seed: int = 0):
    """PCA fit on train, applied to both train/test. Used to pre-reduce
    high-dimensional GEX (thousands of HVGs) before CCA — fitting CCA
    directly on a several-thousand-column matrix is both slow (NIPALS
    iterates over the full feature space) and not how CCA is used in
    practice for single-cell integration (e.g. Seurat's CCA runs on PCA
    components, not raw features). Low-dimensional inputs (ADT, ATAC-LSI)
    skip this step entirely.
    """
    pca = sklearn.decomposition.PCA(n_components=n_components, random_state=seed)
    train_reduced = pca.fit_transform(x_train)
    test_reduced = pca.transform(x_test)
    return train_reduced.astype(np.float32), test_reduced.astype(np.float32)


def select_hvgs(adata_gex: ad.AnnData, n_top_genes: int | None, seed: int = 0) -> list[str]:
    """Return the names of the top `n_top_genes` highly variable genes,
    computed on raw counts with scanpy's seurat_v3 flavor (recommended for
    count data). `n_top_genes=None` returns every gene (the "전체" condition
    in the quantity dial-swipe).
    """
    if n_top_genes is None or n_top_genes >= adata_gex.shape[1]:
        return list(adata_gex.var_names)
    counts = adata_gex.layers["counts"] if "counts" in adata_gex.layers else adata_gex.X
    tmp = ad.AnnData(X=counts.copy(), var=adata_gex.var.copy())
    sc.pp.highly_variable_genes(
        tmp, n_top_genes=n_top_genes, flavor="seurat_v3", subset=False
    )
    hvg_names = tmp.var_names[tmp.var["highly_variable"]].tolist()
    return hvg_names


def normalize_gex(adata_gex: ad.AnnData, gene_subset: list[str] | None = None) -> np.ndarray:
    """normalize_total -> log1p, restricted to `gene_subset` if given.
    Returns a dense (n_cells, n_genes) float32 array.
    """
    a = adata_gex[:, gene_subset].copy() if gene_subset is not None else adata_gex.copy()
    a.X = a.layers["counts"].copy() if "counts" in a.layers else a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    x = a.X.toarray() if scipy.sparse.issparse(a.X) else np.asarray(a.X)
    return x.astype(np.float32)


def clr_normalize_adt(adata_adt: ad.AnnData) -> np.ndarray:
    """Centered log-ratio normalization per feature (across cells), the
    standard CITE-seq ADT normalization (Seurat's default CLR margin=2).
    """
    counts = adata_adt.layers["counts"] if "counts" in adata_adt.layers else adata_adt.X
    x = counts.toarray() if scipy.sparse.issparse(counts) else np.asarray(counts)
    x = x.astype(np.float64)
    log1p_x = np.log1p(x)
    geometric_mean_log = log1p_x.mean(axis=0, keepdims=True)  # per-feature, across cells
    clr = log1p_x - geometric_mean_log
    return clr.astype(np.float32)


class LSITransformer:
    """TF-IDF + truncated-SVD LSI for ATAC peaks.

    Vendored (simplified, no highly-variable gating) from
    AI4SCR/MatchCLOT matchclot/preprocessing/preprocess.py::lsiTransformer.
    """

    def __init__(self, n_components: int = 256, drop_first: bool = True, seed: int = 777):
        self.n_components = n_components + int(drop_first)
        self.drop_first = drop_first
        self.seed = seed
        self._idf = None
        self._svd = sklearn.decomposition.TruncatedSVD(
            n_components=self.n_components, random_state=seed
        )
        self._fitted = False

    @staticmethod
    def _tfidf(x: np.ndarray | scipy.sparse.spmatrix):
        idf = x.shape[0] / np.asarray(x.sum(axis=0)).ravel()
        if scipy.sparse.issparse(x):
            tf = x.multiply(1 / x.sum(axis=1))
            return tf.multiply(idf)
        return (x / x.sum(axis=1, keepdims=True)) * idf

    def fit_transform(self, counts: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        self._idf = counts.shape[0] / np.asarray(counts.sum(axis=0)).ravel()
        x_tfidf = self._tfidf(counts)
        x_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(x_tfidf)
        x_norm = np.log1p(x_norm * 1e4)
        x_lsi = self._svd.fit_transform(x_norm)
        x_lsi = self._standardize(x_lsi)
        self._fitted = True
        return x_lsi[:, int(self.drop_first):].astype(np.float32)

    def transform(self, counts: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LSITransformer must be fit_transform()'d before transform()")
        if scipy.sparse.issparse(counts):
            tf = counts.multiply(1 / counts.sum(axis=1))
            x_tfidf = tf.multiply(self._idf)
        else:
            x_tfidf = (counts / counts.sum(axis=1, keepdims=True)) * self._idf
        x_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(x_tfidf)
        x_norm = np.log1p(x_norm * 1e4)
        x_lsi = self._svd.transform(x_norm)
        x_lsi = self._standardize(x_lsi)
        return x_lsi[:, int(self.drop_first):].astype(np.float32)

    @staticmethod
    def _standardize(x_lsi: np.ndarray) -> np.ndarray:
        x_lsi = x_lsi - x_lsi.mean(axis=1, keepdims=True)
        x_lsi = x_lsi / x_lsi.std(axis=1, ddof=1, keepdims=True)
        return x_lsi


def get_counts(adata: ad.AnnData) -> scipy.sparse.spmatrix:
    return adata.layers["counts"] if "counts" in adata.layers else adata.X
