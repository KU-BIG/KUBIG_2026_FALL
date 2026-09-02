"""Unit tests for the quality-axis gene-selection helpers in
src/experiments/phase2_expA_dial_swipe.py, on tiny synthetic AnnData —
independent of the real BMMC data, per this project's testing discipline
(docs/PLAN.md Phase 0 checklist)."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from src.experiments.phase2_expA_dial_swipe import _adt_matched_genes, _stat_matched_random_genes


def _make_gex(seed=0, n_cells=200, n_genes=50):
    rng = np.random.default_rng(seed)
    gene_ids = [f"ENSG{i:05d}" for i in range(n_genes)]
    # make expression means span a wide range so deciles are meaningful
    means = np.linspace(0.1, 100, n_genes)
    counts = rng.poisson(lam=means, size=(n_cells, n_genes)).astype(np.float32)
    var = pd.DataFrame({"gene_id": gene_ids}, index=[f"gene{i}" for i in range(n_genes)])
    a = ad.AnnData(X=scipy.sparse.csr_matrix(counts), var=var)
    a.layers["counts"] = a.X.copy()
    return a


def _make_adt(gene_ids_subset):
    n = len(gene_ids_subset)
    var = pd.DataFrame({"gene_id": gene_ids_subset}, index=[f"adt{i}" for i in range(n)])
    x = np.zeros((10, n), dtype=np.float32)
    return ad.AnnData(X=x, var=var)


def test_adt_matched_genes_finds_intersection():
    gex = _make_gex()
    # ADT panel targets genes 0, 5, 10 plus one gene_id not present in GEX at all
    adt = _make_adt(["ENSG00000", "ENSG00005", "ENSG00010", "ENSG99999"])
    matched = _adt_matched_genes(gex, adt)
    assert set(matched) == {"gene0", "gene5", "gene10"}


def test_adt_matched_genes_drops_nan_gene_id():
    gex = _make_gex()
    adt = _make_adt(["ENSG00000", None, "ENSG00010"])
    matched = _adt_matched_genes(gex, adt)
    assert set(matched) == {"gene0", "gene10"}


def test_stat_matched_random_genes_excludes_targets():
    gex = _make_gex(n_genes=100)
    target_genes = [f"gene{i}" for i in range(0, 20, 2)]  # 10 genes
    matched = _stat_matched_random_genes(gex, target_genes, seed=0)
    assert len(set(matched) & set(target_genes)) == 0
    assert len(matched) <= len(target_genes)


def test_stat_matched_random_genes_accepts_set_input():
    """Regression test: pandas .loc rejects bare sets as an indexer, and the
    quality-axis caller passes a set() of ADT-matched genes."""
    gex = _make_gex(n_genes=100)
    target_genes = set(f"gene{i}" for i in range(0, 20, 2))
    matched = _stat_matched_random_genes(gex, target_genes, seed=0)  # must not raise
    assert isinstance(matched, list)


def test_stat_matched_random_genes_no_duplicates():
    gex = _make_gex(n_genes=100)
    target_genes = [f"gene{i}" for i in range(0, 30, 2)]
    matched = _stat_matched_random_genes(gex, target_genes, seed=1)
    assert len(matched) == len(set(matched))
