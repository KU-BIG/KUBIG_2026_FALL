"""Phase 1 batch-effect confound analysis (PLAN.md sec 2).

For each modality pair: measure how much of the shared-embedding variance
is explained by batch vs by modality (variance partitioning), compare gap
metrics with/without Harmony batch correction, sanity-check that Harmony
doesn't erase cell-type structure (over-correction check), and repeat the
baseline gap measurement on N-matched subsamples of both datasets.
"""
from __future__ import annotations

import json
import time

import harmonypy
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.data.loading import held_out_split, load_bmmc, split_modalities
from src.data.preprocessing import (
    clr_normalize_adt,
    normalize_gex,
    pca_reduce,
    select_hvgs,
    LSITransformer,
)
from src.encoders.linear_baseline import LinearCCAEncoder
from src.metrics.gap_metrics import gap_report
from src.metrics.variance_partitioning import group_r2, permutation_test_r2

N_TOP_GENES = 2000
GEX_PCA_N_COMPONENTS = 100
CCA_N_COMPONENTS = 32
ATAC_LSI_N_COMPONENTS = 128
TEST_FRAC = 0.2
SEED = 0
SILHOUETTE_MAX_N = 5000


def _harmony_correct(x: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
    """Harmony batch correction. NOTE: this harmonypy version (2.0.0, C++
    backend) returns Z_corr in the *same* orientation as the input
    (cells x features) — verified empirically, since it differs from the
    classic pure-python harmonypy's transposed convention that other code
    (e.g. MatchCLOT's own harmony() wrapper) assumes. Do not transpose here.
    """
    meta = pd.DataFrame({"batch": batch_labels})
    ho = harmonypy.run_harmony(x, meta, vars_use=["batch"], verbose=False)
    return np.asarray(ho.Z_corr).astype(np.float32)


def _preprocess_pair(pair: str, other_modality: str, cell_idx: np.ndarray | None, seed: int):
    """Loads + preprocesses one modality pair, optionally restricted to
    `cell_idx` (used by the matched-N condition). Returns train/test raw
    (pre-harmony, pre-CCA) modality-specific arrays plus batch/cell_type
    labels, so callers can apply harmony (or not) before fitting CCA."""
    adata = load_bmmc(pair)
    if cell_idx is not None:
        adata = adata[cell_idx].copy()
    gex, other = split_modalities(adata, mod1="GEX", mod2=other_modality)
    train_idx, test_idx = held_out_split(adata, test_frac=TEST_FRAC, seed=seed)

    hvg_names = select_hvgs(gex[train_idx], n_top_genes=N_TOP_GENES, seed=seed)
    gex_train = normalize_gex(gex[train_idx], gene_subset=hvg_names)
    gex_test = normalize_gex(gex[test_idx], gene_subset=hvg_names)
    gex_train, gex_test = pca_reduce(gex_train, gex_test, n_components=GEX_PCA_N_COMPONENTS, seed=seed)

    if other_modality == "ADT":
        other_train = clr_normalize_adt(other[train_idx])
        other_test = clr_normalize_adt(other[test_idx])
    else:
        lsi = LSITransformer(n_components=ATAC_LSI_N_COMPONENTS)
        other_train = lsi.fit_transform(other[train_idx].layers["counts"])
        other_test = lsi.transform(other[test_idx].layers["counts"])

    batch_train = adata.obs["batch"].values[train_idx]
    batch_test = adata.obs["batch"].values[test_idx]
    celltype_test = adata.obs["cell_type"].values[test_idx]

    return dict(
        gex_train=gex_train, gex_test=gex_test,
        other_train=other_train, other_test=other_test,
        batch_train=batch_train, batch_test=batch_test,
        celltype_test=celltype_test,
        n_total=adata.shape[0],
    )


def _fit_and_report(d: dict, use_harmony: bool) -> dict:
    gex_train, gex_test = d["gex_train"], d["gex_test"]
    other_train, other_test = d["other_train"], d["other_test"]
    if use_harmony:
        gex_train = _harmony_correct(gex_train, d["batch_train"])
        gex_test = _harmony_correct(gex_test, d["batch_test"])
        other_train = _harmony_correct(other_train, d["batch_train"])
        other_test = _harmony_correct(other_test, d["batch_test"])

    encoder = LinearCCAEncoder(n_components=CCA_N_COMPONENTS)
    emb_gex, emb_other = encoder.fit_transform(gex_train, other_train, gex_test, other_test)
    report = gap_report(emb_gex, emb_other, paired=True)

    # variance partitioning: pool both modalities' embeddings for the same
    # held-out cells, label each row by modality and by batch
    pooled = np.concatenate([emb_gex, emb_other], axis=0)
    modality_labels = np.array(["gex"] * len(emb_gex) + ["other"] * len(emb_other))
    batch_labels = np.concatenate([d["batch_test"], d["batch_test"]])
    r2_modality = group_r2(pooled, modality_labels)
    r2_batch, p_batch = permutation_test_r2(pooled, batch_labels, n_perm=199, seed=SEED)
    report.update({
        "r2_modality": r2_modality,
        "r2_batch": r2_batch,
        "r2_batch_pvalue": p_batch,
    })

    # over-correction sanity check: does cell-type structure survive in the
    # (post-harmony if used) GEX feature space, pre-CCA?
    n_sil = min(SILHOUETTE_MAX_N, gex_test.shape[0])
    rng = np.random.default_rng(SEED)
    sil_idx = rng.choice(gex_test.shape[0], size=n_sil, replace=False)
    labels = d["celltype_test"][sil_idx]
    # silhouette needs >=2 samples per label and >=2 labels
    vc = pd.Series(labels).value_counts()
    keep = np.isin(labels, vc[vc >= 2].index)
    if keep.sum() > 1 and pd.Series(labels[keep]).nunique() > 1:
        report["celltype_silhouette"] = float(
            silhouette_score(gex_test[sil_idx][keep], labels[keep])
        )
    else:
        report["celltype_silhouette"] = float("nan")

    return report


def run_pair(pair: str, other_modality: str, cell_idx: np.ndarray | None = None, tag: str = "") -> list[dict]:
    t0 = time.time()
    print(f"[{pair}{tag}] preprocessing...")
    d = _preprocess_pair(pair, other_modality, cell_idx, seed=SEED)
    print(f"[{pair}{tag}] preprocessed in {time.time()-t0:.1f}s (n_total={d['n_total']})")

    rows = []
    for use_harmony in (False, True):
        label = "harmony_on" if use_harmony else "harmony_off"
        print(f"[{pair}{tag}] fitting ({label})...")
        report = _fit_and_report(d, use_harmony=use_harmony)
        report.update({"pair": pair, "other_modality": other_modality, "harmony": use_harmony, "condition": tag or "full"})
        rows.append(report)
        print(f"[{pair}{tag}] {label}: delta_gap={report['delta_gap']:.4f} "
              f"r2_batch={report['r2_batch']:.4f} (p={report['r2_batch_pvalue']:.4f}) "
              f"r2_modality={report['r2_modality']:.4f} "
              f"celltype_silhouette={report['celltype_silhouette']:.4f} "
              f"({time.time()-t0:.1f}s elapsed)")
    return rows


def main():
    all_rows = []
    # full-data conditions (batch variance partitioning + harmony on/off)
    all_rows += run_pair("cite", "ADT")
    all_rows += run_pair("multiome", "ATAC")

    # matched-N condition: subsample both datasets to the same N cells
    adata_cite = load_bmmc("cite", backed="r")
    adata_multi = load_bmmc("multiome", backed="r")
    n_match = min(adata_cite.shape[0], adata_multi.shape[0])
    rng = np.random.default_rng(SEED)
    idx_cite = np.sort(rng.choice(adata_cite.shape[0], size=n_match, replace=False))
    idx_multi = np.sort(rng.choice(adata_multi.shape[0], size=n_match, replace=False))
    print(f"matched-N condition: n={n_match}")
    all_rows += run_pair("cite", "ADT", cell_idx=idx_cite, tag="_matchedN")
    all_rows += run_pair("multiome", "ATAC", cell_idx=idx_multi, tag="_matchedN")

    df = pd.DataFrame(all_rows)
    out_path = "results/tables/phase1_batch_confound.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
