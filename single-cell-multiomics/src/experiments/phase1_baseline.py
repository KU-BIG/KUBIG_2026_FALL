"""Phase 1 baseline modality-gap measurement (PLAN.md sec 1).

Runs the linear-CCA encoder (encoder (c) in PLAN.md's 3-encoder comparison;
encoder (a) pretrained-MatchCLOT is unavailable — see docs/HISTORY.md
2026-08-13 decision 1 — encoder (b) from-scratch MatchCLOT-architecture is
implemented separately in src/encoders/matchclot_arch.py) on both modality
pairs, on a held-out test split, and reports the full gap metric suite.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

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

N_TOP_GENES_DEFAULT = 2000
GEX_PCA_N_COMPONENTS = 100  # pre-reduce GEX before CCA; see preprocessing.pca_reduce docstring
CCA_N_COMPONENTS = 32
ATAC_LSI_N_COMPONENTS = 128
TEST_FRAC = 0.2
SEED = 0


def run_pair(pair: str, other_modality: str, seed: int = SEED) -> dict:
    t0 = time.time()
    print(f"[{pair}] loading...")
    adata = load_bmmc(pair)
    gex, other = split_modalities(adata, mod1="GEX", mod2=other_modality)
    train_idx, test_idx = held_out_split(adata, test_frac=TEST_FRAC, seed=seed)
    print(f"[{pair}] loaded {adata.shape[0]} cells in {time.time()-t0:.1f}s; "
          f"train={len(train_idx)} test={len(test_idx)}")

    hvg_names = select_hvgs(gex[train_idx], n_top_genes=N_TOP_GENES_DEFAULT, seed=seed)
    gex_train_x = normalize_gex(gex[train_idx], gene_subset=hvg_names)
    gex_test_x = normalize_gex(gex[test_idx], gene_subset=hvg_names)
    gex_train_x, gex_test_x = pca_reduce(gex_train_x, gex_test_x, n_components=GEX_PCA_N_COMPONENTS, seed=seed)
    print(f"[{pair}] GEX preprocessed+PCA: {gex_train_x.shape} train / {gex_test_x.shape} test")

    if other_modality == "ADT":
        other_train_x = clr_normalize_adt(other[train_idx])
        other_test_x = clr_normalize_adt(other[test_idx])
    elif other_modality == "ATAC":
        lsi = LSITransformer(n_components=ATAC_LSI_N_COMPONENTS)
        counts_train = other[train_idx].layers["counts"]
        counts_test = other[test_idx].layers["counts"]
        other_train_x = lsi.fit_transform(counts_train)
        other_test_x = lsi.transform(counts_test)
    else:
        raise ValueError(other_modality)
    print(f"[{pair}] {other_modality} preprocessed: {other_train_x.shape} train / "
          f"{other_test_x.shape} test ({time.time()-t0:.1f}s elapsed)")

    encoder = LinearCCAEncoder(n_components=CCA_N_COMPONENTS)
    print(f"[{pair}] fitting CCA...")
    emb_gex_test, emb_other_test = encoder.fit_transform(
        gex_train_x, other_train_x, gex_test_x, other_test_x
    )
    print(f"[{pair}] CCA fit done ({time.time()-t0:.1f}s elapsed)")

    report = gap_report(emb_gex_test, emb_other_test, paired=True, k=5)
    report.update({
        "pair": pair,
        "other_modality": other_modality,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_top_genes": N_TOP_GENES_DEFAULT,
        "gex_pca_n_components": GEX_PCA_N_COMPONENTS,
        "cca_n_components": CCA_N_COMPONENTS,
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 1),
    })
    print(f"[{pair}] done: {json.dumps({k: v for k, v in report.items() if k != 'elapsed_sec'}, indent=2)}")
    return report


def main():
    results = [
        run_pair("cite", "ADT"),
        run_pair("multiome", "ATAC"),
    ]
    df = pd.DataFrame(results)
    out_path = "results/tables/phase1_baseline_linear_cca.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
