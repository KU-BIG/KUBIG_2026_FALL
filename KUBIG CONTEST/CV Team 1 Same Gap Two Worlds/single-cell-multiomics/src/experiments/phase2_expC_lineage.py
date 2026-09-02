"""Phase 2 Experiment C: single-lineage subsets / heterogeneity dose-response
(PLAN.md sec 3-3).

Retrains the MatchCLOT-arch encoder on progressively less heterogeneous
subsets of the data (all cell types -> a single coarse lineage at a time)
and checks whether delta_gap falls off with heterogeneity, using Shannon
entropy of the fine cell_type distribution within each subset as the
heterogeneity index (PLAN.md's "x축"). Each single-lineage condition is
paired with an N-matched random subsample of the full dataset, so a drop
in gap can't be explained away by "fewer cells means noisier estimate"
rather than "less heterogeneity" (PLAN.md sec 3-3 addition).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.data.cell_lineage import to_lineage
from src.data.loading import held_out_split, load_bmmc, split_modalities
from src.data.preprocessing import clr_normalize_adt, normalize_gex, select_hvgs
from src.encoders.matchclot_arch import encode, train_modality_clip
from src.metrics.gap_metrics import gap_report

PAIR, OTHER_MODALITY = "cite", "ADT"
N_TOP_GENES = 2000
TEST_FRAC = 0.2
SPLIT_SEED = 0
TRAIN_HPARAMS = dict(n_epochs=150, embedding_dim=64, layers_dim_mod1=(512, 256), layers_dim_mod2=(512, 256))
LINEAGES_TO_TEST = ["T_CD4", "T_CD8", "Myeloid_Mono", "B_cell", "NK_ILC"]
MIN_LINEAGE_CELLS = 1000


def _shannon_entropy(labels) -> float:
    counts = pd.Series(labels).value_counts(normalize=True)
    return float(-np.sum(counts * np.log(counts)))


def _run_condition(adata, cell_idx: np.ndarray, condition_name: str, seed: int = SPLIT_SEED) -> dict:
    t0 = time.time()
    sub = adata[cell_idx].copy()
    gex, other = split_modalities(sub, mod1="GEX", mod2=OTHER_MODALITY)
    train_idx, test_idx = held_out_split(sub, test_frac=TEST_FRAC, seed=seed)

    hvg_names = select_hvgs(gex[train_idx], n_top_genes=N_TOP_GENES, seed=seed)
    gex_train = normalize_gex(gex[train_idx], gene_subset=hvg_names)
    gex_test = normalize_gex(gex[test_idx], gene_subset=hvg_names)
    other_train = clr_normalize_adt(other[train_idx])
    other_test = clr_normalize_adt(other[test_idx])

    model = train_modality_clip(gex_train, other_train, hparams=TRAIN_HPARAMS, seed=seed, verbose_every=0)
    emb_gex, emb_other = encode(model, gex_test, other_test)
    report = gap_report(emb_gex, emb_other, paired=True)

    heterogeneity = _shannon_entropy(sub.obs["cell_type"].values)
    report.update({
        "condition": condition_name,
        "n_total": sub.shape[0],
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_cell_types": sub.obs["cell_type"].nunique(),
        "heterogeneity_entropy": heterogeneity,
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 1),
    })
    print(f"[{condition_name}] n={report['n_total']} n_types={report['n_cell_types']} "
          f"H={heterogeneity:.3f} delta_gap={report['delta_gap']:.4f} ({report['elapsed_sec']}s)")
    return report


def main():
    adata = load_bmmc(PAIR)
    lineage_all = np.array(to_lineage(adata.obs["cell_type"].values, PAIR))
    rng = np.random.default_rng(SPLIT_SEED)

    rows = []
    all_idx = np.arange(adata.shape[0])
    rows.append(_run_condition(adata, all_idx, "full_all_lineages"))

    for lineage in LINEAGES_TO_TEST:
        lineage_idx = np.where(lineage_all == lineage)[0]
        if len(lineage_idx) < MIN_LINEAGE_CELLS:
            print(f"[{lineage}] skipped: only {len(lineage_idx)} cells (< {MIN_LINEAGE_CELLS})")
            continue
        rows.append(_run_condition(adata, lineage_idx, f"single_lineage_{lineage}"))

        # N-matched control: same N, randomly sampled from the FULL dataset
        # (all lineages), to separate "less heterogeneity" from "fewer cells"
        matched_idx = rng.choice(all_idx, size=len(lineage_idx), replace=False)
        rows.append(_run_condition(adata, matched_idx, f"matchedN_for_{lineage}"))

    df = pd.DataFrame(rows)
    out_path = "results/tables/phase2_expC_lineage.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
