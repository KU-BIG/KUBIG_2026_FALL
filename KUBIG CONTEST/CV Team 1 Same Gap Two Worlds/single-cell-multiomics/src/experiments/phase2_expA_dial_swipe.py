"""Phase 2 Experiment A: information dial-swipe (PLAN.md sec 3-1).

Quantity axis: retrain the MatchCLOT-architecture encoder (src/encoders/
matchclot_arch.py) with the GEX input restricted to progressively more HVGs
(50 -> 134 -> 500 -> 2000 -> all), holding everything else fixed, and see
whether delta_gap moves monotonically with input width. Unlike the Phase 1
CCA baseline (which needed a PCA pre-reduction purely for CCA's NIPALS
solver speed — docs/HISTORY.md), the neural encoder's first Linear layer
*is* the dimensionality reduction, so we feed the normalized HVG matrix
directly at whatever width the condition specifies — this is a more direct
realization of "manipulate how much information the GEX encoder is given"
than adding an extra PCA step would be.

Quality axis: fixed count (134, matching ADT dimensionality) but varying
*which* 134 genes — random HVGs vs the genes that correspond biologically
to the ADT panel (matched by Ensembl gene_id) vs a statistic-matched random
control. Run only on the cite (GEX-ADT) pair, since "ADT-matched genes"
requires an ADT panel to match against.

Each condition is repeated over multiple training seeds (encoder init +
minibatch order are the sources of randomness here — HVG selection itself
is deterministic under scanpy's seurat_v3 flavor, so seed does not change
*which* genes are picked, only how the encoder trains on them).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.data.loading import held_out_split, load_bmmc, split_modalities
from src.data.preprocessing import clr_normalize_adt, normalize_gex, select_hvgs
from src.encoders.matchclot_arch import encode, train_modality_clip
from src.metrics.gap_metrics import gap_report

PAIR, OTHER_MODALITY = "cite", "ADT"
TEST_FRAC = 0.2
SPLIT_SEED = 0
QUANTITY_CONDITIONS = [50, 134, 500, 2000, None]  # None = all genes (~13,953)
TRAIN_SEEDS = [0, 1, 2]
TRAIN_HPARAMS = dict(n_epochs=150, embedding_dim=64, layers_dim_mod1=(512, 256), layers_dim_mod2=(512, 256))


def _load_split():
    adata = load_bmmc(PAIR)
    gex, other = split_modalities(adata, mod1="GEX", mod2=OTHER_MODALITY)
    train_idx, test_idx = held_out_split(adata, test_frac=TEST_FRAC, seed=SPLIT_SEED)
    other_train = clr_normalize_adt(other[train_idx])
    other_test = clr_normalize_adt(other[test_idx])
    return gex, other_train, other_test, train_idx, test_idx


def run_quantity_axis(gex, other_train, other_test, train_idx, test_idx) -> list[dict]:
    rows = []
    for n_top_genes in QUANTITY_CONDITIONS:
        hvg_names = select_hvgs(gex[train_idx], n_top_genes=n_top_genes, seed=SPLIT_SEED)
        gex_train = normalize_gex(gex[train_idx], gene_subset=hvg_names)
        gex_test = normalize_gex(gex[test_idx], gene_subset=hvg_names)
        n_genes_actual = len(hvg_names)
        for seed in TRAIN_SEEDS:
            t0 = time.time()
            label = f"quantity_n={n_genes_actual}_seed={seed}"
            print(f"[{label}] training...")
            model = train_modality_clip(gex_train, other_train, hparams=TRAIN_HPARAMS, seed=seed, verbose_every=0)
            emb_gex, emb_other = encode(model, gex_test, other_test)
            report = gap_report(emb_gex, emb_other, paired=True)
            report.update({
                "axis": "quantity",
                "condition": f"n_genes={n_genes_actual}",
                "n_genes_requested": "all" if n_top_genes is None else n_top_genes,
                "n_genes_actual": n_genes_actual,
                "seed": seed,
                "elapsed_sec": round(time.time() - t0, 1),
            })
            rows.append(report)
            print(f"[{label}] delta_gap={report['delta_gap']:.4f} "
                  f"top5_retrieval={report['top5_retrieval_acc']:.4f} ({report['elapsed_sec']}s)")
    return rows


def _adt_matched_genes(gex, other_adata) -> list[str]:
    """Genes whose Ensembl gene_id matches an ADT antibody's target gene_id
    (PLAN.md quality axis, condition (6)). Isotype controls and antibodies
    without a mapped gene_id are dropped."""
    adt_gene_ids = other_adata.var["gene_id"].dropna().unique()
    gex_var = gex.var
    matched = gex_var.index[gex_var["gene_id"].isin(adt_gene_ids)].tolist()
    return matched


def _stat_matched_random_genes(gex, target_genes, seed: int) -> list[str]:
    """Random genes drawn to match the mean-expression decile distribution
    of `target_genes` (PLAN.md quality axis, condition (7)): controls for
    "these happen to be highly-expressed genes" as a confound separate from
    "these are the genes ADT actually measures". `target_genes` may be any
    iterable of gene names (pandas .loc rejects bare sets as an indexer)."""
    target_genes = list(target_genes)
    counts = gex.layers["counts"] if "counts" in gex.layers else gex.X
    mean_expr = np.asarray(counts.mean(axis=0)).ravel()
    deciles = pd.qcut(mean_expr, 10, labels=False, duplicates="drop")
    gene_to_decile = pd.Series(deciles, index=gex.var_names)
    target_deciles = gene_to_decile.loc[target_genes]

    rng = np.random.default_rng(seed)
    target_set = set(target_genes)
    chosen = set()
    for decile in target_deciles:
        excluded = target_set | chosen
        pool = gene_to_decile[(gene_to_decile == decile) & (~gene_to_decile.index.isin(excluded))].index
        if len(pool) == 0:
            continue
        chosen.add(rng.choice(pool))
    return list(chosen)


def run_quality_axis(gex, other_adata_test, other_train, other_test, train_idx, test_idx) -> list[dict]:
    rows = []
    random_134 = select_hvgs(gex[train_idx], n_top_genes=134, seed=SPLIT_SEED)
    adt_matched = _adt_matched_genes(gex, other_adata_test)
    stat_matched = _stat_matched_random_genes(gex[train_idx], set(adt_matched), seed=SPLIT_SEED)

    conditions = {
        "random_134hvg": random_134,
        "adt_matched": adt_matched,
        "stat_matched_random": stat_matched,
    }
    for cond_name, gene_list in conditions.items():
        if len(gene_list) < 5:
            print(f"[{cond_name}] skipped: only {len(gene_list)} genes resolved")
            continue
        gex_train = normalize_gex(gex[train_idx], gene_subset=gene_list)
        gex_test = normalize_gex(gex[test_idx], gene_subset=gene_list)
        for seed in TRAIN_SEEDS:
            t0 = time.time()
            label = f"quality_{cond_name}_seed={seed}"
            print(f"[{label}] training on {len(gene_list)} genes...")
            model = train_modality_clip(gex_train, other_train, hparams=TRAIN_HPARAMS, seed=seed, verbose_every=0)
            emb_gex, emb_other = encode(model, gex_test, other_test)
            report = gap_report(emb_gex, emb_other, paired=True)
            report.update({
                "axis": "quality",
                "condition": cond_name,
                "n_genes_actual": len(gene_list),
                "seed": seed,
                "elapsed_sec": round(time.time() - t0, 1),
            })
            rows.append(report)
            print(f"[{label}] delta_gap={report['delta_gap']:.4f} ({report['elapsed_sec']}s)")

    # pair-shuffle control (condition 8): adt_matched genes, but cell
    # correspondence between GEX and ADT permuted
    if len(adt_matched) >= 5:
        gex_train = normalize_gex(gex[train_idx], gene_subset=adt_matched)
        gex_test = normalize_gex(gex[test_idx], gene_subset=adt_matched)
        rng = np.random.default_rng(SPLIT_SEED)
        perm_train = rng.permutation(other_train.shape[0])
        other_train_shuffled = other_train[perm_train]
        for seed in TRAIN_SEEDS:
            t0 = time.time()
            label = f"quality_adt_matched_pair_shuffled_seed={seed}"
            print(f"[{label}] training...")
            model = train_modality_clip(gex_train, other_train_shuffled, hparams=TRAIN_HPARAMS, seed=seed, verbose_every=0)
            emb_gex, emb_other = encode(model, gex_test, other_test)
            report = gap_report(emb_gex, emb_other, paired=True)
            report.update({
                "axis": "quality",
                "condition": "adt_matched_pair_shuffled",
                "n_genes_actual": len(adt_matched),
                "seed": seed,
                "elapsed_sec": round(time.time() - t0, 1),
            })
            rows.append(report)
            print(f"[{label}] delta_gap={report['delta_gap']:.4f} ({report['elapsed_sec']}s)")
    return rows


def main():
    gex, other_train, other_test, train_idx, test_idx = _load_split()
    adata_full = load_bmmc(PAIR)
    _, other_adata = split_modalities(adata_full, mod1="GEX", mod2=OTHER_MODALITY)

    rows = []
    rows += run_quantity_axis(gex, other_train, other_test, train_idx, test_idx)
    rows += run_quality_axis(gex, other_adata[test_idx], other_train, other_test, train_idx, test_idx)

    df = pd.DataFrame(rows)
    out_path = "results/tables/phase2_expA_dial_swipe.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
