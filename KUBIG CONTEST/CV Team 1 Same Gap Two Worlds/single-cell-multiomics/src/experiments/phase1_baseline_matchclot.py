"""Phase 1 baseline modality-gap measurement with encoder (b): the
from-scratch MatchCLOT-architecture encoder (PLAN.md sec 1). This is the
comparison that actually matters for the project's hypothesis — unlike
linear CCA (src/experiments/phase1_baseline.py), which directly optimizes
cross-modal correlation and so cannot help but shrink "gap" as part of its
own training objective (see docs/HISTORY.md 2026-08-13, "계속 3"), the
InfoNCE/contrastive objective here is the same family of objective Liang et
al. (2022) originally showed *preserves* a modality gap even after training.

No PCA pre-reduction: the neural encoder's own first Linear layer performs
dimensionality reduction, so GEX HVG-normalized features are fed directly.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from src.data.loading import held_out_split, load_bmmc, split_modalities
from src.data.preprocessing import clr_normalize_adt, normalize_gex, select_hvgs, LSITransformer
from src.encoders.matchclot_arch import encode, train_modality_clip
from src.metrics.gap_metrics import gap_report

N_TOP_GENES_DEFAULT = 2000
ATAC_LSI_N_COMPONENTS = 128
TEST_FRAC = 0.2
SPLIT_SEED = 0
TRAIN_SEEDS = [0, 1, 2]
TRAIN_HPARAMS = dict(n_epochs=150, embedding_dim=64, layers_dim_mod1=(512, 256), layers_dim_mod2=(512, 256))


def run_pair(pair: str, other_modality: str) -> list[dict]:
    t0 = time.time()
    print(f"[{pair}] loading...")
    adata = load_bmmc(pair)
    gex, other = split_modalities(adata, mod1="GEX", mod2=other_modality)
    train_idx, test_idx = held_out_split(adata, test_frac=TEST_FRAC, seed=SPLIT_SEED)
    print(f"[{pair}] loaded {adata.shape[0]} cells in {time.time()-t0:.1f}s; "
          f"train={len(train_idx)} test={len(test_idx)}")

    hvg_names = select_hvgs(gex[train_idx], n_top_genes=N_TOP_GENES_DEFAULT, seed=SPLIT_SEED)
    gex_train = normalize_gex(gex[train_idx], gene_subset=hvg_names)
    gex_test = normalize_gex(gex[test_idx], gene_subset=hvg_names)
    print(f"[{pair}] GEX preprocessed: {gex_train.shape} train / {gex_test.shape} test")

    if other_modality == "ADT":
        other_train = clr_normalize_adt(other[train_idx])
        other_test = clr_normalize_adt(other[test_idx])
    else:
        lsi = LSITransformer(n_components=ATAC_LSI_N_COMPONENTS)
        other_train = lsi.fit_transform(other[train_idx].layers["counts"])
        other_test = lsi.transform(other[test_idx].layers["counts"])
    print(f"[{pair}] {other_modality} preprocessed: {other_train.shape} train / "
          f"{other_test.shape} test ({time.time()-t0:.1f}s elapsed)")

    rows = []
    for seed in TRAIN_SEEDS:
        t_seed = time.time()
        print(f"[{pair}] training MatchCLOT-arch (seed={seed})...")
        model = train_modality_clip(gex_train, other_train, hparams=TRAIN_HPARAMS, seed=seed, verbose_every=30)
        emb_gex, emb_other = encode(model, gex_test, other_test)
        report = gap_report(emb_gex, emb_other, paired=True, k=5)
        report.update({
            "pair": pair,
            "other_modality": other_modality,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_top_genes": N_TOP_GENES_DEFAULT,
            "seed": seed,
            "elapsed_sec": round(time.time() - t_seed, 1),
        })
        rows.append(report)
        print(f"[{pair}] seed={seed} done: {json.dumps({k: v for k, v in report.items() if k not in ('elapsed_sec','pair','other_modality')}, indent=2)}")
    return rows


def main():
    rows = []
    rows += run_pair("cite", "ADT")
    rows += run_pair("multiome", "ATAC")
    df = pd.DataFrame(rows)
    out_path = "results/tables/phase1_baseline_matchclot_arch.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
