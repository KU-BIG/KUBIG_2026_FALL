"""Extends experiment A's quantity axis (PLAN.md sec 3-1) to the GEX-ATAC
(multiome) pair — not in the original plan, added as a follow-up to check
whether the quantity-axis finding (delta_gap falls as GEX gene count grows,
opposite of PLAN.md's hypothesis — docs/HISTORY.md 2026-08-13 "계속 6") is
specific to GEX-ADT or holds for GEX-ATAC too.

Reuses run_quantity_axis() from phase2_expA_dial_swipe.py unchanged (it
only needs preprocessed other-modality train/test arrays, with no
ADT-specific logic) — the ATAC-specific part is just building those arrays
via LSITransformer once (LSI doesn't depend on the GEX quantity condition,
so it's fit a single time and reused across all 5x3 quantity runs).
"""
from __future__ import annotations

import time

import pandas as pd

from src.data.loading import held_out_split, load_bmmc, split_modalities
from src.data.preprocessing import LSITransformer
from src.experiments.phase2_expA_dial_swipe import run_quantity_axis, TEST_FRAC, SPLIT_SEED

PAIR, OTHER_MODALITY = "multiome", "ATAC"
ATAC_LSI_N_COMPONENTS = 128


def main():
    t0 = time.time()
    adata = load_bmmc(PAIR)
    gex, other = split_modalities(adata, mod1="GEX", mod2=OTHER_MODALITY)
    train_idx, test_idx = held_out_split(adata, test_frac=TEST_FRAC, seed=SPLIT_SEED)
    print(f"loaded {adata.shape[0]} cells in {time.time()-t0:.1f}s; train={len(train_idx)} test={len(test_idx)}")

    lsi = LSITransformer(n_components=ATAC_LSI_N_COMPONENTS)
    other_train = lsi.fit_transform(other[train_idx].layers["counts"])
    other_test = lsi.transform(other[test_idx].layers["counts"])
    print(f"ATAC LSI done ({time.time()-t0:.1f}s elapsed)")

    rows = run_quantity_axis(gex, other_train, other_test, train_idx, test_idx)
    for row in rows:
        row["pair"] = PAIR
        row["other_modality"] = OTHER_MODALITY

    df = pd.DataFrame(rows)
    out_path = "results/tables/phase2_expA_multiome_quantity.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
