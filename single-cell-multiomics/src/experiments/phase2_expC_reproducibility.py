"""Reproducibility check for experiment C (PLAN.md sec 3-3), added as a
follow-up after the single-seed run found single-lineage subsets have a
LARGER delta_gap than N-matched full-heterogeneity controls in all 5
lineages tested — the opposite of PLAN.md's hypothesis (docs/HISTORY.md
2026-08-13 "계속 7"). Before treating that as a real finding rather than a
seed-specific fluke, this reruns every condition across 3 training seeds
(and, for matchedN, 3 independent random subsamples) and checks whether
the single-lineage > matchedN ordering holds consistently.

Reuses _run_condition() from phase2_expC_lineage.py unchanged; only main()
differs (loops over seeds instead of running each condition once).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.cell_lineage import to_lineage
from src.data.loading import load_bmmc
from src.experiments.phase2_expC_lineage import LINEAGES_TO_TEST, MIN_LINEAGE_CELLS, PAIR, _run_condition

SEEDS = [0, 1, 2]


def main():
    adata = load_bmmc(PAIR)
    lineage_all = np.array(to_lineage(adata.obs["cell_type"].values, PAIR))
    all_idx = np.arange(adata.shape[0])

    rows = []
    for seed in SEEDS:
        rows.append(_run_condition(adata, all_idx, "full_all_lineages", seed=seed))

    for lineage in LINEAGES_TO_TEST:
        lineage_idx = np.where(lineage_all == lineage)[0]
        if len(lineage_idx) < MIN_LINEAGE_CELLS:
            print(f"[{lineage}] skipped: only {len(lineage_idx)} cells (< {MIN_LINEAGE_CELLS})")
            continue
        for seed in SEEDS:
            rows.append(_run_condition(adata, lineage_idx, f"single_lineage_{lineage}", seed=seed))
            # independent random subsample per seed, not just a fixed draw reused across seeds
            rng = np.random.default_rng(1000 + seed)
            matched_idx = rng.choice(all_idx, size=len(lineage_idx), replace=False)
            rows.append(_run_condition(adata, matched_idx, f"matchedN_for_{lineage}", seed=seed))

    df = pd.DataFrame(rows)
    out_path = "results/tables/phase2_expC_reproducibility.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    print("\n=== Per-lineage: single vs matchedN, mean +/- std across seeds ===")
    for lineage in LINEAGES_TO_TEST:
        single = df[df["condition"] == f"single_lineage_{lineage}"]
        matched = df[df["condition"] == f"matchedN_for_{lineage}"]
        if len(single) == 0:
            continue
        print(f"{lineage}: single={single['delta_gap'].mean():.4f}+/-{single['delta_gap'].std():.4f}  "
              f"matchedN={matched['delta_gap'].mean():.4f}+/-{matched['delta_gap'].std():.4f}  "
              f"single>matchedN in {(single['delta_gap'].values[:, None] > matched['delta_gap'].values[None, :]).mean()*100:.0f}% of seed pairs")


if __name__ == "__main__":
    main()
