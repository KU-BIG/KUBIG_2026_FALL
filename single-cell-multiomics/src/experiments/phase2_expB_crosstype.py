"""Phase 2 Experiment B: cross-cell-type mismatch probing (PLAN.md sec 3-2).

Trains one MatchCLOT-arch model on true pairs (held-out test set held out
throughout), then at inference time feeds deliberately mismatched
GEX/other-modality combinations and looks at the resulting cosine
similarity under 5 conditions:
  true_pair               - GEX_X + ADT/ATAC_X (same cell)
  random_pair             - GEX of a random cell + ADT/ATAC of another random cell
  same_type_diff_object   - same fine cell_type, different actual cell
  same_lineage_diff_type  - same coarse lineage, different fine cell_type
  diff_lineage            - different coarse lineage entirely

`random_pair` similarities form a permutation null (PLAN.md: run many
random draws rather than a single sample) that the other conditions are
compared against.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.data.cell_lineage import to_lineage
from src.data.loading import held_out_split, load_bmmc, split_modalities
from src.data.preprocessing import clr_normalize_adt, normalize_gex, select_hvgs, LSITransformer
from src.encoders.matchclot_arch import train_modality_clip
from src.metrics.gap_metrics import _unit_normalize

PAIR, OTHER_MODALITY = "cite", "ADT"
N_TOP_GENES = 2000
TEST_FRAC = 0.2
SPLIT_SEED = 0
TRAIN_HPARAMS = dict(n_epochs=150, embedding_dim=64, layers_dim_mod1=(512, 256), layers_dim_mod2=(512, 256))
N_PAIRS_PER_CONDITION = 2000
N_NULL_DRAWS = 500
MIN_CELLS_PER_CONDITION = 30  # PLAN.md: pre-registered minimum sample size


def _cosine_sim_pairs(model, gex_x: np.ndarray, other_x: np.ndarray, idx_gex: np.ndarray, idx_other: np.ndarray) -> np.ndarray:
    """Cosine similarity between encoder_modality1(gex_x[idx_gex]) and
    encoder_modality2(other_x[idx_other]) — modality1/2 assignment matches
    train_modality_clip(gex_train, other_train, ...) in main(), where the
    first positional arg (GEX) becomes encoder_modality1 and the second
    (ADT/ATAC) becomes encoder_modality2 (see matchclot_arch.Modality_CLIP).
    idx_gex and idx_other need not be equal or aligned — that's the whole
    point of this function, feeding deliberately mismatched combinations.
    """
    import torch

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        g = torch.as_tensor(gex_x[idx_gex], dtype=torch.float32).to(device)
        o = torch.as_tensor(other_x[idx_other], dtype=torch.float32).to(device)
        emb_g = torch.nn.functional.normalize(model.encoder_modality1(g), dim=-1)
        emb_o = torch.nn.functional.normalize(model.encoder_modality2(o), dim=-1)
        sims = (emb_g * emb_o).sum(dim=-1)
    return sims.cpu().numpy()


def _sample_condition_pairs(condition: str, cell_type: np.ndarray, lineage: np.ndarray, n_cells: int, n_pairs: int, seed: int):
    rng = np.random.default_rng(seed)
    idx_gex, idx_other = [], []

    if condition == "true_pair":
        chosen = rng.choice(n_cells, size=min(n_pairs, n_cells), replace=False)
        return chosen, chosen

    if condition == "random_pair":
        idx_gex = rng.integers(0, n_cells, size=n_pairs)
        idx_other = rng.integers(0, n_cells, size=n_pairs)
        return idx_gex, idx_other

    by_type: dict = {}
    for i, t in enumerate(cell_type):
        by_type.setdefault(t, []).append(i)
    by_lineage: dict = {}
    for i, l in enumerate(lineage):
        by_lineage.setdefault(l, []).append(i)

    if condition == "same_type_diff_object":
        types = [t for t, members in by_type.items() if len(members) >= 2]
        attempts = 0
        while len(idx_gex) < n_pairs and attempts < n_pairs * 20:
            attempts += 1
            t = rng.choice(types)
            members = by_type[t]
            a, b = rng.choice(members, size=2, replace=False)
            idx_gex.append(a)
            idx_other.append(b)
        return np.array(idx_gex), np.array(idx_other)

    if condition == "same_lineage_diff_type":
        lineages = [l for l, members in by_lineage.items() if len(set(cell_type[members])) >= 2]
        attempts = 0
        while len(idx_gex) < n_pairs and attempts < n_pairs * 20:
            attempts += 1
            l = rng.choice(lineages)
            members = by_lineage[l]
            a = rng.choice(members)
            candidates = [m for m in members if cell_type[m] != cell_type[a]]
            if not candidates:
                continue
            b = rng.choice(candidates)
            idx_gex.append(a)
            idx_other.append(b)
        return np.array(idx_gex), np.array(idx_other)

    if condition == "diff_lineage":
        lineages_list = list(by_lineage.keys())
        attempts = 0
        while len(idx_gex) < n_pairs and attempts < n_pairs * 20:
            attempts += 1
            l1, l2 = rng.choice(lineages_list, size=2, replace=False)
            a = rng.choice(by_lineage[l1])
            b = rng.choice(by_lineage[l2])
            idx_gex.append(a)
            idx_other.append(b)
        return np.array(idx_gex), np.array(idx_other)

    raise ValueError(condition)


def main():
    t0 = time.time()
    adata = load_bmmc(PAIR)
    gex, other = split_modalities(adata, mod1="GEX", mod2=OTHER_MODALITY)
    train_idx, test_idx = held_out_split(adata, test_frac=TEST_FRAC, seed=SPLIT_SEED)

    hvg_names = select_hvgs(gex[train_idx], n_top_genes=N_TOP_GENES, seed=SPLIT_SEED)
    gex_train = normalize_gex(gex[train_idx], gene_subset=hvg_names)
    gex_test = normalize_gex(gex[test_idx], gene_subset=hvg_names)
    other_train = clr_normalize_adt(other[train_idx])
    other_test = clr_normalize_adt(other[test_idx])
    print(f"preprocessed in {time.time()-t0:.1f}s")

    print("training model on true pairs...")
    # train_modality_clip(x_mod1_train, x_mod2_train, ...) assigns
    # mod1=first-arg, mod2=second-arg, so passing (gex, other) here makes
    # encoder_modality1=GEX, encoder_modality2=other — matches
    # _cosine_sim_pairs(model, gex_x, other_x, ...) above.
    model = train_modality_clip(gex_train, other_train, hparams=TRAIN_HPARAMS, seed=0, verbose_every=30)
    print(f"training done ({time.time()-t0:.1f}s elapsed)")

    cell_type_test = adata.obs["cell_type"].values[test_idx]
    lineage_test = np.array(to_lineage(cell_type_test, PAIR))
    n_cells = len(test_idx)

    conditions = ["true_pair", "random_pair", "same_type_diff_object", "same_lineage_diff_type", "diff_lineage"]
    rows = []
    for cond in conditions:
        idx_gex, idx_other = _sample_condition_pairs(cond, cell_type_test, lineage_test, n_cells, N_PAIRS_PER_CONDITION, seed=SPLIT_SEED)
        if len(idx_gex) < MIN_CELLS_PER_CONDITION:
            print(f"[{cond}] skipped: only {len(idx_gex)} pairs available (< {MIN_CELLS_PER_CONDITION})")
            continue
        sims = _cosine_sim_pairs(model, gex_test, other_test, idx_gex, idx_other)
        row = {"condition": cond, "n_pairs": len(sims), "mean_sim": float(np.mean(sims)), "std_sim": float(np.std(sims))}
        rows.append(row)
        print(f"[{cond}] n={row['n_pairs']} mean_sim={row['mean_sim']:.4f} std={row['std_sim']:.4f} ({time.time()-t0:.1f}s)")

    # permutation null over many independent random-pair draws
    null_means = []
    for k in range(N_NULL_DRAWS):
        idx_gex, idx_other = _sample_condition_pairs("random_pair", cell_type_test, lineage_test, n_cells, N_PAIRS_PER_CONDITION, seed=2000 + k)
        sims = _cosine_sim_pairs(model, gex_test, other_test, idx_gex, idx_other)
        null_means.append(np.mean(sims))
    null_means = np.array(null_means)
    print(f"null distribution built from {N_NULL_DRAWS} draws: mean={null_means.mean():.4f} std={null_means.std():.4f}")

    for row in rows:
        if row["condition"] == "random_pair":
            row["null_pvalue"] = float("nan")
            continue
        row["null_pvalue"] = float((np.sum(null_means >= row["mean_sim"]) + 1) / (N_NULL_DRAWS + 1))

    df = pd.DataFrame(rows)
    out_path = "results/tables/phase2_expB_crosstype.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
