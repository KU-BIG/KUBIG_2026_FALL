"""Unit tests for src/experiments/phase2_expB_crosstype.py helpers, on tiny
synthetic data — catches the modality1/modality2 mixup that was found and
fixed in docs/HISTORY.md (a call-site argument swap + an internal encoder
swap that happened to cancel out, which is exactly the kind of fragile
"two wrongs make a right" code that breaks on the next innocent edit).

Using *different* input dimensions for the two modalities makes a
modality mixup fail loudly (shape mismatch) rather than silently, which is
why these tests deliberately pick d_gex != d_other.
"""
import numpy as np
import pytest

from src.encoders.matchclot_arch import train_modality_clip
from src.experiments.phase2_expB_crosstype import _cosine_sim_pairs, _sample_condition_pairs


def _toy_model_and_data(seed=0, n=300, d_gex=30, d_other=18):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 6))
    gex_x = (base @ rng.normal(size=(6, d_gex)) + rng.normal(scale=0.05, size=(n, d_gex))).astype(np.float32)
    other_x = (base @ rng.normal(size=(6, d_other)) + rng.normal(scale=0.05, size=(n, d_other))).astype(np.float32)
    model = train_modality_clip(
        gex_x, other_x,
        hparams=dict(n_epochs=80, embedding_dim=16, layers_dim_mod1=(32, 16), layers_dim_mod2=(32, 16)),
        seed=seed, verbose_every=0,
    )
    return model, gex_x, other_x


def test_cosine_sim_pairs_true_pairs_beat_random_pairs():
    model, gex_x, other_x = _toy_model_and_data()
    n = gex_x.shape[0]
    idx = np.arange(n)
    true_sims = _cosine_sim_pairs(model, gex_x, other_x, idx, idx)

    rng = np.random.default_rng(1)
    shuffled = rng.permutation(n)
    # avoid accidental self-pairs inflating the "random" baseline
    shuffled = np.where(shuffled == idx, (shuffled + 1) % n, shuffled)
    random_sims = _cosine_sim_pairs(model, gex_x, other_x, idx, shuffled)

    assert true_sims.mean() > random_sims.mean()


def test_cosine_sim_pairs_does_not_crash_on_mismatched_dims():
    """If encoder_modality1/2 were ever swapped again, this would raise a
    shape-mismatch error immediately (d_gex=30 != d_other=18)."""
    model, gex_x, other_x = _toy_model_and_data()
    idx = np.arange(gex_x.shape[0])
    sims = _cosine_sim_pairs(model, gex_x, other_x, idx, idx)
    assert np.isfinite(sims).all()


def test_sample_condition_pairs_true_pair_is_identity():
    cell_type = np.array(["A", "A", "B", "B", "C", "C"] * 10)
    lineage = np.array(["L1", "L1", "L1", "L1", "L2", "L2"] * 10)
    idx_gex, idx_other = _sample_condition_pairs("true_pair", cell_type, lineage, len(cell_type), 1000, seed=0)
    assert np.array_equal(idx_gex, idx_other)


def test_sample_condition_pairs_same_type_diff_object():
    cell_type = np.array(["A", "A", "A", "B", "B", "B"] * 10)
    lineage = np.array(["L1"] * 60)
    idx_gex, idx_other = _sample_condition_pairs("same_type_diff_object", cell_type, lineage, len(cell_type), 200, seed=0)
    assert len(idx_gex) > 0
    assert np.all(cell_type[idx_gex] == cell_type[idx_other])
    assert np.all(idx_gex != idx_other)


def test_sample_condition_pairs_same_lineage_diff_type():
    cell_type = np.array(["A", "B", "C", "D"] * 15)
    lineage = np.array(["L1", "L1", "L2", "L2"] * 15)
    idx_gex, idx_other = _sample_condition_pairs("same_lineage_diff_type", cell_type, lineage, len(cell_type), 200, seed=0)
    assert len(idx_gex) > 0
    assert np.all(lineage[idx_gex] == lineage[idx_other])
    assert np.all(cell_type[idx_gex] != cell_type[idx_other])


def test_sample_condition_pairs_diff_lineage():
    cell_type = np.array(["A", "B", "C", "D"] * 15)
    lineage = np.array(["L1", "L1", "L2", "L2"] * 15)
    idx_gex, idx_other = _sample_condition_pairs("diff_lineage", cell_type, lineage, len(cell_type), 200, seed=0)
    assert len(idx_gex) > 0
    assert np.all(lineage[idx_gex] != lineage[idx_other])
