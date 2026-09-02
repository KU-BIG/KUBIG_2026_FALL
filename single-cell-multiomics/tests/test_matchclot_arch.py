"""Smoke tests for src/encoders/matchclot_arch.py on tiny synthetic data.

Purpose is just to catch plumbing breakage (shape errors, NaNs, crashes) —
statistical validation of gap behavior happens in the real experiments
(src/experiments/), not here.
"""
import numpy as np
import torch

from src.encoders.matchclot_arch import Modality_CLIP, encode, train_modality_clip


def _tiny_paired_data(seed=0, n=200, d1=30, d2=20):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 8))
    x1 = (base @ rng.normal(size=(8, d1)) + rng.normal(scale=0.1, size=(n, d1))).astype(np.float32)
    x2 = (base @ rng.normal(size=(8, d2)) + rng.normal(scale=0.1, size=(n, d2))).astype(np.float32)
    return x1, x2


def test_train_and_encode_shapes():
    x1, x2 = _tiny_paired_data()
    model = train_modality_clip(
        x1, x2,
        hparams=dict(n_epochs=5, embedding_dim=12, layers_dim_mod1=(16, 12), layers_dim_mod2=(16, 12)),
        verbose_every=0,
    )
    assert isinstance(model, Modality_CLIP)
    e1, e2 = encode(model, x1, x2)
    assert e1.shape == (200, 12)
    assert e2.shape == (200, 12)
    assert np.isfinite(e1).all() and np.isfinite(e2).all()
    # encoder L2-normalizes its outputs
    norms1 = np.linalg.norm(e1, axis=1)
    assert np.allclose(norms1, 1.0, atol=1e-4)


def test_different_input_dims_supported():
    """The dial-swipe experiments retrain with varying GEX input width —
    confirm the model handles arbitrary (different) dims per modality."""
    x1, x2 = _tiny_paired_data(d1=5, d2=50)
    model = train_modality_clip(
        x1, x2,
        hparams=dict(n_epochs=3, embedding_dim=8, layers_dim_mod1=(16, 8), layers_dim_mod2=(16, 8)),
        verbose_every=0,
    )
    e1, e2 = encode(model, x1, x2)
    assert e1.shape[1] == 8 and e2.shape[1] == 8


def test_reproducible_with_same_seed():
    x1, x2 = _tiny_paired_data()
    hp = dict(n_epochs=5, embedding_dim=10, layers_dim_mod1=(16, 10), layers_dim_mod2=(16, 10))
    m1 = train_modality_clip(x1, x2, hparams=hp, seed=42, verbose_every=0)
    m2 = train_modality_clip(x1, x2, hparams=hp, seed=42, verbose_every=0)
    e1a, _ = encode(m1, x1, x2)
    e1b, _ = encode(m2, x1, x2)
    assert np.allclose(e1a, e1b, atol=1e-5)
