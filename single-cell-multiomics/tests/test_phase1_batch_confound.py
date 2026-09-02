"""Regression test for the harmonypy orientation bug caught in
docs/HISTORY.md 2026-08-13: this harmonypy version (2.0.0, C++ backend)
returns Z_corr in the same (cells x features) orientation as the input,
unlike the classic pure-python harmonypy's transposed convention."""
import numpy as np

from src.experiments.phase1_batch_confound import _harmony_correct


def test_harmony_correct_preserves_input_orientation():
    rng = np.random.default_rng(0)
    n, d = 300, 10
    x = rng.normal(size=(n, d)).astype(np.float32)
    batch = rng.integers(0, 3, size=n)
    corrected = _harmony_correct(x, batch)
    assert corrected.shape == (n, d)


def test_harmony_correct_reduces_batch_separation():
    """Sanity check that correction actually does something: shift each
    batch's cluster by a fixed offset, then confirm harmony pulls the
    per-batch centroids closer together."""
    rng = np.random.default_rng(1)
    n_per_batch, d = 200, 5
    batch = np.repeat([0, 1, 2], n_per_batch)
    offsets = np.array([[0.0] * d, [8.0] * d, [-8.0] * d])
    x = offsets[batch] + rng.normal(scale=0.5, size=(len(batch), d)).astype(np.float32)

    def centroid_spread(y):
        centroids = np.array([y[batch == b].mean(axis=0) for b in np.unique(batch)])
        return np.linalg.norm(centroids - centroids.mean(axis=0), axis=1).mean()

    before = centroid_spread(x)
    corrected = _harmony_correct(x, batch)
    after = centroid_spread(corrected)
    # Directional check only — Harmony's default hyperparameters (theta,
    # max_iter_harmony) don't fully collapse extreme, substructure-free
    # synthetic clusters like this; the real assertion is "did it help".
    assert after < before * 0.8
