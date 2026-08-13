"""Linear CCA baseline "encoder" (PLAN.md Phase 1, encoder (c)).

A lightweight, non-deep-learning baseline: fit CCA between the two
modalities' preprocessed representations on the train split, and use the
resulting canonical variates as the shared embedding space for gap metrics.
Serves as a robustness check — if the gap ranking (GEX-ADT vs GEX-ATAC)
agrees with the from-scratch MatchCLOT-architecture encoder, the ranking is
not an artifact of that one deep model (docs/PLAN.md sec 1).
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA


class LinearCCAEncoder:
    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self._cca = CCA(n_components=n_components, max_iter=2000)
        self._fitted = False

    def fit(self, x_mod1_train: np.ndarray, x_mod2_train: np.ndarray) -> "LinearCCAEncoder":
        self._cca.fit(x_mod1_train, x_mod2_train)
        self._fitted = True
        return self

    def transform(self, x_mod1: np.ndarray, x_mod2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("LinearCCAEncoder must be fit() before transform()")
        emb1, emb2 = self._cca.transform(x_mod1, x_mod2)
        return emb1.astype(np.float32), emb2.astype(np.float32)

    def fit_transform(
        self,
        x_mod1_train: np.ndarray,
        x_mod2_train: np.ndarray,
        x_mod1_test: np.ndarray,
        x_mod2_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(x_mod1_train, x_mod2_train)
        return self.transform(x_mod1_test, x_mod2_test)
