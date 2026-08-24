"""Pairwise BPR models and leakage-safe dynamic negative sampling."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mvp_recommendation.model import (
    FusionGameEncoder,
    IDUserEncoder,
    TabularGameEncoder,
    TextGameEncoder,
)


class TextAnchoredGatedGameEncoder(nn.Module):
    """Game-level vector gate initialized to preserve 90% of the Text signal."""

    def __init__(self, embed_dim: int = 64, initial_text_weight: float = 0.9) -> None:
        super().__init__()
        self.tabular_encoder = TabularGameEncoder(embed_dim)
        self.text_encoder = TextGameEncoder(embed_dim)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.zeros_(self.gate.weight)
        initial_logit = float(np.log(initial_text_weight / (1.0 - initial_text_weight)))
        nn.init.constant_(self.gate.bias, initial_logit)

    def components(
        self, tabular: torch.Tensor, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_tabular = self.tabular_encoder(tabular, text)
        z_text = self.text_encoder(tabular, text)
        text_gate = torch.sigmoid(self.gate(torch.cat([z_tabular, z_text], dim=-1)))
        return z_tabular, z_text, text_gate

    def forward(self, tabular: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        z_tabular, z_text, text_gate = self.components(tabular, text)
        return F.normalize(text_gate * z_text + (1.0 - text_gate) * z_tabular, dim=-1)


class BPRRecommender(nn.Module):
    """User/item dot-product model with MF or content-based game encoders."""

    def __init__(self, num_users: int, num_items: int, mode: str, embed_dim: int = 64) -> None:
        super().__init__()
        self.mode = mode
        self.user_encoder = IDUserEncoder(num_users, embed_dim)
        self.user_tabular_encoder: IDUserEncoder | None = None
        self.user_modality_gate: nn.Embedding | None = None
        self.tabular_game_encoder: TabularGameEncoder | None = None
        self.text_game_encoder: TextGameEncoder | None = None
        if mode == "mf_bpr":
            self.item_embedding = nn.Embedding(num_items, embed_dim)
            nn.init.normal_(self.item_embedding.weight, std=0.02)
            self.game_encoder = None
        elif mode == "tabular_bpr":
            self.game_encoder = TabularGameEncoder(embed_dim)
        elif mode == "text_bpr":
            self.game_encoder = TextGameEncoder(embed_dim)
        elif mode == "tabular_text_fusion_bpr":
            self.game_encoder = FusionGameEncoder(embed_dim)
        elif mode == "text_anchored_gated_bpr":
            self.game_encoder = TextAnchoredGatedGameEncoder(embed_dim)
        elif mode == "user_modality_gated_bpr":
            self.game_encoder = None
            self.user_tabular_encoder = IDUserEncoder(num_users, embed_dim)
            self.user_modality_gate = nn.Embedding(num_users, 1)
            initial_text_weight = 0.9
            initial_logit = float(np.log(initial_text_weight / (1.0 - initial_text_weight)))
            nn.init.constant_(self.user_modality_gate.weight, initial_logit)
            self.tabular_game_encoder = TabularGameEncoder(embed_dim)
            self.text_game_encoder = TextGameEncoder(embed_dim)
        else:
            raise ValueError(f"unknown BPR mode: {mode}")

    def encode_game(
        self,
        item_rows: torch.Tensor,
        tabular: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "mf_bpr":
            return self.item_embedding(item_rows)
        assert self.game_encoder is not None
        return self.game_encoder(tabular, text)

    def score(
        self,
        user_idx: torch.Tensor,
        item_rows: torch.Tensor,
        tabular: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "user_modality_gated_bpr":
            assert self.user_tabular_encoder is not None
            assert self.user_modality_gate is not None
            assert self.tabular_game_encoder is not None
            assert self.text_game_encoder is not None
            user_text = self.user_encoder(user_idx)
            user_tabular = self.user_tabular_encoder(user_idx)
            z_text = self.text_game_encoder(tabular, text)
            z_tabular = self.tabular_game_encoder(tabular, text)
            text_weight = torch.sigmoid(self.user_modality_gate(user_idx)).squeeze(-1)
            scores = text_weight * (user_text * z_text).sum(dim=-1) + (
                1.0 - text_weight
            ) * (user_tabular * z_tabular).sum(dim=-1)
            assert torch.isfinite(scores).all(), "user-gated BPR score contains NaN/Inf"
            return scores
        user = self.user_encoder(user_idx)
        game = self.encode_game(item_rows, tabular, text)
        scores = (user * game).sum(dim=-1)
        assert torch.isfinite(scores).all(), "BPR score contains NaN/Inf"
        return scores

    def pair_scores(
        self,
        user_idx: torch.Tensor,
        positive_rows: torch.Tensor,
        negative_rows: torch.Tensor,
        positive_tabular: torch.Tensor,
        positive_text: torch.Tensor,
        negative_tabular: torch.Tensor,
        negative_text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive = self.score(
            user_idx, positive_rows, positive_tabular, positive_text
        )
        negative = self.score(
            user_idx, negative_rows, negative_tabular, negative_text
        )
        return positive, negative


class FixedGameEmbeddingBPR(nn.Module):
    """Train a user tower against a fixed, precomputed game embedding bank.

    The game bank is deliberately a non-persistent buffer.  Checkpoints therefore
    contain the learned user embeddings and an artifact fingerprint, while the
    comparatively large game bank remains the single source of truth on disk.
    """

    def __init__(self, num_users: int, game_bank: torch.Tensor) -> None:
        super().__init__()
        if game_bank.ndim != 2:
            raise ValueError("game_bank must be a 2D tensor")
        if not torch.isfinite(game_bank).all():
            raise ValueError("game_bank contains NaN or Inf")
        self.user_encoder = IDUserEncoder(num_users, int(game_bank.shape[1]))
        self.register_buffer("game_bank", game_bank.float(), persistent=False)

    def score(
        self,
        user_idx: torch.Tensor,
        item_rows: torch.Tensor,
        tabular: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user = self.user_encoder(user_idx)
        game = self.game_bank[item_rows]
        scores = (user * game).sum(dim=-1)
        assert torch.isfinite(scores).all(), "fixed-game BPR score contains NaN/Inf"
        return scores

    def pair_scores(
        self,
        user_idx: torch.Tensor,
        positive_rows: torch.Tensor,
        negative_rows: torch.Tensor,
        positive_tabular: torch.Tensor | None = None,
        positive_text: torch.Tensor | None = None,
        negative_tabular: torch.Tensor | None = None,
        negative_text: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.score(user_idx, positive_rows),
            self.score(user_idx, negative_rows),
        )


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    difference = positive_scores - negative_scores
    assert torch.isfinite(difference).all()
    return -F.logsigmoid(difference).mean()


class DynamicNegativeSampler:
    """Sample catalog rows absent from each user's train interaction history.

    Validation and test histories are intentionally not consulted, preventing
    future item identity from affecting train-time negative sampling.
    """

    def __init__(
        self,
        num_items: int,
        observed_user_idx: np.ndarray,
        observed_item_rows: np.ndarray,
        seed: int,
    ) -> None:
        self.num_items = int(num_items)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        keys = observed_user_idx.astype(np.int64) * self.num_items + observed_item_rows.astype(np.int64)
        self.observed_keys = np.unique(keys)

    def reset(self, seed: int | None = None) -> None:
        """Reset sampling so model ablations receive identical negatives."""
        self.rng = np.random.default_rng(self.seed if seed is None else int(seed))

    def sample(self, user_idx: np.ndarray) -> np.ndarray:
        user_idx = np.asarray(user_idx, dtype=np.int64)
        negatives = self.rng.integers(0, self.num_items, size=len(user_idx), dtype=np.int64)
        invalid = np.isin(user_idx * self.num_items + negatives, self.observed_keys)
        attempts = 0
        while invalid.any():
            negatives[invalid] = self.rng.integers(
                0, self.num_items, size=int(invalid.sum()), dtype=np.int64
            )
            invalid = np.isin(user_idx * self.num_items + negatives, self.observed_keys)
            attempts += 1
            if attempts > 100:
                raise RuntimeError("negative sampling did not converge")
        assert not np.isin(user_idx * self.num_items + negatives, self.observed_keys).any()
        return negatives
