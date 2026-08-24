"""Reusable user encoder, game encoders, scorer, and known-user model."""

from __future__ import annotations

from importlib import import_module

import torch
from torch import nn
import torch.nn.functional as F

from tabular_embedding.tabular_tower import TabularTower


TextTower = import_module("text_data.08_text_tower").TextTower


class IDUserEncoder(nn.Module):
    def __init__(self, num_users: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_users, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(user_idx)


class TabularGameEncoder(nn.Module):
    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.tower = TabularTower(in_dim=64, out_dim=out_dim)

    def forward(self, tabular: torch.Tensor, text: torch.Tensor | None = None) -> torch.Tensor:
        return self.tower(tabular)


class TextGameEncoder(nn.Module):
    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.tower = TextTower(in_dim=384, out_dim=out_dim)

    def forward(self, tabular: torch.Tensor | None, text: torch.Tensor) -> torch.Tensor:
        return self.tower(text)


class FusionGameEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.tabular_tower = TabularTower(in_dim=64, out_dim=embed_dim)
        self.text_tower = TextTower(in_dim=384, out_dim=embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, tabular: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        z_tab = self.tabular_tower(tabular)
        z_text = self.text_tower(text)
        return F.normalize(self.fusion(torch.cat([z_tab, z_text], dim=-1)), dim=-1)


class RecommendationScorer(nn.Module):
    def __init__(self, embed_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user: torch.Tensor, game: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [user, game, user * game, torch.abs(user - game)], dim=-1
        )
        return self.net(features).squeeze(-1)


class KnownUserRecommender(nn.Module):
    def __init__(self, num_users: int, mode: str, embed_dim: int = 64) -> None:
        super().__init__()
        self.mode = mode
        self.user_encoder = IDUserEncoder(num_users, embed_dim)
        if mode == "tabular_only":
            self.game_encoder = TabularGameEncoder(embed_dim)
        elif mode == "text_only":
            self.game_encoder = TextGameEncoder(embed_dim)
        elif mode == "tabular_text_fusion":
            self.game_encoder = FusionGameEncoder(embed_dim)
        else:
            raise ValueError(f"unknown mode: {mode}")
        self.scorer = RecommendationScorer(embed_dim)

    def forward(
        self,
        user_idx: torch.Tensor,
        tabular: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        user = self.user_encoder(user_idx)
        game = self.game_encoder(tabular, text)
        logits = self.scorer(user, game)
        assert torch.isfinite(logits).all(), "model output has NaN/Inf"
        return logits

