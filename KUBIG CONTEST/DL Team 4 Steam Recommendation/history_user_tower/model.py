"""History-conditioned user tower for the Steam recommendation MVP."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HistoryUserTower(nn.Module):
    """Refine a pooled 64D history vector with a small residual MLP.

    The final layer starts at zero, so the untrained model is exactly the
    log1p(hours)-weighted history baseline. This makes training stable and
    lets the experiment measure what the MLP adds over the same input.
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.adapter = nn.Sequential(
            nn.Linear(self.embedding_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.embedding_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, history_vector: torch.Tensor) -> torch.Tensor:
        if history_vector.ndim != 2 or history_vector.shape[1] != self.embedding_dim:
            raise ValueError(
                f"expected [batch, {self.embedding_dim}] history vectors, "
                f"got {tuple(history_vector.shape)}"
            )
        output = F.normalize(history_vector + self.adapter(history_vector), dim=-1)
        if not torch.isfinite(output).all():
            raise ValueError("user tower produced NaN/Inf")
        return output


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""
    return -F.logsigmoid(positive_scores - negative_scores).mean()
