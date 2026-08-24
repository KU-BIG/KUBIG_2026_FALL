"""Fusion-ready tabular embedding loader and projection tower.

Usage
-----
    from tabular_tower import TabularTower, load_tabular_bank

    bank, id2row = load_tabular_bank(
        "tabular_embedding/emb_tabular_svd64",
        app_ids=games.app_id.values,
    )
    tower = TabularTower(in_dim=64, out_dim=64)
    z_tab = tower(bank[batch_rows])  # (B, 64), L2 normalized
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularTower(nn.Module):
    """Project the fixed 64D SVD bank into a fusion-trained 64D space."""

    def __init__(
        self,
        in_dim: int = 64,
        hidden: int = 128,
        out_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.in_dim:
            raise ValueError(f"Expected (batch, {self.in_dim}), got {tuple(z.shape)}")
        return F.normalize(self.net(z), p=2, dim=-1)


def load_tabular_bank(
    prefix: str | Path,
    app_ids: Iterable[int] | None = None,
    device: str | torch.device = "cpu",
    fill_missing: bool = True,
):
    """Load the fixed embedding bank and build an app_id -> row mapping.

    This mirrors `load_text_bank()` and `load_image_bank()` used by the other
    fusion towers. If `app_ids` is given, the returned bank follows that exact
    order. Unknown IDs are filled with the normalized catalog-mean vector when
    `fill_missing=True`; otherwise a KeyError is raised.

    Returns
    -------
    bank : torch.FloatTensor, shape (N, 64)
    id2row : dict[int, int]
    """
    prefix = Path(prefix)
    embeddings = np.load(prefix.with_suffix(".npy"), allow_pickle=False).astype(np.float32)
    ids = pd.read_csv(prefix.with_suffix(".csv"))["app_id"].to_numpy(np.int64)

    if embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise ValueError(f"Expected embedding shape (N, 64), got {embeddings.shape}")
    if len(embeddings) != len(ids):
        raise ValueError("Embedding rows and app_id rows do not match")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate app_id values in tabular bank")
    if not np.isfinite(embeddings).all():
        raise ValueError("Tabular embedding contains NaN or infinity")

    if app_ids is None:
        bank = embeddings
        output_ids = ids
    else:
        requested_ids = np.asarray(list(app_ids), dtype=np.int64)
        position = {int(app_id): row for row, app_id in enumerate(ids)}
        missing = [int(app_id) for app_id in requested_ids if int(app_id) not in position]
        if missing and not fill_missing:
            raise KeyError(f"Tabular embedding missing for {len(missing)} app_ids: {missing[:5]}")

        mean_vector = embeddings.mean(axis=0)
        mean_vector /= max(float(np.linalg.norm(mean_vector)), 1e-12)
        bank = np.stack(
            [embeddings[position[int(app_id)]] if int(app_id) in position else mean_vector
             for app_id in requested_ids]
        ).astype(np.float32)
        output_ids = requested_ids

        if missing:
            print(
                f"[tabular_tower] Filled {len(missing)} missing app_ids with catalog mean: "
                f"{missing[:5]}"
            )

    tensor_bank = torch.from_numpy(bank).to(device)
    id2row = {int(app_id): row for row, app_id in enumerate(output_ids)}
    return tensor_bank, id2row


if __name__ == "__main__":
    base = Path(__file__).resolve().parent / "emb_tabular_svd64"
    bank, id2row = load_tabular_bank(base)
    tower = TabularTower(in_dim=bank.shape[1], out_dim=64)
    tower.eval()
    with torch.no_grad():
        output = tower(bank[:32])

    norms = output.norm(dim=-1)
    assert output.shape == (32, 64)
    assert torch.isfinite(output).all()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print(f"bank: {tuple(bank.shape)}, unique app_ids: {len(id2row):,}")
    print(f"tower: {tuple(bank[:32].shape)} -> {tuple(output.shape)}")
    print(f"output norm: {norms.min():.6f} ~ {norms.max():.6f}")
    print(
        "trainable parameters: "
        f"{sum(parameter.numel() for parameter in tower.parameters() if parameter.requires_grad):,}"
    )
