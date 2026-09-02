"""MatchCLOT dual-encoder architecture + training loop (PLAN.md Phase 1,
encoder (b): "from-scratch 재학습" baseline, and reused by every Phase 2
experiment that needs to retrain on a modified input, e.g. exp A's dial
swipe).

`Encoder`, `Modality_CLIP`, `symmetric_npair_loss` below are vendored
(near-verbatim) from AI4SCR/MatchCLOT (BSD-3-Clause,
external/MatchCLOT/LICENSE), commit at clone time 2026-08-13 — see
docs/HISTORY.md. We vendor rather than import from external/MatchCLOT at
runtime because external/ is not committed to git (docs/HISTORY.md decision
3): vendoring the ~40 lines of pure-torch model code keeps this module
self-contained and reproducible without depending on a live third-party
clone. The training loop below is new — MatchCLOT's own train.py depends on
`catalyst==22.4` pinned to torch==1.13.1, incompatible with this
environment's torch 2.8 (docs/HISTORY.md decision 2) — so we reproduce the
same architecture and loss with a plain PyTorch loop instead.

Epoch budget: the original paper trains 7000 epochs per fold x 9 folds.
Given this project needs to retrain across dozens of experimental
conditions (dial swipe x seeds, cross-cell-type subsets, lineage subsets),
we default to a much smaller epoch budget (see DEFAULT_HYPERPARAMS) and a
single fold (no k-fold ensembling). This is a deliberate scope reduction for
tractability, documented here and in docs/HISTORY.md — the comparison that
matters for this project (relative gap across conditions) does not require
matching the paper's absolute competition-metric SOTA.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    """Single modality encoder MLP with dropout (vendored from MatchCLOT;
    stochastic-feature-augmentation noise path kept for fidelity, disabled
    by default via noise_amount=0.0)."""

    def __init__(self, n_input, embedding_size, dropout_rates, dims_layers, noise_amount=0.0):
        super().__init__()
        layers = [nn.Linear(n_input, dims_layers[0])]
        for i in range(len(dims_layers) - 1):
            layers.append(nn.Linear(dims_layers[i], dims_layers[i + 1]))
        layers.append(nn.Linear(dims_layers[-1], embedding_size))
        self.fc_list = nn.ModuleList(layers)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=p) for p in dropout_rates])
        self.noise_amount = noise_amount

    def forward(self, x):
        for i in range(len(self.fc_list) - 1):
            if i > 0 and self.training and self.noise_amount > 0:
                x = x * (1 + self.noise_amount * torch.randn_like(x))
                x = x + self.noise_amount * torch.randn_like(x)
            x = F.elu(self.fc_list[i](x))
            if i < len(self.dropout_list):
                x = self.dropout_list[i](x)
        return self.fc_list[-1](x)


class Modality_CLIP(nn.Module):
    """CLIP-style dual encoder (vendored from MatchCLOT)."""

    def __init__(self, layers_dims, dropout_rates, dim_mod1, dim_mod2, output_dim, T, noise_amount=0.0):
        super().__init__()
        self.encoder_modality1 = Encoder(dim_mod1, output_dim, dropout_rates[0], layers_dims[0], noise_amount)
        self.encoder_modality2 = Encoder(dim_mod2, output_dim, dropout_rates[1], layers_dims[1], noise_amount)
        self.logit_scale = nn.Parameter(torch.ones([]) * T)

    def forward(self, features_first, features_second):
        f1 = self.encoder_modality1(features_first)
        f2 = self.encoder_modality2(features_second)
        f1 = f1 / torch.norm(f1, p=2, dim=-1, keepdim=True)
        f2 = f2 / torch.norm(f2, p=2, dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * f1 @ f2.T
        return logits, f1, f2


def symmetric_npair_loss(logits, targets):
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


# Reduced-epoch defaults; see module docstring "Epoch budget" for rationale.
DEFAULT_HYPERPARAMS = dict(
    embedding_dim=128,
    layers_dim_mod1=(1024, 512),
    layers_dim_mod2=(1024, 512),
    dropout_mod1=(0.3, 0.3),
    dropout_mod2=(0.3, 0.3),
    log_t=3.0,
    lr=3e-4,
    weight_decay=1e-4,
    n_epochs=300,
    batch_size=4096,
)


def train_modality_clip(
    x_mod1_train: np.ndarray,
    x_mod2_train: np.ndarray,
    hparams: dict | None = None,
    seed: int = 0,
    device: str | None = None,
    verbose_every: int = 50,
) -> Modality_CLIP:
    """Train a Modality_CLIP model on paired (row-aligned) training arrays.
    features_first=mod1, features_second=mod2 (order only matters for which
    encoder gets which array; embeddings returned by encode() are always
    (emb_mod1, emb_mod2))."""
    hp = {**DEFAULT_HYPERPARAMS, **(hparams or {})}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Modality_CLIP(
        layers_dims=(list(hp["layers_dim_mod1"]), list(hp["layers_dim_mod2"])),
        dropout_rates=(list(hp["dropout_mod1"]), list(hp["dropout_mod2"])),
        dim_mod1=x_mod1_train.shape[1],
        dim_mod2=x_mod2_train.shape[1],
        output_dim=hp["embedding_dim"],
        T=hp["log_t"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    x1 = torch.as_tensor(x_mod1_train, dtype=torch.float32)
    x2 = torch.as_tensor(x_mod2_train, dtype=torch.float32)
    n = x1.shape[0]
    batch_size = min(hp["batch_size"], n)

    model.train()
    for epoch in range(hp["n_epochs"]):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            b1 = x1[idx].to(device)
            b2 = x2[idx].to(device)
            targets = torch.arange(b1.shape[0], device=device)
            logits, _, _ = model(b1, b2)
            loss = symmetric_npair_loss(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if verbose_every and (epoch % verbose_every == 0 or epoch == hp["n_epochs"] - 1):
            print(f"  epoch {epoch}/{hp['n_epochs']} loss={epoch_loss / n_batches:.4f}")

    model.eval()
    return model


@torch.no_grad()
def encode(model: Modality_CLIP, x_mod1: np.ndarray, x_mod2: np.ndarray, device: str | None = None):
    device = device or next(model.parameters()).device
    model.eval()
    x1 = torch.as_tensor(x_mod1, dtype=torch.float32).to(device)
    x2 = torch.as_tensor(x_mod2, dtype=torch.float32).to(device)
    _, f1, f2 = model(x1, x2)
    return f1.cpu().numpy(), f2.cpu().numpy()
