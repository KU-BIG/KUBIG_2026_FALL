"""Fusion modules for multimodal game embedding.

Provides frozen and trainable fusion towers for Step 1-3 embedding generation:
- FusionTower: Step 1-2 frozen concat fusion
- TextTowerTrainable, ImageTowerTrainable, TabularTowerTrainable: Trainable projection layers
- UserEncoder: User embedding for BPR
- BPRModel: Full BPR training model (Step 3)

Usage
-----
    # Step 1-2: Frozen concat baseline
    from fusion_tower import FusionTower
    fusion_tower = FusionTower(in_dim=192, hidden=256, out_dim=64)
    game_embedding = fusion_tower(z_concat)  # (B, 64), L2 normalized
    
    # Step 3: Fine-tuned BPR model
    from fusion_tower import BPRModel
    model = BPRModel(num_users=100, embed_dim=64)
    score, game_emb = model(user_idx, text_emb, image_emb, tab_emb)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionTower(nn.Module):
    """Concatenation-based fusion MLP for multimodal game embeddings.
    
    Takes concatenated modality vectors and projects to unified game embedding space.
    All modality inputs should be L2 normalized before concatenation to ensure
    balanced contribution during fusion.
    
    Parameters
    ----------
    in_dim : int
        Input dimension (sum of all modality dimensions after projection to 64D each).
        For text + tabular: 128 (64 + 64)
    hidden : int, default=256
        Hidden layer dimension
    out_dim : int, default=64
        Output dimension (game embedding size)
    dropout : float, default=0.2
        Dropout rate
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden: int = 256,
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
        """Project concatenated embedding to game embedding space.
        
        Parameters
        ----------
        z : torch.Tensor, shape (batch_size, in_dim)
            Concatenated normalized embeddings from all modalities
            
        Returns
        -------
        torch.Tensor, shape (batch_size, out_dim)
            L2 normalized game embedding
        """
        if z.ndim != 2 or z.shape[1] != self.in_dim:
            raise ValueError(f"Expected (batch, {self.in_dim}), got {tuple(z.shape)}")
        return F.normalize(self.net(z), p=2, dim=-1)


# ============================================================================
# Step 3: Fine-tuning Components (BPR Model)
# ============================================================================


class UserEncoder(nn.Module):
    """User embedding encoder for BPR learning.
    
    Maps user indices to learnable embeddings.
    
    Parameters
    ----------
    num_users : int
        Number of unique users
    embed_dim : int, default=64
        User embedding dimension
    """
    
    def __init__(self, num_users: int, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_users, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        """Get normalized user embeddings.
        
        Parameters
        ----------
        user_idx : torch.Tensor, shape (batch_size,)
            User indices
            
        Returns
        -------
        torch.Tensor, shape (batch_size, embed_dim)
            L2 normalized user embeddings
        """
        return F.normalize(self.embedding(user_idx), dim=-1)


class TextTowerTrainable(nn.Module):
    """Trainable text projection tower (MiniLM 384D → output_dim).
    
    Parameters
    ----------
    in_dim : int, default=384
        Input dimension (MiniLM embedding size)
    hidden : int, default=192
        Hidden layer dimension
    out_dim : int, default=64
        Output dimension
    dropout : float, default=0.2
        Dropout rate
    """
    
    def __init__(
        self,
        in_dim: int = 384,
        hidden: int = 192,
        out_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project text embeddings.
        
        Parameters
        ----------
        z : torch.Tensor, shape (batch_size, in_dim)
            Text embeddings (MiniLM 384D)
            
        Returns
        -------
        torch.Tensor, shape (batch_size, out_dim)
            L2 normalized text projections
        """
        return F.normalize(self.net(z), dim=-1)


class ImageTowerTrainable(nn.Module):
    """Trainable image projection tower for frozen CLIP embeddings."""

    def __init__(
        self,
        in_dim: int = 512,
        hidden: int = 256,
        out_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


class TabularTowerTrainable(nn.Module):
    """Trainable tabular projection tower (SVD 64D → output_dim).
    
    Parameters
    ----------
    in_dim : int, default=64
        Input dimension (SVD embedding size)
    hidden : int, default=128
        Hidden layer dimension
    out_dim : int, default=64
        Output dimension
    dropout : float, default=0.2
        Dropout rate
    """
    
    def __init__(
        self,
        in_dim: int = 64,
        hidden: int = 128,
        out_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project tabular embeddings.
        
        Parameters
        ----------
        z : torch.Tensor, shape (batch_size, in_dim)
            Tabular embeddings (SVD 64D)
            
        Returns
        -------
        torch.Tensor, shape (batch_size, out_dim)
            L2 normalized tabular projections
        """
        return F.normalize(self.net(z), dim=-1)


class FusionTowerTrainable(nn.Module):
    """Trainable fusion tower for Step 3 fine-tuning.
    
    Similar to FusionTower but used within BPRModel for end-to-end training.
    
    Parameters
    ----------
    in_dim : int, default=128
        Input dimension (concatenated embeddings)
    hidden : int, default=256
        Hidden layer dimension
    out_dim : int, default=64
        Output dimension
    dropout : float, default=0.2
        Dropout rate
    """
    
    def __init__(
        self,
        in_dim: int = 128,
        hidden: int = 256,
        out_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Fuse concatenated embeddings.
        
        Parameters
        ----------
        z : torch.Tensor, shape (batch_size, in_dim)
            Concatenated text and tabular projections
            
        Returns
        -------
        torch.Tensor, shape (batch_size, out_dim)
            L2 normalized fused game embeddings
        """
        return F.normalize(self.net(z), dim=-1)


class ResidualFeatureAdapter(nn.Module):
    """Trainable residual adapter for frozen encoder feature banks."""

    def __init__(
        self,
        dim: int,
        bottleneck: int,
        dropout: float = 0.1,
        residual_scale: float = 0.2,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z + self.residual_scale * self.net(z), dim=-1)


class PartialFusionBPRModel(nn.Module):
    """BPR model with trainable text/image adapters plus projection/fusion towers.

    This is the Step 4 model. The heavy MiniLM/CLIP encoders are still represented
    by frozen feature banks, while small residual adapters simulate partial
    top-layer adaptation before projection and fusion.
    """

    def __init__(self, num_users: int, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.user_encoder = UserEncoder(num_users, embed_dim)

        self.text_adapter = ResidualFeatureAdapter(dim=384, bottleneck=96)
        self.image_adapter = ResidualFeatureAdapter(dim=512, bottleneck=128)

        self.text_tower = TextTowerTrainable(in_dim=384, hidden=192, out_dim=embed_dim)
        self.image_tower = ImageTowerTrainable(in_dim=512, hidden=256, out_dim=embed_dim)
        self.tab_tower = TabularTowerTrainable(in_dim=64, hidden=128, out_dim=embed_dim)
        self.fusion_tower = FusionTowerTrainable(in_dim=embed_dim * 3, hidden=384, out_dim=embed_dim)

    def forward(
        self,
        user_idx: torch.Tensor,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        tab_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_encoder(user_idx)

        text_adapted = self.text_adapter(text_emb)
        image_adapted = self.image_adapter(image_emb)

        text_proj = self.text_tower(text_adapted)
        image_proj = self.image_tower(image_adapted)
        tab_proj = self.tab_tower(tab_emb)

        concat = torch.cat([text_proj, image_proj, tab_proj], dim=-1)
        game_emb = self.fusion_tower(concat)
        score = (user_emb * game_emb).sum(dim=-1)
        return score, game_emb

    def forward_game_only(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        tab_emb: torch.Tensor,
    ) -> torch.Tensor:
        text_adapted = self.text_adapter(text_emb)
        image_adapted = self.image_adapter(image_emb)

        text_proj = self.text_tower(text_adapted)
        image_proj = self.image_tower(image_adapted)
        tab_proj = self.tab_tower(tab_emb)

        concat = torch.cat([text_proj, image_proj, tab_proj], dim=-1)
        return self.fusion_tower(concat)


class BPRModel(nn.Module):
    """BPR (Bayesian Personalized Ranking) model for Step 3 fine-tuning.
    
    Learns user embeddings and fine-tunes text/image/tabular/fusion projections
    via BPR loss on positive/negative game pairs.
    
    Architecture:
    - UserEncoder: User ID → 64D embedding
    - TextTowerTrainable: MiniLM 384D → 64D
    - ImageTowerTrainable: CLIP 512D → 64D
    - TabularTowerTrainable: SVD 64D → 64D
    - FusionTowerTrainable: Concat 192D → 64D
    - Scorer: inner_product(user_emb, game_emb)
    
    Parameters
    ----------
    num_users : int
        Number of unique users
    embed_dim : int, default=64
        Embedding dimension for all components
    """
    
    def __init__(self, num_users: int, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.user_encoder = UserEncoder(num_users, embed_dim)
        self.text_tower = TextTowerTrainable(in_dim=384, hidden=192, out_dim=embed_dim)
        self.image_tower = ImageTowerTrainable(in_dim=512, hidden=256, out_dim=embed_dim)
        self.tab_tower = TabularTowerTrainable(in_dim=64, hidden=128, out_dim=embed_dim)
        self.fusion_tower = FusionTowerTrainable(in_dim=embed_dim * 3, hidden=384, out_dim=embed_dim)
    
    def forward(
        self,
        user_idx: torch.Tensor,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        tab_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score user-game pairs and return game embeddings.
        
        Parameters
        ----------
        user_idx : torch.Tensor, shape (batch_size,)
            User indices
        text_emb : torch.Tensor, shape (batch_size, 384)
            Text embeddings (MiniLM)
        image_emb : torch.Tensor, shape (batch_size, 512)
            Image embeddings (CLIP)
        tab_emb : torch.Tensor, shape (batch_size, 64)
            Tabular embeddings (SVD)
        
        Returns
        -------
        score : torch.Tensor, shape (batch_size,)
            Inner product scores between user_emb and game_emb
        game_emb : torch.Tensor, shape (batch_size, embed_dim)
            Game embeddings used in scoring
        """
        # Encode components
        user_emb = self.user_encoder(user_idx)  # (B, embed_dim)
        text_proj = self.text_tower(text_emb)   # (B, embed_dim)
        image_proj = self.image_tower(image_emb)  # (B, embed_dim)
        tab_proj = self.tab_tower(tab_emb)      # (B, embed_dim)
        
        # Fuse text, image, and tabular
        concat = torch.cat([text_proj, image_proj, tab_proj], dim=-1)  # (B, embed_dim*3)
        game_emb = self.fusion_tower(concat)  # (B, embed_dim)
        
        # Score
        score = (user_emb * game_emb).sum(dim=-1)  # (B,)
        
        return score, game_emb
    
    def forward_game_only(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        tab_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Generate game embeddings without user encoding.
        
        Used for inference to generate final game embedding bank.
        
        Parameters
        ----------
        text_emb : torch.Tensor, shape (batch_size, 384)
            Text embeddings (MiniLM)
        tab_emb : torch.Tensor, shape (batch_size, 64)
            Tabular embeddings (SVD)
        
        Returns
        -------
        torch.Tensor, shape (batch_size, embed_dim)
            Game embeddings
        """
        text_proj = self.text_tower(text_emb)
        image_proj = self.image_tower(image_emb)
        tab_proj = self.tab_tower(tab_emb)
        concat = torch.cat([text_proj, image_proj, tab_proj], dim=-1)
        game_emb = self.fusion_tower(concat)
        return game_emb


def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()


def prepare_bpr_data(
    positive_df,
    num_games: int,
    game_id2idx: dict,
    user_id2idx: dict,
) -> list[tuple[int, int, int]]:
    """Create (user, positive game, negative game) samples for BPR training."""
    user_positive_games = {}
    for user_id, group in positive_df.groupby("user_id"):
        user_positive_games[user_id] = {
            game_id2idx[gid]
            for gid in group["app_id"]
            if gid in game_id2idx
        }

    samples = []
    for user_id, group in positive_df.groupby("user_id"):
        if user_id not in user_id2idx:
            continue
        user_idx = user_id2idx[user_id]
        positives = user_positive_games.get(user_id, set())

        for game_id in group["app_id"]:
            if game_id not in game_id2idx:
                continue
            pos_idx = game_id2idx[game_id]
            neg_idx = np.random.randint(num_games)
            for _ in range(20):
                if neg_idx not in positives:
                    break
                neg_idx = np.random.randint(num_games)
            samples.append((user_idx, pos_idx, neg_idx))

    return samples


def train_epoch_bpr(
    model: BPRModel,
    train_loader,
    optimizer: torch.optim.Optimizer,
    text_bank: torch.Tensor,
    image_bank: torch.Tensor,
    tab_bank: torch.Tensor,
    device: str,
) -> float:
    """Run one BPR training epoch over frozen text/image/tabular feature banks."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for user_idx, pos_game_idx, neg_game_idx in train_loader:
        user_idx = user_idx.to(device)
        pos_game_idx = pos_game_idx.to(text_bank.device)
        neg_game_idx = neg_game_idx.to(text_bank.device)

        text_pos = text_bank[pos_game_idx].to(device)
        image_pos = image_bank[pos_game_idx].to(device)
        tab_pos = tab_bank[pos_game_idx].to(device)
        text_neg = text_bank[neg_game_idx].to(device)
        image_neg = image_bank[neg_game_idx].to(device)
        tab_neg = tab_bank[neg_game_idx].to(device)

        pos_score, _ = model(user_idx, text_pos, image_pos, tab_pos)
        neg_score, _ = model(user_idx, text_neg, image_neg, tab_neg)
        loss = bpr_loss(pos_score, neg_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    tower = FusionTower(in_dim=192, hidden=256, out_dim=64)
    z_concat = torch.randn(16, 192)
    out = tower(z_concat)
    print(f"Input {tuple(z_concat.shape)} -> Output {tuple(out.shape)}")
    print(f"Output norm (all should be ~1.0): "
          f"{out.norm(dim=-1).min():.4f} ~ {out.norm(dim=-1).max():.4f}")
    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in tower.parameters() if p.requires_grad):,}")
