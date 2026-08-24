"""Reusable inference API for existing and new-user history/intent vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from history_user_tower.experiment import load_game_bank
from history_user_tower.model import HistoryUserTower


REPO_ROOT = Path(__file__).resolve().parents[1]


class HistoryUserEncoder:
    """Encode game history or onboarding intent in the frozen Game Tower space."""

    def __init__(
        self,
        checkpoint: Path = REPO_ROOT / "history_user_tower/results_seed_42/history_user_tower.pt",
        game_prefix: Path = REPO_ROOT / "game_fusion/emb_game_concat_64",
        genre_prefix: Path | None = REPO_ROOT / "history_user_tower/results_seed_42/genre_prototypes",
        device: str = "cpu",
    ) -> None:
        self.catalog_ids, self.game_bank, self.app_to_row = load_game_bank(Path(game_prefix))
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = HistoryUserTower(
            int(saved.get("embedding_dim", 64)),
            int(saved.get("hidden_dim", 128)),
            float(saved.get("dropout", 0.1)),
        ).to(self.device)
        self.model.load_state_dict(saved["model_state_dict"])
        self.model.eval()
        self.genre_to_row: dict[str, int] = {}
        self.genre_bank: np.ndarray | None = None
        if genre_prefix is not None:
            prefix = Path(genre_prefix)
            if prefix.with_suffix(".npy").exists() and prefix.with_suffix(".csv").exists():
                genre_index = pd.read_csv(prefix.with_suffix(".csv"))
                self.genre_bank = np.load(prefix.with_suffix(".npy"), allow_pickle=False).astype(np.float32)
                if self.genre_bank.shape != (len(genre_index), 64):
                    raise ValueError("genre prototype shape/index mismatch")
                self.genre_to_row = {
                    str(genre).strip().casefold(): int(row)
                    for genre, row in zip(genre_index.genre, genre_index.prototype_row)
                }

    @staticmethod
    def _normalized_pool(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if len(vectors) == 0:
            raise ValueError("at least one history game or genre is required")
        if not np.isfinite(vectors).all() or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("vectors/weights must be finite and weights non-negative")
        pooled = np.average(vectors, axis=0, weights=weights) if weights.sum() > 0 else vectors.mean(axis=0)
        norm = float(np.linalg.norm(pooled))
        if norm == 0:
            raise ValueError("pooled intent vector has zero norm")
        return (pooled / norm).astype(np.float32)

    def encode_user(
        self,
        history_app_ids: Sequence[int],
        history_hours: Sequence[float] | None = None,
        selected_genres: Sequence[str] | None = None,
        apply_mlp: bool = True,
    ) -> np.ndarray:
        """Return one `(64,)` vector from history games and optional genres.

        Existing users should pass observed hours. New users can omit hours;
        selected games and genre prototypes then receive equal weight.
        """
        app_ids = [int(app_id) for app_id in history_app_ids]
        unknown = [app_id for app_id in app_ids if app_id not in self.app_to_row]
        if unknown:
            raise KeyError(f"app_id missing from frozen game bank: {unknown[:5]}")
        game_vectors = self.game_bank[[self.app_to_row[app_id] for app_id in app_ids]] if app_ids else np.empty((0, 64), np.float32)
        if history_hours is None:
            game_weights = np.ones(len(app_ids), dtype=np.float32)
        else:
            if len(history_hours) != len(app_ids):
                raise ValueError("history_hours length must equal history_app_ids length")
            raw_hours = np.asarray(history_hours, dtype=np.float32)
            if not np.isfinite(raw_hours).all() or (raw_hours < 0).any():
                raise ValueError("history_hours must be finite and non-negative")
            game_weights = np.log1p(raw_hours)

        genres = [str(value).strip() for value in (selected_genres or []) if str(value).strip()]
        genre_vectors = np.empty((0, 64), np.float32)
        if genres:
            if self.genre_bank is None:
                raise FileNotFoundError("genre prototype artifacts are not loaded")
            unknown_genres = [value for value in genres if value.casefold() not in self.genre_to_row]
            if unknown_genres:
                raise KeyError(f"unknown genres: {unknown_genres[:5]}")
            genre_vectors = self.genre_bank[[self.genre_to_row[value.casefold()] for value in genres]]

        vectors = np.vstack([game_vectors, genre_vectors])
        weights = np.r_[game_weights, np.ones(len(genre_vectors), dtype=np.float32)]
        history_vector = self._normalized_pool(vectors, weights)
        if not apply_mlp:
            return history_vector
        with torch.no_grad():
            tensor = torch.from_numpy(history_vector[None, :]).to(self.device)
            output = self.model(tensor).cpu().numpy()[0].astype(np.float32)
        if output.shape != (64,) or not np.isclose(np.linalg.norm(output), 1.0, atol=1e-5):
            raise ValueError("invalid User Tower output")
        return output

    def recommend(
        self,
        history_app_ids: Sequence[int],
        history_hours: Sequence[float] | None = None,
        selected_genres: Sequence[str] | None = None,
        top_k: int = 10,
        apply_mlp: bool = True,
    ) -> pd.DataFrame:
        """Full-catalog cosine retrieval, excluding selected/history games."""
        user = self.encode_user(history_app_ids, history_hours, selected_genres, apply_mlp)
        scores = self.game_bank @ user
        blocked = [self.app_to_row[int(app)] for app in history_app_ids if int(app) in self.app_to_row]
        scores[blocked] = -np.inf
        k = min(int(top_k), len(scores) - len(blocked))
        if k <= 0:
            raise ValueError("top_k must be positive")
        rows = np.argpartition(-scores, kth=k - 1)[:k]
        rows = rows[np.argsort(-scores[rows], kind="stable")]
        return pd.DataFrame({"rank": np.arange(1, k + 1), "app_id": self.catalog_ids[rows], "score": scores[rows]})


def encode_user(
    history_app_ids: Sequence[int],
    history_hours: Sequence[float] | None = None,
    selected_genres: Sequence[str] | None = None,
    apply_mlp: bool = True,
) -> np.ndarray:
    """Convenience API matching the fusion handoff contract."""
    return HistoryUserEncoder().encode_user(
        history_app_ids, history_hours, selected_genres, apply_mlp
    )
