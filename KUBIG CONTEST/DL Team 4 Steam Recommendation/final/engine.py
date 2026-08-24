"""Cold and warm final recommendation interface for UI handoff.

Both entry points retrieve from the same frozen 64D multimodal game bank using
dot product.  The only difference is how the user/query vector and baseline
score are built:

* cold: selected genre/game intent vector + train popularity prior
* warm: observed user history vector + MF-BPR collaborative prior
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.bpr import BPRRecommender  # noqa: E402


@dataclass(frozen=True)
class FinalPaths:
    game_prefix: Path = REPO_ROOT / "game_fusion" / "emb_game_concat_64"
    genre_prefix: Path = REPO_ROOT / "history_user_tower" / "results_seed_42" / "genre_prototypes"
    catalog_path: Path = REPO_ROOT / "recommendation_mvp" / "deploy_data" / "catalog_ui.parquet"
    seen_history_path: Path = REPO_ROOT / "recommendation_mvp" / "deploy_data" / "seen_history_all.parquet"
    popularity_path: Path = REPO_ROOT / "recommendation_mvp" / "deploy_data" / "train_positive_counts.csv"
    mf_checkpoint_path: Path = REPO_ROOT / "outputs" / "mvp_50k" / "repro_seed_42" / "checkpoints" / "mf_bpr_best.pt"
    user_profile_prefix: Path = REPO_ROOT / "history_user_tower" / "results_seed_42" / "user_profiles_hours"


class FinalRecommendationEngine:
    """Serve cold/warm Top-K app_id recommendations with fixed I/O contracts."""

    def __init__(
        self,
        paths: FinalPaths | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.paths = paths or FinalPaths()
        self.device = torch.device(device)
        self.app_ids, self.game_bank, self.app_to_row = self._load_game_bank(self.paths.game_prefix)
        self.catalog = self._load_catalog(self.paths.catalog_path)
        self.genre_bank, self.genre_to_row = self._load_genres(self.paths.genre_prefix)
        self.tag_to_rows = self._build_tag_lookup(self.catalog)
        self._tag_profile_cache: dict[str, np.ndarray] = {}
        self.popularity_raw = self._load_popularity(self.paths.popularity_path)
        self.history = self._load_history(self.paths.seen_history_path)
        self.seen_by_user = (
            self.history.groupby("user_id").app_id.agg(lambda values: set(map(int, values))).to_dict()
        )
        self.mf_model, self.user_to_idx = self._load_mf(self.paths.mf_checkpoint_path)
        self.profile_users, self.user_profiles, self.user_to_profile = self._load_user_profiles(
            self.paths.user_profile_prefix
        )
        self.mf_items = self.mf_model.item_embedding.weight.detach().cpu().numpy().astype(np.float64)
        if self.mf_items.shape != self.game_bank.shape:
            raise ValueError(f"MF item shape {self.mf_items.shape} != game bank shape {self.game_bank.shape}")

    @staticmethod
    def _load_game_bank(prefix: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
        index = pd.read_csv(prefix.with_suffix(".csv"))
        ids = index["app_id"].to_numpy(np.int64)
        bank = np.load(prefix.with_suffix(".npy"), allow_pickle=False).astype(np.float32)
        if bank.shape != (len(ids), 64):
            raise ValueError(f"expected {(len(ids), 64)} game bank, got {bank.shape}")
        if len(np.unique(ids)) != len(ids) or not np.isfinite(bank).all():
            raise ValueError("invalid game bank")
        norms = np.linalg.norm(bank, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("game bank must be L2-normalized")
        return ids, bank, {int(app_id): row for row, app_id in enumerate(ids)}

    def _load_catalog(self, path: Path) -> pd.DataFrame:
        catalog = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        if "app_id" not in catalog.columns:
            raise ValueError("catalog must contain app_id")
        catalog = pd.DataFrame({"app_id": self.app_ids}).merge(
            catalog.drop_duplicates("app_id"), on="app_id", how="left"
        )
        return catalog

    @staticmethod
    def _load_genres(prefix: Path) -> tuple[np.ndarray, dict[str, int]]:
        frame = pd.read_csv(prefix.with_suffix(".csv"))
        bank = np.load(prefix.with_suffix(".npy"), allow_pickle=False).astype(np.float32)
        if bank.shape != (len(frame), 64):
            raise ValueError("genre prototype shape mismatch")
        mapping = {
            str(genre).strip().casefold(): int(row)
            for genre, row in zip(frame["genre"], frame["prototype_row"])
        }
        return bank, mapping

    @staticmethod
    def _build_tag_lookup(catalog: pd.DataFrame) -> dict[str, list[int]]:
        if "tags_kaggle" not in catalog.columns:
            return {}
        output: dict[str, list[int]] = {}
        for row, raw in enumerate(catalog["tags_kaggle"]):
            if pd.isna(raw):
                continue
            try:
                tags = ast.literal_eval(str(raw))
            except (ValueError, SyntaxError):
                continue
            if not isinstance(tags, (list, tuple, set)):
                continue
            for tag in tags:
                key = str(tag).strip().casefold()
                if key:
                    output.setdefault(key, []).append(row)
        return output

    def _load_popularity(self, path: Path) -> np.ndarray:
        frame = pd.read_csv(path)
        if "train_positive_count" not in frame.columns:
            raise ValueError("popularity file must contain train_positive_count")
        counts = frame.set_index("app_id")["train_positive_count"]
        return np.log1p(pd.Series(self.app_ids).map(counts).fillna(0).to_numpy(np.float64))

    @staticmethod
    def _load_history(path: Path) -> pd.DataFrame:
        history = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        required = {"user_id", "app_id"}
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"history missing columns: {sorted(missing)}")
        if "is_recommended" in history.columns:
            history = history.loc[history["is_recommended"].astype(bool)].copy()
        return history

    def _load_mf(self, checkpoint_path: Path) -> tuple[BPRRecommender, dict[int, int]]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if checkpoint.get("mode") != "mf_bpr":
            raise ValueError("MF checkpoint mode must be mf_bpr")
        users = {int(key): int(value) for key, value in checkpoint["user_to_idx"].items()}
        model = BPRRecommender(len(users), len(self.app_ids), "mf_bpr", embed_dim=64).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, users

    @staticmethod
    def _load_user_profiles(prefix: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
        index = pd.read_csv(prefix.with_name("user_embeddings").with_suffix(".csv"))
        profiles = np.load(prefix.with_suffix(".npy"), allow_pickle=False).astype(np.float32)
        if profiles.shape != (len(index), 64):
            raise ValueError("user profile shape mismatch")
        users = index["user_id"].to_numpy(np.int64)
        mapping = {int(user): row for row, user in enumerate(users)}
        return users, profiles, mapping

    @staticmethod
    def _validate_weight(value: float) -> float:
        weight = float(value)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("interest_weight must be between 0 and 1")
        return weight

    @staticmethod
    def _validate_k(value: int) -> int:
        k = int(value)
        if k <= 0:
            raise ValueError("k must be positive")
        return k

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("cannot normalize a zero vector")
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _zscore_available(scores: np.ndarray, available: np.ndarray) -> np.ndarray:
        output = np.full(len(scores), -np.inf, dtype=np.float64)
        values = scores[available]
        if len(values) == 0:
            raise ValueError("no available candidates to score")
        output[available] = (values - float(values.mean())) / max(float(values.std()), 1e-12)
        return output

    def _topk_detail(
        self,
        scores: np.ndarray,
        available: np.ndarray,
        k: int,
        extra: dict[str, np.ndarray | float | str],
    ) -> pd.DataFrame:
        rows = np.flatnonzero(available)
        if len(rows) == 0:
            raise ValueError("no games remain after exclusions")
        k = min(k, len(rows))
        if k == len(rows):
            top_rows = rows[np.argsort(-scores[rows], kind="stable")]
        else:
            partition = np.argpartition(scores[rows], -k)[-k:]
            top_rows = rows[partition]
            top_rows = top_rows[np.argsort(-scores[top_rows], kind="stable")]
        top_rows = top_rows[:k]
        output = pd.DataFrame(
            {
                "rank": np.arange(1, k + 1),
                "app_id": self.app_ids[top_rows],
                "score": scores[top_rows].astype(float),
            }
        )
        for key, value in extra.items():
            if isinstance(value, np.ndarray):
                output[key] = value[top_rows].astype(float)
            else:
                output[key] = value
        return output

    def _cold_user_vector(
        self,
        preferred_genres: Sequence[str],
        liked_game_ids: Sequence[int],
    ) -> tuple[np.ndarray, set[int]]:
        vectors: list[np.ndarray] = []
        liked = list(dict.fromkeys(int(app_id) for app_id in liked_game_ids))
        missing_games = [app_id for app_id in liked if app_id not in self.app_to_row]
        if missing_games:
            raise KeyError(f"liked_game_ids absent from game bank: {missing_games[:5]}")
        vectors.extend(self.game_bank[self.app_to_row[app_id]] for app_id in liked)

        genres = [str(value).strip() for value in preferred_genres if str(value).strip()]
        unknown_genres = [
            genre
            for genre in genres
            if genre.casefold() not in self.genre_to_row and genre.casefold() not in self.tag_to_rows
        ]
        if unknown_genres:
            raise KeyError(f"unknown preferred_genres: {unknown_genres[:5]}")
        for genre in genres:
            key = genre.casefold()
            if key in self.genre_to_row:
                vectors.append(self.genre_bank[self.genre_to_row[key]])
            else:
                vectors.append(self._tag_profile(key))
        if not vectors:
            raise ValueError("cold input requires at least one preferred_genre or liked_game_id")
        return self._normalize(np.stack(vectors).mean(axis=0)), set(liked)

    def _tag_profile(self, key: str) -> np.ndarray:
        if key not in self._tag_profile_cache:
            rows = self.tag_to_rows[key]
            self._tag_profile_cache[key] = self._normalize(self.game_bank[rows].mean(axis=0))
        return self._tag_profile_cache[key]

    def recommend_cold_detail(
        self,
        preferred_genres: Sequence[str],
        liked_game_ids: Sequence[int],
        interest_weight: float = 1.0,
        k: int = 10,
    ) -> pd.DataFrame:
        """Return rank/app_id/scores for a new user."""
        alpha = self._validate_weight(interest_weight)
        k = self._validate_k(k)
        user_vector, excluded = self._cold_user_vector(preferred_genres, liked_game_ids)
        available = np.array([int(app_id) not in excluded for app_id in self.app_ids], dtype=bool)
        content_raw = self.game_bank @ user_vector
        content_z = self._zscore_available(content_raw.astype(np.float64), available)
        popularity_z = self._zscore_available(self.popularity_raw, available)
        final = alpha * content_z + (1.0 - alpha) * popularity_z
        detail = self._topk_detail(
            final,
            available,
            k,
            {
                "content_score_z": content_z,
                "baseline_score_z": popularity_z,
                "baseline": "train_popularity",
                "interest_weight": alpha,
                "method": "cold_intent_dot_product",
            },
        )
        return detail

    def recommend_cold(
        self,
        preferred_genres: Sequence[str],
        liked_game_ids: Sequence[int],
        interest_weight: float = 1.0,
        k: int = 10,
    ) -> list[int]:
        """Return only Top-K app_id values for UI integration."""
        detail = self.recommend_cold_detail(preferred_genres, liked_game_ids, interest_weight, k)
        return [int(app_id) for app_id in detail["app_id"]]

    def _warm_user_vector(self, user_id: int) -> tuple[np.ndarray, set[int]]:
        if int(user_id) in self.user_to_profile:
            return (
                self.user_profiles[self.user_to_profile[int(user_id)]],
                self.seen_by_user.get(int(user_id), set()),
            )
        user_history = self.history.loc[self.history["user_id"].astype(np.int64).eq(int(user_id))]
        if user_history.empty:
            raise KeyError(f"user_id={user_id} has no deploy history")
        known = user_history.loc[user_history["app_id"].isin(set(self.app_to_row))]
        if known.empty:
            raise KeyError(f"user_id={user_id} history has no games in the game bank")
        rows = known["app_id"].map(self.app_to_row).to_numpy(np.int64)
        vectors = self.game_bank[rows]
        if "hours" in known.columns:
            weights = np.log1p(known["hours"].clip(lower=0).to_numpy(np.float32))
            pooled = np.average(vectors, axis=0, weights=weights) if weights.sum() > 0 else vectors.mean(axis=0)
        else:
            pooled = vectors.mean(axis=0)
        return self._normalize(pooled), set(map(int, user_history["app_id"]))

    @torch.no_grad()
    def _mf_scores(self, user_id: int) -> np.ndarray:
        if int(user_id) not in self.user_to_idx:
            raise KeyError(f"user_id={user_id} is not in the MF-BPR checkpoint")
        user_idx = torch.tensor([self.user_to_idx[int(user_id)]], device=self.device)
        user_vector = self.mf_model.user_encoder(user_idx).cpu().numpy().astype(np.float64)[0]
        return self.mf_items @ user_vector

    def recommend_warm_detail(
        self,
        user_id: int,
        interest_weight: float = 0.6,
        k: int = 10,
    ) -> pd.DataFrame:
        """Return rank/app_id/scores for an existing user."""
        alpha = self._validate_weight(interest_weight)
        k = self._validate_k(k)
        user_vector, history_excluded = self._warm_user_vector(int(user_id))
        seen = self.seen_by_user.get(int(user_id), set()) | history_excluded
        available = np.array([int(app_id) not in seen for app_id in self.app_ids], dtype=bool)
        content_raw = self.game_bank @ user_vector
        mf_raw = self._mf_scores(int(user_id))
        content_z = self._zscore_available(content_raw.astype(np.float64), available)
        mf_z = self._zscore_available(mf_raw, available)
        final = alpha * content_z + (1.0 - alpha) * mf_z
        detail = self._topk_detail(
            final,
            available,
            k,
            {
                "content_score_z": content_z,
                "baseline_score_z": mf_z,
                "baseline": "mf_bpr",
                "interest_weight": alpha,
                "method": "warm_history_dot_product",
            },
        )
        detail.insert(0, "user_id", int(user_id))
        return detail

    def recommend_warm(
        self,
        user_id: int,
        interest_weight: float = 0.6,
        k: int = 10,
    ) -> list[int]:
        """Return only Top-K app_id values for UI integration."""
        detail = self.recommend_warm_detail(user_id, interest_weight, k)
        return [int(app_id) for app_id in detail["app_id"]]


def recommend_cold(
    preferred_genres: Sequence[str],
    liked_game_ids: Sequence[int],
    interest_weight: float = 1.0,
    k: int = 10,
) -> list[int]:
    return FinalRecommendationEngine().recommend_cold(
        preferred_genres=preferred_genres,
        liked_game_ids=liked_game_ids,
        interest_weight=interest_weight,
        k=k,
    )


def recommend_warm(
    user_id: int,
    interest_weight: float = 0.6,
    k: int = 10,
) -> list[int]:
    return FinalRecommendationEngine().recommend_warm(
        user_id=user_id,
        interest_weight=interest_weight,
        k=k,
    )


def _parse_csv_strings(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Final cold/warm recommendation smoke CLI")
    sub = parser.add_subparsers(dest="mode", required=True)
    cold = sub.add_parser("cold")
    cold.add_argument("--preferred-genres", nargs="*", default=[])
    cold.add_argument("--liked-game-ids", nargs="*", type=int, default=[])
    cold.add_argument("--interest-weight", type=float, default=1.0)
    cold.add_argument("-k", type=int, default=10)
    cold.add_argument("--detail", action="store_true", help="print detail rows instead of app_id JSON")
    warm = sub.add_parser("warm")
    warm.add_argument("--user-id", type=int, required=True)
    warm.add_argument("--interest-weight", type=float, default=0.6)
    warm.add_argument("-k", type=int, default=10)
    warm.add_argument("--detail", action="store_true", help="print detail rows instead of app_id JSON")
    args = parser.parse_args()

    engine = FinalRecommendationEngine()
    if args.mode == "cold":
        genres = _parse_csv_strings(args.preferred_genres)
        detail = engine.recommend_cold_detail(genres, args.liked_game_ids, args.interest_weight, args.k)
    else:
        detail = engine.recommend_warm_detail(args.user_id, args.interest_weight, args.k)

    if args.detail:
        print(detail.to_string(index=False))
    else:
        print(json.dumps([int(app_id) for app_id in detail["app_id"]], ensure_ascii=False))


if __name__ == "__main__":
    main()
