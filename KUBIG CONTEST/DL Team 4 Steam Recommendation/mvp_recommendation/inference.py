"""Inference utilities for known-user MF, Text, and score-level hybrid ranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from mvp_recommendation.bpr import BPRRecommender, FixedGameEmbeddingBPR


MODEL_NAMES = ("mf_bpr", "text_bpr", "balanced_hybrid")
MULTIMODAL_MODEL_NAMES = ("multimodal_bpr", "mf_multimodal_hybrid")
ALL_MODEL_NAMES = MODEL_NAMES + MULTIMODAL_MODEL_NAMES


def read_catalog(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


class KnownUserRecommendationPipeline:
    """Load trained BPR towers once and produce full-catalog recommendations."""

    def __init__(
        self,
        checkpoint_dir: Path,
        hybrid_summary_path: Path,
        text_prefix: Path,
        tabular_prefix: Path | None,
        catalog_path: Path,
        data_dir: Path | None,
        history_path: Path | None = None,
        device: str | torch.device = "cpu",
        batch_size: int = 4096,
        history_scope: str = "all",
        multimodal_prefix: Path | None = None,
        multimodal_checkpoint: Path | None = None,
        multimodal_summary_path: Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.history_scope = history_scope
        if history_scope not in {"train", "train_validation", "all"}:
            raise ValueError("history_scope must be train, train_validation, or all")

        self.index = pd.read_csv(text_prefix.with_suffix(".csv"))
        assert np.array_equal(self.index.row.to_numpy(), np.arange(len(self.index)))
        self.app_ids = self.index.app_id.to_numpy(np.int64)
        self.app_to_row = {int(app_id): row for row, app_id in enumerate(self.app_ids)}
        self.text_bank = torch.from_numpy(
            np.load(text_prefix.with_suffix(".npy"), allow_pickle=False).astype(np.float32)
        )
        assert self.text_bank.shape == (len(self.index), 384)
        self.tabular_bank: torch.Tensor | None = None
        if tabular_prefix is not None:
            tab_index = pd.read_csv(tabular_prefix.with_suffix(".csv"))
            assert np.array_equal(self.index.app_id.to_numpy(), tab_index.app_id.to_numpy())
            self.tabular_bank = torch.from_numpy(
                np.load(tabular_prefix.with_suffix(".npy"), allow_pickle=False).astype(np.float32)
            )
            assert self.tabular_bank.shape == (len(self.index), 64)

        catalog = read_catalog(catalog_path)
        assert catalog.app_id.is_unique
        catalog = self.index[["app_id", "row"]].merge(
            catalog, on="app_id", how="left", validate="one_to_one"
        )
        assert catalog.title.notna().all(), "catalog titles do not cover the embedding index"
        self.catalog = catalog

        self.mf_model, mf_users = self._load_model("mf_bpr", checkpoint_dir)
        self.text_model, text_users = self._load_model("text_bpr", checkpoint_dir)
        assert mf_users == text_users, "MF/Text checkpoints use different user mappings"
        self.user_to_idx = mf_users

        hybrid_summary = json.loads(Path(hybrid_summary_path).read_text(encoding="utf-8"))
        self.alpha_mf = float(hybrid_summary["balanced_alpha_mf"])
        self.alpha_text = 1.0 - self.alpha_mf
        if not 0.0 <= self.alpha_mf <= 1.0:
            raise ValueError("invalid hybrid alpha")

        self.multimodal_model: FixedGameEmbeddingBPR | None = None
        self.multimodal_bank: torch.Tensor | None = None
        self.alpha_multimodal_mf: float | None = None
        supplied = [multimodal_prefix, multimodal_checkpoint, multimodal_summary_path]
        if any(value is not None for value in supplied):
            if not all(value is not None for value in supplied):
                raise ValueError("multimodal prefix, checkpoint, and summary must be supplied together")
            assert multimodal_prefix is not None
            assert multimodal_checkpoint is not None
            assert multimodal_summary_path is not None
            multimodal_index = pd.read_csv(multimodal_prefix.with_suffix(".csv"))
            assert np.array_equal(self.app_ids, multimodal_index.app_id.to_numpy(np.int64))
            bank_array = np.load(
                multimodal_prefix.with_suffix(".npy"), allow_pickle=False
            ).astype(np.float32)
            assert bank_array.shape == (len(self.index), 64) and np.isfinite(bank_array).all()
            self.multimodal_bank = torch.from_numpy(bank_array)
            saved_multi = torch.load(
                multimodal_checkpoint, map_location=self.device, weights_only=False
            )
            multi_users = {
                int(key): int(value) for key, value in saved_multi["user_to_idx"].items()
            }
            assert multi_users == mf_users, "MF/multimodal checkpoints use different user mappings"
            self.multimodal_model = FixedGameEmbeddingBPR(
                len(multi_users), self.multimodal_bank
            ).to(self.device)
            self.multimodal_model.load_state_dict(saved_multi["model_state_dict"])
            self.multimodal_model.eval()
            multimodal_summary = json.loads(
                Path(multimodal_summary_path).read_text(encoding="utf-8")
            )
            self.alpha_multimodal_mf = float(
                multimodal_summary["hybrid"]["selected_balanced_alpha_mf"]
            )
            if not 0.0 <= self.alpha_multimodal_mf <= 1.0:
                raise ValueError("invalid MF/multimodal hybrid alpha")

        if history_path is not None:
            history = pd.read_parquet(history_path, columns=["user_id", "app_id"])
        else:
            if data_dir is None:
                raise ValueError("data_dir is required when history_path is not supplied")
            split_names = {
                "train": ["debug_train.parquet"],
                "train_validation": ["debug_train.parquet", "debug_validation.parquet"],
                "all": ["debug_train.parquet", "debug_validation.parquet", "debug_test.parquet"],
            }[history_scope]
            history_parts = [
                pd.read_parquet(data_dir / name, columns=["user_id", "app_id"])
                for name in split_names
            ]
            history = pd.concat(history_parts, ignore_index=True).drop_duplicates()
        self.seen_by_user = (
            history.groupby("user_id").app_id.agg(lambda x: set(map(int, x))).to_dict()
        )

        self._mf_items = (
            self.mf_model.item_embedding.weight.detach().cpu().numpy().astype(np.float64)
        )
        self._text_items = self._encode_text_catalog()
        assert self._mf_items.shape == self._text_items.shape == (len(self.catalog), 64)
        assert np.isfinite(self._mf_items).all() and np.isfinite(self._text_items).all()

    def _load_model(
        self, mode: str, checkpoint_dir: Path
    ) -> tuple[BPRRecommender, dict[int, int]]:
        checkpoint = torch.load(
            Path(checkpoint_dir) / f"{mode}_best.pt",
            map_location=self.device,
            weights_only=False,
        )
        assert checkpoint["mode"] == mode
        users = {int(key): int(value) for key, value in checkpoint["user_to_idx"].items()}
        model = BPRRecommender(len(users), len(self.index), mode, embed_dim=64).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, users

    @torch.no_grad()
    def _encode_text_catalog(self) -> np.ndarray:
        assert self.text_model.game_encoder is not None
        chunks = []
        for start in range(0, len(self.catalog), self.batch_size):
            end = min(start + self.batch_size, len(self.catalog))
            tabular = (
                self.tabular_bank[start:end].to(self.device)
                if self.tabular_bank is not None
                else None
            )
            output = self.text_model.game_encoder(
                tabular,
                self.text_bank[start:end].to(self.device),
            )
            chunks.append(output.cpu().numpy().astype(np.float64))
        return np.concatenate(chunks)

    @staticmethod
    def _zscore_available(scores: np.ndarray, available: np.ndarray) -> np.ndarray:
        result = np.full(scores.shape, -np.inf, dtype=np.float64)
        values = scores[available]
        std = max(float(values.std()), 1e-12)
        result[available] = (values - float(values.mean())) / std
        return result

    @torch.no_grad()
    def _user_scores(
        self, user_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        if int(user_id) not in self.user_to_idx:
            raise KeyError(
                f"unknown user_id={user_id}; this known-user pipeline supports "
                f"{len(self.user_to_idx):,} users with positive train history"
            )
        user_index = torch.tensor([self.user_to_idx[int(user_id)]], device=self.device)
        mf_user = self.mf_model.user_encoder(user_index).cpu().numpy().astype(np.float64)[0]
        text_user = self.text_model.user_encoder(user_index).cpu().numpy().astype(np.float64)[0]
        mf_raw = self._mf_items @ mf_user
        text_raw = self._text_items @ text_user
        multimodal_raw: np.ndarray | None = None
        if self.multimodal_model is not None and self.multimodal_bank is not None:
            multimodal_user = (
                self.multimodal_model.user_encoder(user_index)
                .cpu()
                .numpy()
                .astype(np.float32)[0]
            )
            multimodal_raw = self.multimodal_bank.numpy() @ multimodal_user

        available = np.ones(len(self.catalog), dtype=bool)
        for app_id in self.seen_by_user.get(int(user_id), set()):
            row = self.app_to_row.get(int(app_id))
            if row is not None:
                available[row] = False
        if not available.any():
            raise RuntimeError(f"user_id={user_id} has no unseen catalog games")
        mf_z = self._zscore_available(mf_raw, available)
        text_z = self._zscore_available(text_raw, available)
        multimodal_z = (
            self._zscore_available(multimodal_raw, available)
            if multimodal_raw is not None
            else None
        )
        return mf_z, text_z, multimodal_z, available

    @staticmethod
    def _ranks(scores: np.ndarray, available: np.ndarray) -> np.ndarray:
        rows = np.flatnonzero(available)
        order = rows[np.argsort(-scores[rows], kind="stable")]
        ranks = np.full(len(scores), -1, dtype=np.int64)
        ranks[order] = np.arange(1, len(order) + 1)
        return ranks

    @staticmethod
    def _reason(
        model: str, mf_rank: int, text_rank: int, agreement_cutoff: int
    ) -> tuple[str, str]:
        if model == "mf_bpr":
            return "collaborative", "비슷한 사용자들의 선호 패턴이 강한 게임"
        if model == "text_bpr":
            return "content", "사용자의 선호와 게임 텍스트 특성이 잘 맞는 게임"
        if model == "multimodal_bpr":
            return "multimodal", "텍스트·이미지·정형 특성이 사용자 취향과 유사한 게임"
        if model == "mf_multimodal_hybrid":
            return "mf+multimodal", "협업 신호와 텍스트·이미지·정형 특성을 함께 반영한 게임"
        if mf_rank <= agreement_cutoff and text_rank <= agreement_cutoff:
            return "mf+text_agreement", "협업 신호와 텍스트 신호가 모두 높은 게임"
        if mf_rank < text_rank:
            return "mf_dominant", "협업 신호가 중심이고 텍스트 신호가 보완한 게임"
        return "text_dominant", "텍스트 신호가 중심이고 협업 신호가 보완한 게임"

    def recommend(
        self,
        user_ids: Iterable[int],
        top_k: int = 10,
        models: Iterable[str] = MODEL_NAMES,
    ) -> pd.DataFrame:
        requested_models = tuple(models)
        unknown_models = set(requested_models) - set(ALL_MODEL_NAMES)
        if unknown_models:
            raise ValueError(f"unknown models: {sorted(unknown_models)}")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if set(requested_models) & set(MULTIMODAL_MODEL_NAMES) and self.multimodal_model is None:
            raise ValueError("multimodal artifacts were not loaded")

        output = []
        metadata_columns = [
            column
            for column in [
                "app_id", "title", "tags_text", "rating", "positive_ratio",
                "user_reviews", "price_final",
            ]
            if column in self.catalog.columns
        ]
        # Keep explanations invariant when a larger candidate pool is requested
        # for downstream diversity reranking.
        agreement_cutoff = 50
        for user_id in map(int, user_ids):
            mf_z, text_z, multimodal_z, available = self._user_scores(user_id)
            mf_rank = self._ranks(mf_z, available)
            text_rank = self._ranks(text_z, available)
            score_map = {
                "mf_bpr": mf_z,
                "text_bpr": text_z,
                "balanced_hybrid": self.alpha_mf * mf_z + self.alpha_text * text_z,
            }
            if multimodal_z is not None:
                assert self.alpha_multimodal_mf is not None
                score_map["multimodal_bpr"] = multimodal_z
                score_map["mf_multimodal_hybrid"] = (
                    self.alpha_multimodal_mf * mf_z
                    + (1.0 - self.alpha_multimodal_mf) * multimodal_z
                )
            for model in requested_models:
                scores = score_map[model]
                eligible_rows = np.flatnonzero(available)
                if top_k >= len(eligible_rows):
                    top_rows = eligible_rows[np.argsort(-scores[eligible_rows], kind="stable")]
                else:
                    partition = np.argpartition(scores[eligible_rows], -top_k)[-top_k:]
                    top_rows = eligible_rows[partition]
                    top_rows = top_rows[np.argsort(-scores[top_rows], kind="stable")]
                top_rows = top_rows[:top_k]
                for rank, row in enumerate(top_rows, start=1):
                    source, reason = self._reason(
                        model, int(mf_rank[row]), int(text_rank[row]), agreement_cutoff
                    )
                    metadata = self.catalog.loc[row, metadata_columns].to_dict()
                    output.append(
                        {
                            "user_id": user_id,
                            "model": model,
                            "rank": rank,
                            **metadata,
                            "score": float(scores[row]),
                            "mf_score_z": float(mf_z[row]),
                            "text_score_z": float(text_z[row]),
                            "multimodal_score_z": (
                                float(multimodal_z[row]) if multimodal_z is not None else np.nan
                            ),
                            "mf_catalog_rank": int(mf_rank[row]),
                            "text_catalog_rank": int(text_rank[row]),
                            "recommendation_source": source,
                            "recommendation_reason": reason,
                            "alpha_mf": (
                                self.alpha_mf
                                if model == "balanced_hybrid"
                                else self.alpha_multimodal_mf
                                if model == "mf_multimodal_hybrid"
                                else float(model == "mf_bpr")
                            ),
                            "alpha_text": self.alpha_text if model == "balanced_hybrid" else float(model == "text_bpr"),
                            "alpha_multimodal": (
                                1.0 - self.alpha_multimodal_mf
                                if model == "mf_multimodal_hybrid"
                                and self.alpha_multimodal_mf is not None
                                else float(model == "multimodal_bpr")
                            ),
                            "excluded_history_scope": self.history_scope,
                        }
                    )
        result = pd.DataFrame(output)
        assert not result.empty
        assert result.groupby(["user_id", "model"]).size().eq(top_k).all()
        assert result.groupby(["user_id", "model"]).app_id.nunique().eq(top_k).all()
        return result
