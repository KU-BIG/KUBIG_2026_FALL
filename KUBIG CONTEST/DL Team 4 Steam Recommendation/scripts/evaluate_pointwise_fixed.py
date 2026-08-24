"""Re-score saved pointwise checkpoints on a supplied fixed candidate bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.metrics import ranking_diagnostics, one_positive_ranking_metrics  # noqa: E402
from mvp_recommendation.model import KnownUserRecommender  # noqa: E402


MODES = ["tabular_only", "text_only", "tabular_text_fusion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--text-prefix", type=Path, default=REPO_ROOT / "text_data" / "emb_text_minilm"
    )
    parser.add_argument(
        "--tabular-prefix",
        type=Path,
        default=REPO_ROOT / "tabular_embedding" / "leakage_safe" / "emb_tabular_safe_svd64",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


@torch.no_grad()
def score(
    model: KnownUserRecommender,
    candidates: pd.DataFrame,
    user_to_idx: dict[int, int],
    app_to_row: np.ndarray,
    tab_bank: torch.Tensor,
    text_bank: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    users = torch.from_numpy(candidates.user_id.map(user_to_idx).to_numpy(np.int64))
    rows = torch.from_numpy(app_to_row[candidates.app_id.to_numpy(np.int64)])
    outputs = []
    model.eval()
    for start in range(0, len(candidates), batch_size):
        u = users[start : start + batch_size].to(device)
        r = rows[start : start + batch_size]
        outputs.append(
            model(u, tab_bank[r].to(device), text_bank[r].to(device)).cpu().numpy()
        )
    result = np.concatenate(outputs)
    assert np.isfinite(result).all()
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    candidates = pd.read_parquet(args.candidates)
    assert candidates.groupby("group_id").size().eq(100).all()
    train = pd.read_parquet(args.data_dir / "debug_train.parquet")
    train_positive_counts = train.loc[train.is_recommended].groupby("app_id").size()
    existing = pd.read_csv(args.checkpoint_dir.parent / "mvp_model_comparison.csv").set_index("model")

    text_idx = pd.read_csv(args.text_prefix.with_suffix(".csv"))
    tab_idx = pd.read_csv(args.tabular_prefix.with_suffix(".csv"))
    assert np.array_equal(text_idx.app_id.to_numpy(), tab_idx.app_id.to_numpy())
    max_app_id = int(text_idx.app_id.max())
    app_to_row = np.full(max_app_id + 1, -1, np.int64)
    app_to_row[text_idx.app_id.to_numpy(np.int64)] = text_idx.row.to_numpy(np.int64)
    tab_bank = torch.from_numpy(np.load(args.tabular_prefix.with_suffix(".npy"), allow_pickle=False).copy())
    text_bank = torch.from_numpy(np.load(args.text_prefix.with_suffix(".npy"), allow_pickle=False).copy())

    results = []
    for mode in MODES:
        saved = torch.load(
            args.checkpoint_dir / f"{mode}_best.pt", map_location=device, weights_only=False
        )
        user_to_idx = {int(key): int(value) for key, value in saved["user_to_idx"].items()}
        assert candidates.user_id.isin(user_to_idx).all()
        model = KnownUserRecommender(len(user_to_idx), mode, embed_dim=64).to(device)
        model.load_state_dict(saved["model_state_dict"])
        scores = score(
            model, candidates, user_to_idx, app_to_row, tab_bank, text_bank, args.batch_size, device
        )
        result = {
            "model": f"{mode}_pointwise",
            "test_auc": float(existing.loc[mode, "test_auc"]),
            **one_positive_ranking_metrics(scores, 100),
            **ranking_diagnostics(
                candidates, scores, train_positive_counts, valid_catalog_size=len(text_idx), k=10
            ),
        }
        results.append(result)
        print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)
    print("EVALUATE_POINTWISE_FIXED_OK")


if __name__ == "__main__":
    main()

