"""Analyze recommendations.csv and build a chronological DEBUG split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.data import (  # noqa: E402
    choose_debug_users,
    chronological_split,
    load_selected_interactions,
    save_prepared_outputs,
    scan_interactions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommendations", type=Path, default=WORKSPACE_ROOT / "recommendations.csv")
    parser.add_argument("--games", type=Path, default=WORKSPACE_ROOT / "games.csv")
    parser.add_argument(
        "--text-prefix", type=Path, default=REPO_ROOT / "text_data" / "emb_text_minilm"
    )
    parser.add_argument(
        "--tabular-prefix",
        type=Path,
        default=REPO_ROOT / "tabular_embedding" / "emb_tabular_svd64",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "mvp")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--debug-users", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_and_validate_ids(args: argparse.Namespace) -> np.ndarray:
    games = pd.read_csv(args.games, usecols=["app_id"])
    text_idx = pd.read_csv(args.text_prefix.with_suffix(".csv"))
    tab_idx = pd.read_csv(args.tabular_prefix.with_suffix(".csv"))
    text_emb = np.load(args.text_prefix.with_suffix(".npy"), mmap_mode="r", allow_pickle=False)
    tab_emb = np.load(args.tabular_prefix.with_suffix(".npy"), mmap_mode="r", allow_pickle=False)

    assert text_emb.shape == (len(text_idx), 384)
    assert tab_emb.shape == (len(tab_idx), 64)
    assert text_idx.app_id.is_unique and tab_idx.app_id.is_unique
    assert np.array_equal(text_idx.row.to_numpy(), np.arange(len(text_idx)))

    valid = np.intersect1d(games.app_id, text_idx.app_id)
    valid = np.intersect1d(valid, tab_idx.app_id)
    assert len(valid) > 0

    text_pos = dict(zip(text_idx.app_id.astype(int), text_idx.row.astype(int)))
    tab_pos = {int(app_id): row for row, app_id in enumerate(tab_idx.app_id)}
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(valid, size=min(20, len(valid)), replace=False)
    for app_id in sample:
        assert int(text_idx.iloc[text_pos[int(app_id)]].app_id) == int(app_id)
        assert int(tab_idx.iloc[tab_pos[int(app_id)]].app_id) == int(app_id)
    print(f"games={games.app_id.nunique():,} text={len(text_idx):,} tabular={len(tab_idx):,}")
    print(f"valid_game_ids={len(valid):,}; random mapping assertions passed={len(sample)}")
    return valid.astype(np.int64)


def main() -> None:
    args = parse_args()
    for path in [args.recommendations, args.games]:
        if not path.exists():
            raise FileNotFoundError(path)
    valid_game_ids = load_and_validate_ids(args)
    total, valid, valid_user_counts = scan_interactions(
        args.recommendations, valid_game_ids, args.chunksize
    )
    selected_users = choose_debug_users(
        valid_user_counts,
        args.min_user_interactions,
        args.debug_users,
        args.seed,
    )
    selected = load_selected_interactions(
        args.recommendations, selected_users, valid_game_ids, args.chunksize
    )
    per_user = selected.groupby("user_id").size()
    assert per_user.ge(args.min_user_interactions).all()
    split = chronological_split(selected)
    assert split.app_id.isin(valid_game_ids).all()
    assert not split.is_recommended.isna().any()
    save_prepared_outputs(
        split,
        total,
        valid,
        args.output_dir,
        {
            "min_user_interactions": args.min_user_interactions,
            "debug_users": args.debug_users,
            "seed": args.seed,
            "chunksize": args.chunksize,
            "valid_game_ids": int(len(valid_game_ids)),
        },
    )
    print("PREPARE_MVP_DATA_OK")


if __name__ == "__main__":
    main()

