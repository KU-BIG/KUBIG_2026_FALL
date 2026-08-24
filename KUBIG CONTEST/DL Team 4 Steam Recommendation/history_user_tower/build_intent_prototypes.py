"""Build genre prototypes by averaging the frozen multimodal game vectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from history_user_tower.experiment import REPO_ROOT, load_game_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "Data_process/games_metadata_enriched.parquet")
    parser.add_argument("--game-prefix", type=Path, default=REPO_ROOT / "game_fusion/emb_game_concat_64")
    parser.add_argument("--output-prefix", type=Path,
                        default=REPO_ROOT / "history_user_tower/results_seed_42/genre_prototypes")
    parser.add_argument("--min-games", type=int, default=10)
    return parser.parse_args()


def split_genres(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    ids, bank, app_to_row = load_game_bank(args.game_prefix)
    metadata = pd.read_parquet(args.metadata, columns=["app_id", "genres"])
    if metadata.app_id.duplicated().any():
        raise ValueError("metadata app_id must be unique")
    records = []
    for row in metadata.itertuples(index=False):
        if int(row.app_id) not in app_to_row:
            continue
        records.extend((genre, app_to_row[int(row.app_id)]) for genre in split_genres(row.genres))
    exploded = pd.DataFrame(records, columns=["genre", "game_row"])
    prototype_rows, index_rows = [], []
    for genre, group in exploded.groupby("genre", sort=True):
        if len(group) < args.min_games:
            continue
        vector = bank[group.game_row.to_numpy(np.int64)].mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            continue
        prototype_rows.append((vector / norm).astype(np.float32))
        index_rows.append({"genre": genre, "prototype_row": len(prototype_rows) - 1, "game_count": len(group)})
    prototypes = np.asarray(prototype_rows, dtype=np.float32)
    index = pd.DataFrame(index_rows)
    if prototypes.shape != (len(index), 64) or not np.allclose(np.linalg.norm(prototypes, axis=1), 1, atol=1e-5):
        raise ValueError("invalid genre prototypes")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_prefix.with_suffix(".npy"), prototypes)
    index.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    print(f"genres={len(index)}; source_games={exploded.game_row.nunique()}; shape={prototypes.shape}")
    print("BUILD_INTENT_PROTOTYPES_OK")


if __name__ == "__main__":
    main()
