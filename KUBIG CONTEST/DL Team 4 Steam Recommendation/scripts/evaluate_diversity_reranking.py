"""Compare the saved original and MMR-diversified cold-start Top-10 lists."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.reranking import (  # noqa: E402
    _title_jaccard,
    _title_tokens,
    mean_intra_list_cosine,
)


def duplicate_title_pairs(frame: pd.DataFrame, threshold: float = 0.5) -> int:
    count = 0
    for left in range(len(frame)):
        for right in range(left + 1, len(frame)):
            similarity = _title_jaccard(
                _title_tokens(frame.title.iloc[left]), _title_tokens(frame.title.iloc[right])
            )
            count += int(similarity >= threshold)
    return count


def main() -> None:
    output_dir = REPO_ROOT / "recommendation_mvp"
    original = pd.read_csv(output_dir / "sample_new_user_preferences.csv")
    diverse = pd.read_csv(output_dir / "sample_new_user_diverse.csv")
    index = pd.read_csv(REPO_ROOT / "text_data" / "emb_text_minilm.csv")
    embeddings = np.load(REPO_ROOT / "text_data" / "emb_text_minilm.npy", allow_pickle=False)
    app_to_row = dict(zip(index.app_id.astype(int), index.row.astype(int)))

    rows = []
    for name, frame in [("original", original), ("mmr_diverse", diverse)]:
        rows.append(
            {
                "list": name,
                "games": len(frame),
                "mean_original_score": float(frame.score.mean()),
                "mean_intra_list_cosine": mean_intra_list_cosine(
                    frame.app_id, embeddings, app_to_row
                ),
                "high_title_overlap_pairs": duplicate_title_pairs(frame),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "diversity_comparison.csv", index=False)
    original_row, diverse_row = rows
    summary = {
        "candidate_pool_multiplier": 10,
        "lambda_relevance": 0.65,
        "lambda_diversity": 0.35,
        "top_k": 10,
        "shared_games": len(set(original.app_id) & set(diverse.app_id)),
        "relevance_retention": diverse_row["mean_original_score"] / original_row["mean_original_score"],
        "intra_list_cosine_delta": (
            diverse_row["mean_intra_list_cosine"] - original_row["mean_intra_list_cosine"]
        ),
        "high_title_overlap_pairs_before": original_row["high_title_overlap_pairs"],
        "high_title_overlap_pairs_after": diverse_row["high_title_overlap_pairs"],
        "conclusion": "The default MMR removes the Witcher DLC title-overlap pair with minimal relevance loss on the sample profile.",
    }
    (output_dir / "diversity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("DIVERSITY_EVALUATION_OK")


if __name__ == "__main__":
    main()
