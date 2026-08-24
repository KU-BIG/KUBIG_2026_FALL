"""Generate cold-start recommendations from explicit preferences or popularity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.cold_start import ColdStartRecommendationPipeline  # noqa: E402
from mvp_recommendation.reranking import mmr_rerank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-name", default="new_user")
    parser.add_argument("--preferred-tags", nargs="*", default=[])
    parser.add_argument("--liked-app-ids", nargs="*", type=int, default=[])
    parser.add_argument("--exclude-app-ids", nargs="*", type=int, default=[])
    parser.add_argument("--content-weight", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/new_user_recommendations.csv"))
    parser.add_argument(
        "--train-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "mvp_50k" / "data_seed_42" / "debug_train.parquet",
    )
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "text_data" / "games_text_ready.csv")
    parser.add_argument("--text-prefix", type=Path, default=REPO_ROOT / "text_data" / "emb_text_minilm")
    parser.add_argument(
        "--multimodal-prefix", type=Path,
        default=REPO_ROOT / "game_fusion" / "emb_game_concat_64",
    )
    parser.add_argument("--save-available-tags", type=Path, default=None)
    parser.add_argument("--diversity", action="store_true")
    parser.add_argument("--diversity-lambda", type=float, default=0.65)
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = ColdStartRecommendationPipeline(
        text_prefix=args.text_prefix,
        catalog_path=args.catalog,
        train_path=args.train_path,
        multimodal_prefix=args.multimodal_prefix,
    )
    if args.save_available_tags is not None:
        args.save_available_tags.parent.mkdir(parents=True, exist_ok=True)
        pipeline.available_tags().to_csv(args.save_available_tags, index=False, encoding="utf-8-sig")
    pool_k = args.top_k * args.candidate_multiplier if args.diversity else args.top_k
    result = pipeline.recommend(
        profile_name=args.profile_name,
        top_k=pool_k,
        preferred_tags=args.preferred_tags,
        liked_app_ids=args.liked_app_ids,
        exclude_app_ids=args.exclude_app_ids,
        content_weight=args.content_weight,
    )
    if args.diversity:
        result = mmr_rerank(
            result,
            pipeline.text_items,
            pipeline.app_to_row,
            top_k=args.top_k,
            lambda_relevance=args.diversity_lambda,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    manifest = {
        "output": str(args.output),
        "shape": list(result.shape),
        "profile_name": args.profile_name,
        "preferred_tags": args.preferred_tags,
        "liked_app_ids": args.liked_app_ids,
        "exclude_app_ids": args.exclude_app_ids,
        "content_weight_requested": args.content_weight,
        "effective_method": result.cold_start_method.iloc[0],
        "effective_content_weight": float(result.content_weight.iloc[0]),
        "top_k": args.top_k,
        "embedding": "frozen multimodal liked-game profile; MiniLM tag profile",
        "diversity": args.diversity,
        "diversity_lambda": args.diversity_lambda if args.diversity else None,
        "candidate_multiplier": args.candidate_multiplier if args.diversity else None,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(result[["profile_name", "rank", "app_id", "title", "score", "cold_start_method", "recommendation_reason"]].to_string(index=False))
    print(f"saved: {args.output}")
    print(f"manifest: {manifest_path}")
    print("RECOMMEND_NEW_USER_OK")


if __name__ == "__main__":
    main()
