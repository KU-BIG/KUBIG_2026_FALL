"""Generate known-user Top-K recommendations from trained MF/Text BPR models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.inference import (  # noqa: E402
    ALL_MODEL_NAMES,
    KnownUserRecommendationPipeline,
)
from mvp_recommendation.reranking import mmr_rerank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-ids", nargs="+", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--models", nargs="+", choices=ALL_MODEL_NAMES, default=list(ALL_MODEL_NAMES))
    parser.add_argument("--output", type=Path, default=Path("outputs/final_recommendations.csv"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "mvp_50k" / "repro_seed_42" / "checkpoints",
    )
    parser.add_argument(
        "--hybrid-summary",
        type=Path,
        default=REPO_ROOT / "outputs" / "mvp_50k" / "repro_seed_42" / "hybrid" / "hybrid_summary.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "mvp_50k" / "data_seed_42",
    )
    parser.add_argument(
        "--catalog", type=Path, default=REPO_ROOT / "text_data" / "games_text_ready.csv"
    )
    parser.add_argument(
        "--text-prefix", type=Path, default=REPO_ROOT / "text_data" / "emb_text_minilm"
    )
    parser.add_argument(
        "--tabular-prefix",
        type=Path,
        default=REPO_ROOT / "tabular_embedding" / "leakage_safe" / "emb_tabular_safe_svd64",
    )
    parser.add_argument(
        "--multimodal-prefix", type=Path,
        default=REPO_ROOT / "game_fusion" / "emb_game_concat_64",
    )
    parser.add_argument(
        "--multimodal-checkpoint", type=Path,
        default=REPO_ROOT / "recommendation_mvp" / "model_artifacts" / "frozen_multimodal_user_bpr_seed42.pt",
    )
    parser.add_argument(
        "--multimodal-summary", type=Path,
        default=REPO_ROOT / "recommendation_mvp" / "model_artifacts" / "multimodal_evaluation_summary_seed42.json",
    )
    parser.add_argument(
        "--history-scope", choices=["train", "train_validation", "all"], default="all"
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--diversity", action="store_true")
    parser.add_argument("--diversity-lambda", type=float, default=0.65)
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    return parser.parse_args()


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    pipeline = KnownUserRecommendationPipeline(
        checkpoint_dir=args.checkpoint_dir,
        hybrid_summary_path=args.hybrid_summary,
        text_prefix=args.text_prefix,
        tabular_prefix=args.tabular_prefix,
        catalog_path=args.catalog,
        data_dir=args.data_dir,
        device=device,
        batch_size=args.batch_size,
        history_scope=args.history_scope,
        multimodal_prefix=args.multimodal_prefix,
        multimodal_checkpoint=args.multimodal_checkpoint,
        multimodal_summary_path=args.multimodal_summary,
    )
    pool_k = args.top_k * args.candidate_multiplier if args.diversity else args.top_k
    result = pipeline.recommend(args.user_ids, top_k=pool_k, models=args.models)
    if args.diversity:
        result = mmr_rerank(
            result,
            pipeline.text_bank.numpy(),
            pipeline.app_to_row,
            top_k=args.top_k,
            lambda_relevance=args.diversity_lambda,
            group_columns=["user_id", "model"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    manifest = {
        "output": str(args.output),
        "shape": list(result.shape),
        "users": args.user_ids,
        "models": args.models,
        "top_k": args.top_k,
        "history_scope": args.history_scope,
        "hybrid_alpha_mf": pipeline.alpha_mf,
        "hybrid_alpha_text": pipeline.alpha_text,
        "hybrid_alpha_multimodal_mf": pipeline.alpha_multimodal_mf,
        "device": device,
        "diversity": args.diversity,
        "diversity_lambda": args.diversity_lambda if args.diversity else None,
        "candidate_multiplier": args.candidate_multiplier if args.diversity else None,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(result[["user_id", "model", "rank", "app_id", "title", "score", "recommendation_source"]].to_string(index=False))
    print(f"saved: {args.output}")
    print(f"manifest: {manifest_path}")
    print("RECOMMEND_USERS_OK")


if __name__ == "__main__":
    main()
