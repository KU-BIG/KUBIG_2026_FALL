"""Tune an MF/Text BPR score hybrid on validation and evaluate once on test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.bpr import BPRRecommender  # noqa: E402
from mvp_recommendation.metrics import one_positive_ranking_metrics, ranking_diagnostics  # noqa: E402
from scripts.train_bpr import candidate_scores, load_banks, pick_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-candidates", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--text-prefix", type=Path, default=REPO_ROOT / "text_data" / "emb_text_minilm"
    )
    parser.add_argument(
        "--tabular-prefix",
        type=Path,
        default=REPO_ROOT / "tabular_embedding" / "leakage_safe" / "emb_tabular_safe_svd64",
    )
    parser.add_argument("--max-validation-positives", type=int, default=10_000)
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument("--candidate-seed", type=int, default=4242)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def build_validation_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    eligible_users: set[int],
    valid_app_ids: np.ndarray,
    max_positives: int,
    num_negatives: int,
    seed: int,
) -> pd.DataFrame:
    """Create fixed validation ranking sets without consulting test history."""
    past = pd.concat([train, validation], ignore_index=True)
    history = past.groupby("user_id").app_id.agg(lambda values: set(map(int, values)))
    positives = validation.loc[
        validation.is_recommended & validation.user_id.isin(eligible_users), ["user_id", "app_id"]
    ].copy()
    if len(positives) > max_positives:
        positives = positives.sample(max_positives, random_state=seed)
    positives = positives.sort_values(["user_id", "app_id"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int, int, int]] = []
    catalog = np.asarray(valid_app_ids, dtype=np.int64)
    for group_id, row in positives.iterrows():
        user_id, positive = int(row.user_id), int(row.app_id)
        blocked = history.loc[user_id]
        negatives: set[int] = set()
        while len(negatives) < num_negatives:
            needed = num_negatives - len(negatives)
            draws = rng.choice(catalog, size=needed * 2, replace=True)
            negatives.update(int(x) for x in draws if int(x) not in blocked)
        chosen = sorted(negatives)[:num_negatives]
        assert positive not in chosen and not (set(chosen) & blocked)
        rows.append((group_id, user_id, positive, 1))
        rows.extend((group_id, user_id, item, 0) for item in chosen)
    result = pd.DataFrame(rows, columns=["group_id", "user_id", "app_id", "target"])
    group_size = num_negatives + 1
    assert result.groupby("group_id").size().eq(group_size).all()
    assert result.groupby("group_id").target.sum().eq(1).all()
    return result


def load_model(
    mode: str,
    checkpoint_dir: Path,
    num_items: int,
    device: torch.device,
) -> tuple[BPRRecommender, dict[int, int]]:
    saved = torch.load(
        checkpoint_dir / f"{mode}_best.pt", map_location=device, weights_only=False
    )
    assert saved["mode"] == mode
    user_to_idx = {int(key): int(value) for key, value in saved["user_to_idx"].items()}
    model = BPRRecommender(len(user_to_idx), num_items, mode, embed_dim=64).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, user_to_idx


def group_standardize(scores: np.ndarray, group_size: int) -> np.ndarray:
    # Use float64 so affine normalization preserves all float32 score ties and
    # strict orderings exactly at the alpha=0/1 single-model endpoints.
    grouped = scores.astype(np.float64).reshape(-1, group_size)
    means = grouped.mean(axis=1, keepdims=True)
    stds = grouped.std(axis=1, keepdims=True)
    standardized = (grouped - means) / np.maximum(stds, 1e-8)
    assert np.isfinite(standardized).all()
    return standardized.reshape(-1)


def hybrid_scores(mf: np.ndarray, text: np.ndarray, alpha_mf: float) -> np.ndarray:
    return alpha_mf * mf + (1.0 - alpha_mf) * text


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index, app_to_row, tab_bank, text_bank = load_banks(args.text_prefix, args.tabular_prefix)
    mf_model, mf_users = load_model("mf_bpr", args.checkpoint_dir, len(index), device)
    text_model, text_users = load_model("text_bpr", args.checkpoint_dir, len(index), device)
    assert mf_users == text_users, "MF/Text checkpoints use different user mappings"

    train = pd.read_parquet(args.data_dir / "debug_train.parquet")
    validation = pd.read_parquet(args.data_dir / "debug_validation.parquet")
    eligible_users = set(mf_users)
    valid_app_ids = index.app_id.to_numpy(np.int64)
    validation_candidates = build_validation_candidates(
        train,
        validation,
        eligible_users,
        valid_app_ids,
        args.max_validation_positives,
        args.num_negatives,
        args.candidate_seed,
    )
    validation_path = args.output_dir / "validation_ranking_candidates.parquet"
    validation_candidates.to_parquet(validation_path, index=False)

    test_candidates = pd.read_parquet(args.test_candidates)
    test_candidates = test_candidates[test_candidates.user_id.isin(eligible_users)].copy()
    group_size = args.num_negatives + 1
    assert test_candidates.groupby("group_id").size().eq(group_size).all()
    assert test_candidates.groupby("group_id").target.sum().eq(1).all()

    def score_pair(candidates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mf = candidate_scores(
            mf_model, candidates, mf_users, app_to_row, tab_bank, text_bank, args.batch_size, device
        )
        text = candidate_scores(
            text_model, candidates, text_users, app_to_row, tab_bank, text_bank, args.batch_size, device
        )
        return group_standardize(mf, group_size), group_standardize(text, group_size)

    print(
        f"device={device}; validation queries={validation_candidates.group_id.nunique():,}; "
        f"test queries={test_candidates.group_id.nunique():,}"
    )
    val_mf, val_text = score_pair(validation_candidates)
    train_positive_counts = train.loc[train.is_recommended].groupby("app_id").size()
    grid_rows = []
    for alpha in np.linspace(0.0, 1.0, 21):
        score = hybrid_scores(val_mf, val_text, float(alpha))
        metrics = one_positive_ranking_metrics(score, group_size)
        diagnostics = ranking_diagnostics(
            validation_candidates, score, train_positive_counts, valid_catalog_size=len(index), k=10
        )
        grid_rows.append(
            {"alpha_mf": float(alpha), "alpha_text": float(1 - alpha), **metrics, **diagnostics}
        )
    grid = pd.DataFrame(grid_rows).sort_values(
        ["ndcg@10", "recall@10", "mrr", "alpha_mf"], ascending=[False, False, False, True]
    )
    best = grid.iloc[0]
    alpha = float(best.alpha_mf)
    # A second, predeclared objective keeps at least 99.5% of the best
    # validation NDCG and maximizes long-tail recall inside that constraint.
    ndcg_floor = float(best["ndcg@10"]) * 0.995
    balanced_pool = grid[grid["ndcg@10"].ge(ndcg_floor)].sort_values(
        ["long_tail_recall@10", "catalog_coverage@10", "ndcg@10", "alpha_mf"],
        ascending=[False, False, False, True],
    )
    balanced_alpha = float(balanced_pool.iloc[0].alpha_mf)
    grid.sort_values("alpha_mf").to_csv(args.output_dir / "validation_alpha_search.csv", index=False)

    test_mf, test_text = score_pair(test_candidates)
    test_rows = []
    for name, score, value in [
        ("text_bpr_rescored", test_text, 0.0),
        ("mf_text_balanced_hybrid", hybrid_scores(test_mf, test_text, balanced_alpha), balanced_alpha),
        ("mf_text_accuracy_hybrid", hybrid_scores(test_mf, test_text, alpha), alpha),
        ("mf_bpr_rescored", test_mf, 1.0),
    ]:
        test_rows.append(
            {
                "model": name,
                "alpha_mf": value,
                "alpha_text": 1.0 - value,
                **one_positive_ranking_metrics(score, group_size),
                **ranking_diagnostics(
                    test_candidates, score, train_positive_counts, valid_catalog_size=len(index), k=10
                ),
            }
        )
    results = pd.DataFrame(test_rows)
    results.to_csv(args.output_dir / "hybrid_test_results.csv", index=False)

    ordered = grid.sort_values("alpha_mf")
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(ordered.alpha_mf, ordered["recall@10"], marker="o", label="Recall@10")
    plt.plot(ordered.alpha_mf, ordered["ndcg@10"], marker="o", label="NDCG@10")
    plt.axvline(alpha, color="black", linestyle="--", label=f"selected α={alpha:.2f}")
    plt.axvline(
        balanced_alpha, color="#59a14f", linestyle=":", label=f"balanced α={balanced_alpha:.2f}"
    )
    plt.xlabel("MF score weight α (Text weight = 1 - α)")
    plt.ylabel("Validation metric")
    plt.title("Leakage-free MF + Text hybrid weight search")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "validation_alpha_search.png", dpi=160)
    plt.close()

    summary = {
        "selection_data": "validation only",
        "test_used_for_alpha_selection": False,
        "score_normalization": "within-query z-score per modality",
        "selected_alpha_mf": alpha,
        "selected_alpha_text": 1.0 - alpha,
        "validation_selection_metric": "ndcg@10, then recall@10, then mrr",
        "balanced_alpha_mf": balanced_alpha,
        "balanced_alpha_text": 1.0 - balanced_alpha,
        "balanced_selection_rule": "maximize validation long-tail Recall@10 while retaining >=99.5% of best validation NDCG@10",
        "balanced_ndcg_floor": ndcg_floor,
        "validation_queries": int(validation_candidates.group_id.nunique()),
        "test_queries": int(test_candidates.group_id.nunique()),
        "test_results": results.to_dict(orient="records"),
    }
    (args.output_dir / "hybrid_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"accuracy alpha: MF={alpha:.2f}, Text={1-alpha:.2f}")
    print(f"balanced alpha: MF={balanced_alpha:.2f}, Text={1-balanced_alpha:.2f}")
    print(results[["model", "recall@10", "ndcg@10", "mrr", "long_tail_recall@10", "catalog_coverage@10"]].to_string(index=False))
    print("MF_TEXT_HYBRID_OK")


if __name__ == "__main__":
    main()
