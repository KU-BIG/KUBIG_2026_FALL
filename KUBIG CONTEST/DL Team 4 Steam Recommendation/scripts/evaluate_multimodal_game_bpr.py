"""Train/evaluate user towers against fixed multimodal game embeddings.

This is the downstream test missing from ``game_fusion``: all game banks use
the same real Steam interactions, dynamic train negatives, validation pairs,
and fixed 100-item test candidate groups as the existing MF/Text BPR models.
Model/bank selection and MF hybrid weights use validation data only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.bpr import (  # noqa: E402
    BPRRecommender,
    DynamicNegativeSampler,
    FixedGameEmbeddingBPR,
    bpr_loss,
)
from mvp_recommendation.metrics import (  # noqa: E402
    one_positive_ranking_metrics,
    ranking_diagnostics,
)
from scripts.evaluate_mf_text_hybrid import (  # noqa: E402
    build_validation_candidates,
    group_standardize,
)
from scripts.train_bpr import load_banks, pick_device  # noqa: E402


DEFAULT_BANKS = [
    "frozen_concat=game_fusion/emb_game_concat_64",
    "synthetic_fusion_tuned=game_fusion/emb_game_finetuned_64",
    "synthetic_partial_adapter=game_fusion/emb_game_partial_fusion_tuned_64",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--banks", nargs="+", default=DEFAULT_BANKS,
                        help="Entries formatted name=path_prefix")
    parser.add_argument("--mf-checkpoint", type=Path, default=None)
    parser.add_argument("--text-prefix", type=Path,
                        default=REPO_ROOT / "text_data" / "emb_text_minilm")
    parser.add_argument("--tabular-prefix", type=Path,
                        default=REPO_ROOT / "tabular_embedding" / "leakage_safe" / "emb_tabular_safe_svd64")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-seed", type=int, default=4242)
    parser.add_argument("--max-validation-positives", type=int, default=10_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_prefix(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"bank must use name=prefix format: {raw}")
    name, value = raw.split("=", 1)
    prefix = Path(value)
    if not prefix.is_absolute():
        prefix = REPO_ROOT / prefix
    return name.strip(), prefix


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixed_bank(prefix: Path, canonical_ids: np.ndarray) -> tuple[torch.Tensor, dict[str, object]]:
    npy_path, csv_path = prefix.with_suffix(".npy"), prefix.with_suffix(".csv")
    if not npy_path.exists() or not csv_path.exists():
        raise FileNotFoundError(f"missing bank pair: {npy_path}, {csv_path}")
    ids = pd.read_csv(csv_path)["app_id"].to_numpy(np.int64)
    bank_np = np.load(npy_path, allow_pickle=False).astype(np.float32)
    assert bank_np.shape == (len(canonical_ids), 64), bank_np.shape
    assert np.array_equal(ids, canonical_ids), "game bank app_id order differs from canonical index"
    assert np.isfinite(bank_np).all()
    norms = np.linalg.norm(bank_np, axis=1)
    metadata = {
        "prefix": str(prefix.relative_to(REPO_ROOT)),
        "shape": list(bank_np.shape),
        "sha256": fingerprint(npy_path),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
    }
    return torch.from_numpy(bank_np), metadata


def build_training_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    app_to_row: np.ndarray,
    seed: int,
) -> tuple[dict[int, int], np.ndarray, np.ndarray, DynamicNegativeSampler,
           np.ndarray, np.ndarray]:
    positives = train.loc[train.is_recommended, ["user_id", "app_id"]]
    users = np.sort(positives.user_id.unique())
    user_to_idx = {int(user): i for i, user in enumerate(users)}
    train_users = positives.user_id.map(user_to_idx).to_numpy(np.int64)
    train_rows = app_to_row[positives.app_id.to_numpy(np.int64)]
    assert (train_rows >= 0).all()

    observed = train[train.user_id.isin(user_to_idx)]
    sampler = DynamicNegativeSampler(
        len((app_to_row >= 0).nonzero()[0]),
        observed.user_id.map(user_to_idx).to_numpy(np.int64),
        app_to_row[observed.app_id.to_numpy(np.int64)],
        seed,
    )
    validation_eligible = validation[validation.user_id.isin(user_to_idx)]
    past = pd.concat(
        [observed[["user_id", "app_id"]], validation_eligible[["user_id", "app_id"]]],
        ignore_index=True,
    ).drop_duplicates()
    val_sampler = DynamicNegativeSampler(
        len((app_to_row >= 0).nonzero()[0]),
        past.user_id.map(user_to_idx).to_numpy(np.int64),
        app_to_row[past.app_id.to_numpy(np.int64)],
        seed + 10_000,
    )
    val_positive = validation_eligible[validation_eligible.is_recommended]
    val_users = val_positive.user_id.map(user_to_idx).to_numpy(np.int64)
    val_positive_rows = app_to_row[val_positive.app_id.to_numpy(np.int64)]
    val_pairs = np.column_stack([val_positive_rows, val_sampler.sample(val_users)])
    return user_to_idx, train_users, train_rows, sampler, val_users, val_pairs


def pair_epoch(
    model: FixedGameEmbeddingBPR,
    users: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(users), torch.from_numpy(positives), torch.from_numpy(negatives)),
        batch_size=batch_size,
        shuffle=optimizer is not None,
    )
    model.train(optimizer is not None)
    total_loss = 0.0
    total_rows = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for user, positive, negative in loader:
            user, positive, negative = user.to(device), positive.to(device), negative.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            pos_score, neg_score = model.pair_scores(user, positive, negative)
            loss = bpr_loss(pos_score, neg_score)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += float(loss.item()) * len(user)
            total_rows += len(user)
    return total_loss / total_rows


@torch.no_grad()
def fixed_scores(
    model: FixedGameEmbeddingBPR,
    frame: pd.DataFrame,
    user_to_idx: dict[int, int],
    app_to_row: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    users = torch.from_numpy(frame.user_id.map(user_to_idx).to_numpy(np.int64))
    rows = torch.from_numpy(app_to_row[frame.app_id.to_numpy(np.int64)])
    assert (rows.numpy() >= 0).all()
    output = []
    model.eval()
    for start in range(0, len(frame), batch_size):
        output.append(model.score(
            users[start:start + batch_size].to(device),
            rows[start:start + batch_size].to(device),
        ).cpu().numpy())
    scores = np.concatenate(output)
    assert np.isfinite(scores).all()
    return scores


@torch.no_grad()
def mf_scores(
    model: BPRRecommender,
    frame: pd.DataFrame,
    user_to_idx: dict[int, int],
    app_to_row: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    users = torch.from_numpy(frame.user_id.map(user_to_idx).to_numpy(np.int64))
    rows = torch.from_numpy(app_to_row[frame.app_id.to_numpy(np.int64)])
    output = []
    model.eval()
    for start in range(0, len(frame), batch_size):
        u = users[start:start + batch_size].to(device)
        r = rows[start:start + batch_size].to(device)
        dummy = torch.empty((len(u), 1), device=device)
        output.append(model.score(u, r, dummy, dummy).cpu().numpy())
    return np.concatenate(output)


def metrics_row(
    name: str,
    frame: pd.DataFrame,
    scores: np.ndarray,
    train_counts: pd.Series,
    catalog_size: int,
) -> dict[str, object]:
    group_size = int(frame.groupby("group_id").size().iloc[0])
    assert frame.groupby("group_id").size().eq(group_size).all()
    return {
        "model": name,
        **one_positive_ranking_metrics(scores, group_size),
        **ranking_diagnostics(frame, scores, train_counts, catalog_size, k=10),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    canonical, app_to_row, _, _ = load_banks(args.text_prefix, args.tabular_prefix)
    canonical_ids = canonical.app_id.to_numpy(np.int64)
    banks = dict(resolve_prefix(value) for value in args.banks)
    loaded = {name: load_fixed_bank(prefix, canonical_ids) for name, prefix in banks.items()}
    train = pd.read_parquet(args.data_dir / "debug_train.parquet")
    validation = pd.read_parquet(args.data_dir / "debug_validation.parquet")
    test_candidates = pd.read_parquet(args.test_candidates)
    user_to_idx, train_users, train_rows, sampler, val_users, val_pairs = build_training_data(
        train, validation, app_to_row, args.seed
    )
    test_candidates = test_candidates[test_candidates.user_id.isin(user_to_idx)].copy()
    assert test_candidates.groupby("group_id").size().eq(100).all()
    validation_candidates = build_validation_candidates(
        train, validation, set(user_to_idx), canonical_ids,
        args.max_validation_positives, 99, args.candidate_seed,
    )
    validation_candidates.to_parquet(args.output_dir / "validation_candidates.parquet", index=False)
    train_counts = train.loc[train.is_recommended].groupby("app_id").size()
    print(f"device={device}; users={len(user_to_idx):,}; train positives={len(train_rows):,}; "
          f"validation queries={validation_candidates.group_id.nunique():,}; "
          f"test queries={test_candidates.group_id.nunique():,}")

    val_score_by_name: dict[str, np.ndarray] = {}
    test_score_by_name: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    bank_metadata: dict[str, dict[str, object]] = {}
    for name, (bank, metadata) in loaded.items():
        seed_everything(args.seed)
        sampler.reset(args.seed)
        model = FixedGameEmbeddingBPR(len(user_to_idx), bank).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss, stale = float("inf"), 0
        history: list[dict[str, float]] = []
        checkpoint = checkpoint_dir / f"{name}_best.pt"
        print(f"\n[{name}] trainable parameters={sum(p.numel() for p in model.parameters()):,}")
        for epoch in range(1, args.epochs + 1):
            negatives = sampler.sample(train_users)
            train_loss = pair_epoch(model, train_users, train_rows, negatives,
                                    args.batch_size, device, optimizer)
            val_loss = pair_epoch(model, val_users, val_pairs[:, 0], val_pairs[:, 1],
                                  args.batch_size, device, None)
            history.append({"epoch": epoch, "train_bpr_loss": train_loss, "val_bpr_loss": val_loss})
            print(f"epoch={epoch} train_bpr={train_loss:.6f} val_bpr={val_loss:.6f}")
            if val_loss < best_loss - 1e-5:
                best_loss, stale = val_loss, 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "user_to_idx": user_to_idx,
                    "mode": "fixed_game_bpr",
                    "bank_name": name,
                    "bank_metadata": metadata,
                    "seed": args.seed,
                }, checkpoint)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stopping at epoch {epoch}")
                    break
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        val_scores = fixed_scores(model, validation_candidates, user_to_idx, app_to_row,
                                  args.batch_size, device)
        test_scores = fixed_scores(model, test_candidates, user_to_idx, app_to_row,
                                   args.batch_size, device)
        val_score_by_name[name], test_score_by_name[name] = val_scores, test_scores
        row = {
            **metrics_row(name, test_candidates, test_scores, train_counts, len(canonical)),
            "best_val_bpr_loss": best_loss,
            "validation_ndcg@10": one_positive_ranking_metrics(val_scores, 100)["ndcg@10"],
            "validation_recall@10": one_positive_ranking_metrics(val_scores, 100)["recall@10"],
        }
        rows.append(row)
        histories[name] = history
        bank_metadata[name] = metadata
        pd.DataFrame(history).to_csv(args.output_dir / f"{name}_history.csv", index=False)
        print(pd.Series(row).to_string())

    # Selection never examines test metrics.
    comparison = pd.DataFrame(rows)
    comparison["selected_by_validation"] = False
    selected_idx = comparison.sort_values(
        ["validation_ndcg@10", "validation_recall@10"], ascending=False
    ).index[0]
    comparison.loc[selected_idx, "selected_by_validation"] = True
    selected = str(comparison.loc[selected_idx, "model"])
    comparison.to_csv(args.output_dir / "multimodal_bpr_comparison.csv", index=False)

    hybrid_rows: list[dict[str, object]] = []
    alpha_grid: list[dict[str, float]] = []
    if args.mf_checkpoint is not None:
        saved_mf = torch.load(args.mf_checkpoint, map_location=device, weights_only=False)
        mf_users = {int(k): int(v) for k, v in saved_mf["user_to_idx"].items()}
        assert mf_users == user_to_idx, "MF and multimodal user mappings differ"
        mf_model = BPRRecommender(len(user_to_idx), len(canonical), "mf_bpr", 64).to(device)
        mf_model.load_state_dict(saved_mf["model_state_dict"])
        val_mf = group_standardize(
            mf_scores(mf_model, validation_candidates, user_to_idx, app_to_row, args.batch_size, device), 100
        )
        test_mf = group_standardize(
            mf_scores(mf_model, test_candidates, user_to_idx, app_to_row, args.batch_size, device), 100
        )
        val_multi = group_standardize(val_score_by_name[selected], 100)
        test_multi = group_standardize(test_score_by_name[selected], 100)
        for alpha in np.linspace(0.0, 1.0, 21):
            score = float(alpha) * val_mf + (1.0 - float(alpha)) * val_multi
            alpha_grid.append({"alpha_mf": float(alpha), "alpha_multimodal": 1.0 - float(alpha),
                               **metrics_row("validation", validation_candidates, score,
                                             train_counts, len(canonical))})
        grid = pd.DataFrame(alpha_grid).sort_values(
            ["ndcg@10", "recall@10", "mrr", "alpha_mf"], ascending=[False, False, False, True]
        )
        best_alpha = float(grid.iloc[0].alpha_mf)
        ndcg_floor = float(grid.iloc[0]["ndcg@10"]) * 0.995
        balanced = grid[grid["ndcg@10"].ge(ndcg_floor)].sort_values(
            ["long_tail_recall@10", "catalog_coverage@10", "ndcg@10", "alpha_mf"],
            ascending=[False, False, False, True],
        )
        balanced_alpha = float(balanced.iloc[0].alpha_mf)
        grid.sort_values("alpha_mf").to_csv(args.output_dir / "validation_hybrid_alpha_search.csv", index=False)
        for label, alpha in [("multimodal_only", 0.0),
                             ("mf_multimodal_balanced", balanced_alpha),
                             ("mf_multimodal_accuracy", best_alpha),
                             ("mf_only", 1.0)]:
            scores = alpha * test_mf + (1.0 - alpha) * test_multi
            hybrid_rows.append({"alpha_mf": alpha, "alpha_multimodal": 1.0 - alpha,
                                **metrics_row(label, test_candidates, scores,
                                              train_counts, len(canonical))})
        pd.DataFrame(hybrid_rows).to_csv(args.output_dir / "mf_multimodal_hybrid_test.csv", index=False)
    else:
        best_alpha = balanced_alpha = None

    summary = {
        "seed": args.seed,
        "selection_data": "validation only",
        "test_used_for_selection": False,
        "real_interaction_training": True,
        "game_embeddings_frozen": True,
        "selected_bank": selected,
        "selected_validation_ndcg@10": float(comparison.loc[selected_idx, "validation_ndcg@10"]),
        "selected_test_metrics": comparison.loc[selected_idx].to_dict(),
        "bank_metadata": bank_metadata,
        "hybrid": {
            "selected_accuracy_alpha_mf": best_alpha,
            "selected_balanced_alpha_mf": balanced_alpha,
            "test_results": hybrid_rows,
        },
    }
    (args.output_dir / "multimodal_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("\nSelected by validation:", selected)
    print(comparison[["model", "validation_ndcg@10", "recall@10", "ndcg@10",
                      "long_tail_recall@10", "catalog_coverage@10",
                      "selected_by_validation"]].to_string(index=False))
    if hybrid_rows:
        print("\nMF + selected multimodal:")
        print(pd.DataFrame(hybrid_rows)[["model", "alpha_mf", "recall@10", "ndcg@10",
                                         "long_tail_recall@10", "catalog_coverage@10"]].to_string(index=False))
    print("MULTIMODAL_GAME_BPR_OK")


if __name__ == "__main__":
    main()
