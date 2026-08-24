"""Train and evaluate a history-based User Tower against a frozen game bank.

No recommendation.csv is used. Inputs are the chronological interaction
splits and the final 64D multimodal game embedding bank.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from history_user_tower.model import HistoryUserTower, bpr_loss  # noqa: E402
from mvp_recommendation.bpr import DynamicNegativeSampler  # noqa: E402


@dataclass
class ProfileBank:
    users: np.ndarray
    simple: np.ndarray
    hours: np.ndarray

    def mapping(self) -> dict[int, int]:
        return {int(user): row for row, user in enumerate(self.users)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "outputs/mvp_50k/data_seed_42")
    parser.add_argument("--game-prefix", type=Path, default=REPO_ROOT / "game_fusion/emb_game_concat_64")
    parser.add_argument("--test-candidates", type=Path,
                        default=REPO_ROOT / "outputs/mvp_50k/models_seed_42/ranking_candidates.parquet")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "history_user_tower/results_seed_42")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--full-catalog", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(raw)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)


def load_game_bank(prefix: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    ids = pd.read_csv(prefix.with_suffix(".csv"))["app_id"].to_numpy(np.int64)
    bank = np.load(prefix.with_suffix(".npy"), allow_pickle=False).astype(np.float32)
    if bank.shape != (len(ids), 64):
        raise ValueError(f"expected {(len(ids), 64)} game bank, got {bank.shape}")
    if len(np.unique(ids)) != len(ids) or not np.isfinite(bank).all():
        raise ValueError("invalid game bank IDs or values")
    norms = np.linalg.norm(bank, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            f"game bank must be L2-normalized; norm range={norms.min():.6f}..{norms.max():.6f}"
        )
    return ids, bank, {int(app): row for row, app in enumerate(ids)}


def validate_split_chronology(train: pd.DataFrame, validation: pd.DataFrame,
                              test: pd.DataFrame) -> dict[str, int | str]:
    """Assert user-wise (date, review_id) chronology across split boundaries."""
    result: dict[str, int | str] = {"split_method": "per-user chronological 80/10/10"}
    for left, left_name, right, right_name in (
        (train, "train", validation, "validation"),
        (validation, "validation", test, "test"),
    ):
        last = left.sort_values(["user_id", "date", "review_id"]).groupby("user_id").tail(1).set_index("user_id")
        first = right.sort_values(["user_id", "date", "review_id"]).groupby("user_id").head(1).set_index("user_id")
        common = last.index.intersection(first.index)
        violations = (
            (last.loc[common, "date"] > first.loc[common, "date"])
            | (
                (last.loc[common, "date"] == first.loc[common, "date"])
                & (last.loc[common, "review_id"] > first.loc[common, "review_id"])
            )
        )
        count = int(violations.sum())
        if count:
            raise ValueError(f"{left_name}->{right_name} chronology violations: {count}")
        result[f"{left_name}_to_{right_name}_users_checked"] = int(len(common))
        result[f"{left_name}_to_{right_name}_violations"] = count
    return result


def add_item_rows(frame: pd.DataFrame, app_to_row: dict[int, int]) -> pd.DataFrame:
    output = frame.copy()
    output["item_row"] = output.app_id.map(app_to_row)
    missing = int(output.item_row.isna().sum())
    if missing:
        raise ValueError(f"{missing:,} interactions have no game-bank row")
    output["item_row"] = output.item_row.astype(np.int64)
    return output


def history_weight(hours: float) -> float:
    return float(np.log1p(max(float(hours), 0.0)))


def build_prefix_examples(positive_train: pd.DataFrame, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create (prior-history, user, next-positive) without target leakage."""
    ordered = positive_train.sort_values(["user_id", "date", "review_id"], kind="stable")
    user_values = ordered.user_id.to_numpy(np.int64)
    item_rows = ordered.item_row.to_numpy(np.int64)
    vectors = bank[item_rows]
    weights = np.log1p(ordered.hours.clip(lower=0).to_numpy(np.float32))
    row_index = np.arange(len(ordered), dtype=np.int64)
    starts = np.r_[True, user_values[1:] != user_values[:-1]]
    start_index = np.maximum.accumulate(np.where(starts, row_index, 0))
    seen_count = row_index - start_index

    # Global cumulative sums minus the value just before each group give the
    # group-local prefix. Shift by one so the current target is never included.
    cumulative_simple = np.cumsum(vectors, axis=0, dtype=np.float32)
    prior_simple_global = np.vstack([np.zeros((1, bank.shape[1]), np.float32), cumulative_simple[:-1]])
    prior_simple = prior_simple_global - prior_simple_global[start_index]
    cumulative_weighted = np.cumsum(vectors * weights[:, None], axis=0, dtype=np.float32)
    prior_weighted_global = np.vstack([np.zeros((1, bank.shape[1]), np.float32), cumulative_weighted[:-1]])
    prior_weighted = prior_weighted_global - prior_weighted_global[start_index]
    cumulative_weights = np.cumsum(weights, dtype=np.float32)
    prior_weight_global = np.r_[np.float32(0), cumulative_weights[:-1]]
    prior_weight = prior_weight_global - prior_weight_global[start_index]

    eligible = seen_count > 0
    histories = prior_simple[eligible] / seen_count[eligible, None]
    weighted_ok = prior_weight[eligible] > 0
    histories[weighted_ok] = prior_weighted[eligible][weighted_ok] / prior_weight[eligible][weighted_ok, None]
    norms = np.linalg.norm(histories, axis=1, keepdims=True)
    histories = (histories / np.maximum(norms, 1e-12)).astype(np.float32)
    assert np.isfinite(histories).all()
    return histories, user_values[eligible], item_rows[eligible]


def build_profiles(positive_history: pd.DataFrame, bank: np.ndarray) -> ProfileBank:
    """Build one simple-mean and hours-weighted profile per user."""
    users, inverse = np.unique(positive_history.user_id.to_numpy(np.int64), return_inverse=True)
    vectors = bank[positive_history.item_row.to_numpy(np.int64)]
    weights = np.log1p(positive_history.hours.clip(lower=0).to_numpy(np.float32))
    simple_sum = np.zeros((len(users), bank.shape[1]), dtype=np.float32)
    weighted_sum = np.zeros_like(simple_sum)
    counts = np.zeros(len(users), dtype=np.float32)
    weight_sums = np.zeros(len(users), dtype=np.float32)
    np.add.at(simple_sum, inverse, vectors)
    np.add.at(weighted_sum, inverse, vectors * weights[:, None])
    np.add.at(counts, inverse, 1)
    np.add.at(weight_sums, inverse, weights)
    simple = simple_sum / counts[:, None]
    weighted = simple.copy()
    has_weight = weight_sums > 0
    weighted[has_weight] = weighted_sum[has_weight] / weight_sums[has_weight, None]
    simple /= np.maximum(np.linalg.norm(simple, axis=1, keepdims=True), 1e-12)
    weighted /= np.maximum(np.linalg.norm(weighted, axis=1, keepdims=True), 1e-12)
    return ProfileBank(users, simple.astype(np.float32), weighted.astype(np.float32))


def make_sampler(train: pd.DataFrame, users: np.ndarray, num_items: int, seed: int) -> tuple[DynamicNegativeSampler, dict[int, int]]:
    user_to_idx = {int(user): row for row, user in enumerate(users)}
    observed = train[train.user_id.isin(user_to_idx)]
    sampler = DynamicNegativeSampler(
        num_items,
        observed.user_id.map(user_to_idx).to_numpy(np.int64),
        observed.item_row.to_numpy(np.int64),
        seed,
    )
    return sampler, user_to_idx


def build_validation_pairs(validation: pd.DataFrame, profiles: ProfileBank, bank: np.ndarray,
                           train: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profile_map = profiles.mapping()
    positive = validation[validation.is_recommended & validation.user_id.isin(profile_map)].copy()
    histories = profiles.hours[positive.user_id.map(profile_map).to_numpy(np.int64)]
    targets = positive.item_row.to_numpy(np.int64)
    users = profiles.users
    user_to_idx = {int(user): row for row, user in enumerate(users)}
    past = pd.concat([train, validation], ignore_index=True).drop_duplicates(["user_id", "item_row"])
    observed = past[past.user_id.isin(user_to_idx)]
    sampler = DynamicNegativeSampler(
        len(bank), observed.user_id.map(user_to_idx).to_numpy(np.int64),
        observed.item_row.to_numpy(np.int64), seed,
    )
    negative = sampler.sample(positive.user_id.map(user_to_idx).to_numpy(np.int64))
    return histories, targets, negative


def run_epoch(model: HistoryUserTower, histories: np.ndarray, positives: np.ndarray,
              negatives: np.ndarray, game_bank: torch.Tensor, batch_size: int,
              device: torch.device, optimizer: torch.optim.Optimizer | None) -> float:
    model.train(optimizer is not None)
    total, rows = 0.0, 0
    order = np.random.permutation(len(histories)) if optimizer is not None else np.arange(len(histories))
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            history = torch.from_numpy(histories[index]).to(device)
            pos = torch.from_numpy(positives[index]).to(device)
            neg = torch.from_numpy(negatives[index]).to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            user = model(history)
            pos_score = (user * game_bank[pos]).sum(dim=1)
            neg_score = (user * game_bank[neg]).sum(dim=1)
            loss = bpr_loss(pos_score, neg_score)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total += float(loss.detach()) * len(history)
            rows += len(history)
    return total / rows


@torch.no_grad()
def encode_profiles(model: HistoryUserTower, profiles: np.ndarray, batch_size: int,
                    device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(profiles), batch_size):
        batch = torch.from_numpy(profiles[start:start + batch_size]).to(device)
        output.append(model(batch).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def candidate_scores(candidates: pd.DataFrame, profile_users: np.ndarray, profiles: np.ndarray,
                     bank: np.ndarray, app_to_row: dict[int, int]) -> tuple[pd.DataFrame, np.ndarray]:
    user_to_profile = {int(user): row for row, user in enumerate(profile_users)}
    kept_groups = candidates.loc[candidates.user_id.isin(user_to_profile), "group_id"].unique()
    frame = candidates[candidates.group_id.isin(kept_groups)].sort_values(["group_id"], kind="stable").copy()
    if not frame.groupby("group_id").size().eq(100).all():
        raise ValueError("candidate groups must contain 100 rows")
    if not frame.groupby("group_id", sort=False).first().target.eq(1).all():
        raise ValueError("candidate 0 must be the positive")
    user_rows = frame.user_id.map(user_to_profile).to_numpy(np.int64)
    item_rows = frame.app_id.map(app_to_row).to_numpy(np.int64)
    scores = np.einsum("ij,ij->i", profiles[user_rows], bank[item_rows]).astype(np.float32)
    return frame, scores


def metrics_from_ranks(ranks: np.ndarray, prefix: str = "") -> dict[str, float | int]:
    result: dict[str, float | int] = {f"{prefix}queries": int(len(ranks))}
    for k in (5, 10, 20):
        result[f"{prefix}recall@{k}"] = float((ranks <= k).mean()) if len(ranks) else float("nan")
        result[f"{prefix}ndcg@{k}"] = float(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0).mean()) if len(ranks) else float("nan")
    result[f"{prefix}mrr"] = float((1 / ranks).mean()) if len(ranks) else float("nan")
    return result


def sampled_metrics(frame: pd.DataFrame, scores: np.ndarray, train_items: set[int],
                    catalog_size: int) -> dict[str, float | int]:
    grouped = scores.reshape(-1, 100)
    ranks = 1 + (grouped[:, 1:] > grouped[:, :1]).sum(axis=1)
    positives = frame.groupby("group_id", sort=False).first().reset_index()
    cold = ~positives.app_id.isin(train_items).to_numpy()
    top_rows = np.argpartition(-grouped, kth=9, axis=1)[:, :10]
    item_matrix = frame.app_id.to_numpy(np.int64).reshape(-1, 100)
    unique_top = np.unique(np.take_along_axis(item_matrix, top_rows, axis=1))
    return {
        **metrics_from_ranks(ranks),
        **metrics_from_ranks(ranks[~cold], "warm_"),
        **metrics_from_ranks(ranks[cold], "cold_"),
        "catalog_coverage@10": float(len(unique_top) / catalog_size),
        "unique_recommended@10": int(len(unique_top)),
    }


def full_catalog_metrics(query_frame: pd.DataFrame, profile_users: np.ndarray, profiles: np.ndarray,
                         bank: np.ndarray, train_items: set[int], seen: pd.DataFrame,
                         train_positive_counts: pd.Series, batch_size: int = 128) -> dict[str, float | int]:
    """Exact ranks against the whole catalog, excluding each user's past items."""
    user_to_profile = {int(user): row for row, user in enumerate(profile_users)}
    queries = query_frame[query_frame.user_id.isin(user_to_profile)].copy()
    seen_by_user = {int(u): g.item_row.to_numpy(np.int64) for u, g in seen.groupby("user_id")}
    ranks, cold_flags, top_items = [], [], []
    query_rows = list(queries.itertuples(index=False))
    for start in range(0, len(query_rows), batch_size):
        batch = query_rows[start:start + batch_size]
        user_vectors = profiles[[user_to_profile[int(row.user_id)] for row in batch]]
        scores = user_vectors @ bank.T
        target_rows = np.asarray([int(row.item_row) for row in batch], dtype=np.int64)
        target_scores = scores[np.arange(len(batch)), target_rows].copy()
        for local_row, row in enumerate(batch):
            blocked = seen_by_user.get(int(row.user_id))
            if blocked is not None:
                scores[local_row, blocked] = -np.inf
        scores[np.arange(len(batch)), target_rows] = target_scores
        ranks.extend((1 + (scores > target_scores[:, None]).sum(axis=1)).tolist())
        cold_flags.extend(int(row.app_id) not in train_items for row in batch)
        top_items.extend(np.argpartition(-scores, kth=9, axis=1)[:, :10].ravel().tolist())
    ranks_np = np.asarray(ranks, dtype=np.int64)
    cold_np = np.asarray(cold_flags, dtype=bool)
    top_rows = np.asarray(top_items, dtype=np.int64)
    popularity = train_positive_counts.reindex(query_frame.attrs["catalog_ids"][top_rows], fill_value=0).to_numpy()
    return {
        **metrics_from_ranks(ranks_np),
        **metrics_from_ranks(ranks_np[~cold_np], "warm_"),
        **metrics_from_ranks(ranks_np[cold_np], "cold_"),
        "catalog_coverage@10": float(len(np.unique(top_rows)) / len(bank)),
        "unique_recommended@10": int(len(np.unique(top_rows))),
        "mean_train_positive_popularity@10": float(popularity.mean()),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_ids, bank, app_to_row = load_game_bank(args.game_prefix)
    train = add_item_rows(pd.read_parquet(args.data_dir / "debug_train.parquet"), app_to_row)
    validation = add_item_rows(pd.read_parquet(args.data_dir / "debug_validation.parquet"), app_to_row)
    test = add_item_rows(pd.read_parquet(args.data_dir / "debug_test.parquet"), app_to_row)
    chronology_audit = validate_split_chronology(train, validation, test)
    candidates = pd.read_parquet(args.test_candidates)

    positive_train = train[train.is_recommended].copy()
    prefix_history, prefix_users, prefix_targets = build_prefix_examples(positive_train, bank)
    sampler, user_to_sampler = make_sampler(train, np.sort(positive_train.user_id.unique()), len(bank), args.seed)
    train_user_idx = np.asarray([user_to_sampler[int(u)] for u in prefix_users], dtype=np.int64)
    train_profiles = build_profiles(positive_train, bank)
    val_history, val_pos, val_neg = build_validation_pairs(validation, train_profiles, bank, train, args.seed + 1)

    model = HistoryUserTower().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bank_tensor = torch.from_numpy(bank).to(device)
    best_loss, best_state, stale, history_rows = math.inf, None, 0, []
    print(f"device={device}; prefix examples={len(prefix_targets):,}; users={len(train_profiles.users):,}")
    for epoch in range(1, args.epochs + 1):
        negatives = sampler.sample(train_user_idx)
        train_loss = run_epoch(model, prefix_history, prefix_targets, negatives, bank_tensor,
                               args.batch_size, device, optimizer)
        val_loss = run_epoch(model, val_history, val_pos, val_neg, bank_tensor,
                             args.batch_size, device, None)
        history_rows.append({"epoch": epoch, "train_bpr_loss": train_loss, "val_bpr_loss": val_loss})
        print(f"epoch={epoch:02d} train_bpr={train_loss:.6f} val_bpr={val_loss:.6f}")
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    pd.DataFrame(history_rows).to_csv(args.output_dir / "training_history.csv", index=False)
    torch.save({"model_state_dict": best_state, "embedding_dim": 64, "hidden_dim": 128,
                "dropout": 0.1, "seed": args.seed, "best_val_bpr_loss": best_loss,
                "game_prefix": str(args.game_prefix)}, args.output_dir / "history_user_tower.pt")

    # Test profile uses all information available before the test event.
    pretest_positive = pd.concat([positive_train, validation[validation.is_recommended]], ignore_index=True)
    test_profiles = build_profiles(pretest_positive, bank)
    mlp_profiles = encode_profiles(model, test_profiles.hours, args.batch_size, device)
    models = {"simple_mean": test_profiles.simple, "hours_weighted_mean": test_profiles.hours,
              "history_mlp_bpr": mlp_profiles}
    train_catalog = set(train.app_id.unique())  # strict cold: no Train interaction of either sign
    rows = []
    for name, profile_matrix in models.items():
        frame, scores = candidate_scores(candidates, test_profiles.users, profile_matrix, bank, app_to_row)
        row = {"model": name, "evaluation": "sampled_1_positive_99_negative",
               **sampled_metrics(frame, scores, train_catalog, len(bank))}
        rows.append(row)
        print(name, "sampled Recall@10=", f"{row['recall@10']:.4f}",
              "NDCG@10=", f"{row['ndcg@10']:.4f}", "Cold Recall@10=", f"{row['cold_recall@10']:.4f}")

    if args.full_catalog:
        # Use the same 10k held-out targets as sampled evaluation, but rank them
        # against all 50,872 games. This keeps exact evaluation computationally
        # manageable and makes the sampled/full-catalog comparison aligned.
        query_users = candidates.loc[candidates.target.eq(1), ["group_id", "user_id", "app_id"]]
        query = add_item_rows(query_users, app_to_row)
        query = query[query.user_id.isin(test_profiles.mapping())].copy()
        query.attrs["catalog_ids"] = catalog_ids
        seen = pd.concat([train, validation], ignore_index=True).drop_duplicates(["user_id", "item_row"])
        popularity = train.loc[train.is_recommended].groupby("app_id").size()
        for name in ("hours_weighted_mean", "history_mlp_bpr"):
            row = {"model": name, "evaluation": "full_catalog",
                   **full_catalog_metrics(query, test_profiles.users, models[name], bank,
                                          train_catalog, seen, popularity,
                                          batch_size=min(args.batch_size, 256))}
            rows.append(row)
            print(name, "full Recall@10=", f"{row['recall@10']:.4f}",
                  "Cold Recall@10=", f"{row['cold_recall@10']:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "evaluation_metrics.csv", index=False)
    np.save(args.output_dir / "user_embeddings.npy", mlp_profiles)
    np.save(args.output_dir / "user_profiles_hours.npy", test_profiles.hours)
    np.save(args.output_dir / "user_profiles_simple.npy", test_profiles.simple)
    pd.DataFrame({"user_id": test_profiles.users, "embedding_row": np.arange(len(test_profiles.users))}).to_csv(
        args.output_dir / "user_embeddings.csv", index=False)
    audit = {
        "recommendation_csv_used": False,
        "interaction_split": "chronological per user; train -> validation -> test",
        "train_rows": len(train), "train_positive_rows": len(positive_train),
        "prefix_training_examples": len(prefix_targets), "profile_users": len(test_profiles.users),
        "catalog_games": len(catalog_ids), "strict_cold_definition": "app_id absent from every Train interaction",
        "test_positive_strict_cold": int((~test.loc[test.is_recommended, "app_id"].isin(train_catalog)).sum()),
        "best_validation_bpr_loss": best_loss,
        "chronology_audit": chronology_audit,
        "cohort_selection_caveat": "the upstream 50k debug cohort was sampled after filtering users by full-period valid interaction count; User Tower histories and targets remain chronological, but cohort eligibility is not train-only",
        "catalog_feature_caveat": "catalog-cold evaluation; the provided game bank includes catalog aggregates built outside the interaction split",
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("HISTORY_USER_TOWER_OK")


if __name__ == "__main__":
    main()
