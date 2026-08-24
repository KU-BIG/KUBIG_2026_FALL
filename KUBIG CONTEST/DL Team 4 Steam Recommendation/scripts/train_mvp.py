"""Train and compare Tabular-only, Text-only, and early-fusion DEBUG models."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.metrics import ranking_diagnostics, one_positive_ranking_metrics  # noqa: E402
from mvp_recommendation.model import KnownUserRecommender  # noqa: E402


MODES = ["tabular_only", "text_only", "tabular_text_fusion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "outputs" / "mvp")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "mvp")
    parser.add_argument(
        "--text-prefix", type=Path, default=REPO_ROOT / "text_data" / "emb_text_minilm"
    )
    parser.add_argument(
        "--tabular-prefix",
        type=Path,
        default=REPO_ROOT / "tabular_embedding" / "emb_tabular_svd64",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument("--max-ranking-positives", type=int, default=2_000)
    parser.add_argument(
        "--ranking-candidates",
        type=Path,
        default=None,
        help="Existing fixed candidate parquet. Created at output-dir when omitted.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--use-pos-weight", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(data_dir / "debug_train.parquet")
    val = pd.read_parquet(data_dir / "debug_validation.parquet")
    test = pd.read_parquet(data_dir / "debug_test.parquet")
    assert not set(train.review_id) & set(val.review_id)
    assert not set(train.review_id) & set(test.review_id)
    assert not set(val.review_id) & set(test.review_id)
    return train, val, test


def build_mappings(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    text_prefix: Path,
    tabular_prefix: Path,
) -> tuple[dict[int, int], np.ndarray, torch.Tensor, torch.Tensor]:
    users = np.sort(train.user_id.unique())
    user_to_idx = {int(user_id): idx for idx, user_id in enumerate(users)}
    assert val.user_id.isin(user_to_idx).all() and test.user_id.isin(user_to_idx).all()

    text_idx = pd.read_csv(text_prefix.with_suffix(".csv"))
    tab_idx = pd.read_csv(tabular_prefix.with_suffix(".csv"))
    assert np.array_equal(text_idx.app_id.to_numpy(), tab_idx.app_id.to_numpy())
    assert np.array_equal(text_idx.row.to_numpy(), np.arange(len(text_idx)))
    max_app_id = int(max(text_idx.app_id.max(), train.app_id.max(), val.app_id.max(), test.app_id.max()))
    app_to_row = np.full(max_app_id + 1, -1, dtype=np.int64)
    app_to_row[text_idx.app_id.to_numpy(np.int64)] = text_idx.row.to_numpy(np.int64)
    for frame in (train, val, test):
        assert (app_to_row[frame.app_id.to_numpy(np.int64)] >= 0).all()

    tab_bank = torch.from_numpy(
        np.load(tabular_prefix.with_suffix(".npy"), allow_pickle=False).copy()
    )
    text_bank = torch.from_numpy(
        np.load(text_prefix.with_suffix(".npy"), allow_pickle=False).copy()
    )
    assert tab_bank.shape == (50872, 64) and text_bank.shape == (50872, 384)
    assert torch.isfinite(tab_bank).all() and torch.isfinite(text_bank).all()
    return user_to_idx, app_to_row, tab_bank, text_bank


def make_loader(
    frame: pd.DataFrame,
    user_to_idx: dict[int, int],
    app_to_row: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    user_idx = frame.user_id.map(user_to_idx).to_numpy(np.int64)
    game_rows = app_to_row[frame.app_id.to_numpy(np.int64)]
    targets = frame.is_recommended.astype(np.float32).to_numpy()
    assert np.isfinite(targets).all() and (game_rows >= 0).all()
    dataset = TensorDataset(
        torch.from_numpy(user_idx), torch.from_numpy(game_rows), torch.from_numpy(targets)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: KnownUserRecommender,
    loader: DataLoader,
    tab_bank: torch.Tensor,
    text_bank: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_rows = 0
    all_targets: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for user_idx, game_rows, target in loader:
            user_idx = user_idx.to(device)
            target = target.to(device)
            tab = tab_bank[game_rows].to(device)
            text = text_bank[game_rows].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(user_idx, tab, text)
            loss = criterion(logits, target)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += float(loss.item()) * len(target)
            total_rows += len(target)
            all_targets.append(target.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
    return total_loss / total_rows, np.concatenate(all_targets), np.concatenate(all_logits)


def build_ranking_candidates(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    valid_app_ids: np.ndarray,
    num_negatives: int,
    max_positives: int,
    seed: int,
) -> pd.DataFrame:
    all_history = pd.concat([train, val, test], ignore_index=True)
    history = all_history.groupby("user_id").app_id.agg(lambda values: set(map(int, values)))
    positives = test.loc[test.is_recommended, ["user_id", "app_id"]].copy()
    if len(positives) > max_positives:
        positives = positives.sample(max_positives, random_state=seed)
    positives = positives.sort_values(["user_id", "app_id"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int, int, int]] = []
    for group_id, row in positives.iterrows():
        user_id, positive = int(row.user_id), int(row.app_id)
        blocked = history.loc[user_id]
        negatives: set[int] = set()
        while len(negatives) < num_negatives:
            draws = rng.choice(valid_app_ids, size=(num_negatives - len(negatives)) * 2)
            negatives.update(int(app_id) for app_id in draws if int(app_id) not in blocked)
        chosen = sorted(negatives)[:num_negatives]
        assert positive not in chosen and not (set(chosen) & blocked)
        rows.append((group_id, user_id, positive, 1))
        rows.extend((group_id, user_id, app_id, 0) for app_id in chosen)
    candidates = pd.DataFrame(rows, columns=["group_id", "user_id", "app_id", "target"])
    assert candidates.groupby("group_id").size().eq(num_negatives + 1).all()
    assert candidates.groupby("group_id").target.sum().eq(1).all()
    return candidates


@torch.no_grad()
def score_candidates(
    model: KnownUserRecommender,
    candidates: pd.DataFrame,
    user_to_idx: dict[int, int],
    app_to_row: np.ndarray,
    tab_bank: torch.Tensor,
    text_bank: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    users = torch.from_numpy(candidates.user_id.map(user_to_idx).to_numpy(np.int64))
    rows = torch.from_numpy(app_to_row[candidates.app_id.to_numpy(np.int64)])
    scores = []
    for start in range(0, len(candidates), batch_size):
        u = users[start : start + batch_size].to(device)
        r = rows[start : start + batch_size]
        logits = model(u, tab_bank[r].to(device), text_bank[r].to(device))
        scores.append(logits.cpu().numpy())
    output = np.concatenate(scores)
    assert np.isfinite(output).all()
    return output


def plot_curves(history: list[dict[str, float]], mode: str, figure_dir: Path) -> None:
    frame = pd.DataFrame(history)
    plt.figure(figsize=(7, 4))
    plt.plot(frame.epoch, frame.train_loss, label="train loss")
    plt.plot(frame.epoch, frame.val_loss, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title(mode.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / f"{mode}_training_curve.png", dpi=150)
    plt.close()


def plot_comparison(results: pd.DataFrame, metric: str, output: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(results.model, results[metric])
    plt.ylabel(metric)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    figure_dir = args.output_dir / "figures"
    checkpoint_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    train, val, test = load_splits(args.data_dir)
    user_to_idx, app_to_row, tab_bank, text_bank = build_mappings(
        train, val, test, args.text_prefix, args.tabular_prefix
    )
    train_loader = make_loader(train, user_to_idx, app_to_row, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val, user_to_idx, app_to_row, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test, user_to_idx, app_to_row, args.batch_size, False, args.num_workers)
    valid_app_ids = np.flatnonzero(app_to_row >= 0).astype(np.int64)
    candidate_path = args.ranking_candidates or (args.output_dir / "ranking_candidates.parquet")
    if candidate_path.exists():
        candidates = pd.read_parquet(candidate_path)
        print(f"reusing fixed ranking candidates: {candidate_path}")
    else:
        candidates = build_ranking_candidates(
            train, val, test, valid_app_ids, args.num_negatives, args.max_ranking_positives, args.seed
        )
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_parquet(candidate_path, index=False)
        print(f"saved fixed ranking candidates: {candidate_path}")
    assert candidates.groupby("group_id").size().eq(args.num_negatives + 1).all()
    assert candidates.groupby("group_id").target.sum().eq(1).all()
    assert candidates.user_id.isin(user_to_idx).all()
    assert (app_to_row[candidates.app_id.to_numpy(np.int64)] >= 0).all()
    train_positive_counts = train.loc[train.is_recommended].groupby("app_id").size()

    pos = int(train.is_recommended.sum())
    neg = len(train) - pos
    print(f"device={device}; train positive={pos:,} negative={neg:,} ratio={pos/len(train):.2%}")
    print(f"text bank={args.text_prefix}")
    print(f"tabular bank={args.tabular_prefix}")
    print(f"users={len(user_to_idx):,}; ranking positives={candidates.group_id.nunique():,}")
    print("pos_weight enabled:", args.use_pos_weight)
    pos_weight = torch.tensor([neg / pos], device=device) if args.use_pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    results: list[dict[str, float | str]] = []

    for mode in MODES:
        seed_everything(args.seed)
        model = KnownUserRecommender(len(user_to_idx), mode, embed_dim=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"\n[{mode}] parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        best_loss = float("inf")
        stale = 0
        history: list[dict[str, float]] = []
        checkpoint = checkpoint_dir / f"{mode}_best.pt"
        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = run_epoch(
                model, train_loader, tab_bank, text_bank, criterion, device, optimizer
            )
            val_loss, val_target, val_logits = run_epoch(
                model, val_loader, tab_bank, text_bank, criterion, device
            )
            val_auc = roc_auc_score(val_target, val_logits)
            history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc}
            )
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} "
                f"val_loss={val_loss:.5f} val_auc={val_auc:.5f}"
            )
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                stale = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "user_to_idx": user_to_idx,
                        "config": vars(args),
                        "mode": mode,
                        "embedding_dimensions": {"user": 64, "game": 64, "tabular": 64, "text": 384},
                    },
                    checkpoint,
                )
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stopping at epoch {epoch}")
                    break

        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        test_loss, test_target, test_logits = run_epoch(
            model, test_loader, tab_bank, text_bank, criterion, device
        )
        test_auc = roc_auc_score(test_target, test_logits)
        ranking_scores = score_candidates(
            model, candidates, user_to_idx, app_to_row, tab_bank, text_bank, args.batch_size, device
        )
        ranking = one_positive_ranking_metrics(ranking_scores, args.num_negatives + 1)
        diagnostics = ranking_diagnostics(
            candidates,
            ranking_scores,
            train_positive_counts,
            valid_catalog_size=len(valid_app_ids),
            k=10,
        )
        result: dict[str, float | str] = {
            "model": mode,
            "test_loss": test_loss,
            "test_auc": test_auc,
            **ranking,
            **diagnostics,
        }
        results.append(result)
        pd.DataFrame(history).to_csv(args.output_dir / f"{mode}_history.csv", index=False)
        plot_curves(history, mode, figure_dir)
        print(json.dumps(result, indent=2))

    comparison = pd.DataFrame(results)
    comparison.to_csv(args.output_dir / "mvp_model_comparison.csv", index=False)
    plot_comparison(comparison, "recall@10", figure_dir / "model_recall_at_10.png")
    plot_comparison(comparison, "ndcg@10", figure_dir / "model_ndcg_at_10.png")
    best_unimodal = comparison[comparison.model != "tabular_text_fusion"].set_index("model")
    fusion = comparison.set_index("model").loc["tabular_text_fusion"]
    improvements = {}
    for metric in ["recall@10", "ndcg@10"]:
        baseline = float(best_unimodal[metric].max())
        absolute = float(fusion[metric]) - baseline
        improvements[metric] = {
            "best_unimodal": baseline,
            "fusion": float(fusion[metric]),
            "absolute_difference": absolute,
            "relative_improvement": absolute / baseline if baseline else None,
        }
    (args.output_dir / "fusion_improvement.json").write_text(
        json.dumps(improvements, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(json.dumps(improvements, ensure_ascii=False, indent=2))
    print("TRAIN_MVP_OK")


if __name__ == "__main__":
    main()
