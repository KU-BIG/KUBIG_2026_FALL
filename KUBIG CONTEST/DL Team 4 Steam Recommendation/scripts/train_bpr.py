"""Train MF/content BPR models on the fixed 50k chronological experiment."""

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
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.bpr import BPRRecommender, DynamicNegativeSampler, bpr_loss  # noqa: E402
from mvp_recommendation.metrics import ranking_diagnostics, one_positive_ranking_metrics  # noqa: E402


BASE_MODES = ["mf_bpr", "tabular_bpr", "text_bpr", "tabular_text_fusion_bpr"]
ALL_MODES = BASE_MODES + ["text_anchored_gated_bpr", "user_modality_gated_bpr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--text-prefix", type=Path, default=REPO_ROOT / "text_data" / "emb_text_minilm"
    )
    parser.add_argument(
        "--tabular-prefix",
        type=Path,
        default=REPO_ROOT / "tabular_embedding" / "leakage_safe" / "emb_tabular_safe_svd64",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--limit-positive", type=int, default=None)
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=BASE_MODES)
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


def load_banks(
    text_prefix: Path, tabular_prefix: Path
) -> tuple[pd.DataFrame, np.ndarray, torch.Tensor, torch.Tensor]:
    text_idx = pd.read_csv(text_prefix.with_suffix(".csv"))
    tab_idx = pd.read_csv(tabular_prefix.with_suffix(".csv"))
    assert np.array_equal(text_idx.app_id.to_numpy(), tab_idx.app_id.to_numpy())
    assert np.array_equal(text_idx.row.to_numpy(), np.arange(len(text_idx)))
    tab_bank = torch.from_numpy(np.load(tabular_prefix.with_suffix(".npy"), allow_pickle=False).copy())
    text_bank = torch.from_numpy(np.load(text_prefix.with_suffix(".npy"), allow_pickle=False).copy())
    assert tab_bank.shape == (len(text_idx), 64)
    assert text_bank.shape == (len(text_idx), 384)
    assert torch.isfinite(tab_bank).all() and torch.isfinite(text_bank).all()
    max_app_id = int(text_idx.app_id.max())
    app_to_row = np.full(max_app_id + 1, -1, dtype=np.int64)
    app_to_row[text_idx.app_id.to_numpy(np.int64)] = text_idx.row.to_numpy(np.int64)
    return text_idx, app_to_row, tab_bank, text_bank


def prepare_data(
    data_dir: Path,
    candidates_path: Path,
    app_to_row: np.ndarray,
    limit_positive: int | None,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[int, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    DynamicNegativeSampler,
    np.ndarray,
    np.ndarray,
]:
    train = pd.read_parquet(data_dir / "debug_train.parquet")
    val = pd.read_parquet(data_dir / "debug_validation.parquet")
    test = pd.read_parquet(data_dir / "debug_test.parquet")
    train_positive = train.loc[train.is_recommended, ["user_id", "app_id"]].copy()
    if limit_positive is not None and len(train_positive) > limit_positive:
        train_positive = train_positive.sample(limit_positive, random_state=seed)
    users = np.sort(train_positive.user_id.unique())
    user_to_idx = {int(user): idx for idx, user in enumerate(users)}
    train_users = train_positive.user_id.map(user_to_idx).to_numpy(np.int64)
    train_rows = app_to_row[train_positive.app_id.to_numpy(np.int64)]
    assert (train_rows >= 0).all()

    train_observed = train[train.user_id.isin(user_to_idx)].copy()
    observed_users = train_observed.user_id.map(user_to_idx).to_numpy(np.int64)
    observed_rows = app_to_row[train_observed.app_id.to_numpy(np.int64)]
    sampler = DynamicNegativeSampler(
        num_items=int((app_to_row >= 0).sum()),
        observed_user_idx=observed_users,
        observed_item_rows=observed_rows,
        seed=seed,
    )

    # Fixed validation pairs. The validation sampler may inspect validation
    # history, but the train sampler above intentionally cannot.
    val_eligible = val[val.user_id.isin(user_to_idx)].copy()
    validation_observed = pd.concat(
        [train_observed[["user_id", "app_id"]], val_eligible[["user_id", "app_id"]]],
        ignore_index=True,
    ).drop_duplicates()
    validation_sampler = DynamicNegativeSampler(
        num_items=int((app_to_row >= 0).sum()),
        observed_user_idx=validation_observed.user_id.map(user_to_idx).to_numpy(np.int64),
        observed_item_rows=app_to_row[validation_observed.app_id.to_numpy(np.int64)],
        seed=seed + 10_000,
    )
    val_positive = val_eligible.loc[val_eligible.is_recommended]
    val_users = val_positive.user_id.map(user_to_idx).to_numpy(np.int64)
    val_rows = app_to_row[val_positive.app_id.to_numpy(np.int64)]
    val_negatives = validation_sampler.sample(val_users)

    candidates = pd.read_parquet(candidates_path)
    trained_positive_users = set(user_to_idx)
    original_groups = candidates.group_id.nunique()
    candidates = candidates[candidates.user_id.isin(trained_positive_users)].copy()
    kept_groups = candidates.group_id.nunique()
    candidates.to_parquet(candidates_path.parent / "ranking_candidates_bpr_users.parquet", index=False)
    assert candidates.groupby("group_id").size().eq(100).all()
    assert candidates.groupby("group_id").target.sum().eq(1).all()
    print(f"BPR candidates: {kept_groups:,}/{original_groups:,} groups retained")
    return (
        train,
        val,
        test,
        candidates,
        user_to_idx,
        train_users,
        train_rows,
        users,
        sampler,
        val_users,
        np.column_stack([val_rows, val_negatives]),
    )


def pair_epoch(
    model: BPRRecommender,
    user_idx: np.ndarray,
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
    tab_bank: torch.Tensor,
    text_bank: torch.Tensor,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    dataset = TensorDataset(
        torch.from_numpy(user_idx),
        torch.from_numpy(positive_rows),
        torch.from_numpy(negative_rows),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=optimizer is not None)
    model.train(optimizer is not None)
    total_loss = total_rows = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for users, positive, negative in loader:
            users = users.to(device)
            positive_device = positive.to(device)
            negative_device = negative.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            pos_score, neg_score = model.pair_scores(
                users,
                positive_device,
                negative_device,
                tab_bank[positive].to(device),
                text_bank[positive].to(device),
                tab_bank[negative].to(device),
                text_bank[negative].to(device),
            )
            loss = bpr_loss(pos_score, neg_score)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += float(loss.item()) * len(users)
            total_rows += len(users)
    return total_loss / total_rows


@torch.no_grad()
def candidate_scores(
    model: BPRRecommender,
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
        score = model.score(u, r.to(device), tab_bank[r].to(device), text_bank[r].to(device))
        outputs.append(score.cpu().numpy())
    scores = np.concatenate(outputs)
    assert np.isfinite(scores).all()
    return scores


@torch.no_grad()
def observed_test_auc(
    model: BPRRecommender,
    test: pd.DataFrame,
    user_to_idx: dict[int, int],
    app_to_row: np.ndarray,
    tab_bank: torch.Tensor,
    text_bank: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    frame = test[test.user_id.isin(user_to_idx)].copy()
    scores = candidate_scores(
        model,
        frame.assign(group_id=np.arange(len(frame)), target=frame.is_recommended.astype(int)),
        user_to_idx,
        app_to_row,
        tab_bank,
        text_bank,
        batch_size,
        device,
    )
    return float(roc_auc_score(frame.is_recommended.astype(int), scores))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    figure_dir = args.output_dir / "figures"
    checkpoint_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    index, app_to_row, tab_bank, text_bank = load_banks(args.text_prefix, args.tabular_prefix)
    (
        train, val, test, candidates, user_to_idx, train_users, train_rows, users,
        sampler, val_users, val_pairs,
    ) = prepare_data(args.data_dir, args.candidates, app_to_row, args.limit_positive, args.seed)
    train_positive_counts = train.loc[train.is_recommended].groupby("app_id").size()
    print(
        f"device={device}; users={len(users):,}; train positives={len(train_rows):,}; "
        f"validation positives={len(val_users):,}; candidates={len(candidates):,}"
    )
    results = []
    for mode in args.modes:
        seed_everything(args.seed)
        # Fair ablation: every model sees the same epoch-wise negative samples.
        sampler.reset(args.seed)
        model = BPRRecommender(len(users), len(index), mode, embed_dim=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        checkpoint = checkpoint_dir / f"{mode}_best.pt"
        history = []
        best_loss = float("inf")
        stale = 0
        print(f"\n[{mode}] parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        for epoch in range(1, args.epochs + 1):
            train_negative = sampler.sample(train_users)
            train_loss = pair_epoch(
                model, train_users, train_rows, train_negative, tab_bank, text_bank,
                args.batch_size, device, optimizer,
            )
            val_loss = pair_epoch(
                model, val_users, val_pairs[:, 0], val_pairs[:, 1], tab_bank, text_bank,
                args.batch_size, device, None,
            )
            history.append({"epoch": epoch, "train_bpr_loss": train_loss, "val_bpr_loss": val_loss})
            print(f"epoch={epoch} train_bpr={train_loss:.6f} val_bpr={val_loss:.6f}")
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                stale = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "user_to_idx": user_to_idx,
                        "mode": mode,
                        "config": vars(args),
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
        scores = candidate_scores(
            model, candidates, user_to_idx, app_to_row, tab_bank, text_bank, args.batch_size, device
        )
        ranking = one_positive_ranking_metrics(scores, 100)
        diagnostics = ranking_diagnostics(
            candidates, scores, train_positive_counts, valid_catalog_size=len(index), k=10
        )
        result = {
            "model": mode,
            "test_auc": observed_test_auc(
                model, test, user_to_idx, app_to_row, tab_bank, text_bank, args.batch_size, device
            ),
            "best_val_bpr_loss": best_loss,
            **ranking,
            **diagnostics,
        }
        if mode == "user_modality_gated_bpr":
            assert model.user_modality_gate is not None
            gates = torch.sigmoid(model.user_modality_gate.weight.detach()).cpu().numpy().ravel()
            result.update(
                {
                    "text_gate_mean": float(gates.mean()),
                    "text_gate_std": float(gates.std()),
                    "text_gate_p10": float(np.quantile(gates, 0.10)),
                    "text_gate_p50": float(np.quantile(gates, 0.50)),
                    "text_gate_p90": float(np.quantile(gates, 0.90)),
                }
            )
        elif mode == "text_anchored_gated_bpr":
            gate_chunks = []
            model.eval()
            with torch.no_grad():
                for start in range(0, len(index), args.batch_size):
                    end = min(start + args.batch_size, len(index))
                    encoder = model.game_encoder
                    assert hasattr(encoder, "components")
                    _, _, gate = encoder.components(
                        tab_bank[start:end].to(device), text_bank[start:end].to(device)
                    )
                    gate_chunks.append(gate.cpu().numpy().ravel())
            gates = np.concatenate(gate_chunks)
            result.update(
                {
                    "text_gate_mean": float(gates.mean()),
                    "text_gate_std": float(gates.std()),
                    "text_gate_p10": float(np.quantile(gates, 0.10)),
                    "text_gate_p50": float(np.quantile(gates, 0.50)),
                    "text_gate_p90": float(np.quantile(gates, 0.90)),
                }
            )
        results.append(result)
        pd.DataFrame(history).to_csv(args.output_dir / f"{mode}_history.csv", index=False)
        frame = pd.DataFrame(history)
        plt.figure(figsize=(7, 4))
        plt.plot(frame.epoch, frame.train_bpr_loss, label="train BPR")
        plt.plot(frame.epoch, frame.val_bpr_loss, label="validation BPR")
        plt.xlabel("Epoch")
        plt.ylabel("BPR loss")
        plt.title(mode.replace("_", " ").title())
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / f"{mode}_training_curve.png", dpi=150)
        plt.close()
        print(json.dumps(result, indent=2))

    comparison = pd.DataFrame(results)
    comparison.to_csv(args.output_dir / "bpr_model_comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print("TRAIN_BPR_OK")


if __name__ == "__main__":
    main()
