"""Aggregate three-seed downstream multimodal recommendation results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "multimodal_game_bpr"
OUTPUT = ROOT / "game_fusion" / "downstream_evaluation"
SEEDS = (42, 7, 2026)
METRICS = ["recall@10", "ndcg@10", "mrr", "long_tail_recall@10", "catalog_coverage@10"]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seed in SEEDS:
        seed_dir = SOURCE / f"seed_{seed}"
        multimodal = pd.read_csv(seed_dir / "multimodal_bpr_comparison.csv")
        hybrid = pd.read_csv(seed_dir / "mf_multimodal_hybrid_test.csv")
        for _, row in hybrid.iterrows():
            all_rows.append({"seed": seed, **row.to_dict()})
        if seed == 42:
            ablations = multimodal.copy()
            ablations.insert(0, "seed", seed)
            ablations.to_csv(OUTPUT / "fusion_bank_ablation_seed42.csv", index=False)

    results = pd.DataFrame(all_rows)
    results.to_csv(OUTPUT / "multiseed_results.csv", index=False)
    summary_rows = []
    for model, group in results.groupby("model", sort=False):
        row: dict[str, object] = {"model": model, "seeds": len(group)}
        row["alpha_mf_mean"] = float(group.alpha_mf.mean())
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    previous_path = (
        ROOT / "outputs" / "mvp_50k" / "hybrid_reproducibility" / "three_seed_mean_std.csv"
    )
    previous = pd.read_csv(previous_path).set_index("model").loc["mf_text_balanced_hybrid"]
    previous_row: dict[str, object] = {
        "model": "mf_text_balanced_hybrid",
        "seeds": int(previous["n_seeds"]),
        "alpha_mf_mean": float("nan"),
    }
    for metric in METRICS:
        previous_row[f"{metric}_mean"] = float(previous[f"{metric}_mean"])
        previous_row[f"{metric}_std"] = float(previous[f"{metric}_std"])
    summary = pd.concat([summary, pd.DataFrame([previous_row])], ignore_index=True)
    summary.to_csv(OUTPUT / "multiseed_summary.csv", index=False)

    lookup = summary.set_index("model")
    hybrid = lookup.loc["mf_multimodal_balanced"]
    mf = lookup.loc["mf_only"]
    payload = {
        "seeds": list(SEEDS),
        "selection": "bank and alpha selected using validation only",
        "selected_bank": "emb_game_concat_64 (frozen concat)",
        "balanced_alpha_mf_each_seed": results.loc[
            results.model.eq("mf_multimodal_balanced"), ["seed", "alpha_mf"]
        ].to_dict(orient="records"),
        "mean_delta_hybrid_vs_mf": {
            metric: float(hybrid[f"{metric}_mean"] - mf[f"{metric}_mean"])
            for metric in METRICS
        },
        "synthetic_bank_warning": (
            "Step 3/4 banks were tuned on synthetic interactions and performed far below "
            "the frozen concat bank on real recommendation data. Do not deploy them."
        ),
    }
    (OUTPUT / "evaluation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def pm(model: str, metric: str) -> str:
        row = lookup.loc[model]
        return f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"

    ablation = pd.read_csv(OUTPUT / "fusion_bank_ablation_seed42.csv").set_index("model")
    readme = f"""# Multimodal Game Embedding Downstream Evaluation

`game_fusion`의 64차원 게임 임베딩을 실제 Steam interaction에 연결해 검증한 결과입니다.
게임 임베딩은 고정하고 49,742명의 user embedding만 BPR로 학습했습니다. 기존 실험과 같은
train/validation/test split, dynamic negative sampling, 9,984개 고정 test query를 사용했습니다.

## 결론

- 최종 game bank: `emb_game_concat_64` (frozen text + image + tabular concat)
- 최종 known-user 모델: MF-BPR + frozen multimodal BPR score hybrid
- validation에서 선택된 balanced MF 비중: 세 seed 모두 `0.40`
- Step 3/4 synthetic-tuned bank는 실제 interaction 성능이 크게 낮아 사용하지 않습니다.

## Seed 42 bank 비교

| Game bank | Recall@10 | NDCG@10 | Long-tail Recall@10 | Coverage@10 |
|---|---:|---:|---:|---:|
| Frozen concat | {ablation.loc['frozen_concat', 'recall@10']:.4f} | {ablation.loc['frozen_concat', 'ndcg@10']:.4f} | {ablation.loc['frozen_concat', 'long_tail_recall@10']:.4f} | {ablation.loc['frozen_concat', 'catalog_coverage@10']:.4f} |
| Synthetic fusion tuned | {ablation.loc['synthetic_fusion_tuned', 'recall@10']:.4f} | {ablation.loc['synthetic_fusion_tuned', 'ndcg@10']:.4f} | {ablation.loc['synthetic_fusion_tuned', 'long_tail_recall@10']:.4f} | {ablation.loc['synthetic_fusion_tuned', 'catalog_coverage@10']:.4f} |
| Synthetic partial adapter | {ablation.loc['synthetic_partial_adapter', 'recall@10']:.4f} | {ablation.loc['synthetic_partial_adapter', 'ndcg@10']:.4f} | {ablation.loc['synthetic_partial_adapter', 'long_tail_recall@10']:.4f} | {ablation.loc['synthetic_partial_adapter', 'catalog_coverage@10']:.4f} |

## 3-seed 결과 (평균 ± 표본표준편차)

| 모델 | Recall@10 | NDCG@10 | MRR | Long-tail Recall@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Frozen multimodal | {pm('multimodal_only', 'recall@10')} | {pm('multimodal_only', 'ndcg@10')} | {pm('multimodal_only', 'mrr')} | {pm('multimodal_only', 'long_tail_recall@10')} | {pm('multimodal_only', 'catalog_coverage@10')} |
| MF + Multimodal balanced | {pm('mf_multimodal_balanced', 'recall@10')} | {pm('mf_multimodal_balanced', 'ndcg@10')} | {pm('mf_multimodal_balanced', 'mrr')} | {pm('mf_multimodal_balanced', 'long_tail_recall@10')} | {pm('mf_multimodal_balanced', 'catalog_coverage@10')} |
| 기존 MF + Text balanced | {pm('mf_text_balanced_hybrid', 'recall@10')} | {pm('mf_text_balanced_hybrid', 'ndcg@10')} | {pm('mf_text_balanced_hybrid', 'mrr')} | {pm('mf_text_balanced_hybrid', 'long_tail_recall@10')} | {pm('mf_text_balanced_hybrid', 'catalog_coverage@10')} |
| MF only | {pm('mf_only', 'recall@10')} | {pm('mf_only', 'ndcg@10')} | {pm('mf_only', 'mrr')} | {pm('mf_only', 'long_tail_recall@10')} | {pm('mf_only', 'catalog_coverage@10')} |

Balanced hybrid는 MF-only보다 Recall@10과 NDCG@10을 모두 높이면서 catalog coverage와
long-tail hit도 회복했습니다. 단, 이 평가는 1 positive + 99 sampled negatives 설정이므로
배포 전 full-catalog 검증이나 실제 사용자 A/B test가 추가로 필요합니다.

## 재현

```bash
python scripts/evaluate_multimodal_game_bpr.py \\
  --data-dir outputs/mvp_50k/data_seed_42 \\
  --test-candidates outputs/mvp_50k/models_seed_42/ranking_candidates.parquet \\
  --output-dir outputs/multimodal_game_bpr/seed_42 \\
  --mf-checkpoint outputs/mvp_50k/repro_seed_42/checkpoints/mf_bpr_best.pt \\
  --seed 42

python scripts/summarize_multimodal_evaluation.py
```

선택과 hybrid weight 탐색에는 validation만 사용했고 test 결과로 설정을 고르지 않았습니다.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(summary.to_string(index=False))
    print("SUMMARIZE_MULTIMODAL_EVALUATION_OK")


if __name__ == "__main__":
    main()
