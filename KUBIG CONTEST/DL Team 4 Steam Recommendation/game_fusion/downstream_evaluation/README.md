# Multimodal Game Embedding Downstream Evaluation

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
| Frozen concat | 0.8026 | 0.5182 | 0.2713 | 0.1596 |
| Synthetic fusion tuned | 0.4261 | 0.2423 | 0.2186 | 0.3703 |
| Synthetic partial adapter | 0.3217 | 0.1751 | 0.2031 | 0.4288 |

## 3-seed 결과 (평균 ± 표본표준편차)

| 모델 | Recall@10 | NDCG@10 | MRR | Long-tail Recall@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Frozen multimodal | 0.8039 ± 0.0013 | 0.5185 ± 0.0008 | 0.4381 ± 0.0008 | 0.2744 ± 0.0031 | 0.1588 ± 0.0007 |
| MF + Multimodal balanced | 0.8165 ± 0.0014 | 0.5609 ± 0.0006 | 0.4898 ± 0.0012 | 0.0341 ± 0.0041 | 0.1133 ± 0.0002 |
| 기존 MF + Text balanced | 0.7906 ± 0.0086 | 0.5404 ± 0.0032 | 0.4713 ± 0.0034 | 0.0279 ± 0.0198 | 0.1020 ± 0.0025 |
| MF only | 0.7960 ± 0.0007 | 0.5372 ± 0.0014 | 0.4656 ± 0.0016 | 0.0000 ± 0.0000 | 0.0974 ± 0.0003 |

Balanced hybrid는 MF-only보다 Recall@10과 NDCG@10을 모두 높이면서 catalog coverage와
long-tail hit도 회복했습니다. 단, 이 평가는 1 positive + 99 sampled negatives 설정이므로
배포 전 full-catalog 검증이나 실제 사용자 A/B test가 추가로 필요합니다.

## 재현

```bash
python scripts/evaluate_multimodal_game_bpr.py \
  --data-dir outputs/mvp_50k/data_seed_42 \
  --test-candidates outputs/mvp_50k/models_seed_42/ranking_candidates.parquet \
  --output-dir outputs/multimodal_game_bpr/seed_42 \
  --mf-checkpoint outputs/mvp_50k/repro_seed_42/checkpoints/mf_bpr_best.pt \
  --seed 42

python scripts/summarize_multimodal_evaluation.py
```

선택과 hybrid weight 탐색에는 validation만 사용했고 test 결과로 설정을 고르지 않았습니다.
