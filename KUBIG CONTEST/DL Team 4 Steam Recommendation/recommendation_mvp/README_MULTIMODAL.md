# Multimodal Steam Recommendation MVP

> 서비스 실행과 팀 공유를 위한 최신 통합 안내서는 [`README.md`](./README.md)입니다.
> 이 문서는 멀티모달 전환 시점의 모델 선택과 실험 결과를 간단히 보존합니다.

기존 MF/Text 추천 파이프라인에 팀의 `game_fusion` 결과를 연결한 버전입니다.
게임당 text(MiniLM), image(CLIP), tabular(SVD) 정보를 하나의 64차원 벡터로 사용합니다.

## 최종 선택

- Game bank: `game_fusion/emb_game_concat_64.npy`
- User tower: 실제 train positive interaction으로 학습한 64차원 BPR user embedding
- Known-user 기본 모델: `MF 40% + Multimodal 60%`
- New-user 정책:
  - 좋아하는 게임 입력: multimodal game vector 평균으로 취향 프로필 생성
  - 태그만 입력: MiniLM tag profile 사용
  - 입력 없음: train positive popularity fallback

Step 3/4의 `finetuned`, `partial_fusion_tuned` 파일은 synthetic interaction smoke test로
만든 산출물이어서 최종 추천에는 사용하지 않습니다. 실제 interaction downstream 평가에서
frozen concat보다 성능이 크게 낮았습니다.

## 성능

같은 9,984개 test query와 seed 42/7/2026에서 얻은 평균 ± 표본표준편차입니다.

| 모델 | Recall@10 | NDCG@10 | MRR | Long-tail Recall@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Frozen multimodal | 0.8039 ± 0.0013 | 0.5185 ± 0.0008 | 0.4381 ± 0.0008 | 0.2744 ± 0.0031 | 0.1588 ± 0.0007 |
| **MF + Multimodal** | **0.8165 ± 0.0014** | **0.5609 ± 0.0006** | **0.4898 ± 0.0012** | 0.0341 ± 0.0041 | 0.1133 ± 0.0002 |
| 기존 MF + Text | 0.7906 ± 0.0086 | 0.5404 ± 0.0032 | 0.4713 ± 0.0034 | 0.0279 ± 0.0198 | 0.1020 ± 0.0025 |
| MF only | 0.7960 ± 0.0007 | 0.5372 ± 0.0014 | 0.4656 ± 0.0016 | 0.0000 ± 0.0000 | 0.0974 ± 0.0003 |

모델과 hybrid weight는 validation에서만 선택했으며 test 결과로 설정을 고르지 않았습니다.
MF + Multimodal은 기존 MF + Text보다 Recall@10 `+0.0259`, NDCG@10 `+0.0205`였습니다.
상세 결과는 [`game_fusion/downstream_evaluation/`](../game_fusion/downstream_evaluation/)에 있습니다.

## UI 실행

repository root에서 실행합니다.

```bash
pip install -r recommendation_mvp/requirements.txt
streamlit run recommendation_mvp/app.py
```

기존 사용자 화면의 기본 모델은 `MF + Multimodal Hybrid (recommended)`입니다. 여러 모델을
동시에 선택해 기존 MF/Text 결과와 비교할 수도 있습니다. 신규 사용자 화면에서는 선호 태그와
좋아하는 게임을 함께 또는 따로 입력할 수 있습니다.

추천 결과는 기본적으로 2열 이미지 카드로 표시됩니다. 각 카드에는 Steam header image,
순위, 게임명, Steam 평가, 긍정 비율, 가격과 추천 이유가 포함됩니다. 게임명을 누르면 해당
Steam Store 페이지가 새 탭에서 열립니다. 최신 이미지 경로가 없으면 legacy CDN 경로를 한 번
더 확인하고, 두 경로가 모두 없으면 깨진 이미지 대신 placeholder 배경을 표시합니다. 기존 표는
`표 형태로 보기`에서 확인할 수 있고 CSV 다운로드도 그대로 지원합니다.

## CLI

기존 사용자:

```bash
python scripts/recommend_users.py \
  --user-ids 13 \
  --models mf_multimodal_hybrid multimodal_bpr \
  --top-k 10 \
  --output outputs/user_13_multimodal.csv
```

신규 사용자:

```bash
python scripts/recommend_new_user.py \
  --profile-name witcher_fan \
  --preferred-tags RPG "Open World" \
  --liked-app-ids 292030 \
  --top-k 10 \
  --output outputs/witcher_fan.csv
```

## 핵심 파일

| 파일 | 역할 |
|---|---|
| `game_fusion/emb_game_concat_64.npy/.csv` | 최종 frozen multimodal game bank와 app_id 순서 |
| `model_artifacts/frozen_multimodal_user_bpr_seed42.pt` | 실제 interaction으로 학습한 user tower |
| `model_artifacts/multimodal_evaluation_summary_seed42.json` | 선택 bank와 hybrid weight |
| `mvp_recommendation/inference.py` | known-user full-catalog scoring |
| `mvp_recommendation/cold_start.py` | 신규 사용자 tag/liked-game profile |
| `scripts/evaluate_multimodal_game_bpr.py` | 동일 조건 downstream 학습·평가 재현 |

## 주의

- offline 평가는 한 query당 positive 1개와 sampled negative 99개를 사용했습니다.
- 이미지가 없던 소수 게임은 fusion 원본 제작 단계의 대체 벡터 정책을 따릅니다.
- 실제 만족도는 클릭·찜·플레이 데이터가 쌓인 뒤 online A/B test로 다시 확인해야 합니다.
- `.npy`와 `.csv`의 행 순서는 반드시 함께 유지해야 합니다.
