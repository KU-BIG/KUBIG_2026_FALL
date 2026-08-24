# Tabular + Text Fusion Recommendation MVP

이 폴더의 코드는 Steam known-user 추천에서 다음 세 모델을 동일한 데이터 split과
동일한 ranking 후보로 비교하기 위한 개인 MVP입니다.

1. Tabular only
2. Text only
3. Tabular + Text early fusion

이미지는 이번 실험에 사용하지 않습니다.

## 현재 완료 상태

- `recommendations.csv` 41,154,794건 전체 구조 및 통계 확인
- `date` 기반 user-wise chronological split 구현 및 검증
- MiniLM text embedding 50,872개 재생성 및 mapping 검증
- Tabular/Text/Fusion 세 모델 DEBUG 학습 완료
- 1 positive + 99 sampled negatives ranking 평가 완료
- 체크포인트, 학습 곡선, 비교 CSV 생성 완료

## 입력 파일

기본 실행은 repository와 원본 데이터 폴더가 나란히 있는 현재 구조를 사용합니다.

```text
26_2_Contest/
├── games.csv
├── games_metadata.json
├── recommendations.csv
└── github_upload_repo/
    ├── tabular_embedding/
    ├── text_data/
    ├── mvp_recommendation/
    └── scripts/
```

### Text representation

```text
title + tags + description
→ all-MiniLM-L6-v2
→ mean pooling
→ L2 normalization
→ 384D fixed embedding
```

| 파일 | shape / 역할 |
|---|---|
| `text_data/games_text_ready.csv` | 50,872행, encoder 입력 텍스트 |
| `text_data/emb_text_minilm.npy` | `(50872, 384)`, `float32` |
| `text_data/emb_text_minilm.csv` | `app_id`, `row` mapping |
| `text_data/08_text_tower.py` | `384 → 192 → 64` trainable projection |

검증 결과 전 행이 finite이며 L2 norm은 약 1입니다. 빈 encoder 입력은 0개이고,
256 token 초과는 기존 README와 동일하게 21개입니다.

### Tabular representation

| 파일 | shape / 역할 |
|---|---|
| `tabular_embedding/emb_tabular_svd64.npy` | `(50872, 64)`, `float32` |
| `tabular_embedding/emb_tabular_svd64.csv` | NPY 행 순서의 `app_id` |
| `tabular_embedding/tabular_tower.py` | `64 → 128 → 64` trainable projection |

Text, Tabular, `games.csv`의 50,872개 `app_id`는 집합과 행 순서가 모두 일치합니다.
학습 코드에서는 그래도 행 순서를 가정하지 않고 `app_id → row` mapping을 다시 만듭니다.

## Interaction 통계

| 항목 | 결과 |
|---|---:|
| 전체 interaction | 41,154,794 |
| unique users | 13,781,059 |
| unique games | 37,610 |
| positive ratio | 85.78% |
| negative ratio | 14.22% |
| date 범위 | 2010-10-15 ~ 2022-12-31 |
| date 파싱 실패 | 0 |
| embedding catalog에 포함된 interaction | 100% |
| 최소 5회 조건 충족 사용자 | 1,910,831 |

2GB CSV를 전체 DataFrame으로 한 번에 읽지 않고 100만 행 chunk로 두 번 순회합니다.
첫 pass는 통계와 사용자별 count를 계산하고, 두 번째 pass는 선택된 DEBUG 사용자의
전체 interaction만 가져옵니다.

## DEBUG split

고정 seed 42로 최소 interaction 5개인 사용자 5,000명을 먼저 선택한 뒤,
각 사용자의 전체 이력을 날짜순으로 정렬해 80/10/10으로 분리했습니다.

| split | interactions | users | games | positive ratio |
|---|---:|---:|---:|---:|
| Train | 47,417 | 5,000 | 7,684 | 85.23% |
| Validation | 6,610 | 5,000 | 2,646 | 83.89% |
| Test | 6,610 | 5,000 | 2,544 | 82.63% |

각 사용자에 대해 다음 조건을 assert합니다.

```text
max(train_date) <= min(validation_date)
max(validation_date) <= min(test_date)
```

`review_id`가 split 사이에 중복되지 않는지, target이 결측이 아닌지, 모든 `app_id`가
embedding bank에 존재하는지도 검사합니다.

## 모델 구조

```text
IDUserEncoder: user_id → 64D

Tabular only: 64D SVD → TabularTower → 64D
Text only: 384D MiniLM → TextTower → 64D
Fusion: [TabularTower 64D ; TextTower 64D] → concat → MLP → 64D

Scorer input:
[user, game, user * game, abs(user - game)] = 256D
→ 128 → 64 → logit
```

Loss는 `BCEWithLogitsLoss`입니다. DEBUG train의 positive 비율이 85.23%이지만,
첫 비교에서는 원 분포를 유지하기 위해 class weight를 적용하지 않았습니다.
필요하면 `--use-pos-weight`를 켤 수 있습니다.

## DEBUG 결과

Validation loss 기준 early stopping을 사용했습니다. 각 모델은 epoch 5에서 멈췄고,
best checkpoint는 epoch 2였습니다.

| model | Test AUC | Recall@10 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|
| Tabular only | 0.7923 | 0.2095 | 0.0850 | 0.0779 |
| Text only | 0.7266 | 0.0915 | 0.0421 | 0.0519 |
| Tabular + Text Fusion | 0.7784 | **0.2185** | **0.0994** | **0.0919** |

Fusion과 가장 좋은 단일 modality인 Tabular only의 차이:

| metric | absolute | relative |
|---|---:|---:|
| Recall@10 | +0.0090 | +4.30% |
| NDCG@10 | +0.01438 | +16.92% |

DEBUG 실험에서는 Fusion이 pointwise AUC는 Tabular only보다 낮았지만 실제 sampled ranking의
Recall@10, NDCG@10, MRR은 더 높았습니다. 개인 MVP의 목적에는 긍정적인 결과지만,
5,000명 표본이므로 전체 데이터 일반화 성능으로 해석해서는 안 됩니다.

## 실행 방법

repository root에서 실행합니다.

### 1. Text 입력과 embedding 재생성

```bash
python text_data/steam_text_preprocessing.py
python text_data/09_encode_text_embeddings.py --batch-size 128 --device cpu
```

### 2. 전체 interaction 분석과 DEBUG split

```bash
python scripts/prepare_mvp_data.py \
  --chunksize 1000000 \
  --debug-users 5000 \
  --min-user-interactions 5 \
  --seed 42
```

### 3. 세 모델 학습과 평가

```bash
python scripts/train_mvp.py \
  --epochs 10 \
  --patience 3 \
  --batch-size 1024 \
  --num-negatives 99 \
  --max-ranking-positives 2000 \
  --device auto
```

## 주요 결과 파일

```text
outputs/mvp/
├── data_summary.json
├── debug_train.parquet
├── debug_validation.parquet
├── debug_test.parquet
├── ranking_candidates.parquet
├── mvp_model_comparison.csv
├── fusion_improvement.json
├── *_history.csv
├── checkpoints/
│   └── *_best.pt
└── figures/
    ├── *_training_curve.png
    ├── model_recall_at_10.png
    └── model_ndcg_at_10.png
```

## 중요한 한계: temporal leakage 가능성

현재 Tabular embedding에는 `rating`, `positive_ratio`, `user_reviews`, HF의 positive/negative,
recommendations, playtime, peak CCU 같은 수집 시점 누적 통계가 포함돼 있습니다.
이 값이 test interaction 이후 시점까지 반영한 snapshot이면 chronological evaluation에서
미래 정보를 간접적으로 포함할 수 있습니다.

따라서 현재 결과는 **개인 MVP 및 파이프라인 검증 결과**로 사용해야 합니다. 정식 비교 전에는
시변 리뷰·인기도·플레이타임 집계 컬럼을 제외한 leakage-safe Tabular bank를 별도로 생성해
동일 실험을 반복하는 것을 권장합니다.

또한 sampled ranking은 positive 1개와 unseen game 99개를 사용하므로, positive가 하나인 현재
설정에서는 Recall@K와 HitRate@K가 동일합니다.

## Leakage-safe 후속 실험

초기 Tabular bank의 시간 누수 가능성을 줄이기 위해 더 보수적인 `leakage_safe_v1` bank를
추가했습니다.

### 제외한 정보

- rating, positive ratio, review 수
- HF positive/negative/recommendations
- 평균·중앙 playtime, peak CCU, owner estimate
- 현재 가격과 할인율
- Metacritic 점수
- 현재 achievement 및 DLC 수

현재 가격·할인·DLC·도전과제도 시간에 따라 바뀔 수 있어 보수적으로 제외했습니다.

### 남긴 정보

- 출시 연도와 월
- 지원 OS와 Steam Deck 지원
- required age
- genres, categories
- 지원 언어와 full-audio 언어
- 개발사와 배급사

Text와 겹치는 title, description, tags는 계속 제외했습니다.

```text
safe feature bank: (50,872, 1,305)
safe SVD bank:     (50,872, 64), float32, L2 normalized
SVD explained variance: 0.9345
```

### 3-seed 결과

사용자 표본, 모델 초기화, negative sampling seed를 `42, 123, 2026`으로 바꾸고 각 seed에서
사용자 5,000명의 chronological split을 새로 만들었습니다.

| model | Test AUC | Recall@10 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|
| Safe Tabular only | 0.7432 ± 0.0061 | 0.1595 ± 0.0232 | 0.0838 ± 0.0138 | 0.0824 ± 0.0115 |
| Text only | 0.7344 ± 0.0078 | 0.1040 ± 0.0110 | 0.0481 ± 0.0054 | 0.0553 ± 0.0034 |
| Safe Tabular + Text | 0.7429 ± 0.0017 | **0.2180 ± 0.0141** | **0.1318 ± 0.0135** | **0.1270 ± 0.0122** |

각 seed의 가장 좋은 단일 modality 대비 Fusion의 paired absolute improvement:

| metric | mean ± SD |
|---|---:|
| Recall@10 | **+0.0585 ± 0.0372** |
| NDCG@10 | **+0.0480 ± 0.0268** |

Fusion의 Test AUC는 Safe Tabular와 거의 같지만 ranking metric은 세 seed 모두 높았습니다.
따라서 Text의 증분은 pointwise positive/negative 분류보다는 후보 순위를 더 잘 배치하는 데서
나타난 것으로 해석합니다. 상대 개선율은 단일 baseline이 작을 때 커 보일 수 있으므로
absolute difference를 중심 결과로 사용합니다.

Safe bank 생성과 multi-seed 집계:

```bash
python scripts/build_leakage_safe_tabular.py
python scripts/aggregate_multiseed.py
```

## 이미지 modality 결정

팀 이미지 실험 문서의 최종 채택안은 다음과 같습니다.

```text
CLIP ViT-B/32 + squash resize
encoder frozen
ImageTower projection만 학습
emb_clip_squash.npy: (50,864, 512), float16, L2 normalized
missing 8 games: catalog mean vector
```

팀 실험에서는 CLIP squash의 태그 linear-probe macro-AUC가 `0.7683 ± 0.0019`로 가장 높았고,
ResNet18 concat의 추가 기여는 순열 대조 후 확인되지 않았습니다.

그러나 현재 workspace에는 다음 파일이 없습니다.

- `emb_clip_squash.npy` 및 mapping CSV
- `steam_image_links.csv`
- 다운로드된 약 2.42GB 이미지 50,864장

CPU 환경에서 이 자료를 다시 수집하고 CLIP으로 5만 장을 재인코딩하는 것은 개인 MVP의 다음
검증보다 비용이 크므로 이번 실행에서는 이미지를 제외했습니다. 팀원의 최종 NPY를 나중에 받을
수 있다면 기존 `Data_process/07_image_tower.py`의 평균 벡터 결측 처리와 `512 → 256 → 64`
projection을 사용해 다음 ablation만 추가하면 됩니다.

```text
Safe Tabular + Text
Safe Tabular + Text + Image
```

이때 세 모델은 동일 split과 동일 ranking candidates를 사용해야 하며, 이미지 헤더에 게임명이
포함돼 Text와 신호가 중복될 수 있으므로 반드시 image 추가 전후를 비교해야 합니다.

## 50,000-user 중간 규모 실험

이미지는 제외하고 사용자 표본을 5,000명에서 50,000명으로 확대했습니다. 최소 interaction
5개 조건을 만족한 사용자에서 seed 42로 사용자 ID를 먼저 표본 추출한 뒤 전체 이력을 가져와
동일한 chronological split을 적용했습니다.

| split | interactions | users | games | positive ratio |
|---|---:|---:|---:|---:|
| Train | 451,741 | 50,000 | 18,276 | 85.23% |
| Validation | 63,754 | 50,000 | 8,518 | 84.02% |
| Test | 63,754 | 50,000 | 8,122 | 83.41% |

Test positive 10,000개를 고정하고 각각 사용자의 train/validation/test 이력에 없는 게임 99개를
붙여 1,000,000행의 `ranking_candidates.parquet`를 만들었습니다. 세 신경망과 Popularity
baseline이 모두 이 파일을 재사용합니다.

### 50k 신경망 결과

| model | Test AUC | Recall@10 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|
| Safe Tabular only | 0.7663 | 0.2886 | 0.1716 | 0.1573 |
| Text only | 0.7672 | 0.2801 | 0.1781 | 0.1673 |
| Safe Tabular + Text | **0.7847** | **0.2978** | **0.1845** | **0.1695** |

Fusion은 가장 좋은 단일 modality 대비 Recall@10 `+0.0092`, NDCG@10 `+0.0063`으로 여전히
우세하지만, 5k 실험보다 증분은 작아졌습니다. 규모가 커질수록 단일 modality도 더 안정적으로
학습되기 때문으로 볼 수 있습니다.

### Popularity baseline과 핵심 발견

Popularity는 train의 `is_recommended=True` 횟수만 사용합니다. Validation/Test 정보나 현재
Steam 리뷰 수는 사용하지 않습니다.

| model | Recall@10 | NDCG@10 | MRR | Long-tail Recall@10 | Catalog coverage@10 |
|---|---:|---:|---:|---:|---:|
| Train-positive Popularity | **0.7915** | **0.5316** | **0.4600** | 0.0000 | 0.0952 |
| Safe Tabular | 0.2886 | 0.1716 | 0.1573 | **0.1192** | **0.1025** |
| Text | 0.2801 | 0.1781 | 0.1673 | 0.0960 | 0.0984 |
| Fusion | 0.2978 | 0.1845 | 0.1695 | 0.1146 | 0.0981 |

Long-tail은 train positive 횟수의 하위 25%로 정의했습니다. 현재 데이터에서는 threshold가
1회이며 평가 positive 646개가 해당합니다.

Popularity가 전체 ranking metric을 크게 앞서는 이유는 두 가지입니다.

1. Test positive는 실제 사용자가 추천한 게임이라 인기 게임 비율이 높습니다.
2. Negative 99개는 전체 catalog의 미관측 게임에서 균등 추출하므로 대부분 덜 인기 있습니다.

반면 현재 신경망은 관측된 review의 추천/비추천을 `BCEWithLogitsLoss`로 분류할 뿐, 미관측
게임을 negative로 직접 학습하지 않습니다. 따라서 이 모델은 엄밀히 말하면 personalized item
ranking보다 review sentiment prediction에 더 가깝습니다.

Popularity의 높은 평균 성능도 비용이 있습니다. train positive가 1회 이하인 long-tail positive는
Recall@10이 0이지만 Fusion은 0.1146을 회수했습니다. 평균 정확도와 신규·희소 게임 노출이
서로 다른 목표임을 보여주는 결과입니다.

### 다음 모델링 단계

이 결과 이후에는 단순히 현재 BCE 모델의 사용자 수를 더 늘리는 것이 우선이 아닙니다. 다음은
train positive interaction마다 미관측 게임을 negative로 sampling하고 개인화 순위를 직접
학습하는 모델이어야 합니다.

권장 비교:

```text
Popularity
Matrix Factorization + BPR
Tabular BPR
Text BPR
Tabular + Text Fusion BPR
```

동일한 50k split과 고정 ranking candidates를 그대로 사용하면 현재 결과와 공정하게 비교할 수
있습니다. Popularity를 넘는지와 함께 long-tail Recall 및 catalog coverage가 유지되는지도 봐야
합니다.

실행 파일:

```bash
python scripts/evaluate_popularity.py ...
python scripts/summarize_50k.py
```

## BPR personalized ranking 후속 실험

50k pointwise 모델이 Popularity를 넘지 못한 원인을 반영해 학습 objective를 pairwise ranking으로
바꿨습니다.

```text
train positive: is_recommended=True
train negative: 같은 사용자의 train 관측 이력에 없는 catalog game
loss: -log sigmoid(score_positive - score_negative)
```

Train negative sampler는 validation/test의 게임 ID를 보지 않습니다. Validation negative sampler만
validation 평가 시 train+validation 관측 이력을 제외합니다. Test ranking candidates는 기존과
동일하게 사용자의 train/validation/test 전체 관측 이력을 제외합니다.

Train positive가 한 개 이상인 사용자는 49,742명입니다. 나머지 258명은 BPR preference를 학습할
positive가 없어 BPR 공통 평가에서 제외했습니다. 기존 10,000개 query 중 9,984개가 남았으며,
Popularity·pointwise·BPR 모든 최종 비교는 이 9,984개와 동일한 99 negatives를 사용합니다.

### 비교 모델

| 모델 | Game representation | 학습 objective |
|---|---|---|
| MF-BPR | 학습 가능한 item embedding 64D | BPR |
| Tabular-BPR | Safe TabularTower 64D | BPR |
| Text-BPR | TextTower 384→64D | BPR |
| Fusion-BPR | Tabular 64D + Text 64D → concat MLP 64D | BPR |

각 모델은 최대 10 epoch의 동일 예산으로 실행했습니다. 네 모델 모두 epoch 10에도 validation BPR
loss가 감소하고 있었으므로 아래 결과는 완전 수렴값이 아니라 **고정 예산 MVP 결과**입니다.

### 최종 고정-candidate 결과

| model | Recall@10 | NDCG@10 | MRR | Long-tail Recall@10 | Catalog coverage@10 |
|---|---:|---:|---:|---:|---:|
| **MF-BPR** | **0.7968** | **0.5386** | **0.4671** | 0.0000 | 0.0975 |
| Text-BPR | 0.7935 | 0.5309 | 0.4561 | 0.0729 | 0.1038 |
| Popularity | 0.7915 | 0.5316 | 0.4601 | 0.0000 | 0.0952 |
| Fusion-BPR | 0.7627 | 0.4441 | 0.3536 | 0.0326 | 0.0992 |
| Tabular-BPR | 0.6871 | 0.3999 | 0.3237 | 0.0698 | **0.1050** |
| Fusion pointwise | 0.2977 | 0.1844 | 0.1695 | 0.1147 | 0.0981 |
| Tabular pointwise | 0.2887 | 0.1717 | 0.1574 | **0.1194** | 0.1025 |
| Text pointwise | 0.2802 | 0.1782 | 0.1673 | 0.0961 | 0.0984 |

MF-BPR은 Popularity 대비 Recall@10 `+0.00531`, NDCG@10 `+0.00693`, MRR `+0.00699`로
소폭 앞섰습니다. Pairwise objective로 바꾸자 pointwise 모델과 Popularity 사이의 큰 격차가
사라졌다는 것이 핵심입니다.

Text-BPR은 Recall@10에서 Popularity보다 `+0.00200` 높고 NDCG@10은 `-0.00078`로 사실상
비슷하지만, Popularity와 MF-BPR이 전혀 찾지 못한 long-tail positive를 `7.29%` 회수했습니다.
Catalog coverage도 Popularity보다 높습니다. 따라서 평균 성능과 신규·희소 게임 노출을 함께
고려하면 Text-BPR이 가장 균형 잡힌 MVP 후보입니다.

### Fusion에 대한 수정된 결론

Pointwise BCE에서는 Tabular+Text Fusion이 각 단일 modality보다 좋았지만, BPR에서는 단순 concat
Fusion이 Text-BPR보다 Recall@10 `0.0307` 낮았습니다. 따라서 현재 결과는 “Fusion이 항상
좋다”가 아니라 다음을 의미합니다.

- Text와 Tabular는 pointwise sentiment prediction에서 상호보완적입니다.
- Personalized pairwise ranking에서는 단순 동일비중 concat이 최선이 아닙니다.
- Safe Tabular block의 약한 신호가 강한 Text ranking 신호를 희석할 수 있습니다.
- 다음 Fusion은 gated/residual fusion 또는 modality dropout, 별도 user-modality preference가
  필요합니다.

또한 BPR 모델의 낮은 observed-test AUC는 실패 지표로 해석하지 않습니다. BPR은 추천/비추천 리뷰
분류가 아니라 positive와 미관측 item의 상대 순서를 학습하므로 pointwise AUC와 직접 비교할 수
없습니다.

### BPR 실행

```bash
python scripts/train_bpr.py \
  --data-dir outputs/mvp_50k/data_seed_42 \
  --candidates outputs/mvp_50k/models_seed_42/ranking_candidates.parquet \
  --output-dir outputs/mvp_50k/bpr_seed_42 \
  --epochs 10 --patience 3 --batch-size 2048 --seed 42

python scripts/evaluate_pointwise_fixed.py ...
python scripts/summarize_bpr.py
```

주요 결과는 `outputs/mvp_50k/bpr_seed_42/final_fixed_candidate_comparison.csv`와
`bpr_summary.json`에 저장됩니다.

## Gated Fusion 후속 실험

단순 concat fusion이 Text-BPR보다 낮았던 원인을 확인하기 위해 Text 신호를 90%로
초기화한 두 가지 선택적 fusion을 같은 50k split과 같은 고정 후보에서 비교했습니다.

| 모델 | 구조 | Recall@10 | NDCG@10 | MRR | Long-tail Recall@10 | Coverage@10 |
|---|---|---:|---:|---:|---:|---:|
| Text-BPR | Text 단독 | **0.7935** | **0.5309** | **0.4561** | **0.0729** | 0.1038 |
| User gated BPR | 사용자별 Text/Tabular 선호도 | 0.7795 | 0.4952 | 0.4148 | 0.0620 | **0.1096** |
| Item gated BPR | 게임·차원별 Text/Tabular gate | 0.7746 | 0.4792 | 0.3954 | 0.0171 | 0.1022 |
| Concat Fusion-BPR | 단순 concat MLP | 0.7627 | 0.4441 | 0.3536 | 0.0326 | 0.0992 |

User gate는 단순 concat보다 Recall@10을 `+0.0168`, NDCG@10을 `+0.0510`
개선했습니다. 따라서 modality를 선택적으로 섞는 방향 자체는 유효합니다. 하지만 가장
강한 Text-BPR보다 Recall@10이 `-0.0139`, NDCG@10이 `-0.0357` 낮아, 현재 MVP의
최종 content 모델은 **Text-BPR**로 유지합니다.

학습 후 User gate의 Text 가중치는 평균 `0.8992`(p10 `0.8788`, 중앙값 `0.9021`,
p90 `0.9163`)였습니다. 사용자가 대체로 Text를 강하게 선호하면서 일부에만 Tabular를
보조적으로 쓰는 패턴입니다. Item-level vector gate는 평균 `0.4796`까지 내려가 Tabular가
과도하게 개입했고, Text 단독 성능을 충분히 보존하지 못했습니다.

재현 명령:

```bash
python scripts/train_bpr.py \
  --data-dir outputs/mvp_50k/data_seed_42 \
  --candidates outputs/mvp_50k/models_seed_42/ranking_candidates.parquet \
  --output-dir outputs/mvp_50k/gated_seed_42 \
  --epochs 10 --patience 3 --batch-size 2048 --seed 42 \
  --modes text_anchored_gated_bpr user_modality_gated_bpr

python scripts/summarize_gated_fusion.py
```

주요 산출물:

- `outputs/mvp_50k/gated_seed_42/all_model_fixed_candidate_comparison.csv`
- `outputs/mvp_50k/gated_seed_42/gated_fusion_deltas_vs_text.csv`
- `outputs/mvp_50k/gated_seed_42/gated_fusion_summary.json`
- `outputs/mvp_50k/gated_seed_42/figures/gated_fusion_recall_at_10.png`
- `outputs/mvp_50k/gated_seed_42/figures/learned_text_gate_summary.png`

## MF + Text score Hybrid와 3-seed 재현성

Gated fusion 이후에는 MF-BPR와 Text-BPR를 각각 학습한 뒤 추천 점수 수준에서 결합했습니다.
두 점수는 사용자 질의별 z-score로 표준화해 모델별 점수 범위 차이를 제거했습니다.

가중치는 test가 아닌 validation 후보 10,000개에서만 선택합니다.

```text
hybrid_score = α × standardized_MF + (1 - α) × standardized_Text
```

두 가지 선택 규칙을 비교했습니다.

1. Accuracy hybrid: validation NDCG@10을 최대화
2. Balanced hybrid: 최고 validation NDCG@10의 99.5% 이상을 유지하면서 long-tail Recall@10 최대화

또한 모델 간 공정한 비교를 위해 각 모델 학습 시작 시 동적 negative sampler를 같은 seed로
재설정했습니다. 따라서 같은 seed의 MF와 Text는 epoch별로 동일한 negative 표본을 봅니다.

### 3-seed 결과

동일한 50k split과 동일한 9,984개 test 후보 질의를 유지하고 학습 초기화와 negative sampling
seed만 `42`, `7`, `2026`으로 변경했습니다. 아래 표는 평균 ± 표본 표준편차입니다.

| 모델 | Recall@10 | NDCG@10 | MRR | Long-tail Recall@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| MF-BPR | **0.7960 ± 0.0007** | 0.5372 ± 0.0014 | 0.4656 ± 0.0016 | 0.0000 ± 0.0000 | 0.0974 ± 0.0003 |
| Text-BPR | 0.7858 ± 0.0064 | 0.5163 ± 0.0053 | 0.4397 ± 0.0046 | **0.0543 ± 0.0123** | **0.1039 ± 0.0011** |
| Balanced MF+Text | 0.7906 ± 0.0086 | **0.5404 ± 0.0032** | **0.4713 ± 0.0034** | 0.0279 ± 0.0198 | 0.1020 ± 0.0025 |

Accuracy hybrid는 세 seed 모두 `α=1.0`, 즉 MF 단독을 선택했습니다. 반면 balanced 규칙은
MF 가중치를 seed별로 `0.20`, `0.65`, `0.10`으로 선택했습니다. Balanced hybrid는 세 seed
모두 MF보다 NDCG@10과 MRR이 높았고, MF가 전혀 추천하지 못한 long-tail positive도 일부
회복했습니다. 대신 Recall@10 평균은 MF보다 약 `0.0054` 낮았습니다.

따라서 현재 선택은 목표에 따라 나뉩니다.

- 정답 게임의 Top-10 진입률만 우선하면 `MF-BPR`
- 상위 순위 품질, long-tail 노출, catalog coverage까지 함께 보면 `Balanced MF+Text hybrid`
- 신규 사용자에는 두 모델 모두 별도 cold-start 정책이 필요함
- 신규 게임에는 ID 기반 MF보다 Text-BPR 경로가 유리함

세 seed는 학습 안정성을 확인하기에는 유용하지만 강한 통계적 유의성을 주장하기에는 적습니다.
또한 balanced α의 seed 간 변동이 크므로 실제 서비스에서는 validation 재튜닝을 유지해야 합니다.

재현 명령 예시:

```bash
python scripts/train_bpr.py \
  --data-dir outputs/mvp_50k/data_seed_42 \
  --candidates outputs/mvp_50k/models_seed_42/ranking_candidates.parquet \
  --output-dir outputs/mvp_50k/repro_seed_42 \
  --epochs 10 --patience 3 --batch-size 2048 --seed 42 \
  --modes mf_bpr text_bpr

python scripts/evaluate_mf_text_hybrid.py \
  --data-dir outputs/mvp_50k/data_seed_42 \
  --test-candidates outputs/mvp_50k/models_seed_42/ranking_candidates.parquet \
  --checkpoint-dir outputs/mvp_50k/repro_seed_42/checkpoints \
  --output-dir outputs/mvp_50k/repro_seed_42/hybrid

python scripts/summarize_hybrid_reproducibility.py
```

최종 집계 파일은 `outputs/mvp_50k/hybrid_reproducibility/`에 있습니다.
이 절의 3-seed 결과가 앞선 단일 seed BPR 표보다 우선하는 최종 검증 결과입니다.

## 최종 Top-K 추천 생성

학습된 seed 42 체크포인트를 사용해 기존 사용자에게 전체 50,872개 게임을 점수화하는 inference
파이프라인을 추가했습니다. MF-BPR, Text-BPR, Balanced Hybrid를 한 번에 비교할 수 있으며
train·validation·test에서 이미 관측된 게임은 기본적으로 제외합니다.

```bash
python scripts/recommend_users.py \
  --user-ids 13 7654189 14306011 \
  --top-k 10 \
  --output recommendation_mvp/sample_recommendations.csv
```

출력에는 `app_id`, 게임명, 태그, MF/Text 표준화 점수, 각 모델의 전체 카탈로그 순위,
추천 신호와 가중치가 포함됩니다. 자세한 사용법과 컬럼 설명은
`recommendation_mvp/README.md`를 참고합니다.

학습에 없는 신규 사용자는 `scripts/recommend_new_user.py`로 처리합니다. 선호 태그·좋아하는
게임이 있으면 MiniLM 콘텐츠 프로필과 Train Popularity를 결합하고, 입력이 전혀 없으면 Train
Popularity로 fallback합니다. 신규 사용자 경로에서도 validation/test 정보는 사용하지 않습니다.

## Streamlit UI와 다양성 옵션

최종 사용자 흐름은 Streamlit UI로 연결했습니다.

```bash
streamlit run recommendation_mvp/app.py
```

UI에서 사용자 유형, 태그, 좋아하는 게임, 추천 수, 모델을 선택하고 결과를 CSV로 내려받을 수
있습니다. 다양성 옵션은 기본 ON이며 상위 후보 10배에 MMR을 적용합니다. 기본 가중치는 원래
관련성 0.65, 중복 억제 0.35입니다. Witcher 샘플에서는 평균 원본 점수 99.74%를 유지하면서
고중복 DLC 제목 pair가 1개에서 0개로 줄었습니다. 상세 사용법은
`recommendation_mvp/README.md`에 있습니다.
