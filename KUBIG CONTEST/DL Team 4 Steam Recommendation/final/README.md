# Final Recommendation Handoff

이 폴더는 UI 구현 담당자에게 넘기기 위한 최종 추천 API 산출물입니다.
입출력 계약은 [api_contract.json](api_contract.json)에도 같은 내용으로 고정해두었습니다.

핵심 구조는 cold와 warm 모두 같습니다.

```text
user/query vector 64D
    dot product
game_fusion/emb_game_concat_64.npy  (50,872 x 64)
    top-k app_id
```

즉 두 버전의 차이는 “user/query vector를 어떻게 만들 것인가”뿐입니다. 후보 게임은 동일한 frozen multimodal game bank에서 가져오고, 최종 검색은 dot product로 수행합니다.

## 빠른 사용법

```python
from final import FinalRecommendationEngine

engine = FinalRecommendationEngine()

cold_ids = engine.recommend_cold(
    preferred_genres=["RPG", "Open World"],
    liked_game_ids=[292030],
    interest_weight=1.0,
    k=10,
)

warm_ids = engine.recommend_warm(
    user_id=13,
    interest_weight=0.6,
    k=10,
)
```

UI용 공개 출력은 항상 단순한 `list[int]`입니다.

```python
[12345, 67890, ...]  # top-k app_id
```

디버깅이나 화면 표시용 점수가 필요하면 아래 detail 함수를 사용합니다.

```python
engine.recommend_cold_detail(...)
engine.recommend_warm_detail(...)
```

detail 출력 컬럼은 다음과 같습니다.

```text
rank, app_id, score, content_score_z, baseline_score_z, baseline,
interest_weight, method
```

## Cold 입력/출력 계약

입력:

```python
preferred_genres: list[str]
liked_game_ids: list[int]
interest_weight: float  # 0.0 이상 1.0 이하
k: int
```

출력:

```python
list[int]  # 추천된 top-k app_id
```

`preferred_genres`에는 `RPG`, `Adventure`, `Strategy`처럼 저장된 genre prototype 이름을 넣을 수 있습니다. 또한 catalog의 Steam tag도 받을 수 있습니다. 예를 들어 `Open World`는 별도 genre prototype에는 없지만 catalog tag로 존재하므로, 해당 tag가 붙은 게임들의 game vector 평균을 prototype처럼 만들어 사용합니다.

Cold 점수 계산은 다음과 같습니다.

```text
cold_user_vector = normalize(mean(selected game vectors + genre/tag prototypes))
content_score = game_bank @ cold_user_vector

final_score = interest_weight * z(content_score)
            + (1 - interest_weight) * z(train_popularity)
```

선택한 `liked_game_ids`는 추천 결과에서 제외합니다. `preferred_genres`와 `liked_game_ids`가 모두 비어 있으면 user/query vector를 만들 수 없으므로 에러를 냅니다.

## Warm 입력/출력 계약

입력:

```python
user_id: int
interest_weight: float  # 0.0 이상 1.0 이하
k: int
```

출력:

```python
list[int]  # 추천된 top-k app_id
```

Warm은 먼저 `history_user_tower/results_seed_42/user_profiles_hours.npy`에 저장된 positive-history 기반 user profile을 사용합니다. 이 profile은 기존 사용자의 positive history를 frozen game embedding 공간에서 pooling한 64D vector입니다.

Warm 점수 계산은 다음과 같습니다.

```text
warm_user_vector = saved positive-history profile
content_score = game_bank @ warm_user_vector

final_score = interest_weight * z(content_score)
            + (1 - interest_weight) * z(MF-BPR score)
```

이미 본 게임은 추천 결과에서 제외합니다.

## interest_weight 의미

`interest_weight`는 cold와 warm에서 같은 의미로 사용합니다.

```text
0.0 = baseline 중심
1.0 = content/user intent dot product 중심
```

Cold의 baseline은 `train_popularity`입니다. 신규 사용자는 collaborative 정보가 없기 때문입니다.

Warm의 baseline은 `MF-BPR`입니다. 기존 사용자는 interaction 기반 collaborative signal을 쓸 수 있기 때문입니다.

점수 결합 전에는 항상 available catalog 기준으로 z-score normalization을 수행합니다. raw dot product와 raw popularity count를 바로 더하면 scale이 달라져 popularity가 점수를 압도할 수 있기 때문입니다.

## 학습 여부

[engine.py](engine.py) 자체에서는 새 train을 수행하지 않습니다. 이 파일은 이미 학습되거나 생성된 artifact를 조합하는 inference wrapper입니다.

사용하는 학습/생성 artifact는 다음과 같습니다.

```text
Game Tower output:
  game_fusion/emb_game_concat_64.npy

Warm content profile:
  history_user_tower/results_seed_42/user_profiles_hours.npy

Warm baseline:
  outputs/mvp_50k/repro_seed_42/checkpoints/mf_bpr_best.pt

Cold baseline:
  recommendation_mvp/deploy_data/train_positive_counts.csv

Cold genre prototype:
  history_user_tower/results_seed_42/genre_prototypes.npy
```

따라서 dot product 과정 자체는 학습이 아니라 retrieval입니다. 학습은 그 전에 끝났습니다. Game Tower embedding, MF-BPR user/item embedding, history user profile 또는 MLP 실험이 upstream에서 만들어졌고, final 단계는 이를 로드해 점수를 계산합니다.

## 성능 평가 근거

이 final wrapper와 직접 같은 형태의 UI 입력을 평가한 별도 online metric은 아직 없습니다. 대신 현재 시스템의 성능 근거는 아래 upstream 평가 결과를 사용합니다.

### 1. Multimodal Game Bank + BPR 평가

파일:

```text
recommendation_mvp/model_artifacts/multimodal_evaluation_summary_seed42.json
```

주요 결과:

```text
Multimodal only:
  Recall@10 = 0.8026
  NDCG@10  = 0.5182
  Long-tail Recall@10 = 0.2713

MF + Multimodal balanced:
  Recall@10 = 0.8178
  NDCG@10  = 0.5610
  Long-tail Recall@10 = 0.0388

MF only:
  Recall@10 = 0.7968
  NDCG@10  = 0.5386
```

이 평가는 `user_id` 기반 warm 추천에서 MF signal과 multimodal game bank signal을 결합할 근거입니다.

### 2. History/User Profile dot product 평가

파일:

```text
history_user_tower/results_seed_42/evaluation_metrics.csv
```

주요 결과:

```text
sampled 1 positive + 99 negatives:
  simple_mean Recall@10         = 0.7680
  hours_weighted_mean Recall@10 = 0.7809
  history_mlp_bpr Recall@10     = 0.8175

full catalog:
  hours_weighted_mean Recall@10 = 0.0222
  history_mlp_bpr Recall@10     = 0.0165
```

이 결과 때문에 현재 final warm content vector는 MLP를 기본으로 쓰지 않고, 저장된 `user_profiles_hours.npy` 기반 profile을 사용합니다. sampled 후보에서는 MLP가 강하지만, full catalog에서는 hours-weighted profile이 더 안정적이었기 때문입니다.

## CLI 예시

```bash
python -m final.engine cold --preferred-genres RPG "Open World" --liked-game-ids 292030 -k 10
python -m final.engine warm --user-id 13 --interest-weight 0.6 -k 10
python -m final.engine cold --preferred-genres RPG --liked-game-ids 292030 -k 5 --detail
```

