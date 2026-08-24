# History-based User Tower 실험

사용자의 과거 Steam interaction으로 64차원 user embedding을 만드는 실험입니다.
`recommendation.csv` 같은 추천 결과 파일은 학습 데이터로 사용하지 않습니다. 실제 chronological
Train/Validation/Test interaction과 팀의 최종 multimodal game bank를 직접 연결합니다.

## 결론부터 보기

| 모델 | 100개 후보 Recall@10 | 100개 후보 NDCG@10 | 전체 catalog Recall@10 | 전체 catalog Cold Recall@10 |
|---|---:|---:|---:|---:|
| Simple mean | 0.7680 | 0.4870 | 미측정 | 미측정 |
| `log1p(hours)` weighted mean | 0.7809 | 0.5036 | **0.0222** | 0.0000 |
| History MLP + BPR | **0.8175** | **0.5359** | 0.0165 | 0.0000 |

MLP는 1 positive + 99 negative sampled 평가에서는 좋아졌지만, 더 어려운 50,872개 전체
catalog 평가에서는 hours-weighted mean보다 낮았습니다. 따라서 현재의 정직한 판단은 다음과 같습니다.

- **재현 가능한 User Tower와 user embedding 생성은 완료**했습니다.
- sampled 지표만 보면 MLP가 가장 좋지만, 현재 배포 기본값으로 즉시 교체할 근거는 부족합니다.
- 전체 catalog 추천의 현재 권장 user representation은 **hours-weighted mean**입니다.
- 다음 개선 실험은 random negative보다 어려운 hard negative 또는 in-batch negative 학습입니다.

## 사용한 데이터

| 입력 | 역할 |
|---|---|
| `outputs/mvp_50k/data_seed_42/debug_train.parquet` | User Tower 학습과 과거 이력 |
| `debug_validation.parquet` | early stopping과 Test 직전 이력 |
| `debug_test.parquet` | 최종 held-out positive |
| `outputs/mvp_50k/models_seed_42/ranking_candidates.parquet` | 기존 모델과 동일한 100개 후보 평가 |
| `game_fusion/emb_game_concat_64.npy/.csv` | Text + Image + Tabular 64D game bank |

Train은 451,741 interactions, 그중 positive는 385,030개입니다. 각 사용자의 두 번째 positive부터
직전까지의 이력만 사용해 335,288개 prefix 학습 샘플을 만들었습니다. 현재 target 게임을 history에
포함하지 않으므로 target leakage가 없습니다.

## User embedding 생성 방식

```text
과거 positive game embeddings
        ↓ log1p(hours) 가중 평균
64D history vector
        ↓ residual MLP: 64 → 128 → 64
        ↓ L2 normalization
64D user embedding
        ↓ dot product with frozen 64D game embeddings
추천 점수
```

MLP 마지막 층은 0으로 초기화합니다. 따라서 학습 전 출력은 hours-weighted baseline과 정확히 같고,
BPR 학습은 이 baseline에 필요한 보정만 residual로 학습합니다. Game bank는 동결합니다.

학습 loss는 다음 pairwise objective입니다.

```text
BPR loss = -log sigmoid(score(user, positive) - score(user, negative))
```

negative는 해당 사용자의 Train 관측 게임이 아닌 catalog 게임에서 매 epoch 동적으로 뽑습니다.

## 평가 규칙

- Train example: 같은 사용자의 현재 positive보다 **앞선 Train positive만** history로 사용
- Validation profile: Train positive 전체 사용
- Test profile: Train + Validation positive 전체 사용
- 추천 시 제외: 사용자에게 이미 관측된 게임
- strict Cold: positive/negative 여부와 무관하게 **Train에 app_id가 한 번도 없던 게임**
- 모델 선택에 Test 결과를 사용하지 않음

시간 split은 전역 달력 cutoff가 아닙니다. 각 사용자의 interaction을 `(date, review_id)`로 정렬한 뒤
앞 80%/다음 10%/마지막 10%로 나눈 **사용자별 chronological split**입니다. 50,000명 전체에 대해
Train→Validation 및 Validation→Test 경계 순서 위반이 0건임을 다시 확인했습니다.

다만 upstream 50k debug cohort는 split 전에 전체 기간의 유효 interaction이 5개 이상인 사용자를
후보로 삼아 무작위 추출했습니다. User Tower의 history/target에는 미래 데이터가 들어가지 않지만,
사용자 cohort eligibility 자체는 train-only가 아닙니다. 기존 모델과의 동일 조건 비교를 위해 현재
cohort를 유지했으며, 엄격한 최종 실험에서는 전체 사용자에게 먼저 시간 split을 적용한 다음 Train
interaction만으로 사용자를 고르는 것이 맞습니다.

`sampled` 평가는 기존 파이프라인과 같은 1 positive + 99 negative 문제입니다. `full_catalog`는 동일한
9,992개 test query의 target을 전체 50,872개 게임과 비교합니다. sampled 성능이 실제 전체 catalog
추천 성능을 과대평가할 수 있다는 점이 이번 실험에서 확인됐습니다.

Cold 수치는 `catalog-cold`로 해석해야 합니다. 제공된 multimodal game bank의 정형 부분에는 리뷰 수,
플레이타임 같은 catalog aggregate가 포함되며, 이 feature가 interaction split의 당시 시점으로
동결되었다는 보장은 없기 때문입니다. 이는 user history target 누수와는 별개의 평가 한계입니다.

## 실행

저장소 루트에서 실행합니다.

```bash
pip install -r history_user_tower/requirements.txt
python history_user_tower/test_history_user_tower.py
python history_user_tower/experiment.py --epochs 15 --patience 4 --full-catalog
```

CPU 기준 seed 42 전체 실험은 이 환경에서 약 1분대가 걸렸습니다. 경로는 CLI 인자로 바꿀 수 있습니다.

## 기존 사용자와 신규 사용자 추론

첨부 설계에서 요구한 공통 추론 계약을 `inference.py`에 구현했습니다.

```python
from history_user_tower.inference import HistoryUserEncoder

encoder = HistoryUserEncoder()

# 기존 사용자: 실제 positive history와 관측 플레이타임
existing_user = encoder.encode_user(
    history_app_ids=[292030, 489830],
    history_hours=[120.0, 45.0],
)

# 신규 사용자: 선택 게임 + 장르 prototype, 동일한 MLP 사용
new_user = encoder.encode_user(
    history_app_ids=[292030],
    selected_genres=["RPG", "Adventure"],
)

recommendations = encoder.recommend(
    history_app_ids=[292030],
    selected_genres=["RPG"],
    top_k=10,
)
```

`encode_user(...)`는 항상 L2-normalized `(64,)` `float32`를 반환합니다. 신규 사용자는 플레이타임이
없으므로 선택 게임과 장르 prototype을 동일 가중치로 pooling합니다. 장르 prototype은 해당 장르의
frozen multimodal game embedding 평균이며 `genre_prototypes.npy/.csv`에 저장했습니다.

현재 MLP는 sampled 후보에는 강하지만 full-catalog에서는 hours pooling보다 낮았으므로 운영 비교 시
`apply_mlp=False`도 반드시 함께 평가해야 합니다. `False`는 MLP 전의 normalized intent/history
vector를 반환합니다.

## 주의사항 재감사 결과

| 점검 항목 | 결과 |
|---|---|
| User/Game 출력 차원 | 모두 64D |
| 동일 latent space 학습 | frozen 실제 Game bank와 dot-product BPR 사용 |
| Game bank 고정 | 학습 중 frozen |
| history 입력 | Train positive만 사용 |
| `hours` 사용 위치 | 과거 history 가중치에만 사용 |
| target 포함 여부 | prefix 생성 후 target을 추가하므로 leakage 없음 |
| Validation/Test history | Train / Train+Validation까지만 사용 |
| User ID 의존성 | 없음; 임의 history/intent를 입력 가능 |
| negative | 사용자의 Train 관측 게임을 제외한 uniform random negative |
| 정규화 | Game/User/prototype 모두 L2 norm 1 |
| `app_id` alignment | `.csv` 순서와 `.npy` row를 assert |
| 재사용 함수 | `encode_user(...) -> np.ndarray(shape=(64,))` 제공 |

따라서 기존 학습 결과를 폐기하거나 다시 학습할 오류는 없었습니다. 수정이 필요했던 부분은 공통 추론
함수와 장르 intent 경로의 부재, 그리고 split/cohort 설명의 정확성입니다. 이 부분을 보완했습니다.

## 산출물

`results_seed_42/`의 파일은 다음과 같습니다.

| 파일 | 설명 |
|---|---|
| `evaluation_metrics.csv` | sampled/full-catalog, Warm/Cold 평가 결과 |
| `training_history.csv` | epoch별 Train/Validation BPR loss |
| `history_user_tower.pt` | 학습된 residual MLP checkpoint |
| `user_embeddings.npy` | MLP가 만든 `(49,848, 64)` user embedding |
| `user_profiles_hours.npy` | 현재 권장 hours-weighted `(49,848, 64)` profile |
| `user_profiles_simple.npy` | 비교용 simple-mean profile |
| `user_embeddings.csv` | `user_id → embedding_row` 매핑 |
| `genre_prototypes.npy/.csv` | 신규 사용자 장르 intent용 26개 normalized prototype |
| `run_summary.json` | 입력 규모, Cold 정의, 주의사항 |

세 `.npy` 파일은 모두 동일한 `user_embeddings.csv` 행 매핑을 사용합니다. Fusion 담당자는 먼저
`user_profiles_hours.npy`를 baseline으로 연결하고, `user_embeddings.npy`는 MLP ablation으로 비교하는
것을 권장합니다.

## 다음 실험

1. validation에서 full-catalog proxy를 만들고 모델 선택 기준을 sampled loss에서 바꾸기
2. in-batch negative 또는 content-similar hard negative로 MLP 재학습
3. 인기도 편향을 제어한 hybrid 및 reranking 실험
4. seed 42/43/44 반복으로 평균과 표준편차 보고
5. 출시 시점 snapshot feature로 엄밀한 item-cold 평가
