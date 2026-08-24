# Game Fusion Embedding

Text, Image, Tabular 정보를 하나의 64D **game embedding**으로 통합하는 fusion pipeline입니다.


## 전체 Pipeline

```text
Step 1. Frozen Feature Fusion

MiniLM text bank  [frozen] -> TextTower    -> 64D
CLIP image bank   [frozen] -> ImageTower   -> 64D
SVD tabular bank  [frozen] -> TabularTower -> 64D
                                      |
                                Concat 192D
                                      |
                                  Fusion MLP
                                      |
                             Game Embedding 64D
```

```text
Step 2. Ablation

Text-only 64D
Image-only 64D
Tabular-only 64D

목적: 각 modality 단독 embedding을 만들어 fusion 결과와 비교할 수 있게 함.
```

```text
Step 3. Projection/Fusion Tuning

MiniLM/CLIP/SVD feature bank는 그대로 고정합니다.
다만 각 modality projection tower와 Fusion MLP를 pairwise ranking objective 형태의 코드 경로로 조정합니다.

Positive game score > Negative game score가 되도록 학습
```

```text
Step 4. Partial Adapter + Fusion Tuning

MiniLM/CLIP encoder parameter를 직접 unfreeze하지는 않습니다.
대신 frozen MiniLM/CLIP output 위에 residual adapter를 추가합니다.

MiniLM text bank [frozen] -> Text Adapter  [trainable] -> TextTower  [trainable]
CLIP image bank  [frozen] -> Image Adapter [trainable] -> ImageTower [trainable]
SVD tabular bank [frozen] ------------------------------> TabularTower [trainable]
                                                               |
                                                          Concat 192D
                                                               |
                                                        Fusion MLP [trainable]
                                                               |
                                                     Game Embedding 64D
```

## 핵심 차이

| 구분 | Step 1 | Step 3 | Step 4 |
|---|---|---|---|
| 한 줄 요약 | 이전 단계의 text/image/tabular embedding을 MLP로 묶은 frozen baseline | Step 1 구조에서 projection/fusion 경로를 pairwise ranking 방식으로 조정 | Step 3 구조에 text/image residual adapter를 추가 |
| 입력 | MiniLM, CLIP, SVD feature bank | MiniLM, CLIP, SVD feature bank | MiniLM, CLIP, SVD feature bank |
| 원본 encoder | frozen | frozen | frozen |
| Text/Image adapter | 없음 | 없음 | 있음 |
| 학습되는 부분 | 새로 초기화된 tower/fusion forward로 embedding 생성 | projection towers, FusionTower, 학습용 UserEncoder | text/image adapters, projection towers, FusionTower, 학습용 UserEncoder |
| 결과 해석 | multi-modal frozen fusion baseline | projection/fusion tuning 결과 | partial adapter tuning까지 포함한 확장 결과 |
| 산출물 | `emb_game_concat_64.npy/csv` | `emb_game_finetuned_64.npy/csv` | `emb_game_partial_fusion_tuned_64.npy/csv` |

정리하면:

- **Step 1**: 이미 만들어진 text/image/tabular representation을 각각 64D로 projection하고, concat 후 Fusion MLP로 묶어 64D game embedding을 만듭니다.
- **Step 3**: Step 1과 같은 입력 feature bank를 사용하지만, positive/negative game pair를 이용해 projection/fusion 경로를 조정합니다. 현재 저장된 결과는 실제 interaction이 아니라 synthetic sample 기반 smoke test로 생성되었습니다.
- **Step 4**: Step 3에 비해 text/image feature bank 바로 위에 trainable residual adapter를 하나 더 둡니다. MiniLM/CLIP 자체를 직접 재학습한 것은 아니고, encoder output space를 조정하는 partial adapter tuning입니다.

## 다음 단계 인계 산출물

Step 3과 Step 4 중, recommendation metric에서 비교하여 우수한 결과물을 채택합니다.

```text
game_fusion/emb_game_finetuned_64.npy
game_fusion/emb_game_finetuned_64.csv
game_fusion/emb_game_partial_fusion_tuned_64.npy
game_fusion/emb_game_partial_fusion_tuned_64.csv
```

Step 1 frozen baseline까지 함께 비교하려면 아래도 선택적으로 전달합니다.

```text
game_fusion/emb_game_concat_64.npy
game_fusion/emb_game_concat_64.csv
```

## 산출물 설명

| 산출물 | 의미 | shape |
|---|---|---:|
| `emb_game_finetuned_64.npy` | Step 3 projection/fusion tuned game embedding | `(50872, 64)` |
| `emb_game_finetuned_64.csv` | 위 `.npy` row 순서에 대응하는 `app_id` | `(50872, 1)` |
| `emb_game_partial_fusion_tuned_64.npy` | Step 4 partial adapter + fusion tuned game embedding | `(50872, 64)` |
| `emb_game_partial_fusion_tuned_64.csv` | 위 `.npy` row 순서에 대응하는 `app_id` | `(50872, 1)` |
| `emb_game_concat_64.npy` | Step 1 frozen fusion baseline embedding | `(50872, 64)` |
| `emb_game_concat_64.csv` | Step 1 `.npy` row 순서에 대응하는 `app_id` | `(50872, 1)` |

`.npy`와 `.csv`는 반드시 같은 prefix끼리 함께 사용해야 합니다.

## Step 3과 Step 4 비교 방법

Step 3과 Step 4의 우열은 embedding geometry만으로 유의하지 않습니다.
두 산출물을 다음 단계의 같은 evaluation pipeline에 넣고 recommendation metric으로 비교해야 합니다.

같게 맞춰야 하는 조건:

- 같은 user tower 또는 user embedding 생성 방식
- 같은 train/test split
- 같은 candidate set
- 같은 negative sampling 방식
- 같은 metric 계산 코드

권장 metric:

```text
Recall@K
NDCG@K
```

## 산출물 로드 코드

```python
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path.cwd()
if ROOT.name == "game_fusion":
    ROOT = ROOT.parent

def load_game_embedding(prefix: str):
    emb = np.load(ROOT / "game_fusion" / f"{prefix}.npy").astype(np.float32)
    ids = pd.read_csv(ROOT / "game_fusion" / f"{prefix}.csv")["app_id"].to_numpy()

    assert emb.shape[0] == len(ids)
    assert emb.shape[1] == 64
    return emb, ids

step3_emb, step3_ids = load_game_embedding("emb_game_finetuned_64")
step4_emb, step4_ids = load_game_embedding("emb_game_partial_fusion_tuned_64")

assert np.array_equal(step3_ids, step4_ids)

print(step3_emb.shape)  # (50872, 64)
print(step4_emb.shape)  # (50872, 64)
```

## app_id를 embedding row로 바꾸기

```python
game_emb, game_ids = load_game_embedding("emb_game_partial_fusion_tuned_64")
app_id_to_row = {int(app_id): row for row, app_id in enumerate(game_ids)}

interaction_app_ids = [730, 570, 440]
rows = [app_id_to_row[app_id] for app_id in interaction_app_ids if app_id in app_id_to_row]

batch_game_emb = game_emb[rows]
print(batch_game_emb.shape)  # (valid_items, 64)
```

## 평가 단계에서의 사용 예시

아래 코드는 형태 예시입니다. 실제 user tower는 다음 단계 pipeline에서 정의해야 합니다.

```python
def score_games(user_emb, game_emb):
    # user_emb: (64,)
    # game_emb: (num_games, 64)
    return game_emb @ user_emb

scores_step3 = score_games(user_emb, step3_emb)
scores_step4 = score_games(user_emb, step4_emb)

# 이후 같은 positive/test item과 같은 candidate set으로 Recall@K, NDCG@K 등을 계산합니다.
```

## 현재 산출물의 실제 학습 방식

Step 3/4 노트북에는 BPR loss 형태의 pairwise ranking objective가 구현되어 있습니다.

```python
def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
```

하지만 현재 repo 실행 시점에는 실제 user-item interaction 파일이 발견되지 않았습니다.  
그래서 03/04 노트북은 아래 방식으로 synthetic interaction을 만들어 모델 구조만 smoke test했습니다.

```text
1. game catalog에서 일부 app_id를 임의 샘플링
2. synthetic user_id를 부여
3. 임의 label 또는 positive row를 생성
4. user별 positive game을 기준으로 negative game을 랜덤 샘플링
5. pos_score가 neg_score보다 커지도록 pairwise loss를 계산
6. 학습된 game tower 경로로 전체 game embedding을 export
```

따라서 현재 저장된 Step 3/4 embedding은 **BPR loss 코드 경로를 실행해 만든 결과는 맞지만, 실제 사용자 interaction으로 학습된 추천 모델 결과는 아닙니다.**

실제 성능 비교는 다음 단계에서 실제 interaction split을 사용해 recommendation metric으로 다시 계산해야 합니다.

## 주의사항

- 이 폴더의 결과물은 game embedding입니다.
- user tower는 이 폴더에서 학습하거나 완성하지 않았습니다.
- 최종 추천 성능은 다음 단계에서 user tower와 결합한 downstream evaluation으로 판단해야 합니다.
- 현재 03/04 산출물은 실제 interaction 기반 학습 결과가 아니라 synthetic smoke test 기반 결과입니다.
- Step 4는 MiniLM/CLIP encoder parameter를 직접 재학습한 full fine-tuning이 아닙니다.
- Step 4는 frozen encoder output 위에 residual adapter를 학습한 partial adapter tuning입니다.
