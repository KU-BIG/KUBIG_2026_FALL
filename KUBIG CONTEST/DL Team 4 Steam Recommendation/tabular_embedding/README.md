# Tabular Embedding for Multimodal Fusion

Fusion 담당자가 Text/Image Tower와 동일한 방식으로 바로 사용할 수 있도록 정형데이터를 최소 패키지로 정리한 폴더입니다.

## 바로 사용하기

```python
from tabular_embedding.tabular_tower import TabularTower, load_tabular_bank

# fusion 기준 games DataFrame의 app_id 순서로 bank를 정렬
tab_bank, tab_id2row = load_tabular_bank(
    "tabular_embedding/emb_tabular_svd64",
    app_ids=games.app_id.values,
    device="cpu",
)

tabular_tower = TabularTower(in_dim=64, out_dim=64)

# batch_rows가 games/app_id 기준 공통 row index일 때
z_tab = tabular_tower(tab_bank[batch_rows])  # (B, 64), L2 normalized
```

Text/Image Tower와 결합:

```python
z_txt = text_tower(text_bank[batch_rows])
z_img = image_tower(image_bank[batch_rows])
z_tab = tabular_tower(tab_bank[batch_rows])

fusion_input = torch.cat([z_txt, z_img, z_tab], dim=-1)  # (B, 192)
```

> 세 bank를 만들 때 반드시 동일한 `games.app_id.values`를 `app_ids=`로 넘겨 같은 행 순서로 정렬하세요.

## 파일

| 파일 | Fusion에서의 역할 |
|---|---|
| `emb_tabular_svd64.npy` | 게임 50,872개 × 64차원 고정 tabular representation |
| `emb_tabular_svd64.csv` | embedding row에 대응하는 `app_id` |
| `tabular_tower.py` | `64 → 128 → 64` trainable projection과 loader |
| `feature_schema.json` | 원본 1,348개 정형 feature와 변환 규칙 기록 |
| `requirements.txt` | 최소 실행 패키지 |

## Shape 및 검증 결과

| 항목 | 결과 |
|---|---:|
| 게임 수 | 50,872 |
| pre-SVD 정형 feature | 1,348차원 |
| 전달 embedding | `(50872, 64)` |
| dtype | `float32` |
| NaN / Inf | 0 |
| zero-norm row | 0 |
| L2 norm | 약 1.0 |
| SVD 설명분산 합 | 0.9531 |
| Top-50 tag linear probe macro-AUC | 0.9318 ± 0.0065 |
| row-shuffled control | 0.5009 ± 0.0105 |

Linear probe에서 tag 컬럼은 정형 입력에서 제외했습니다. 장르·카테고리와 tag의 상관성이 크므로 이 결과는 최종 추천 성능이 아니라 representation sanity check로 해석해야 합니다.

## 어떤 정보를 사용했나?

- 수치형: 가격, 할인, 평가 수, 긍정 비율, 플레이타임, Peak CCU, 도전과제, DLC, Metacritic, 추정 소유자 수
- Boolean: Windows/macOS/Linux/Steam Deck, HF 매칭 및 정보 유무 플래그
- 범주형: rating
- Multi-hot: genres, categories, 지원 언어, 전체 음성 언어
- 빈도 vocabulary: 개발사, 퍼블리셔
- 날짜 파생: 출시 연도와 월의 sin/cos

텍스트 모달리티와 중복 및 tag probe 누출을 막기 위해 다음은 제외했습니다.

```text
app_id                 # 모델 입력이 아니라 row mapping
title
모든 description 컬럼
description_source
tags_kaggle
tags_huggingface
```

## 결측 처리

- 수치형: catalog 중앙값 대체 + 명시적 missing indicator
- 범주형/multi-label: `UNKNOWN`
- 희귀하거나 vocabulary 밖인 값: `OTHER`
- HF 숫자의 실제 0은 보존
- `hf_matched`, `has_*`는 정보 유무 특징으로 보존

따라서 fusion 담당자가 추가로 결측치를 대체할 필요는 없습니다.

## 중요한 구분

`emb_tabular_svd64.npy`는 1,348차원 정형 feature를 L2-normalized 64D로 압축한 **비지도 SVD representation**입니다. 이미 고정된 bank이므로 Text의 MiniLM 및 Image의 CLIP bank처럼 로드할 수 있습니다.

`TabularTower`는 random initialization 상태로 제공되며 fusion loss로 학습해야 합니다. 학습 전 Tower 출력을 최종 embedding으로 저장하면 안 됩니다.

```text
fixed SVD bank 64D
→ TabularTower 64 → 128 → 64
→ L2 normalization
→ Text 64D + Image 64D + Tabular 64D fusion
```

## 누락 app_id 처리

현재 Kaggle 기준 50,872개 게임이 모두 tabular bank에 존재합니다. 그래도 다른 catalog 순서를 전달할 경우를 대비해 `load_tabular_bank()`는 다음처럼 작동합니다.

- `fill_missing=True`: 누락 ID를 L2-normalized catalog 평균 벡터로 대체
- `fill_missing=False`: 누락 ID가 있으면 즉시 `KeyError`

엄격한 실험에서는 `fill_missing=False`를 권장합니다.

## 로컬 smoke test

```bash
pip install -r tabular_embedding/requirements.txt
python tabular_embedding/tabular_tower.py
```

정상 출력 예시:

```text
bank: (50872, 64), unique app_ids: 50,872
tower: (32, 64) -> (32, 64)
output norm: 1.000000 ~ 1.000000
```
