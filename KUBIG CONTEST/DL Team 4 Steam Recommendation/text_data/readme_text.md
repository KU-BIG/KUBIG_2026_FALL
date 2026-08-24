# Steam 게임 추천용 텍스트 데이터

Kaggle Steam Games 데이터를 기준으로 `title`, `description`, `tags`를 정리해
텍스트 encoder에 넣을 문장을 만든 파일입니다. 이미지 파트는 별도로 진행하므로
여기서는 게임 1개당 텍스트 임베딩 1개를 만드는 데 필요한 데이터와 코드만 둡니다.

## 빠른 요약

| 항목 | 결과 |
|---|---:|
| 기준 데이터 크기 | 50,872행 |
| 기준 key | `app_id` |
| 사용 텍스트 컬럼 | `title`, `tags`, `description` |
| `title` 빈 값 | 0개 (0.00%) |
| `description` 빈 값 | 10,373개 (20.39%) |
| `tags` 빈 값 | 1,244개 (2.45%) |
| baseline text p95 token length | 105 |
| baseline text p99 token length | 121 |
| 256 token 초과 | 21개 (0.04%) |
| baseline encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| encoder embedding dim | 384 |
| fusion projection dim | 64 |

## 파일 안내

| 파일 | 용도 |
|---|---|
| `games_text_ready.csv` | 텍스트 모델 입력용 최종 테이블. `text_for_embedding` 컬럼 사용 |
| `text_token_length_summary.csv` | text 조합별 token 길이 요약 |
| `text_over_256_samples.csv` | 256 token을 넘는 샘플 확인용 |
| `steam_text_preprocessing.py` | 전처리와 token length check 재실행용 스크립트 |
| `09_encode_text_embeddings.py` | MiniLM으로 텍스트 임베딩을 생성하는 스크립트 |
| `08_text_tower.py` | fusion 모델에 연결할 텍스트 타워 |
| `10_text_neighbors.py` | 생성된 텍스트 임베딩의 가까운 이웃을 확인하는 점검 스크립트 |
| `11_text_recommendation_baseline.py` | interaction split 기준 Popularity vs Text-only 추천 성능 평가 |
| `12_text_recommendation_improvements.py` | hours weighting, positive history threshold, negative subtraction 개선 실험 |
| `13_build_text_recommender_artifacts.py` | 실제 추천 실행을 위한 item/user artifact 생성 |
| `14_recommend_text_only.py` | user_id를 넣어 실제 text-only Top-K 추천 생성 |
| `15_visualize_text_recommendation.py` | 입력 history, embedding map, 추천 결과를 HTML 리포트로 시각화 |
| `text_neighbors_sample.csv` | 대표 게임 몇 개의 nearest neighbor 결과 |
| `emb_text_minilm.npy` | 텍스트 임베딩 행렬. 생성 후 저장됨 |
| `emb_text_minilm.csv` | `app_id`와 embedding 행 번호 매핑 |
| `text_recommendation_results/` | Recall/NDCG/Coverage, cold-start 결과, 그래프, 정성 사례, 최종 요약 |
| `text_recommendation_improvements/` | 기존 baseline 보존 후 user vector 개선 실험 결과 |
| `text_recommender_artifacts/` | 추천 엔진이 바로 읽는 item embedding, user vector, history 파일 |

## 왜 이 텍스트를 사용했나?

`description`은 게임 내용을 직접 설명해 주지만 빈 값이 약 20%라서 단독으로 쓰기에는 불안정합니다.
반대로 `title`은 항상 존재하지만 정보량이 적습니다. `tags`는 대부분 존재하고 장르, 분위기,
플레이 방식 같은 추천에 중요한 정보를 짧게 담고 있습니다.

그래서 첫 baseline은 다음처럼 잡았습니다.

```text
text_for_embedding = title + tags + description
```

description이 비어 있으면 자연스럽게 `title + tags`만 남고, tags와 description이 둘 다 비면
최소한 title은 남게 했습니다. 원본 컬럼은 덮어쓰지 않고 `description_original`,
`tags_kaggle`, `text_v1_title`, `text_v2_title_tags`, `text_v3_title_tags_description`으로
나누어 두었습니다.

## 긴 텍스트 처리

MiniLM tokenizer로 실제 token 길이를 확인했습니다.

| text column | p95 | p99 | max | 256 token 초과 |
|---|---:|---:|---:|---:|
| `text_v1_title` | 14 | 20 | 68 | 0 |
| `text_v2_title_tags` | 46 | 52 | 93 | 0 |
| `text_v3_title_tags_description` | 105 | 121 | 347 | 21 |

전체 50,872개 중 21개만 256 token을 넘습니다. 비율로는 0.04% 정도라서
첫 실험부터 chunking을 넣지는 않았습니다. embedding 생성 시 tokenizer의
`truncation=True, max_length=256`을 사용하고, chunking은 나중에 성능이 아쉬울 때
비교 실험으로 남깁니다.

## 컬럼 설명

| 컬럼 | 설명 |
|---|---|
| `app_id` | Steam app 고유 ID |
| `title` | 원본 게임명 |
| `title_clean` | 반복 공백만 정리한 게임명 |
| `description_original` | Kaggle metadata의 원본 설명을 보수적으로 정리한 값 |
| `tags_kaggle` | Kaggle metadata의 tag list 원본 |
| `tags_text` | tag list를 공백으로 이어 붙인 문자열 |
| `text_v1_title` | title만 사용한 비교용 텍스트 |
| `text_v2_title_tags` | title과 tags를 사용한 비교용 텍스트 |
| `text_v3_title_tags_description` | title, tags, description을 모두 사용한 비교용 텍스트 |
| `text_for_embedding` | 첫 baseline embedding에 실제로 사용할 텍스트 |
| `token_len_minilm` | MiniLM tokenizer 기준 token 수 |
| `needs_truncation_256` | 256 token 초과 여부 |

## embedding 생성

필요한 패키지:

```bash
pip install pandas numpy torch transformers
```

전처리 재실행:

```bash
python steam_text_preprocessing.py
```

텍스트 임베딩 생성:

```bash
python 09_encode_text_embeddings.py --batch-size 128 --device cpu
```

짧게 테스트할 때:

```bash
python 09_encode_text_embeddings.py --limit 10 --output-prefix emb_text_minilm_test
```

생성되는 임베딩은 다음 형태입니다.

```python
import numpy as np

emb = np.load("Data_process/emb_text_minilm.npy")
print(emb.shape)  # (50872, 384)
```

fusion 쪽에서는 `08_text_tower.py`의 `TextTower`로 384차원 임베딩을 64차원으로 사영해서
이미지 타워 출력과 같은 크기로 맞춥니다.

## 간단한 이웃 점검

임베딩 생성 후에는 가까운 이웃을 한 번 확인했습니다.

```bash
python 10_text_neighbors.py
```

예를 들어 `Prince of Persia: Warrior Within™`의 가까운 이웃으로 같은 Prince of Persia
시리즈가 먼저 나오고, `Portal 2`의 가까운 이웃으로 `Portal`, `Portal Reloaded`,
`Portal Stories: Mel` 등이 나옵니다. 이 결과는 정량 성능 평가가 아니라
임베딩 파일과 `app_id` 매핑을 확인하는 기본 점검입니다.

## 추천 baseline 검증

MiniLM embedding이 실제 추천 signal로 쓸 수 있는지 보려면 다음 스크립트를 실행합니다.

```bash
python 11_text_recommendation_baseline.py --max-users 100000 --max-eval-users 5000 --eval-split test
```

이 스크립트는 `recommendations.csv`를 날짜 기준으로 나누고, train 기간 interaction 수가
5개 이상인 user만 train 정보로 먼저 고른 뒤, 같은 evaluation user set에서 다음 두 모델을
비교합니다.

- `Popularity`: train 기간 interaction 수가 많은 게임부터 추천
- `Text-only`: user의 train positive 게임 text embedding 평균과 전체 catalog embedding의 cosine similarity로 추천

주요 산출물은 `text_recommendation_results/text_recommendation_result_summary.md`에 정리됩니다.
그래프는 `text_vs_popularity_accuracy.png`, `text_warm_cold_comparison.png`,
`text_coverage_comparison.png`로 저장됩니다.

기존 baseline 결과를 덮어쓰지 않고 user vector 생성법만 바꿔 비교하려면 다음을 실행합니다.

```bash
python 12_text_recommendation_improvements.py --max-users 100000 --max-eval-users 5000 --eval-split test
```

비교 대상은 simple mean baseline, `log1p(hours)` weighted mean, `positive_mean - alpha * negative_mean`
방식이며, train positive history `>=1`, `>=3`, `>=5` 조건을 함께 비교합니다. 결과는
`text_recommendation_improvements/text_recommendation_improvement_summary.md`에 저장됩니다.

## 실제 추천 artifact 생성

추천 함수나 웹 데모가 바로 읽을 수 있는 파일은 다음 스크립트로 만듭니다.

```bash
python 13_build_text_recommender_artifacts.py --max-users 100000 --min-train-interactions 5 --min-positive-history 1
```

이 단계에서는 train 기간에 본 게임을 user별로 저장하고, train positive 게임들의 text embedding을
`log1p(hours)` 가중 평균해서 user 취향 벡터를 만듭니다. 생성 결과는
`text_recommender_artifacts/`에 저장됩니다.

## 실제 추천 실행

artifact 생성 후 특정 user에게 추천을 뽑으려면 다음을 실행합니다.

```bash
python 14_recommend_text_only.py --user-id 58 --top-k 10 --show-history \
  --output-csv text_recommender_artifacts/recommend_user_58_top10.csv
```

출력에는 추천 rank, `app_id`, title, cosine similarity score, train popularity, cold item 여부,
추천 이유, tags가 포함됩니다. 추천 이유는 추천된 게임과 가장 비슷한 user의 train positive
게임을 찾아 만듭니다. 기본적으로 train 기간에 이미 본 게임은 추천에서 제외합니다.

## 추천 결과 HTML 시각화

입력 history, 2D embedding map, 추천 카드와 추천 이유를 한 페이지로 보려면 다음을 실행합니다.

```bash
python 15_visualize_text_recommendation.py --user-id 58 --top-k 10
```

기본 출력은 `text_recommender_artifacts/recommend_user_58_embedding_map.html`입니다. 이 HTML은
외부 서버 없이 로컬 브라우저에서 바로 열 수 있고, 공유 링크가 필요하면 GitHub Pages나 별도
웹 호스팅에 올릴 수 있습니다.

## 파일 관리

`games_text_ready.csv`와 `emb_text_minilm.npy`는 용량이 커질 수 있으므로 Git에 일반 파일로
올리기보다 Git LFS로 관리하는 편이 안전합니다. 특히 `.npy`와 큰 `.csv`는 행 순서가
서로 맞아야 하므로 `emb_text_minilm.npy`와 `emb_text_minilm.csv`는 항상 같이 두어야 합니다.
