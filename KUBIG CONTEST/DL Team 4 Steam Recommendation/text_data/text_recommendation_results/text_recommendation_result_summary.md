# Text Recommendation Result Summary

## 실험 목적

MiniLM text embedding이 실제 사용자 추천에 사용할 수 있는지 검증.

## 방법

사용자의 train positive 게임 embedding 평균을 user vector로 만들고, 전체 game catalog와 cosine similarity를 계산해 Top-K를 추천했다. 비교 기준은 train interaction count 기반 Popularity baseline이다.

## 데이터 구성

- Train: date < 2022-04-01
- Validation: 2022-04-01 <= date < 2022-07-01
- Test: 2022-07-01 <= date < 2023-01-01
- 평가 split: test
- User filter: train 기간 interaction 수 >= 5
- Train 기준 sampling seed: 42
- Train-eligible sampled users: 100,000
- Final eval users: 5,000

## App ID Alignment Check

- 전체 catalog game 수: 50,872
- text embedding이 존재하는 game 수: 50,872
- recommendations에 등장하는 game 중 text embedding이 없는 수: 0
- interaction 0건 게임 중 text embedding이 존재하는 수: 13,262
- duplicate app_id 여부: games=0, text_ready=0, embedding_index=0
- embedding row와 app_id mapping mismatch 여부: False

## 핵심 결과

| 지표 | Popularity | Text-only | 차이 |
| --- | --- | --- | --- |
| Recall@10 | 0.021262 | 0.007153 | -0.014109 |
| NDCG@10 | 0.011281 | 0.004811 | -0.006470 |
| Coverage | 0.000393 | 0.131015 | +0.130622 |
| Cold Recall@10 | 0.000000 | 0.004859 | +0.004859 |

## 전체 @K 결과

| model | k | recall | ndcg | coverage | cold_recall | cold_ndcg | warm_recall | warm_ndcg | avg_train_popularity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | 5 | 0.011873 | 0.007905 | 0.000236 | 0.000000 | 0.000000 | 0.017175 | 0.011137 | 212258.812520 |
| Popularity | 10 | 0.021262 | 0.011281 | 0.000393 | 0.000000 | 0.000000 | 0.031171 | 0.016089 | 195382.703140 |
| Popularity | 20 | 0.036523 | 0.015901 | 0.000668 | 0.000000 | 0.000000 | 0.053419 | 0.022696 | 170433.964960 |
| Text-only | 5 | 0.004611 | 0.003942 | 0.084604 | 0.002118 | 0.001528 | 0.005932 | 0.004877 | 4245.058200 |
| Text-only | 10 | 0.007153 | 0.004811 | 0.131015 | 0.004859 | 0.002417 | 0.007889 | 0.005610 | 3421.391580 |
| Text-only | 20 | 0.009957 | 0.005604 | 0.192837 | 0.008104 | 0.003264 | 0.010405 | 0.006307 | 2824.807000 |

## 한 줄 결론

부분적으로 유효함: 일반 정확도는 popularity보다 약할 수 있지만 cold-start/coverage/long-tail 측면의 가치가 있다.

## 해석

Popularity는 인기 게임을 안정적으로 맞히는지 보는 기준이고, Text-only는 사용자 history와 게임 설명/tag/title의 의미적 유사도가 실제 추천으로 이어지는지 보는 기준이다. 따라서 Recall/NDCG, Cold 지표, Coverage와 Avg Train Popularity를 함께 봐야 한다.

정성 추천 사례는 `text_recommendation_qualitative_cases.md`에서 확인할 수 있다.

## 생성 파일

- `text_recommendation_metrics.csv`
- `text_recommendation_alignment_stats.json`
- `text_vs_popularity_accuracy.png`
- `text_warm_cold_comparison.png`
- `text_coverage_comparison.png`
- `text_recommendation_qualitative_cases.csv`
- `text_recommendation_qualitative_cases.md`