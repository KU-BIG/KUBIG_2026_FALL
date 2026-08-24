# Text Recommender Artifacts

이 폴더는 실제 text-only 추천을 실행하기 위한 중간 산출물입니다.

## 생성 방식

- Train 기간: date < 2022-04-01
- User vector: train positive 게임 embedding의 log1p(hours) weighted mean
- Candidate set: text embedding이 있는 전체 game catalog
- 추천 시 제외: train 기간에 user가 이미 본 게임

## 주요 파일

| 파일 | 설명 |
|---|---|
| `item_embeddings_text_norm.npy` | L2 normalize된 게임 text embedding |
| `item_catalog.csv` | 게임 title/tags/popularity/cold flag/embedding row |
| `user_vectors_hours_weighted.npy` | user별 취향 벡터 |
| `user_vectors_index.csv` | user_id와 user vector row 매핑 |
| `user_train_seen.csv` | 추천에서 제외할 user별 train seen 게임 목록 |
| `user_train_positive_history.csv` | 추천 이유 생성에 쓸 positive history와 hours |
| `artifact_summary.json` | 생성 설정과 row 수 요약 |

## 요약

- recommendation_rows_scanned: 41154794
- catalog_games: 50872
- embedding_games: 50872
- sampled_users: 100000
- users_with_vector: 99805
- user_vector_dim: 384
- positive_history_rows: 985386
- train_eligible_users: 1550295
- min_train_interactions: 5
- min_positive_history: 1
- max_users: 100000
- seed: 42
- train_end: 2022-04-01
- app_id_mapping_ok: True