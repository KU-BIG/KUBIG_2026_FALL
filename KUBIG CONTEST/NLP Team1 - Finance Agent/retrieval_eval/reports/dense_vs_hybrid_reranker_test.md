# Final Korean Dense+Reranker vs Hybrid+Reranker

## Experiment configuration

- Corpus: Final 400/60, 563 chunks (BAAI/bge-m3 tokenizer)
- Split/query: Test 40, Korean
- Candidate/final depth: Top-20 → Top-5
- Dense: BAAI/bge-m3
- Reranker: BAAI/bge-reranker-v2-m3
- Hybrid: BM25 top-50 + Dense top-50, RRF(k=60)
- Metrics: eval_retrieval.py의 기존 evidence-group-aware 구현 재사용

## Final comparison

| Pipeline | Recall@20 | Hit@20 | MRR@20 | Recall@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense -> Reranker | 0.9333 | 0.9500 | 0.6244 | 0.7792 | 0.8250 | 0.6446 | 0.6376 |
| Hybrid(RRF) -> Reranker | 0.9125 | 0.9250 | 0.5415 | 0.7958 | 0.8250 | 0.6342 | 0.6334 |

기존 `results/results_400_60_ko_test.json`의 Hybrid candidate 및 Hybrid+Reranker 지표와 이번 재실행 결과는 full precision에서 일치했습니다.

## Query-level complementarity

- Dense Top-20만 gold hit: 2 — ['FRD_006', 'FRD_020']
- Hybrid Top-20만 gold hit: 1 — ['FRD_015']
- 둘 다 gold hit: 36
- 둘 다 gold miss: 1 — ['ACC_015']

### Reranker 이후 first-gold rank

- Dense+Reranker 우위: 2 — ['ACC_008', 'NOT_005']
- Hybrid+Reranker 우위: 2 — ['ACC_010', 'REM_023']
- 동일: 36

first-gold가 Top-5에 없으면 rank 6 상당으로 처리해 두 pipeline을 비교했습니다. 세부 rank와 후보 chunk ID는 JSON의 `query_details`에 기록했습니다.

### 대표 사례

- **BM25가 Dense를 보완한 사례 — FRD_015**: 악성 앱 설치 후 전원 차단·초기화 조치를
  묻는 문항입니다. Dense Top-20에는 gold가 없었지만 Hybrid에서는 12위에 포함됐습니다.
  다만 reranker 후에는 두 pipeline 모두 gold가 Top-5에 들지 못했습니다.
- **BM25/RRF 결합으로 Dense 후보가 밀린 사례 — FRD_006, FRD_020**: 각각 보이스피싱
  지급정지 조치와 악성 앱 감염 후 금융정보·지인 보호 조치를 묻는 문항입니다. Dense에서는
  gold가 각각 8위와 5위였지만 Hybrid Top-20에서는 사라졌고, 두 문항 모두 reranker
  Top-5에는 gold가 들지 못했습니다.
- reranker의 first-gold rank는 Dense 우위 2건(ACC_008, NOT_005), Hybrid 우위 2건
  (ACC_010, REM_023), 동일 36건으로 어느 후보 방식도 일관된 rank 우위를 보이지 않았습니다.

## Architecture decision

추천: **Dense → Reranker → Top-5**

- **Candidate retrieval**: Dense가 Recall@20(0.9333 vs 0.9125), Hit@20(0.9500 vs
  0.9250), MRR@20(0.6244 vs 0.5415)에서 모두 우세합니다.
- **Reranker 이후**: Hybrid는 evidence Recall@5가 0.0167 높지만, Hit@5는 0.825로
  동일하고 Dense가 MRR@5(0.6446 vs 0.6342)와 nDCG@5(0.6376 vs 0.6334)에서
  소폭 우세합니다.
- **Complementarity**: BM25가 보완한 문항은 1건이지만 Dense 후보를 잃은 문항은
  2건입니다. BM25의 complementary signal은 존재하지만 일관되거나 순이익이라고 보기
  어렵습니다.
- **복잡도**: Dense-only 후보 방식은 BM25 index, 한국어 lexical tokenization, RRF
  결합과 관련 튜닝·운영을 제거합니다.

따라서 Test 40 범위에서는 Hybrid의 작은 Recall@5 이점보다 Dense의 후보 안정성,
랭킹 품질, 단순성이 더 큰 근거입니다. 표본이 40문항이므로 차이를 큰 성능 격차로
해석하지 않되, 현재 Korean production architecture는 Dense → Reranker가 합리적입니다.
