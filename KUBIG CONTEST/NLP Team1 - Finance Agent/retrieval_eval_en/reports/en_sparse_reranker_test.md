# English BGE-M3 Sparse → Reranker (Final 400/60 Test40)

## Implementation

- `FlagEmbedding 1.4.0`의 `BGEM3FlagModel`을 `return_sparse=True`로 사용했습니다.
- query/corpus representation은 BGE-M3 tokenizer token ID별 learned lexical weight dictionary입니다.
- sparse score는 공식 구현과 동일하게 공유 token ID에 대해 `Σ(query_weight × chunk_weight)`로 계산했습니다.
- 40×563 pair를 공식 `compute_lexical_matching_score`로 exact brute-force 계산했습니다. 이 규모에서는 별도 inverted index가 필요하지 않습니다.
- 실제 sparse index 동작에 맞춰 score가 0보다 큰 chunk만 candidate로 인정했으며, 임의의 zero-score chunk로 Top-20을 채우지 않았습니다. 모든 질문은 positive-score chunk가 20개 이상이었습니다.

## Fixed setup

- Final 400/60 corpus, 563 chunks; English Test 40; 동일 Final gold
- candidate_k=20, final_k=5
- reranker: `BAAI/bge-reranker-v2-m3`
- 기존 `eval_retrieval.py`의 evidence-group-aware metric을 그대로 사용

## Dense vs Sparse

| Signal | Recall@20 | Hit@20 | MRR@20 | Recall@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN Dense | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| EN Sparse | 0.5696 | 0.6500 | 0.4090 | 0.5529 | 0.6500 | 0.5571 | 0.5072 |

## Candidate complementarity

- Dense hit / Sparse hit: 25
- Dense hit / Sparse miss: 13 — ['ACC_006', 'ACC_008', 'ACC_010', 'ACC_016', 'ACC_017', 'REM_003', 'REM_023', 'REM_027', 'REM_029', 'FRD_006', 'FRD_015', 'FRD_016', 'FRD_020']
- Dense miss / Sparse hit: 1 — ['ACC_015']
- Dense miss / Sparse miss: 1 — ['ACC_021']
- mean Top-20 set overlap: 7.78/20; mean Jaccard: 0.2490; identical sets: 0/40

| ID | Dense first gold | Sparse first gold | Gold chunk | English question |
|---|---:|---:|---|---|
| ACC_015 | not in 20 | 9 | WEB003_0001 | I tried to send a deposit at an ATM at 11 p.m., but the machine said the service was unavailable. I am not sure whether it was broken or simply outside operating hours. When should I try again? |
| ACC_006 | 2 | not in 20 | PDF005_0001, PDF005_0002 | I could not prepare documents showing why I need the account. Will the bank refuse to open it, or can it be opened first with lower limits on transfers and withdrawals? |
| ACC_008 | 2 | not in 20 | PDF005_0001, PDF005_0002 | My account was opened with a low transfer limit because I had no supporting documents at the time. I can now provide an employment certificate. Can the bank remove the limit after reviewing it? |
| ACC_010 | 17 | not in 20 | PDF005_0002 | The bank has asked me to prove that I have lived in Korea for at least 183 days. Is my residence card enough, and what records can I use? |
| ACC_016 | 2 | not in 20 | PDF005_0001 | My employer asked me to open an account for my salary. I heard the bank may not accept an old pay statement. What should I keep in mind when preparing it? |
| ACC_017 | 8 | not in 20 | WEB005_0004 | I am choosing a transit card in Korea. What is the difference between topping up in advance and paying later through a card bill? |
| ACC_021 | not in 20 | not in 20 | PDF005_0004 | I want to apply for an account, debit card, and online banking at Shinhan Bank's video consultation center. Do I still need my passport if I have a residence card? |

## Dense miss focus

- `ACC_015`: Dense=not in 20, Sparse=9, Sparse→Reranker=3. Gold `WEB003_0001`은 영어로 작성된 ATM 영업시간 chunk이라 English sparse lexical signal이 직접 작동한 사례입니다.
- `ACC_021`: Dense=not in 20, Sparse=not in 20. Gold `PDF005_0004`는 한국어 표 형식의 신한은행 화상상담·여권 안내이므로 English original sparse의 직접 lexical overlap으로는 보완되지 않았습니다.

## Fusion judgment

- Sparse는 Dense 대체재로는 부족합니다. Candidate Recall@20은 0.5696 vs 0.8958, Hit@20은 0.6500 vs 0.9500이고 Dense-only hit가 13건입니다.
- 그럼에도 Sparse-only hit `ACC_015` 1건이 있고, Dense∪Sparse candidate hit의 이론적 upper bound는 39/40(0.975)입니다. 또한 평균 Top-20 Jaccard 0.2490으로 두 ranking은 충분히 다릅니다.
- 따라서 **Dense + Sparse fusion을 다음의 제한적 실험으로 할 가치가 있습니다.** 단, union upper bound는 Top-20 fusion 성능이 아니며 Sparse noise가 Dense 후보를 밀어낼 수 있으므로 실제 fusion metric으로 확인해야 합니다. 이번 단계에서 fusion은 실행하지 않았습니다.
- `ACC_021`을 둘 다 놓쳐, translated Nori BM25도 그 다음 독립 signal로 평가할 가치가 유지됩니다.
