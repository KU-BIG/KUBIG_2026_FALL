# Experiment 4A — Paired-KO Lexical Retrieval Diagnostic

## Scope

`rag_evaluation_dataset.jsonl`의 paired `question.ko`를 번역 결과의 대리값으로 사용하는 controlled upper-bound/feasibility diagnostic입니다. Machine translation 오류가 제거된 조건이며 실제 translation pipeline이나 production translated BM25 성능이 아닙니다.

## Backend and fixed setup

- Nori/Elasticsearch 구현·실행 환경은 repository와 현재 환경에서 발견되지 않았습니다.
- 기존 `eval_retrieval.py`의 `BM25Retriever`를 수정 없이 사용했습니다: `rank_bm25 0.2.2` `BM25Okapi(k1=1.5, b=0.75, epsilon=0.25)`.
- tokenizer: lowercase; 연속 한글 span 전체 token + 길이 3 이상이면 overlapping character bigram; Latin/alphanumeric span token.
- 따라서 결과 명칭은 **Paired-KO lexical BM25 diagnostic**이며 Nori BM25가 아닙니다.
- Final 400/60, 563 chunks, Test40, 동일 gold, candidate_k=20, final_k=5, 기존 evidence-group-aware metrics.
- Candidate query=`question.ko`; reranker query=`question.en`; reranker=`BAAI/bge-reranker-v2-m3`.
- BM25/tokenizer/reranker parameter sweep 없음. 반복 사용된 Test40의 exploratory architecture diagnostic입니다.

## English Dense vs Paired-KO Lexical

| Pipeline | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| English Dense | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| Paired-KO Lexical | 0.7492 | 0.7750 | 0.3951 | 0.7075 | 0.7500 | 0.6562 | 0.6228 |
| Δ KO lexical−Dense | -0.1467 | -0.1750 | -0.1698 | -0.0925 | -0.1250 | -0.0300 | -0.0315 |

## Candidate complementarity

- Dense hit / KO lexical hit: 30 — ['ACC_008', 'ACC_016', 'ACC_017', 'ACC_023', 'ACC_026', 'ACC_030', 'REM_009', 'REM_012', 'REM_016', 'REM_020', 'REM_023', 'REM_027', 'REM_029', 'NOT_003', 'NOT_004', 'NOT_005', 'NOT_007', 'NOT_013', 'NOT_017', 'NOT_019', 'NOT_024', 'NOT_029', 'NOT_030', 'FRD_003', 'FRD_004', 'FRD_015', 'FRD_016', 'FRD_021', 'FRD_025', 'FRD_028']
- Dense hit / KO lexical miss: 8 — ['ACC_006', 'ACC_010', 'REM_003', 'REM_014', 'REM_017', 'FRD_006', 'FRD_009', 'FRD_020']
- Dense miss / KO lexical hit: 1 — ['ACC_021']
- Dense miss / KO lexical miss: 1 — ['ACC_015']

## Dense miss case studies

| ID | Dense rank | KO lexical rank | After reranker | Gold | Paired Korean question |
|---|---:|---:|---:|---|---|
| ACC_015 | miss | miss | miss | WEB003_0001 | 밤 11시에 ATM으로 보증금을 보내려 했는데 이용할 수 없다는 화면이 떴어요. 기계가 고장 난 건지 원래 그 시간에는 안 되는 건지 모르겠어요. 몇 시에 다시 가면 될까요? |
| ACC_021 | miss | 1 | 1 | PDF005_0004 | 신한은행 화상상담센터에서 통장과 체크카드, 온라인뱅킹까지 한 번에 신청하려고 해요. 체류카드가 있어도 여권을 꼭 가져가야 하나요? |

JSON에는 각 문항의 English/Korean question, gold ID/text, lexical Top-20 ID/score, first-gold rank, reranked Top-5와 query-gold shared tokens를 보존했습니다.

## Theoretical complementarity

- Dense hit: 38/40
- Paired-KO lexical hit: 31/40
- Dense∪Paired-KO lexical hit upper bound: 39/40 = 0.9750
- Union evidence-group coverage upper bound: 0.9625
- Mean Top-20 overlap: 4.72/20; mean Jaccard: 0.1420; identical sets: 0/40

Union은 최대 40개 후보를 포함하는 theoretical coverage upper bound이며 실제 Top-20 fusion 성능이 아닙니다.

## Interpretation

- Paired-KO lexical은 Dense보다 candidate Recall@20 -0.1467, Hit@20 -0.1750, MRR@20 -0.1698이므로 Dense 대체재는 아닙니다. Reranker 후에도 모든 지표가 Dense보다 낮습니다.
- 그럼에도 Dense miss `ACC_021`을 lexical rank 1로 회수하고 English-query reranker에서도 rank 1을 유지했습니다. Paired Korean query와 gold에 `신한은행`, `화상`, `상담`, `센터`, `통장`, `체크카드`, `온라인뱅킹`, `여권` 등 구체적 cue가 공유된 결과입니다.
- `ACC_015`는 Paired-KO lexical에서도 miss였습니다. Gold `WEB003_0001`은 영어 문서이고 Korean query와의 공통 token이 `ATM`뿐이어서, Korean translation은 이 사례의 corpus-language mismatch를 해결하지 못했습니다.
- Dense∪Paired-KO lexical은 Dense 단독보다 hit upper bound를 38/40→39/40으로 높입니다. 평균 Jaccard 0.1420으로 candidate 다양성도 크지만, union은 최대 40개 후보이며 실제 Top-20 fusion 성능이 아닙니다.
- 결과는 **Case B**에 해당합니다. 평균 성능은 Dense보다 낮지만 Dense failure를 명확히 보완한 사례가 있어, actual English→Korean translation + Nori BM25를 auxiliary signal로 독립 검증할 가치가 있습니다.
- 다만 Paired-KO는 machine translation 오류가 제거된 조건이고 이 backend도 Nori가 아니므로, 실제 translation+Nori production 성능으로 일반화하지 않습니다.
