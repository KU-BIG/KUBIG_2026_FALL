# English Retrieval Experiment Progress

## 1. Goal

준석님 피드백에 따라 다음 최종 후보 구조를 단계적으로 검토합니다.

English Query

- English original → BGE-M3 Dense
- English original → BGE-M3 Sparse
- English → Korean translation → Elastic Nori BM25
- 필요 시 retrieval signal fusion
- Reranker
- Top-5

처음부터 fusion하지 않고 각 retrieval signal을 동일 조건에서 독립 평가한 뒤 question-level complementarity가 확인되는 경우에만 fusion합니다.

## 2. Fixed Evaluation Setup

- Final corpus 400/60, 563 chunks
- Experiments 1–4A: Test40 exploratory record; Experiment 4B onward: architecture selection uses Validation80 and Test40 remains sealed
- 동일 Final gold
- candidate_k=20, final_k=5
- Reranker=`BAAI/bge-reranker-v2-m3`
- Metrics: Recall@20 / Hit@20 / MRR@20 / Recall@5 / Hit@5 / MRR@5 / nDCG@5

## 3. Experiment 1 — BGE-M3 Dense

| Query | Signal → Reranker | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Korean | Dense → Reranker | 0.9333 | 0.9500 | 0.6244 | 0.7792 | 0.8250 | 0.6446 | 0.6376 |
| English | Dense → Reranker | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |

English Dense는 강한 baseline이며 Hit@20은 KO와 같은 0.95입니다. Candidate Recall/MRR은 KO보다 낮지만 reranker 이후 이번 Test40에서는 EN 지표가 오히려 소폭 높았습니다. 작은 Test40 결과이므로 영어가 일반적으로 더 우수하다고 해석하지 않습니다. `ACC_021`은 EN Dense cross-lingual miss였고, `FRD_015`는 반대로 EN에서만 candidate hit였습니다. 따라서 Dense만으로도 경쟁력이 있지만 sparse/translated lexical signal의 실제 보완 효과를 확인할 가치가 있습니다.

## 4. Current Decision

Sealed Test40 최종 평가 결과 **English Dense Top-20 → Reranker → Top-5**를 final English architecture로 확정합니다. Dense15+Nori5는 Dense hit를 잃지 않고 evidence Recall/MRR/nDCG를 소폭 높였지만, Dense miss를 새로 회수한 question은 0건이었습니다. Translation 위험과 Elasticsearch/Nori 운영 복잡도를 감안하면 held-out 이점이 main path 채택을 정당화하지 못합니다. Test 이후 추가 tuning은 하지 않습니다.

## 5. Experiment Matrix

| Experiment | Split | Candidate R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| EN Dense | Test40 | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 | DONE |
| EN Sparse | Test40 | 0.5696 | 0.6500 | 0.4090 | 0.5529 | 0.6500 | 0.5571 | 0.5072 | DONE |
| Dense + Sparse Equal RRF | Test40 | 0.8708 | 0.9250 | 0.5046 | 0.8000 | 0.8750 | 0.6737 | 0.6487 | DONE |
| Dense + Sparse Weighted RRF 2:1 | Test40 | 0.8958 | 0.9500 | 0.4869 | 0.8000 | 0.8750 | 0.6863 | 0.6543 | DONE |
| Dense + Sparse Weighted RRF 3:1 | Test40 | 0.8958 | 0.9500 | 0.4659 | 0.8000 | 0.8750 | 0.6863 | 0.6543 | DONE |
| Paired-KO Lexical Diagnostic | Test40 | 0.7492 | 0.7750 | 0.3951 | 0.7075 | 0.7500 | 0.6562 | 0.6228 | DONE |
| EN Dense | Validation80 | 0.8281 | 0.8625 | 0.4906 | 0.7198 | 0.7875 | 0.6440 | 0.6271 | DONE |
| Actual Translated Nori BM25 | Validation80 | 0.5427 | 0.5875 | 0.2578 | 0.5385 | 0.5750 | 0.4981 | 0.4740 | DONE |
| Dense + Nori Equal RRF 1:1 | Validation80 | 0.8417 | 0.8875 | 0.4421 | 0.7604 | 0.8250 | 0.6706 | 0.6546 | DONE |
| Dense + Nori Weighted RRF 2:1 | Validation80 | 0.8281 | 0.8625 | 0.4503 | 0.7198 | 0.7875 | 0.6440 | 0.6271 | DONE |
| Dense + Nori Weighted RRF 3:1 | Validation80 | 0.8281 | 0.8625 | 0.4670 | 0.7198 | 0.7875 | 0.6440 | 0.6271 | DONE |
| Dense15 + Nori5 | Validation80 | 0.8750 | 0.9125 | 0.4937 | 0.7573 | 0.8250 | 0.6738 | 0.6555 | SELECTED |
| Dense18 + Nori2 | Validation80 | 0.8656 | 0.9000 | 0.4926 | 0.7573 | 0.8250 | 0.6669 | 0.6504 | DONE |
| EN Dense | Sealed Test40 | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 | FINAL |
| Dense15 + Nori5 | Sealed Test40 | 0.9125 | 0.9500 | 0.5647 | 0.8000 | 0.8750 | 0.6988 | 0.6636 | EVALUATED |

## 6. Experiment 2 — BGE-M3 Sparse

Sparse candidate는 R@20=0.5696, Hit@20=0.6500, MRR@20=0.4090이며 reranker 후 R@5=0.5529, Hit@5=0.6500, MRR@5=0.5571, nDCG@5=0.5072입니다.

Question-level candidate 비교는 Dense/Sparse 둘 다 hit 25, Dense-only hit 13, Sparse-only hit 1, 둘 다 miss 1입니다. Sparse-only ID는 ['ACC_015']이며, 평균 Top-20 Jaccard는 0.2490입니다.

- `ACC_015`: Dense miss, Sparse rank 9, Sparse→Reranker rank 3
- `ACC_021`: Dense/Sparse 둘 다 miss
- Dense∪Sparse candidate hit upper bound: 39/40(0.975). 이는 실제 fusion Top-20 성능이 아니라 보완 가능성을 보여주는 union 기준입니다.

Sparse는 Dense 대체재로는 부족하지만, Sparse-only hit 1건과 낮은 ranking overlap은 제한적 fusion 실험을 정당화합니다.

## Experiment 3 — Dense + Sparse Fusion

Sparse-only hit `ACC_015`를 살리면서 Dense의 기존 hit 38건을 유지할 수 있는지 확인하기 위해 Equal RRF와 Dense-heavy 2:1, 3:1 weighted RRF를 제한적으로 비교했습니다. RRF k=60이며 기존 candidate ranking을 재사용했습니다.

| Pipeline | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| Dense + Sparse Equal RRF | 0.8708 | 0.9250 | 0.5046 | 0.8000 | 0.8750 | 0.6737 | 0.6487 |
| Dense + Sparse Weighted RRF 2:1 | 0.8958 | 0.9500 | 0.4869 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| Dense + Sparse Weighted RRF 3:1 | 0.8958 | 0.9500 | 0.4659 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |

- **Dense + Sparse Equal RRF**: new hit 1 ['ACC_015']; new miss 2 ['ACC_010', 'FRD_015']; first-gold rank improved/worsened/same = 8/16/16; ACC_015=15, ACC_021=miss
- **Dense + Sparse Weighted RRF 2:1**: new hit 0 []; new miss 0 []; first-gold rank improved/worsened/same = 6/16/18; ACC_015=miss, ACC_021=miss
- **Dense + Sparse Weighted RRF 3:1**: new hit 0 []; new miss 0 []; first-gold rank improved/worsened/same = 5/14/21; ACC_015=miss, ACC_021=miss

이 Test40 결과는 사전 지정한 세 weight의 탐색적 sensitivity check이며 확정적인 weight tuning이 아닙니다.

Equal RRF는 `ACC_015`를 살렸지만 `ACC_010`, `FRD_015`를 candidate에서 잃었습니다. 2:1/3:1은 Dense hit를 유지했지만 `ACC_015`를 잃었고, reranker 결과는 Dense와 정확히 같아 추가 이점이 없었습니다. 세 설정 중 Dense 성능 유지와 Sparse-only signal 보존을 동시에 달성한 경우는 없었습니다.

## Experiment 4A — Paired-KO Lexical Retrieval Diagnostic

Dataset의 `question.ko`를 machine translation 오류가 제거된 대리 query로 사용했습니다. Candidate backend는 Nori가 아닌 기존 evaluator의 `rank_bm25 0.2.2` BM25Okapi와 한글 span+character-bigram tokenizer입니다. Korean query로 Top-20을 검색한 뒤 original English query와 multilingual reranker를 사용했습니다.

| Pipeline | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| English Dense | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| Paired-KO Lexical | 0.7492 | 0.7750 | 0.3951 | 0.7075 | 0.7500 | 0.6562 | 0.6228 |
| Δ KO lexical−Dense | -0.1467 | -0.1750 | -0.1698 | -0.0925 | -0.1250 | -0.0300 | -0.0315 |

- Dense hit / KO lexical hit: 30
- Dense hit / KO lexical miss: 8
- Dense miss / KO lexical hit: 1 — `ACC_021`
- Dense miss / KO lexical miss: 1 — `ACC_015`
- `ACC_021`: lexical rank 1, English-query reranker rank 1. `신한은행`, `화상상담센터`, `통장`, `체크카드`, `온라인뱅킹`, `여권` cue가 gold와 공유됩니다.
- `ACC_015`: lexical/reranker miss. Korean query와 English gold chunk의 공통 lexical token은 `ATM`뿐입니다.
- Dense∪Paired-KO lexical hit upper bound: 39/40=0.9750. Mean Top-20 overlap 4.72/20, Jaccard 0.1420, identical set 0/40.

이 결과는 actual translated Nori performance가 아닌 controlled feasibility diagnostic입니다. 평균 성능은 낮지만 `ACC_021`을 명확히 보완했으므로 Case B에 해당하며, auxiliary lexical signal의 실제 구현 가치가 있습니다. Union은 실제 Top-20 fusion 성능이 아닙니다.

## Experiment 4B — Actual Translation + Nori BM25

Validation80 only에서 `question.en → NLLB-200 translation → actual Elasticsearch Nori BM25 Top-20 → original EN query reranker → Top-5`를 평가했습니다. NLLB-200 번역 80건의 sanity audit은 LOW/MEDIUM/HIGH=50/18/12이며, 빈 번역은 0건입니다. Actual Nori는 Elasticsearch 8.17.0의 official `analysis-nori` plugin, mixed decompound analyzer, fixed BM25(k1=1.2, b=0.75)를 사용했습니다.

| Pipeline (Validation80) | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN Dense | 0.8281 | 0.8625 | 0.4906 | 0.7198 | 0.7875 | 0.6440 | 0.6271 |
| Actual Translation → Nori | 0.5427 | 0.5875 | 0.2578 | 0.5385 | 0.5750 | 0.4981 | 0.4740 |

Candidate complementarity는 both 40, Dense-only 29, Nori-only 7, neither 4입니다. Nori-only는 `ACC_003`, `ACC_005`, `ACC_012`, `ACC_020`, `REM_025`, `NOT_014`, `NOT_015`이며 모두 Nori reranker Top-5에도 남았습니다. Dense∪Nori theoretical upper bound는 Hit 76/80=0.9500, evidence recall 0.9156입니다. 평균 overlap은 3.9125/20, Jaccard는 0.1152, identical set은 0건입니다. 이는 실제 fusion 성능이 아닙니다.

Nori는 평균 성능상 Dense를 대체할 수 없지만 Dense miss 11건 중 7건을 보완하므로 auxiliary signal 가치가 있습니다. 동시에 HIGH-risk 번역 12건이 있어 translation bottleneck을 분리해 해석해야 합니다.

## Experiment 5 — Dense + Nori Fusion (Validation80)

Experiment 4B의 저장된 Dense/Nori Top-20을 재사용해 RRF 1:1·2:1·3:1과 quota 15+5·18+2만 비교했습니다. Translation, Dense retrieval, Nori retrieval은 재실행하지 않았고 original EN query의 동일 reranker를 적용했습니다.

| Pipeline | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.8281 | 0.8625 | 0.4906 | 0.7198 | 0.7875 | 0.6440 | 0.6271 |
| Equal RRF 1:1 | 0.8417 | 0.8875 | 0.4421 | 0.7604 | 0.8250 | 0.6706 | 0.6546 |
| Weighted RRF 2:1 | 0.8281 | 0.8625 | 0.4503 | 0.7198 | 0.7875 | 0.6440 | 0.6271 |
| Weighted RRF 3:1 | 0.8281 | 0.8625 | 0.4670 | 0.7198 | 0.7875 | 0.6440 | 0.6271 |
| Dense15 + Nori5 | 0.8750 | 0.9125 | 0.4937 | 0.7573 | 0.8250 | 0.6738 | 0.6555 |
| Dense18 + Nori2 | 0.8656 | 0.9000 | 0.4926 | 0.7573 | 0.8250 | 0.6669 | 0.6504 |

Dense15+Nori5는 Dense candidate hit 69건 중 68건을 유지하고 Nori-only 7건 중 5건을 candidate에 추가했습니다. Reranker 후에는 Dense final hit 손실 없이 `ACC_005`, `REM_025`, `NOT_014` 세 hit를 추가했습니다. 이 중 앞의 두 건은 LOW-risk 번역이고 `NOT_014`는 핵심 조건 누락이 있는 HIGH-risk caution 사례입니다. Equal RRF는 Nori-only를 더 많이 담지만 Dense candidate 4건과 final hit 2건을 잃어 quota보다 불안정합니다. Dense-heavy RRF는 Dense를 보존하지만 Nori-only를 하나도 추가하지 못했습니다.

## Recommended English Architecture Candidate

`EN query → BGE-M3 Dense Top-15` + `EN query → NLLB-200 KO translation → Nori BM25 Top-5` → deterministic de-dup/fill Top-20 → original EN query reranker → Top-5.

Validation80에서는 Dense-only보다 모든 aggregate metric이 높고 Dense final hit 손실이 없어 이 후보를 sealed Test40 평가 대상으로 선택했습니다.

## Final Test — Dense15 + Nori5 (Test40)

Validation에서 고정한 quota, NLLB, Nori/BM25, reranker를 변경하지 않고 Test40에서 한 번 평가했습니다. Dense Top-20은 기존 Test artifact와 40/40 exact match했습니다.

| Pipeline | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN Dense | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| Dense15 + Actual Nori5 | 0.9125 | 0.9500 | 0.5647 | 0.8000 | 0.8750 | 0.6988 | 0.6636 |

Fusion은 Dense candidate/final hit를 하나도 잃지 않았지만 Dense miss를 새로 hit로 바꾼 문항도 0건입니다. R@20 증가는 `NOT_030`의 다중 evidence coverage가 1/3에서 3/3으로 증가한 결과입니다. Final first-gold rank 개선은 `REM_003` 1건이며, Hit@20/Hit@5 및 R@5는 동일합니다.

Test 번역 risk는 LOW/MEDIUM/HIGH=21/11/8입니다. Nori Top-5 gold 9건 중 LOW/MEDIUM/HIGH=4/3/2였으나 모두 Dense가 이미 candidate hit한 문항입니다. HIGH-risk `ACC_023`, `NOT_004`는 category overlap 성공으로 번역 성공으로 해석하지 않습니다.

## Final English Architecture

`English Query → BGE-M3 Dense Top-20 → bge-reranker-v2-m3 → Top-5`

Validation에서 관찰된 Nori의 Dense-miss 보완이 Test에서 재현되지 않았습니다. Fusion의 작은 evidence/ranking 개선보다 NLLB translation 위험, Elasticsearch 운영, latency와 failure mode 증가가 더 크다고 판단합니다.

## Final Decision and Remaining Limitations

Dense-only를 최종 architecture로 확정하고 Test 결과를 이용한 quota/weight/model 재조정은 하지 않습니다. Test40 규모가 작고 번역 품질 audit가 전문 번역 검증은 아니라는 한계가 있습니다. Translation+Nori는 연구용 auxiliary result로 보존하지만 production main path에서는 제외합니다.

## Next Step

Retrieval architecture selection과 sealed Test evaluation은 종료했습니다. 다음 단계는 Dense→Reranker 구조를 downstream agent/RAG 평가에 연결하는 것이며, retrieval quota/weight/model tuning은 더 하지 않습니다.

## Artifacts

- `retrieval_eval_en/reports/en_dense_reranker_test.md`
- `retrieval_eval_en/results/results_400_60_en_dense_reranker_test.json`
- `retrieval_eval_en/reports/en_sparse_reranker_test.md`
- `retrieval_eval_en/results/results_400_60_en_sparse_reranker_test.json`
- `retrieval_eval_en/reports/en_dense_sparse_fusion_test.md`
- `retrieval_eval_en/results/results_400_60_en_dense_sparse_fusion_test.json`
- `retrieval_eval_en/reports/en_paired_ko_lexical_test.md`
- `retrieval_eval_en/results/results_400_60_en_paired_ko_lexical_test.json`
- `retrieval_eval_en/data/validation_en_to_ko_translations.json`
- `retrieval_eval_en/data/nori_index_config.json`
- `retrieval_eval_en/reports/en_translated_nori_validation.md`
- `retrieval_eval_en/results/results_400_60_en_translated_nori_validation.json`
- `retrieval_eval_en/reports/en_dense_nori_fusion_validation.md`
- `retrieval_eval_en/results/results_400_60_en_dense_nori_fusion_validation.json`
- `retrieval_eval_en/data/test_en_to_ko_translations.json`
- `retrieval_eval_en/reports/en_dense_nori_final_test.md`
- `retrieval_eval_en/results/results_400_60_en_dense_nori_final_test.json`
- `retrieval_eval_en/reports/english_retrieval_final_summary.md`
