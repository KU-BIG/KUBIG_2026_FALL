# Final English Retrieval Evaluation — Sealed Test40

## Frozen architecture and protocol

The Validation-selected candidate was frozen before Test access:

`EN Dense Top-15 + NLLB EN→KO → Nori BM25 Top-5 → deterministic de-duplication/Dense fill → Top-20 → original EN query reranker → Top-5`.

No quota, weight, model, decoding, analyzer, BM25, sparse, or reranker tuning was performed after seeing Test results. Test40 was translated once and only the frozen architecture was evaluated. Validation questions were not retrieved or reranked in this run.

## Translation quality

The exact Validation configuration was reused: `facebook/nllb-200-distilled-600M` revision `f8d333a098d19b4fd9a8b18f94170487ad3f821d`, `eng_Latn→kor_Hang`, beam 4, no sampling, input/output limit 256/128, batch 8, MPS. `question.ko` was only an audit reference.

- LOW: 21
- MEDIUM: 11
- HIGH: 8
- Empty: 0; no-Hangul: 0

HIGH examples include `ACC_006` (account-purpose/limited-account intent distorted), `REM_012` (designated-bank question omitted), `NOT_004` (truncated question), `NOT_029` (First Meeting Voucher severely mistranslated), and `FRD_004`/`FRD_025` (core safety-decision question omitted). Outputs were not corrected or regenerated.

## Retrieval and provenance

- Final 400/60 corpus, 563 unchanged chunks, Final evidence-group gold
- BGE-M3 corpus cache reused; recomputed Test40 Dense Top-20 matched the prior Dense artifact exactly for all 40 questions, and baseline metrics matched at full precision
- Elasticsearch 8.17.0 with official analysis-nori 8.17.0
- Same mixed-decompound Nori analyzer and fixed BM25(k1=1.2, b=0.75, discount_overlaps=true)
- Analyzer probe confirmed actual Nori tokens
- Same `BAAI/bge-reranker-v2-m3`, original English query

## Sealed Test40 result

| Pipeline | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| English Dense only | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| Dense15 + Actual Translated Nori5 | 0.9125 | 0.9500 | 0.5647 | 0.8000 | 0.8750 | 0.6988 | 0.6636 |
| Δ Fusion − Dense | +0.0167 | 0.0000 | -0.0002 | 0.0000 | 0.0000 | +0.0125 | +0.0092 |

## Query-level transitions

Candidate Top-20:

- Dense hit→Fusion hit: 38
- Dense hit→Fusion miss: 0
- Dense miss→Fusion hit: 0
- Dense miss→Fusion miss: 2 (`ACC_015`, `ACC_021`)
- First-gold improved/worsened/same: 0/1/39; worsened ID `ACC_010` (rank 17→20)

Reranker Top-5:

- Dense hit→Fusion hit: 35
- Dense hit→Fusion miss: 0
- Dense miss→Fusion hit: 0
- Dense miss→Fusion miss: 5
- First-gold improved/worsened/same: 1/0/39; improved ID `REM_003`

Nori did not recover a Dense-miss question on Test40. The candidate Recall gain comes from `NOT_030`: Dense retrieved one of three evidence groups, while fusion retrieved all three (evidence recall 1/3→3/3). Thus the gain is real multi-evidence coverage, but not an additional question-level hit.

Nori Top-5 contained gold for nine questions: LOW 4, MEDIUM 3, HIGH 2. These golds mostly overlapped already-successful Dense questions. HIGH-risk hits `ACC_023` and `NOT_004` are category-overlap successes and are not evidence of successful translation.

## Validation → Test generalization

| Split | Pipeline | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Validation80 | Dense | 0.8281 | 0.8625 | 0.4906 | 0.7198 | 0.7875 | 0.6440 | 0.6271 |
| Validation80 | Dense15+Nori5 | 0.8750 | 0.9125 | 0.4937 | 0.7573 | 0.8250 | 0.6738 | 0.6555 |
| Test40 | Dense | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| Test40 | Dense15+Nori5 | 0.9125 | 0.9500 | 0.5647 | 0.8000 | 0.8750 | 0.6988 | 0.6636 |

Validation gains in candidate/final Hit did not generalize. Fusion preserved every Dense hit and retained modest evidence-recall and ranking-quality gains, but Nori's Dense-miss complementarity did not reproduce on Test40.

## Final decision

The final English retrieval architecture is **Option 1: English BGE-M3 Dense Top-20 → bge-reranker-v2-m3 → Top-5**.

The frozen fusion was safe on this Test40—no Dense hit was lost—and slightly improved evidence Recall, MRR, and nDCG. However, it produced zero new question-level candidate or final hits, while requiring an NLLB translation model, Elasticsearch/Nori infrastructure, extra latency, and exposure to 8/40 HIGH-risk translations. The held-out benefit is not large or consistent enough to justify that production complexity. No post-Test tuning will be performed.

