# Experiment 4B — Actual Translation + Nori BM25 (Validation80)

## Purpose and scope

This experiment evaluates the actual path `question.en → machine translation → Korean Nori BM25 Top-20 → original English query + bge-reranker-v2-m3 → Top-5`. It uses **Validation80 only**. Test40 was not translated, retrieved, reranked, or used for configuration decisions. Dense+Nori fusion was not run.

## Fixed setup

- Corpus: Final 400/60, 563 unchanged chunks
- Split: Validation80, same Final evidence-group gold
- Candidate/final depth: 20/5
- Dense: `BAAI/bge-m3`, existing 563×1024 corpus embedding cache
- Reranker: `BAAI/bge-reranker-v2-m3`; original English question is paired with each Korean/multilingual chunk
- Metrics: `retrieval_eval/eval_retrieval.py` evidence-group-aware Recall, Hit, MRR, and nDCG
- No parameter, analyzer, or BM25 sweep

## Machine translation

- Model: `facebook/nllb-200-distilled-600M`, revision `f8d333a098d19b4fd9a8b18f94170487ad3f821d`
- Direction: `eng_Latn → kor_Hang`
- Decoding: beam search (`num_beams=4`, `do_sample=false`, `early_stopping=true`), input max length 256, output max 128, batch 8
- Runtime: local model, Apple MPS; no paid API
- Input was strictly `question.en`. Paired `question.ko` was used only as an audit reference.

All 80 outputs were non-empty and contained Hangul. A complete paired-reference sanity audit classified 50 LOW, 18 MEDIUM, and 12 HIGH risk translations. This is not a professional translation certification. Important HIGH-risk cases include:

- `ACC_004`: the ATM-specific question and daily-limit request were dropped.
- `REM_005`: SentBe, receipt mode, and per-transfer-limit comparison were dropped.
- `NOT_027`: diapers/formula were severely mistranslated.
- `FRD_001`: caller ID was mistranslated as a resident-registration number and the trust question was dropped.

Thus this is an end-to-end translation-component evaluation; Nori misses cannot all be attributed to the lexical backend.

## Elasticsearch and Nori implementation

- Elasticsearch 8.17.0 tar distribution with bundled JDK, isolated under `/private/tmp`
- Official `analysis-nori` 8.17.0 plugin
- Dedicated index: `kubig_finance_final_400_60_validation`, exactly 563 documents
- Custom analyzer `kubig_nori`: `nori_tokenizer`, `decompound_mode=mixed`, `discard_punctuation=true`, followed by `lowercase`
- Explicit BM25: k1=1.2, b=0.75, `discount_overlaps=true`
- One shard, zero replicas; `text` is the scored field and `chunk_id` is preserved for gold linkage

The analyzer probe `주택청약통장과 외국인등록증` produced Nori morpheme/decompound tokens, confirming that the actual Nori plugin—not generic whitespace or character-bigram tokenization—was used. Exact settings and mappings are in `data/nori_index_config.json`.

## Validation80 results

| Pipeline | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN Dense → Reranker | 0.8281 | 0.8625 | 0.4906 | 0.7198 | 0.7875 | 0.6440 | 0.6271 |
| Actual Translation → Nori BM25 → Reranker | 0.5427 | 0.5875 | 0.2578 | 0.5385 | 0.5750 | 0.4981 | 0.4740 |
| Δ Nori − Dense | -0.2854 | -0.2750 | -0.2328 | -0.1813 | -0.2125 | -0.1458 | -0.1530 |

Dense is substantially stronger on average. Nori candidate recall almost entirely survives reranking (0.5427→0.5385), but the candidate pool itself is much weaker.

## Candidate complementarity

- Dense hit / Nori hit: 40
- Dense hit / Nori miss: 29
- Dense miss / Nori hit: 7 — `ACC_003`, `ACC_005`, `ACC_012`, `ACC_020`, `REM_025`, `NOT_014`, `NOT_015`
- Dense miss / Nori miss: 4 — `ACC_002`, `ACC_022`, `REM_018`, `FRD_017`

All seven Nori-only candidates survived into Nori's reranked Top-5: their Nori candidate/reranker first-gold ranks were respectively `1/2`, `2/4`, `6/1`, `3/5`, `1/1`, `1/1`, and `15/1`. Representative lexical cues include:

- `ACC_005`: domestic/overseas card and annual-fee wording; Nori rank 2, reranker rank 4.
- `ACC_020`: Shinhan video-consultation center and foreigner ID wording; Nori rank 3, reranker rank 5.
- `REM_025`: exchange student, admission letter, and overseas study/remittance terms; Nori rank 1, reranker rank 1.
- `NOT_015`: savings-account early termination and partial withdrawal/loan context; Nori rank 15, reranker rank 1.

`NOT_014` is a cautionary case: Nori retrieved its gold at rank 1 despite a HIGH-risk translation that omitted the requested marriage/family documents. Its success reflects broad product/category overlap and must not be treated as proof of translation fidelity.

Among all 33 Nori misses, translation risk was LOW 23, MEDIUM 6, HIGH 4. Four Dense+Nori common misses include one MEDIUM-risk translation (`ACC_022`) and three LOW-risk translations. Therefore translation errors explain part, but not most, of Nori's misses under this audit; lexical mismatch and corpus wording remain separate limitations.

## Theoretical Dense ∪ Nori coverage

- Dense hits: 69/80 (0.8625)
- Nori hits: 47/80 (0.5875)
- Union hits: 76/80 (0.9500)
- Union evidence-recall upper bound: 0.9156
- Mean Top-20 overlap: 3.9125 chunks
- Mean Jaccard: 0.1152
- Identical Top-20 sets: 0/80

This is theoretical union coverage, not the performance of a fused Top-20 ranking. The seven Nori-only hits and low overlap show real diversity, while the much lower Nori average means an actual fusion must protect Dense candidates.

## Decision

This result is **Case B**: translated Nori is too weak to replace English Dense, but it recovers seven of Dense's eleven candidate misses and raises the theoretical hit ceiling from 0.8625 to 0.9500. A constrained Dense+Nori fusion experiment on Validation80 is warranted, with Dense as the primary signal and no broad weight sweep. Translation quality is a material bottleneck and should be reported alongside fusion outcomes. Final architecture remains unconfirmed; Test40 stays sealed until selection is complete.

## Artifacts

- `retrieval_eval_en/data/validation_en_to_ko_translations.json`
- `retrieval_eval_en/data/nori_index_config.json`
- `retrieval_eval_en/results/results_400_60_en_translated_nori_validation.json`

