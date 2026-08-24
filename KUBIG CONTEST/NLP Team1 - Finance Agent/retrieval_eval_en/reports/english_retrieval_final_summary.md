# English Retrieval Final Summary

## Why this evaluation was needed

The corpus is primarily Korean, while users may ask in English. We tested whether multilingual semantic retrieval alone is sufficient and whether sparse or translated Korean lexical signals add reliable coverage.

## Decision path

1. **Korean baseline:** BGE-M3 Dense Top-20 → bge-reranker-v2-m3 → Top-5 was selected over Korean BM25+Dense RRF because it was simpler and had stronger candidate/ranking metrics overall.
2. **English Dense:** Cross-lingual BGE-M3 was already strong on Test40: R@20 0.8958, Hit@20 0.9500; reranked R@5 0.8000, Hit@5 0.8750.
3. **English BGE-M3 Sparse:** Much weaker on average. It found one Dense miss, but Dense+Sparse RRF could not preserve that gain without losing other candidates; no final improvement over Dense.
4. **Paired-KO lexical diagnostic:** Using gold paired Korean questions as a translation upper-bound showed lexical complementarity and recovered a hard Dense miss, motivating an actual translation experiment.
5. **Actual NLLB translation + Nori:** On Validation80, Nori alone was weaker than Dense but recovered 7/11 Dense candidate misses. The theoretical union justified controlled fusion.
6. **Validation architecture selection:** Dense15+Nori5 quota fusion improved every aggregate Validation metric and added three final hits without losing a Dense final hit. It was frozen before Test access.
7. **Sealed Test40:** Fusion preserved all Dense hits and slightly improved evidence Recall/MRR/nDCG, but added zero new question-level hits. Validation's Nori miss-recovery effect did not reproduce.

## Key metrics

| Split | Pipeline | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Validation80 | EN Dense | 0.8281 | 0.8625 | 0.4906 | 0.7198 | 0.7875 | 0.6440 | 0.6271 |
| Validation80 | Dense15+Nori5 | 0.8750 | 0.9125 | 0.4937 | 0.7573 | 0.8250 | 0.6738 | 0.6555 |
| Sealed Test40 | EN Dense | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| Sealed Test40 | Dense15+Nori5 | 0.9125 | 0.9500 | 0.5647 | 0.8000 | 0.8750 | 0.6988 | 0.6636 |

## Final English architecture

`English query → BAAI/bge-m3 Dense Top-20 → BAAI/bge-reranker-v2-m3 → Top-5`

Dense-only is retained because the sealed Test showed no additional question-level hit from translated Nori. Fusion's small ranking/evidence-coverage gains do not justify running NLLB plus Elasticsearch/Nori in the main path. The fixed Test architecture was not retuned after evaluation.

## Translation and Nori limitations

- NLLB Test translations: LOW/MEDIUM/HIGH risk = 21/11/8; several core conditions or Korean finance terms were lost or mistranslated.
- Nori relies on translation wording and added no Dense-miss recovery on Test40.
- Translation and Elasticsearch increase latency, memory, deployment complexity, monitoring needs, and failure modes.
- Test40 is small; metric differences should not be overgeneralized.

## PPT-ready points

- Cross-lingual BGE-M3 Dense achieved 0.9500 Hit@20 on English Test40 against the unchanged Korean-majority corpus.
- BGE-M3 Sparse and Dense+Sparse fusion did not improve the final pipeline.
- Actual translation+Nori showed complementarity on Validation, so a fixed Dense15+Nori5 architecture was tested once on sealed Test40.
- On Test, fusion lost no Dense hits and improved R@20/MRR/nDCG slightly, but produced zero new question-level hits.
- Considering accuracy, translation risk, latency, and infrastructure complexity, the final production candidate is Dense→Reranker.

## Main artifacts

- `retrieval_eval_en/reports/retrieval_en_progress.md`
- `retrieval_eval_en/reports/en_dense_reranker_test.md`
- `retrieval_eval_en/reports/en_sparse_reranker_test.md`
- `retrieval_eval_en/reports/en_dense_sparse_fusion_test.md`
- `retrieval_eval_en/reports/en_paired_ko_lexical_test.md`
- `retrieval_eval_en/reports/en_translated_nori_validation.md`
- `retrieval_eval_en/reports/en_dense_nori_fusion_validation.md`
- `retrieval_eval_en/reports/en_dense_nori_final_test.md`
- `retrieval_eval_en/results/results_400_60_en_dense_nori_final_test.json`

