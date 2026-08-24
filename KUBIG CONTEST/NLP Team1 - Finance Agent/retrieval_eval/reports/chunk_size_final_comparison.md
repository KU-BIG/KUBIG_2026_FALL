# Final Chunk Size Comparison

## Experimental Control

This experiment re-evaluates 300/50, 400/60, and 500/80 under the same Final conditions. The historical results in `retrieval_eval/reports/initial_method_chunk_comparison.md` and the old `results_300_50.json` / `results_500_80.json` artifacts were not reused: those artifacts contain only 26 evaluated questions and predate the Final corpus and gold correction.

- Documents: the same 21 Final documents, including PDF002 pages 17–19 only
- Tokenizer: `BAAI/bge-m3`
- Evaluation data: the same 120 questions and source evidence
- Evaluation split: Korean Test40 only (Validation80 was not used for the reported metrics)
- Gold: the same source-span matching and boundary-aware evidence-group mapping rule for every setting
- Dense model: `BAAI/bge-m3`
- Reranker: `BAAI/bge-reranker-v2-m3`
- Candidate / final cutoff: Top-20 / Top-5
- Metrics: the evidence-group-aware implementation in `retrieval_eval/eval_retrieval.py`

The regenerated 400/60 chunks are byte-identical to the canonical Final chunks, and the recomputed 400/60 evidence gold is an exact match to the canonical gold. Its Dense Top-20 and reranked Top-5 rankings also exactly match the existing Final Test40 artifact.

## Chunk Statistics

| Setting | Chunks | Avg. tokens | Max tokens | Avg. characters | Chunk hash (SHA-256) |
|---|---:|---:|---:|---:|---|
| 300/50 | 715 | 244.12 | 300 | 666.42 | `b8ed7a885c69fc2e44cd64a0f36447b80a2f14a67157a5fe88d968394cf1f71e` |
| 400/60 | 563 | 301.09 | 400 | 820.48 | `085d8ddf536741e227467bf00f5d67fa5d88b4ab396f82e91834ca6714aa1b3b` |
| 500/80 | 489 | 345.06 | 500 | 939.28 | `91d82518b3654dab3422f3610c65ee079633046c5792b38996539a5769216934` |

All settings use the same document-local chunk ID rule: `{document_id}_{1-indexed sequence:04d}`. Complete document-level counts and hashes are in the backing JSON.

## Gold Mapping Validation

| Setting | Evidence | Matched | Coverage | Single gold | Multi gold | Boundary-spanning fixed | Partial | Invalid | Corpus error | Unmatched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300/50 | 167 | 167 | 100% | 140 | 27 | 21 | 0 | 0 | 0 | 0 |
| 400/60 | 167 | 167 | 100% | 125 | 42 | 30 | 0 | 0 | 0 | 0 |
| 500/80 | 167 | 167 | 100% | 146 | 21 | 16 | 0 | 0 | 0 | 0 |

The 33 evidence items previously identified for boundary review were processed with one shared rule. A larger multi-gold count does not mean lower quality; it means the evidence group spans more than one valid chunk under that boundary layout.

## Test40 Dense → Reranker Results

| Chunk | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300/50 | **0.9333** | **0.9750** | 0.5957 | **0.7804** | **0.8500** | **0.6621** | **0.6414** |
| 400/60 | **0.9333** | 0.9500 | **0.6244** | 0.7792 | 0.8250 | 0.6446 | 0.6376 |
| 500/80 | 0.9083 | 0.9250 | 0.6155 | 0.7229 | 0.7750 | 0.6196 | 0.5927 |

Compared with 400/60, 300/50 adds one candidate-level hit and one final Top-5 hit. Its final MRR is higher by 0.0175 and nDCG by 0.0039, while 400/60 candidate MRR is higher by 0.0288. The two settings have the same evidence Recall@20 to four decimals. The 500/80 setting is lower on both candidate coverage and final ranking metrics.

## Query-level Differences

### Hit patterns

| Stage | All three hit | All three miss | 300 only | 400 only | 500 only |
|---|---:|---:|---:|---:|---:|
| Candidate Top-20 | 37 | 1 (`FRD_015`) | 1 (`ACC_015`) | 0 | 0 |
| Reranker Top-5 | 30 | 5 | 2 (`ACC_015`, `FRD_009`) | 0 | 0 |

- Candidate Top-20: `REM_003` is retrieved by 300/50 and 400/60 but missed by 500/80. The other 37 questions are hits in every setting.
- Reranker Top-5: `ACC_010` is a hit for 400/60 and 500/80 but not 300/50. `FRD_021` and `REM_003` are hits for 300/50 and 400/60 but not 500/80.
- The five common Top-5 misses are `ACC_006`, `ACC_016`, `FRD_006`, `FRD_015`, and `FRD_020`.

### Representative boundary and ranking cases

1. **Small chunks isolate a useful cue — `ACC_015`.** The 300/50 layout retrieves the English ATM operating-hours chunk at candidate rank 2 and reranks it to rank 1. The corresponding 400/60 and 500/80 gold chunks do not enter Top-20. The shorter layout separates the explicit `10:00 PM ~ 8:00 AM` cue from surrounding material.
2. **Small chunks can omit helpful context — `ACC_010`.** The query asks for proof of 183-day Korean residence. The 400/60 and 500/80 gold chunks contain the residence-document list together and reach candidate rank 1, then final ranks 4 and 3. In 300/50, the valid evidence is split across three gold chunks; its first candidate appears at rank 18 and finishes outside Top-5.
3. **A larger chunk can dilute a focused distinction — `REM_003`.** The question distinguishes cash exchange from remittance exchange rates. The gold is candidate rank 1 / final rank 2 for 300/50 and candidate rank 6 / final rank 5 for 400/60, but is absent from the 500/80 Top-20.
4. **Reranking is sensitive to surrounding text — `FRD_009`.** All three settings retrieve a valid fraud-response gold candidate, but it finishes at final rank 5 only for 300/50 (400/60 rank 7; 500/80 rank 6). The smaller chunk concentrates the stop-contact / refuse-information action cues.
5. **Very large context can lose a borderline Top-5 placement — `FRD_021`.** The same 1394 / 112 / 1332 contact table is retrieved by all settings. It is final rank 4 for 300/50 and 400/60, but rank 6 for 500/80.

These cases are observations from the actual gold chunks and rankings. They do not establish a universal causal rule for every query.

## Interpretation

The comparison is mixed rather than a clean win for one setting:

- **300/50** has the best question-level Hit@20 and Hit@5 and the best final MRR/nDCG, but creates 715 chunks—152 more than 400/60—and has lower candidate MRR.
- **400/60** ties the best evidence Recall@20, has the best candidate MRR@20, and is within 0.0039 nDCG@5 of 300/50 with 21.3% fewer chunks.
- **500/80** reduces the corpus to 489 chunks, but the loss in candidate and final recall/ranking is material on this Test40.

Because Test40 differences between 300/50 and 400/60 are small and metric-dependent, this is **Case C**, not evidence that 400/60 wins every metric. The current 400/60 setting remains a defensible balance of candidate ranking, context granularity, corpus size, and operational continuity. A performance-driven switch to 300/50 would require rerunning the downstream Korean and English architecture artifacts, which are all based on 400/60.

## Final Chunk Decision

**Retain 400/60**, with a qualified interpretation: it is the balanced operating point, not the unique performance optimum. The new controlled comparison shows a modest Top-5 advantage for 300/50, while 400/60 offers the best candidate MRR and substantially fewer chunks with nearly identical final ranking quality. No existing Korean or English result is rewritten by this experiment.

## PPT-ready Takeaway

Under the same Final corpus, gold mapping, Test40, Dense model, and reranker, 300/50 and 400/60 performed similarly, while 500/80 was weaker. We retain 400/60 as the balanced setting: it matches the best Recall@20, gives the best candidate MRR, and uses 21% fewer chunks than 300/50, with only a small final-ranking difference.

## Artifacts

- Raw comparison: `retrieval_eval/results/results_chunk_size_final_comparison.json`
- Reproduction script: `retrieval_eval/run_chunk_size_final_comparison.py`
- New Final chunks and remapped gold: `retrieval_eval/chunk_size_final_artifacts/`
- Canonical 400/60 chunks remain at `retriever_dataset/chunks/chunk_400_60/chunks.jsonl`
