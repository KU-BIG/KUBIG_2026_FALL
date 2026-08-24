# Experiment 5 — Dense + Actual Translation→Nori Fusion (Validation80)

## Goal and fixed scope

This experiment asks whether actual translated Nori candidates can recover English Dense misses without materially damaging the strong Dense baseline. It uses the unchanged Final 400/60 corpus (563 chunks), Final gold, Validation80 only, candidate/final depth 20/5, the evidence-group-aware metrics in `retrieval_eval/eval_retrieval.py`, and `BAAI/bge-reranker-v2-m3` with the original English query. Test40 was not translated, retrieved, reranked, or tuned.

Dense and Nori Top-20 rankings and scores were reused from `results/results_400_60_en_translated_nori_validation.json`. Translation, Nori retrieval, Dense embedding, analyzer configuration, and BM25 were not rerun or changed. Only the unique 2,887 Validation question–chunk pairs in the Dense∪Nori pools were scored once by the reranker and reused across all variants.

## Fusion definitions

- Equal RRF 1:1: `1/(60+dense_rank) + 1/(60+nori_rank)`
- Weighted RRF 2:1 and 3:1: multiply the Dense contribution by 2 or 3; RRF k=60
- Dense15+Nori5: Dense ranks 1–15, then unseen Nori ranks 1–5, then unseen Dense ranks 16–20 until 20 candidates
- Dense18+Nori2: Dense ranks 1–18, then unseen Nori ranks 1–2, then unseen Dense ranks 19–20 until 20 candidates
- Signal-internal order is preserved. Duplicate chunks are removed. A defensive unseen-Nori fallback is defined but the Dense Top-20 is sufficient to fill every list.

Only these five predeclared variants were evaluated; there was no broad weight or quota search.

## Overall results

| Pipeline | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.8281 | 0.8625 | 0.4906 | 0.7198 | 0.7875 | 0.6440 | 0.6271 |
| Equal RRF 1:1 | 0.8417 | 0.8875 | 0.4421 | 0.7604 | 0.8250 | 0.6706 | 0.6546 |
| Weighted RRF 2:1 | 0.8281 | 0.8625 | 0.4503 | 0.7198 | 0.7875 | 0.6440 | 0.6271 |
| Weighted RRF 3:1 | 0.8281 | 0.8625 | 0.4670 | 0.7198 | 0.7875 | 0.6440 | 0.6271 |
| Dense15 + Nori5 | 0.8750 | 0.9125 | 0.4937 | 0.7573 | 0.8250 | 0.6738 | 0.6555 |
| Dense18 + Nori2 | 0.8656 | 0.9000 | 0.4926 | 0.7573 | 0.8250 | 0.6669 | 0.6504 |

Dense15+Nori5 is the best balance: versus Dense it improves candidate R/Hit/MRR by +0.0469/+0.0500/+0.0031 and final R/Hit/MRR/nDCG by +0.0375/+0.0375/+0.0298/+0.0285. Equal RRF has slightly higher final Recall than the quota (0.7604 vs 0.7573), but weaker candidate metrics, loses four Dense hits, and has lower final MRR/nDCG.

## Dense-hit protection and candidate transitions

| Pipeline | Dense hits kept | Dense hits lost | Dense miss→hit | Dense hit→miss |
|---|---:|---:|---:|---:|
| Equal RRF | 65/69 | 4 | 6 | 4 |
| Weighted RRF 2:1 | 69/69 | 0 | 0 | 0 |
| Weighted RRF 3:1 | 69/69 | 0 | 0 | 0 |
| Dense15 + Nori5 | 68/69 | 1 | 5 | 1 |
| Dense18 + Nori2 | 68/69 | 1 | 4 | 1 |

- Equal RRF loses `REM_004`, `REM_013`, `FRD_001`, `FRD_005`.
- Both quota variants lose only `REM_004` at candidate Top-20.
- Dense-heavy RRF protects every Dense hit but admits none of the seven Nori-only hits, so it adds complexity without coverage gain.

Candidate first-gold change counts versus Dense (`improved / worsened / same / miss→hit / hit→miss`) are:

- Equal RRF: 20 / 30 / 20 / 6 / 4
- Weighted RRF 2:1: 20 / 30 / 30 / 0 / 0
- Weighted RRF 3:1: 20 / 29 / 31 / 0 / 0
- Dense15+Nori5: 0 / 1 / 73 / 5 / 1
- Dense18+Nori2: 0 / 0 / 75 / 4 / 1

For quota fusion, added Nori items form a block after the Dense quota, so their candidate ranks are not intended to optimize MRR; the reranker decides final order.

## Nori-only seven tracking

`candidate rank / reranker rank`; `miss` means no gold at that stage.

| Question | Risk | Equal RRF | RRF 2:1 | RRF 3:1 | Dense15+Nori5 | Dense18+Nori2 |
|---|---|---|---|---|---|---|
| ACC_003 | MEDIUM | 2 / miss | miss / miss | miss / miss | 16 / miss | 19 / miss |
| ACC_005 | LOW | 7 / 4 | miss / miss | miss / miss | 16 / 2 | 19 / 2 |
| ACC_012 | MEDIUM | 13 / 1 | miss / miss | miss / miss | miss / miss | miss / miss |
| ACC_020 | MEDIUM | 8 / miss | miss / miss | miss / miss | 17 / miss | miss / miss |
| REM_025 | LOW | 5 / 3 | miss / miss | miss / miss | 16 / 3 | 19 / 3 |
| NOT_014 | HIGH | 5 / 1 | miss / miss | miss / miss | 16 / 1 | 19 / 1 |
| NOT_015 | LOW | miss / miss | miss / miss | miss / miss | miss / miss | miss / miss |

Preserved at candidate / final Top-5:

- Equal RRF: 6/7 → 4/7
- RRF 2:1: 0/7 → 0/7
- RRF 3:1: 0/7 → 0/7
- Dense15+Nori5: 5/7 → 3/7
- Dense18+Nori2: 4/7 → 3/7

`NOT_014` is HIGH risk: translation omitted the requested marriage/family-document condition, so its category-overlap retrieval should not be overinterpreted. The cleaner new final hits are `ACC_005` and `REM_025` (both LOW risk). `ACC_012` is an additional MEDIUM-risk final hit available only under Equal RRF.

## Common misses

`ACC_002`, `ACC_022`, `REM_018`, and `FRD_017` remain candidate and reranker misses for every variant. Neither input ranking contained a gold chunk, so rank-only fusion cannot recover them.

## Reranker-level changes

Compared with Dense→Reranker (`improved / worsened / same / miss→hit / hit→miss`):

- Equal RRF: 4 / 0 / 69 / 5 / 2; new hits `ACC_005`, `ACC_012`, `REM_025`, `NOT_014`, `FRD_002`; lost hits `REM_013`, `FRD_027`
- RRF 2:1: 0 / 0 / 80 / 0 / 0
- RRF 3:1: 0 / 0 / 80 / 0 / 0
- Dense15+Nori5: 2 / 0 / 75 / 3 / 0; new hits `ACC_005`, `REM_025`, `NOT_014`
- Dense18+Nori2: 0 / 0 / 77 / 3 / 0; new hits `ACC_005`, `REM_025`, `NOT_014`

Dense15+Nori5 gains three final hits without losing any Dense final hit and improves two existing first-gold ranks (`ACC_029`, `FRD_012`). It also outperforms Dense18+Nori2 on candidate coverage and final MRR/nDCG.

## Decision

The recommended English architecture candidate is:

`English query → BGE-M3 Dense Top-15`  
`English query → NLLB-200 EN→KO → Nori BM25 Top-5`  
`deterministic de-duplication/fill → Top-20 → bge-reranker-v2-m3 with original English query → Top-5`

This is selected over Dense-only because every reported aggregate metric improves on Validation80, no Dense final Top-5 hit is lost, and two LOW-risk Nori-only cases become final hits. The added translation, Elasticsearch/Nori, latency, and operational complexity remain real costs; `NOT_014` also shows translation-dependent false confidence risk. Therefore the improvement is promising but must be confirmed once on the sealed Test40 before it becomes the final architecture.

