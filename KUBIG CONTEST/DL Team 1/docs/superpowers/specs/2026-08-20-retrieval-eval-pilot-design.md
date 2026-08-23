# Source-Seeded Retrieval Evaluation Design

## Purpose

Build a 50-question, source-seeded evaluation of the existing Dense and Hybrid
retrievers. Fifty unique source articles are sampled deterministically from the
full 432-article corpus; each is an initial gold, but other pooled articles may
also be relevant. Search always uses all 432 articles and 1,377 chunks.

## Workflow

1. Stratify by 2026-07-31–08-05, 08-06, 08-07, and 08-08; sample 10, 10, 18, and 12 unique sources with a fixed seed inside each stratum.
2. Split every stratum equally between annotators (5/5, 5/5, 9/9, 6/6) and retain every unselected article as a deterministic same-stratum reserve list.
3. Assign categories within strata as `3/3/2/2`, `3/2/3/2`, `4/5/4/5`, and `3/3/3/3` for exact_token/abstract/multi_aspect/factoid, totaling 13/13/12/12.
4. Claude proposes a draft question from each source article. The author edits and freezes the question, then performs the required self-check.
5. Dense and Hybrid search the full corpus. Candidates are merged by `article_id` and stripped of system identity and original rank.
6. Claude judges blind candidates as relevant, not_relevant, or uncertain and records evidence. The author resolves uncertainty and may add relevant candidates to gold.
7. Final validation requires approval, complete self-check, the source in gold, and no uncertain evidence.
8. Paired article-level evaluation produces Hit@1/3/5 and MRR@5 plus diagnostic rankings and overlap.

## Schema

- `construction_method` must be `source_seeded`.
- `seed_article_id` is required, exists in the corpus, and is present in `gold_article_ids` for approved records.
- `category` is exact_token, abstract, multi_aspect, or factoid; `date_stratum` records the fixed source-date stratum.
- `reviewer` is a string or null; cross-review is not required.
- `review_mode` must be `ai_assisted_self_check`.
- `self_check` contains five required booleans: `answer_supported_by_source`, `natural_question`, `not_title_copy`, `not_duplicate`, and `source_article_id_verified`. All must be true for approval.
- Evidence relevance is relevant, not_relevant, or uncertain. Relevant evidence belongs to gold; final evaluation rejects uncertain items.

## Assignment and Replacement

Assignment uses only article IDs and dates from the cleaned corpus plus `random.Random` with an explicit seed. Retrieval is never involved. Each stratum is shuffled independently; its primary quota is selected and all remaining articles become ordered reserves. Replacement consumes the next reserve from the same stratum and requires one predefined reason: incomplete article, question generation impossible, or near-duplicate event. Category swaps are allowed only inside the same stratum and must preserve global quotas.

## AI Safety Boundaries

`generate-question` and `judge` reuse the existing Claude client but emit draft JSON to stdout or an explicitly named output file. They never edit the evaluation JSONL, never approve a record, never alter a frozen question, and never see system names or ranks during relevance judgment. AI output is validated before display or storage.

## Results and Limitations

Results record full corpus counts separately from `source_article_count=50`, plus overall, date-stratum, and question-category metrics and per-stratum corpus/sample counts. Date comparisons are exploratory because samples are small. Summaries state that questions and relevance judgments were AI-assisted with single-annotator self-check and no independent cross-review. Unlike strict known-item retrieval, the source is an initial gold rather than the only acceptable answer; blind pooled candidates may expand gold when they answer the frozen question.

## Boundaries and Testing

Do not change Chroma, source/clean/chunk data, models, `candidate_k`, `rrf_k`, RRF, or the full corpus. Tests use fake retrievers and fake LLMs for schema constraints, deterministic allocation, category quotas, reserve replacement, blind pooling, additional gold, unresolved uncertainty, AI draft-only behavior, metadata, and existing metrics.
