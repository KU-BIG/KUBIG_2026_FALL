# Source-Seeded Retrieval Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the framework to a deterministic 50-question source-seeded evaluation with AI-assisted drafting and blind relevance judgment.

**Architecture:** Strict schema and assignment modules own evaluation policy. AI helpers emit draft-only structured artifacts, while existing retrievers provide full-corpus candidates and the runner retains article-level metrics.

**Tech Stack:** Python 3.13 standard library, existing ClaudeClient, DenseRetriever, HybridRetriever, pytest

**Spec:** `docs/superpowers/specs/2026-08-20-retrieval-eval-pilot-design.md`

## Global Constraints

- Search all 432 articles and 1,377 chunks; never reduce the corpus to 50 sources.
- Do not change Chroma, cleaned/chunk data, model, candidate_k, rrf_k, or RRF.
- AI output is draft-only; approval requires author self-check and no uncertainty.
- Do not commit or push without a separate user request.

---

### Task 1: Source-seeded schema

**Files:** Modify `evaluation/schema.py`, example JSONL, and `tests/test_evaluation_schema.py`.

**Interfaces:** `validate_records(records, article_ids, *, allow_draft) -> None` accepts only the final schema.

- [ ] Add failing tests for rejected query_first, required seed, nullable reviewer, fixed review mode, five self-check booleans, initial source gold, additional relevant gold, and final uncertain rejection.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement minimal validation and update the draft example.
- [ ] Re-run focused tests and confirm PASS.

### Task 2: Deterministic assignment

**Files:** Create `evaluation/assignment.py`, `tests/test_evaluation_assignment.py`; modify `evaluation/cli.py`.

**Interfaces:** `create_assignment(articles, seed) -> dict`, `replace_source(assignment, position, reason_code) -> dict`, and `swap_categories(assignment, first_position, second_position) -> dict`.

- [ ] Add failing tests for 10/10/18/12 strata, 25/25 authors with 5/5/9/9/6/6 splits, 13/13/12/12 categories spread within dates, reproducibility, same-stratum reserves, predefined replacement reasons, and quota-preserving within-stratum category swaps.
- [ ] Run focused tests and confirm missing behavior.
- [ ] Implement deterministic sampling and an `assign` CLI command without retrieval imports.
- [ ] Re-run focused tests and confirm PASS.

### Task 3: AI-assisted drafting and blind judging

**Files:** Create `evaluation/ai_assist.py`, `tests/test_evaluation_ai_assist.py`; modify `evaluation/pooling.py`, `evaluation/cli.py`, and pooling/CLI tests.

**Interfaces:** `generate_question(article, category, llm) -> dict`, `judge_candidates(question, candidates, llm) -> list[dict]`, and always-blind `build_pool(...)`.

- [ ] Add failing fake-LLM tests proving draft-only output, no system/rank leakage, frozen question preservation, three relevance labels, evidence support, and no automatic evaluation-file edit.
- [ ] Run focused tests and confirm failures.
- [ ] Implement strict JSON parsing, prompt boundaries, always-blind pools, and `generate-question`/`judge` CLI commands.
- [ ] Re-run focused tests and confirm PASS.

### Task 4: Results, documentation, and verification

**Files:** Modify `evaluation/runner.py`, `evaluation/README.md`, reporting/runner tests, design and plan docs.

**Interfaces:** result metadata distinguishes `corpus.article_count=432` from `source_article_count=50`; summaries include overall, date-stratum, category metrics, stratum corpus/sample counts, and the AI/self-check limitation.

- [ ] Add failing tests for full-corpus metadata and limitation text.
- [ ] Implement metadata/summary changes and replace the two-method documentation with the final 50-question workflow and known-item distinction.
- [ ] Run all evaluation tests, then `uv run pytest -q`, `uv lock --check`, and `git diff --check`.
- [ ] Inspect status and report changes without committing or pushing.
