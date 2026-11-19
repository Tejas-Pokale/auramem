# CLAUDE.md — Memory Layer

## What this is
A self-hosted long-term memory layer for LLMs. Full spec: docs/PLAN.md.
Read the relevant PLAN.md section before working on any module.

## Non-negotiable rules
- Raw turns are never modified or deleted except by an explicit forget request.
- Facts are ADD-only. Changes set is_latest=false and add SUPERSEDES; never delete or overwrite.
- Every store function takes user_id. Every vector/full-text query filters user_id in-index.
- All Cypher lives in src/memory/store/. No Cypher strings anywhere else.
- Every LLM call returns a Pydantic model from src/memory/models/ via with_structured_output.
- Prompts live in src/memory/prompts/ with a version suffix; store the version on outputs.
- All thresholds come from config.py. No magic numbers in code.
- EmbeddingGemma prompts are prepended manually:
  query: "task: search result | query: "  document: "title: none | text: "
- Rerank against the ORIGINAL user question, never a rewrite.
- The write path is async (arq). Nothing in the read path may wait on extraction.
- Never change evals/custom/judge_v1.md without bumping its version. Never add question-specific rules to answer prompts. Tune on the dev split only.

## Models
- claude-haiku-5-5: planner, extractor, observer, entity summaries, default answerer,
  deep-mode answerer (medium effort; user decision 2026-10-25)
- claude-sonnet-5-5: reflector, entity tiebreaks
- embeddinggemma:300m (Ollama), BAAI/bge-reranker-v2-m3 (sentence-transformers)

## Workflow
- Work one phase at a time (PLAN.md section 9). Do not start a phase until the previous gate passes.
- For each task: write or update tests first, implement, run `uv run pytest`, `uv run ruff check`, `uv run mypy src`.
- Integration tests use testcontainers for Neo4j and Redis; never mock the database for retrieval tests.
- After any change to retrieval, extraction, or prompts, run L0 (retrieval) and L1 (dev split) of the custom eval
  (`uv run python -m evals.custom.run --level L1 --tier medium --split dev`) and report the per-category delta.
  LongMemEval (evals/longmemeval/) is an optional one-time L3 check after Phase 8 only.
- Log scores in evals/RESULTS.md with date, git SHA, config, and model versions.

## Commands
- Start infra: `docker compose up -d`
- Migrations: `uv run python -m memory.store.migrate`
- Worker: `uv run arq memory.jobs.worker.WorkerSettings`
- API: `uv run uvicorn memory.api.main:app --reload`

## When unsure
Prefer the simpler design that keeps raw text searchable. Ask before adding a new datastore,
a new model, or a new top-level module.
