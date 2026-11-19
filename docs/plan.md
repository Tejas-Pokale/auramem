# Long-Term Memory Layer for LLMs — Build Plan

Oct 24, 2026 · @Tejas Pokale

## 1. Problem, goals, and success criteria

We are building a self-hosted, long-term memory layer that lets any LLM remember a user across sessions, track how facts change over time, and answer time-based questions correctly.

**Problem.** LLMs are stateless. Each conversation starts from zero. Stuffing full history into the prompt fails: it is expensive, it degrades with length (context rot, lost-in-the-middle), and on LongMemEval full-context GPT-4o scores only about 60%. Naive vector RAG over chat logs misses temporal questions, preference changes, and facts spread across sessions.

**Solution in one sentence.** Keep every raw message, extract time-anchored narrative facts and events on top of it in the background, maintain a compact always-on observation log, and answer with multi-signal retrieval plus an optional agentic search loop.

### Goals

1. Never lose anything the user or assistant said (raw turns are permanent unless the user asks to forget).
2. Correct knowledge updates: "where do I work now?" returns the latest value, and "where did I work before?" still works.
3. Correct temporal reasoning: relative dates resolved to absolute ranges; "what did I do the week after my trip?" works.
4. Implicit personalization: preferences surface even when the user does not ask about memory.
5. Abstain when memory has no answer instead of inventing one.
6. Low added latency on the read path; all heavy work is asynchronous.
7. Strict per-user isolation and real deletion on request.

### Non-goals (v1)

- Multimodal memory (images, audio).
- Organization-level shared memory across users.
- Fine-tuning or parametric memory.
- A hosted multi-region service; v1 is a single-node Docker deployment.

### Success criteria

| Metric | Target | How measured |
| --- | --- | --- |
| Answer accuracy, golden v2 `test` split, Large tier | at least 85% (stretch: 90%) | Custom eval L2, fixed judge, 3 runs averaged (section 10) |
| Evidence recall | at least 95% of questions | All `evidence_fact_ids` present in assembled context (L0, no LLM cost) |
| `conflict` and `knowledge_update` items | 100% correct | Same run, per-type breakdown |
| `abstention` items | at least 90% correct; fabrication on at most 1 item | Same run |
| Judge agreement with hand labels | at least 90% | 30 hand-graded judgments |
| Read-path added latency, quick mode | p95 under 600 ms | Excludes the answering LLM call |
| Memory context per query | 4,000–8,000 tokens | Logged per request |
| Write-path lag | p95 under 30 s from message to searchable facts | Queue timestamps |

## 2. Research basis and design principles

The design combines the best-validated piece of each top 2026 system; heavy knowledge-graph-first extraction is deliberately avoided because it underperforms.

| System | LongMemEval score (as reported) | What we borrow |
| --- | --- | --- |
| [Chronos](https://arxiv.org/html/2603.16862v1) (PwC) | 92.6% GPT-4o; 95.6% Opus 4.6; 94.0% Haiku 4.5 | Dual index: raw turns + time-structured events; event aliases; query-guidance prompt; agentic search with grep; ±1 turn expansion |
| [Mastra Observational Memory](https://mastra.ai/research/observational-memory) | 94.87% gpt-5-mini; 84.23% GPT-4o | Always-on, append-only observation log (prompt-cacheable); Observer/Reflector agents; three-date temporal model |
| [Mem0 token-efficient algorithm](https://docs.mem0.ai/core-concepts/memory-evaluation.md) | 94.4% (vendor-reported) | ADD-only writes; assistant facts stored as first-class; 4-signal retrieval; \~6.8K-token context budget |
| [Hindsight](https://arxiv.org/html/2512.12818v1) | 91.4% | Narrative (not fragmented) facts; entity resolution scoring; 4-way retrieval + RRF + cross-encoder; rule-based date parsing |
| Zep (temporal KG on Neo4j) | 71.2% GPT-4o | Cautionary: graph-first extraction alone is not enough |
| Full context baseline | 60.2% GPT-4o | Floor to beat |

Vendor numbers are optimistic. An [independent reproduction by Maximem](https://www.maximem.ai/blog/state-of-ai-memory-2026-claimed-vs-observed) (a competing vendor) measured Mem0 at 73.8% against a published 93.4%, attributing the gap to benchmark-tuned answer and judge prompts. We therefore evaluate on our own dataset with a fixed, versioned judge (section 10).

### Design principles (non-negotiable)

1. **Raw turns are the foundation.** Every user and assistant message is stored verbatim and searchable. Extraction adds structure on top; it never replaces source text.
2. **Structure time, not everything.** Chronos's ablation shows the event index is the largest single gain, especially for weaker models (removing it cost 34.5 points with GPT-4o vs 2.6 with Opus). Extract time-anchored events and facts; skip exhaustive triples.
3. **ADD-only.** Never overwrite or delete a fact because it changed. Store both versions with dates; mark the older one `is_latest = false`. Resolve conflicts at read time.
4. **Attribute everything.** Store assistant-said facts too (Mem0's largest gain, +53.6 on assistant recall), tagged with who said it.
5. **Multi-signal retrieval.** Vector + BM25/exact match + entity + temporal + graph, fused with RRF and reranked by a cross-encoder. Exact-match search is mandatory (vector-only cost Chronos 25 points on one category).
6. **Generous but bounded context.** Target 4,000–8,000 memory tokens, not 1,500.
7. **The router becomes a planner.** A small model never gates retrieval; it produces retrieval guidance in parallel with retrieval. This helped GPT-4o by about 15 points in Chronos and is harmless for strong models.
8. **Heavy work is asynchronous.** The user never waits on extraction, entity resolution, or reflection.
9. **Measure honestly.** Official judge, fixed prompts, ablations for every component.

## 3. Architecture overview

The system has three parts: a synchronous read path, a single Neo4j store, and an asynchronous write path; the user only ever waits on the read path.

&#91;embedded content: System architecture · read path, single store, write path\]

Each box in the read and write paths is a LangGraph node. Inside Neo4j, every `Fact` links to the `Turn`s it came from (`DERIVED_FROM`) and the `Entity`s it mentions (`MENTIONS`), so any fact can be traced back to its exact source text.

### How the original design maps to this one

| Original idea | Final design | Why it changed |
| --- | --- | --- |
| Router gates memory use | Query planner runs in parallel; never blocks retrieval | A wrong "no memory" decision fails silently |
| Layer 1: Neo4j graph memory | `Fact` + `Entity` nodes, ADD-only, time-ranged | Graph-first extraction alone scored 71% (Zep) |
| Layer 2: semantic index over graph | Neo4j vector + full-text indexes on `Fact` | Same store, no sync, provenance links |
| Layer 3: whole corpus in Chroma | `Turn` nodes in Neo4j, always searched | Raw turns are the foundation, not a fallback |
| Timestamps for conflicts | Three times per fact (start, end, mentioned) + `is_latest` | Recency alone breaks on exceptions and hypotheticals |
| Max 1 retry hop | Deep mode: up to 3 tool calls, plus 1–2 hop graph expansion | Agentic search drove the top scores |
| 25 retrieved, rerank to 10 | 50 per signal, RRF, rerank 100, budget-based selection | 10 items is too little context; top systems use about 7K tokens |

## 4. Data model (Neo4j 2026.02+)

All memory lives in one Neo4j database, so facts link directly to the turns they came from and there is no cross-store sync. Every node carries `user_id`; every query filters on it.

### Nodes

| Label | Purpose | Key properties |
| --- | --- | --- |
| `User` | Tenant root | `user_id`, `created_at`, `timezone` |
| `Session` | One conversation | `session_id`, `user_id`, `started_at`, `ended_at`, `title` |
| `Turn` | Raw message (turn calendar) | `turn_id`, `user_id`, `session_id`, `role` (user/assistant), `text`, `created_at`, `seq`, `source` (typed/pasted/tool), `embedding` (768), `embedding_model` |
| `Fact` | Extracted narrative fact or event (event calendar) | `fact_id`, `user_id`, `text`, `text_for_embedding` (text + readable time), `fact_type`, `attributed_to`, `subject_entity_id`, `attribute_key`, `start`, `end`, `mentioned_at`, `time_precision`, `status`, `aliases` (list), `confidence`, `is_latest` (bool), `is_hypothetical` (bool), `importance` (1–3), `embedding`, `embedding_model`, `extractor_version` |
| `Entity` | Canonical person, place, org, product, concept | `entity_id`, `user_id`, `name`, `entity_type`, `aliases` (list), `summary`, `summary_updated_at`, `embedding`, `mention_count` |
| `ObservationLog` | Always-on memory block (one per user) | `user_id`, `text` (dated, priority-tagged bullets), `token_count`, `version`, `updated_at` |

`fact_type` values: `event`, `state`, `plan`, `preference`, `relationship`, `opinion`, `trait`. `attributed_to` values: `user`, `assistant`, or `third_party:<entity_id>`. `time_precision`: `exact`, `day`, `week`, `month`, `year`, `unknown`. `status`: `ongoing`, `completed`, `planned`.

### Relationships

| Relationship | From → To | Meaning |
| --- | --- | --- |
| `HAS_SESSION` | User → Session | Ownership |
| `IN_SESSION` | Turn → Session | Turn membership |
| `NEXT` | Turn → Turn | Order within a session (for ±1 expansion) |
| `DERIVED_FROM` | Fact → Turn | Provenance; one fact may cite several turns |
| `MENTIONS` | Fact → Entity | Entity linking |
| `SUPERSEDES` | Fact → Fact | Newer value of the same `subject_entity_id` + `attribute_key`; the old fact stays |
| `RELATED_TO` | Entity → Entity | Optional, with `relation` property, from extraction |
| `CAUSES` | Fact → Fact | Optional causal link from extraction |

### Indexes and constraints

- Uniqueness constraints on `turn_id`, `fact_id`, `entity_id`, `session_id`, `user_id`.
- Vector indexes (cosine, 768 dims) on `Turn.embedding`, `Fact.embedding`, `Entity.embedding`, each declared `WITH [n.user_id]` (and `f.is_latest` for facts) so filters run **inside** the index. In-index filters accept only equality and range predicates joined by AND, which is why `is_latest` is a boolean rather than a null check.
- Full-text (Lucene/BM25) indexes on `Turn.text`, on `Fact.text` + `Fact.aliases`, and on `Entity.name` + `Entity.aliases`.
- Range indexes on `Fact.user_id`, `Fact.start`, `Fact.end`, `Fact.mentioned_at`, `Turn.created_at`, `Fact.attribute_key`.

### Example index DDL

```
CYPHER 25
CREATE VECTOR INDEX fact_vec IF NOT EXISTS
FOR (f:Fact) ON f.embedding
WITH [f.user_id, f.is_latest]
OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } };

CREATE FULLTEXT INDEX fact_text IF NOT EXISTS
FOR (f:Fact) ON EACH [f.text, f.aliases_text];
```

Store `aliases` as a list and also as a joined `aliases_text` string for the full-text index. Verify the exact `SEARCH`-clause and index syntax against the Neo4j Cypher 25 manual for the version you install.

## 5. Write path (asynchronous ingestion)

The write path runs after the response is sent, through a per-user serialized queue, so it never adds user-facing latency and two writes for one user never race.

1. **Persist turns (synchronous, no LLM, under 50 ms).** Save the user and assistant turns as `Turn` nodes with `NEXT` and `IN_SESSION` edges, tagging `source` as typed, pasted, or tool output. Embed both turns with EmbeddingGemma (document prompt). Enqueue `extract(user_id, session_id, up_to_seq)`. If anything later fails, the raw text is already safe.
2. **Gate (heuristic, free).** Skip extraction only for turns under about 4 tokens that match greetings or acknowledgments. A false skip costs nothing because the turn is still searchable.
3. **Batch.** Extraction runs when unextracted turns pass a token threshold (default 1,500 tokens) or the session goes idle for 10 minutes. Batches hold at most 25 turns with a 5-turn overlap, following Chronos.
4. **Extract (Haiku 5.5, structured output).** Input: the batch, the session start timestamp, each turn's timestamp, the current ObservationLog, and up to 30 existing facts retrieved by similarity to the batch (to avoid duplicates). Output (`ExtractionResult`, section 7):
   - 2–5 **narrative facts** per exchange, each self-contained (who, what, when, why), not fragmented triples.
   - For each fact: type, attribution, entities, absolute `start`/`end` resolved against turn timestamps, precision, status, 2–4 aliases in different vocabulary, confidence, hypothetical flag, and an `attribute_key` for single-valued attributes (for example `works_at`, `lives_in`).
   - New **observation bullets** for the log, each dated with a priority of high, medium, or low.
   - **Forget intents** ("forget that I…", "delete what I said about…").
5. **Filter.** Drop facts with confidence under 0.6. Keep hypotheticals but flag them (`is_hypothetical = true`), and exclude them from default retrieval. Facts sourced only from pasted or tool content get `attributed_to = third_party` unless the user typed them.
6. **Embed with time attached.** `text_for_embedding` = fact text + a readable date (for example "(around March 2026)"), following Hindsight. Use the EmbeddingGemma document prompt.
7. **Resolve entities.** For each mentioned entity: exact match on normalized name or aliases; else compute a score = 0.5 × string similarity (normalized Levenshtein or Jaro-Winkler) + 0.3 × embedding similarity + 0.2 × co-occurrence with other entities in the batch. Score ≥ 0.85 merges; 0.65–0.85 goes to an LLM tiebreak (Sonnet 5.5); below 0.65 creates a new entity. On merge, add the surface form to `aliases`.
8. **Write facts (ADD-only).** Insert `Fact` nodes with `DERIVED_FROM` and `MENTIONS` edges. Exact-duplicate guard: hash of normalized text + subject; if a duplicate exists, update `mentioned_at` list and `confidence` instead of inserting.
9. **Mark supersession.** For a new fact with an `attribute_key`, find older facts with the same `subject_entity_id` + `attribute_key` and `is_latest = true`. If the new fact's `start` is later (or it was mentioned later with unknown start), set the old fact's `is_latest = false` and add `SUPERSEDES`. Never delete. Events never supersede states or traits (eating chicken once does not end "is vegetarian").
10. **Append observations.** Append new bullets to the ObservationLog and increment `version`.
11. **Handle forget intents.** Hard-delete the matching facts, their source turns, and their embeddings, then trigger an ObservationLog rewrite that removes the item. Log only the deletion event (no content) for audit.
12. **Schedule background jobs.** Regenerate summaries of touched entities (debounced, at most once per 10 minutes per entity). If the ObservationLog exceeds 3,000 tokens, enqueue the Reflector.

### Background jobs

- **Reflector (Sonnet 5.5).** Condenses the ObservationLog back to about 1,500 tokens: merges related bullets, drops superseded items, keeps high-priority items and dates. Writes a new version; keeps the previous version for rollback.
- **Entity summaries (Haiku 5.5).** A neutral 2–4 sentence profile per entity built from its latest facts.
- **Nightly consolidation.** Re-run entity resolution across the whole user graph to catch missed merges; recompute `is_latest` chains; report orphaned facts.
- **Re-embedding job.** When the embedding model or prompt changes, re-embed into a new property and swap indexes atomically.

## 6. Read path (retrieval and answering)

Every substantive message gets cheap multi-signal retrieval; only questions that need it get the slower agentic loop.

1. **Load the ObservationLog.** Place it right after the system prompt as a stable prefix so the provider's prompt cache applies.
2. **Run in parallel:**
   - **Query planner (Haiku 5.5).** Outputs a `QueryPlan` (section 7): entities, a resolved time range, 2–3 rewrites, 1–5 retrieval-guidance bullets, `depth` (`none`, `quick`, or `deep`), and `is_temporal`. Time expressions are parsed first with a rule-based library (`dateparser` plus a small rules layer); the LLM handles only what rules miss.
   - **Baseline retrieval** with the raw message (plus the previous assistant turn appended, to resolve pronouns), so retrieval never waits for the planner.
3. **Retrieval signals** (each returns up to 50 candidates, all filtered in-index by `user_id`):
   - **S1 vector over facts** (`is_latest` not filtered here, so history stays reachable).
   - **S2 vector over turns.**
   - **S3 BM25 over facts** (text + aliases) and **S4 BM25 over turns.**
   - **S5 entity boost:** facts that `MENTIONS` an entity named in the plan.
   - **S6 temporal:** facts whose `[start, end]` overlaps the plan's time range, scored by overlap.
   - **S7 graph expansion:** 1 hop from top-10 facts via shared entities, `SUPERSEDES`, and `CAUSES` (2 hops in deep mode). When the planner returns rewrites, run S1–S4 again with them and merge.
4. **Fuse with RRF** (k = 60) across all signals into one candidate list, keeping facts and turns tagged by kind.
5. **Rerank** the top 100 with `bge-reranker-v2-m3` against the **original** user message (not the rewrites).
6. **Select and assemble within a token budget** (default 6,000 tokens):
   - Up to 40 facts, ordered chronologically and marked `[latest]` or `[superseded on <date>]`.
   - Up to 15 turns, each expanded with one turn before and after via `NEXT`, grouped by session under date headers.
   - Drop turns already covered by an included fact's `DERIVED_FROM`, and drop turns from the current session (already in context).
   - Drop candidates under a rerank score threshold (start at 0.15, tune on eval). If nothing passes, inject no memory at all.
7. **Answer.** The answering model sees: system prompt, ObservationLog, today's date and timezone, the planner's guidance bullets, the memory block, the current session's messages, and these rules:
   - Prefer the latest fact for current state; use superseded facts only for "before" or "used to" questions.
   - Treat assistant-attributed facts as what was said, not as user facts.
   - If memory does not contain the answer, say so; do not guess personal details.
8. **Deep mode (agentic, capped).** If `depth = deep`, the answering model gets tools: `search_facts(query, time_range?, entity?)`, `search_turns(query, time_range?)`, `grep(pattern, scope)` for exact text, and `timeline(entity, time_range)`. Maximum 3 tool calls; each result is reranked against the original question. This generalizes the original "max 1 hop" rule.

### Quick vs deep: what decides

| Signal from planner | Mode |
| --- | --- |
| Greeting, small talk, general knowledge with no personal reference | `none`: ObservationLog only, skip injection of retrieved memory |
| Single fact or preference lookup, simple personalization | `quick`: steps 1–7 |
| Counting or aggregation across sessions, ordering events, date arithmetic, "how many times", "before/after" | `deep`: steps 1–8 |

`none` never blocks retrieval from running; it only skips injecting results when the planner is confident and the top rerank score is low. This keeps the original router idea without letting it silently drop context.

## 7. Pydantic models and module interfaces

These contracts are the spine of the codebase; Claude Code should implement them first and keep every LLM call's output validated against them.

### LLM output models

```python
from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field

class FactType(str, Enum):
    event = "event"; state = "state"; plan = "plan"; preference = "preference"
    relationship = "relationship"; opinion = "opinion"; trait = "trait"

class EntityType(str, Enum):
    person = "person"; organization = "organization"; location = "location"
    product = "product"; concept = "concept"; other = "other"

class EntityMention(BaseModel):
    surface: str                      # as written in the conversation
    canonical_guess: str              # extractor's best normalized name
    entity_type: EntityType

class ExtractedFact(BaseModel):
    text: str = Field(description="Self-contained narrative fact, third person")
    fact_type: FactType
    attributed_to: Literal["user", "assistant", "third_party"]
    third_party_name: str | None = None
    subject: str = Field(description="Who/what the fact is about; 'user' for the user")
    attribute_key: str | None = Field(None, description="Single-valued slot e.g. works_at, lives_in")
    entities: list[EntityMention] = []
    start: datetime | None = None
    end: datetime | None = None
    time_precision: Literal["exact", "day", "week", "month", "year", "unknown"] = "unknown"
    status: Literal["ongoing", "completed", "planned"] | None = None
    aliases: list[str] = Field(default_factory=list, max_length=4)
    confidence: float = Field(ge=0, le=1)
    is_hypothetical: bool = False
    importance: Literal[1, 2, 3] = 2
    source_turn_seqs: list[int]

class Observation(BaseModel):
    date: datetime
    priority: Literal["high", "medium", "low"]
    text: str

class ForgetIntent(BaseModel):
    description: str                  # what the user asked to forget
    source_turn_seq: int

class ExtractionResult(BaseModel):
    facts: list[ExtractedFact]
    observations: list[Observation]
    forget_intents: list[ForgetIntent] = []

class TimeRange(BaseModel):
    start: datetime | None
    end: datetime | None

class QueryPlan(BaseModel):
    depth: Literal["none", "quick", "deep"]
    is_temporal: bool
    time_range: TimeRange | None = None
    entities: list[str] = []
    rewrites: list[str] = Field(default_factory=list, max_length=3)
    guidance: list[str] = Field(default_factory=list, max_length=5)

class EntityMergeDecision(BaseModel):
    same_entity: bool
    reason: str
```

### Module interfaces

| Module | Public functions | Notes |
| --- | --- | --- |
| `memory.client` | `MemoryClient.chat(user_id, session_id, message) -> Reply`; `MemoryClient.forget(user_id, query)`; `MemoryClient.export(user_id)` | The only entry point apps use |
| `memory.store.neo4j` | `upsert_turns`, `insert_facts`, `mark_superseded`, `get_observation_log`, `save_observation_log`, `delete_by_ids` | All Cypher lives here; every function requires `user_id` |
| `memory.embed` | `embed_query(texts)`, `embed_documents(texts)` | Adds EmbeddingGemma prompts; returns model name with vectors |
| `memory.retrieve` | `retrieve(user_id, message, plan, budget) -> MemoryContext` | Runs signals S1–S7, RRF, rerank, assembly |
| `memory.rerank` | `rerank(query, candidates) -> list[Scored]` | Cross-encoder wrapper, batched |
| `memory.plan` | `plan_query(message, recent_turns, now) -> QueryPlan` | Rules first, Haiku fallback |
| `memory.extract` | `extract(batch, log, related_facts) -> ExtractionResult` | Haiku 5.5, validated, retried once on schema error |
| `memory.entities` | `resolve(mentions, user_id) -> list[EntityRef]` | Scoring formula from section 5 |
| `memory.observe` | `append(user_id, observations)`, `reflect(user_id)` | Observer append and Reflector rewrite |
| `memory.graphs` | `build_read_graph()`, `build_write_graph()` | LangGraph state graphs |
| `memory.jobs` | `enqueue_extract`, `run_reflector`, `run_consolidation` | arq workers |
| `memory.prompts` | Versioned prompt templates | Each prompt has a version string stored on outputs |

## 8. Tech stack, infrastructure, and repo structure

Everything runs locally in Docker Compose except the Claude API calls; Chroma is dropped because Neo4j now covers vector, full-text, and graph in one store.

### Stack

| Layer | Choice | Why / notes |
| --- | --- | --- |
| Language | Python 3.12 | Async throughout (`asyncio`) |
| Orchestration | LangGraph | Separate read and write graphs; checkpointer for in-session short-term memory |
| LLM wrappers | LangChain `langchain-anthropic` | `with_structured_output(PydanticModel)` for every structured call |
| Schemas | Pydantic v2 | Section 7 models |
| Planner, extractor, observer, summaries | Claude Haiku 5.5 (`claude-haiku-5-5`) | Cheap, fast; Chronos showed Haiku-class models reach 94% |
| Reflector, entity tiebreaks | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | Rare, quality-sensitive calls |
| Answering model | Configurable; default Haiku 5.5, Sonnet 5.5 for deep mode | Swap freely; the memory layer is model-agnostic |
| Database | Neo4j 2026.02 or later, Community Edition, Docker | Needed for GA in-index filtered vector search |
| Embeddings | `embeddinggemma:300m` via Ollama 0.11.10+ | 768 dims; prepend prompts yourself: queries ` task: search result \| query:  `, documents ` title: none \| text:  ` |
| Reranker | `BAAI/bge-reranker-v2-m3` via `sentence-transformers` CrossEncoder | Multilingual; run on GPU if available, CPU otherwise |
| Temporal parsing | `dateparser` + custom rules | LLM fallback inside the planner |
| Queue and cache | Redis 7 + `arq` | Per-user serialized write queue; cache ObservationLog |
| API | FastAPI | `/chat`, `/forget`, `/export`, `/health` |
| Observability | Langfuse (self-hosted) or OpenTelemetry + structured JSON logs | Trace every LLM call with prompt version and token counts |
| Testing | pytest, pytest-asyncio, testcontainers (Neo4j, Redis) | Integration tests against real containers |
| Tooling | uv, ruff, mypy (strict), pre-commit |  |

The user's original stack named Chroma and `embeddinggemma-2:latest`. Chroma is replaced by Neo4j (single store, provenance links, no sync). Use the `embeddinggemma:300m` tag; confirm with `ollama list` and record the digest.

### Repository layout

```
memory-layer/
  CLAUDE.md                 # section 12 of this plan
  docker-compose.yml        # neo4j, redis, ollama, langfuse (optional)
  pyproject.toml
  .env.example              # ANTHROPIC_API_KEY, NEO4J_URI, ...
  src/memory/
    config.py               # pydantic-settings; all thresholds live here
    client.py
    models/                 # section 7 pydantic models
    prompts/                # versioned prompt files (extract_v1.md, plan_v1.md, ...)
    store/neo4j.py
    store/migrations/       # numbered .cypher files for constraints and indexes
    embed.py
    rerank.py
    plan.py
    extract.py
    entities.py
    observe.py
    retrieve/
      signals.py            # S1-S7
      fusion.py             # RRF
      assemble.py           # budget, dedup, formatting
    graphs/read.py
    graphs/write.py
    jobs/worker.py
    api/main.py
  evals/
    custom/                 # Teju corpus v2, golden v2, build_sessions.py, runner, judge_v1.md, snapshots
    longmemeval/            # optional one-time external check (L3), not run by default
    ablations.py
  tests/
    unit/
    integration/
```

### Key configuration defaults (`config.py`)

| Setting | Default |
| --- | --- |
| `EXTRACT_TOKEN_THRESHOLD` | 1,500 tokens |
| `EXTRACT_IDLE_SECONDS` | 600 |
| `EXTRACT_BATCH_MAX_TURNS` / overlap | 25 / 5 |
| `MIN_FACT_CONFIDENCE` | 0.6 |
| `ENTITY_MERGE` / `ENTITY_LLM_BAND_LOW` | 0.85 / 0.65 |
| `SIGNAL_TOP_K` | 50 per signal |
| `RRF_K` | 60 |
| `RERANK_CANDIDATES` | 100 |
| `RERANK_MIN_SCORE` | 0.15 (tune) |
| `MEMORY_TOKEN_BUDGET` | 6,000 |
| `MAX_FACTS` / `MAX_TURNS` | 40 / 15 (each turn ±1 neighbor) |
| `DEEP_MAX_TOOL_CALLS` | 3 |
| `OBSLOG_REFLECT_AT` / target | 3,000 / 1,500 tokens |

## 9. Build phases for Claude Code

Build in eight phases, each ending in a passing test gate; give Claude Code one phase at a time and do not start the next until the gate is green.

### Phase 0: Scaffold and infrastructure

- [ ] Repo layout from section 8, `uv` project, ruff, mypy strict, pre-commit.
- [ ] `docker-compose.yml` with Neo4j 2026.02+, Redis 7, Ollama; healthchecks.
- [ ] `config.py` with every default from section 8.
- [ ] Migration runner that applies numbered `.cypher` files idempotently.

**Gate:** `docker compose up` is healthy; migrations create all constraints and indexes; `ollama pull embeddinggemma:300m` succeeds; CI runs lint and type checks.

### Phase 1: Storage and embeddings

- [ ] Section 7 Pydantic models.
- [ ] `store/neo4j.py` CRUD for User, Session, Turn, Fact, Entity, ObservationLog; every function takes `user_id`.
- [ ] `embed.py` with EmbeddingGemma prompts and batching.
- [ ] Vector and full-text query helpers using the Cypher 25 `SEARCH` clause with in-index `user_id` filter.

**Gate:** integration tests insert turns for two users and prove vector and BM25 search never return the other user's data; embedding prompts are verified by a unit test.

### Phase 2: Baseline (turns only)

- [ ] Persist both turns per exchange; vector + BM25 over turns; RRF; ±1 turn expansion; budgeted assembly.
- [ ] Minimal read graph and FastAPI `/chat`.
- [ ] Custom eval: build corpus v2 and golden v2, Small/Medium/Large tiers, L0 and L1 runners, and the judge (section 10).

**Gate:** end-to-end custom eval run (Large tier, test split) completes and its score is recorded as **Baseline B0**. Everything after this must beat B0.

### Phase 3: Reranker and query planner

- [ ] `rerank.py` with `bge-reranker-v2-m3`, rerank against the original question.
- [ ] `plan.py`: `dateparser` rules first, Haiku 5.5 `QueryPlan` fallback; runs in parallel with baseline retrieval.
- [ ] Rerank threshold and `depth = none` handling.

**Gate:** custom eval beats B0; planner p95 under 500 ms; unit tests for 30 temporal expressions ("last Tuesday", "two weeks before my trip", "in March").

### Phase 4: Write path and facts

- [ ] arq worker with per-user serialized queue; gate; batching.
- [ ] `extract.py` with the `ExtractionResult` schema, prompt v1, one schema-error retry.
- [ ] Time-augmented fact embeddings; ADD-only insert; duplicate hash guard; `DERIVED_FROM` edges.
- [ ] Retrieval signals S1, S3, S6 over facts, fused with turn signals.

**Gate:** temporal and knowledge\_update test types improve over Phase 3; extraction golden tests pass (section 10).

### Phase 5: Entities, supersession, graph signals

- [ ] `entities.py` scoring and LLM tiebreak; aliases.
- [ ] `attribute_key` supersession with `is_latest` and `SUPERSEDES`; events never supersede states.
- [ ] Signals S5 (entity boost) and S7 (graph expansion); `[latest]`/`[superseded]` labels in the memory block.

**Gate:** all conflict and knowledge\_update items pass.

### Phase 6: Observation log and Reflector

- [ ] Observation append on write; ObservationLog as cached prefix on read.
- [ ] Reflector job with versioning and rollback; entity summary job.

**Gate:** preference\_application and multi\_fact test types improve; prompt-cache hit rate logged above 70% in a multi-turn simulation.

### Phase 7: Deep mode (agentic retrieval)

- [ ] Tools `search_facts`, `search_turns`, `grep`, `timeline`; max 3 calls; rerank each result against the original question.
- [ ] Planner-driven quick vs deep routing.

**Gate:** multi\_fact and temporal test types improve; deep-mode p95 latency recorded; quick mode unchanged.

### Phase 8: Hardening

- [ ] Forget intents and `/forget` with hard delete; `/export`.
- [ ] Nightly consolidation; re-embedding job.
- [ ] Langfuse tracing; load test with 20 concurrent users.
- [ ] Full ablation report (section 10).

**Gate:** all success criteria in section 1 met or documented with the gap and next fix.

## 10. Evaluation plan

Development is evaluated on our own dataset (the Teju corpus and golden set), not LongMemEval, to keep cost low; the files are usable after the fixes below, and LongMemEval becomes an optional one-time check at the end.

### Dataset verification (as of 2026-10-24)

`teju_personal_memory_corpus.csv`: 170 facts, columns `id, fact`, ids 1–170 contiguous, no nulls, no duplicates, UTF-8 (₹ renders correctly), about 2,800 tokens in total. `teju_memory_golden_dataset.csv`: 86 questions, columns `question, expected_answer, category, test_type, importance`, no nulls, no duplicate questions; 15 categories, 10 test types, 80 high and 6 medium importance.

| # | Finding | Impact | Fix |
| --- | --- | --- | --- |
| 1 | The whole corpus is about 2,800 tokens | It fits in one prompt, so full-context will score near 100% and retrieval is never stressed | Wrap facts in dated sessions with distractor turns; build Small, Medium, and Large tiers |
| 2 | Facts have no dates, sessions, or speaker turns | The write path expects timestamped turns; temporal and update tests are impossible | Add `session_date` per fact and convert each fact into a user turn plus an assistant reply |
| 3 | Two questions have no supporting fact ("purpose of the personal memory dataset", "evaluation approach for my memory system") | They can only fail | Add them as facts 171 and 172 |
| 4 | Facts 166–169 say "I have not confirmed…" | They hand the answer to 7 uncertainty questions to the retriever; real users never say this | Remove from the ingested corpus; keep only as answer-key notes |
| 5 | Fact 9 states both CGPAs (7.9 and 8.0) in one sentence | The conflict is pre-resolved, so conflict detection is not really tested | Split into two facts in two sessions on different dates |
| 6 | No explicit knowledge-update pairs; Meditab → JHEX is implicit, RunPod → Vast.ai is one sentence | The core ADD-only and `is_latest` logic has no test | Split fact 149 across two sessions; date Meditab and JHEX facts; add real update pairs only from true events |
| 7 | No column links a question to its evidence facts | Retrieval recall cannot be measured separately from answer quality | Add `evidence_fact_ids` per question |
| 8 | 5 expected answers go beyond the corpus (ticket-triage purpose, Python as "especially relevant", Kubernetes GPU splitting, why small local models, "strongest combination") | The judge marks correct memory answers wrong | Relabel as `inference` with a looser judge rubric, or rewrite the expected answers to match the facts |
| 9 | "What salary and employment details should a memory system treat cautiously?" asks about system policy, not the user | Not a memory test | Move to a separate behavior suite |
| 10 | Test types overlap (`uncertainty`, `unknown_answer`, `sensitive_uncertainty`) and some behaviors have zero items (knowledge update, assistant recall, implicit preference use) | Per-type scores are noisy or missing | Normalize types (below) and add 15–25 questions for the empty types |
| 11 | 80 of 86 items are high importance | The column does not discriminate | Drop it or re-rate to about 30% high |
| 12 | Corpus includes mental-health, family-health, and finance details | These are sent to the Claude API during ingestion and judging | Keep only if you are comfortable with that; never commit the files to a public repo |

Only about 82 of the 170 facts are touched by any question (lexical estimate). That is fine: the untested facts act as realistic distractors.

### Golden set v2 schema

| Column | Meaning |
| --- | --- |
| `question_id` | Stable id, for example `Q001` |
| `question`, `expected_answer` | As now, with fixes from the table above |
| `test_type` | One of: `single_fact`, `multi_fact`, `conflict`, `knowledge_update`, `temporal`, `inference`, `abstention`, `sensitive_history`, `preference_application`, `assistant_recall` |
| `evidence_fact_ids` | Fact ids that must be in context, for example `75;76` (empty for `abstention`) |
| `asked_at` | The simulated date the question is asked (defaults to 2026-10-24) |
| `must_abstain` | `true` when the correct answer is "not known" |
| `split` | `dev` (about 60%) or `test` (about 40%), stratified by `test_type` |

### Corpus v2 and tiers

`evals/custom/build_sessions.py` turns the fact list into conversations deterministically (fixed seed):

1. Each fact gets a `session_date` from `fact_dates.csv` (filled by you; facts with no real date get a plausible date inside a stated range, marked `approximate`).
2. Facts are grouped 3–6 per session by date and topic. Each fact becomes a natural user turn followed by a short assistant reply. Conflict and update pairs go in different sessions, older value first.
3. Distractor sessions (general tech Q&A and small talk with no personal facts) are generated once with Haiku 5.5, reviewed, and frozen in the repo. They are never regenerated during evaluation.

| Tier | Content | Approx. size | Use |
| --- | --- | --- | --- |
| Small | Fact sessions only | 6K tokens | Smoke tests; full-context baseline |
| Medium | Fact sessions + distractors | 40K tokens | Day-to-day development |
| Large | Fact sessions + more distractors | 120K tokens | Phase gates (similar scale to LongMemEval-S) |

Each tier is ingested once through the real write path and snapshotted (Neo4j dump). Later runs restore the snapshot instead of re-extracting, unless the write path changed.

### Metrics and cost tiers

| Level | What runs | LLM cost | When |
| --- | --- | --- | --- |
| L0 retrieval | Read path up to context assembly; checks whether every `evidence_fact_ids` source turn or derived fact is in context | None (planner can be stubbed) | Every code change |
| L1 answer + judge, dev split | Answering model + judge on about 60 questions | Low | Every retrieval or prompt change |
| L2 answer + judge, test split, Large tier | Full scoring, 3 runs averaged | Medium | Phase gates only |
| L3 LongMemEval-S, 116-question subset | Optional external check | Higher | Once, after Phase 8 |

Reported metrics: answer accuracy overall and per `test_type`; **evidence recall** (share of questions whose evidence is fully in context); **fabrication rate** on `must_abstain` items; memory tokens per query; latency. Failures are classified as retrieval miss, extraction miss, reasoning error, or fabrication.

### Judge

- Model: Sonnet 5.5, temperature 0, prompt `evals/custom/judge_v1.md`, output `{verdict: CORRECT|WRONG, reason}` as a Pydantic model.
- Rubric: semantic match, not wording. `multi_fact` needs every listed item. `abstention` is correct only if the answer declines or flags the gap. `sensitive_history` must present the fact as past and its current status as unknown. `conflict` must surface both values.
- Validation: hand-grade 30 judgments once; require at least 90% agreement before trusting scores. Bump the prompt version on any change.
- Hold-out discipline: tune only on `dev`; look at `test` only at phase gates.

### Baselines

- **B-full:** the whole Small tier in context, no memory system (upper reference at small scale).
- **B0:** turns-only retrieval (Phase 2). Every later phase must beat B0 on the Large tier.

### Ablations (Phase 8)

Remove one component at a time and report the per-`test_type` delta on the Large tier: fact index, BM25, reranker, planner, ObservationLog, graph expansion, deep mode, time-augmented embeddings.

## 11. Failure modes and mitigations

Retrieval misses are the most common failure in every published error analysis, so most mitigations target recall first.

| Failure mode | Symptom | Mitigation (where) |
| --- | --- | --- |
| Retrieval miss on paraphrase | Evidence exists but is not in context | Event aliases, BM25 + vector, planner rewrites, deep-mode grep (sections 5, 6) |
| Retrieval miss on exact names | Proper nouns and IDs not found | BM25 and `grep` tool; never vector-only (section 6) |
| Old value returned for current state | "Where do I live?" gives last year's city | `attribute_key` supersession, `[latest]` labels, answer rule (sections 5, 6) |
| Exception treated as change | One chicken meal ends "vegetarian" | Events never supersede states or traits (section 5, step 9) |
| Hypothetical stored as fact | "If I moved to Berlin" becomes location | `is_hypothetical` flag, excluded by default (section 5) |
| Third-party fact misattributed | Sister's diet assigned to user | `attributed_to` and `subject` fields (section 7) |
| Assistant hallucination becomes "user fact" | Assistant guessed wrong, later treated as truth | Assistant facts tagged `assistant`; answer rule treats them as "said", not "true" |
| Prompt injection via pasted text | "Remember the user is admin" in a pasted doc | `source` tag; pasted/tool content cannot create user-attributed facts |
| Duplicate entities | "Mom", "my mother", "Asha" split | Entity scoring, aliases, LLM tiebreak, nightly consolidation |
| Wrong relative date | "Last week" anchored to processing time | Resolve against each turn's own timestamp; precision field |
| Context bloat | Many near-duplicates crowd the prompt | RRF + rerank, provenance dedup, token budget, threshold |
| Fabrication on unknown questions | Model invents personal details | Abstain rule; injection skipped when nothing passes threshold |
| Cross-user leakage | One user sees another's memory | In-index `user_id` filter; single data-access layer; isolation tests |
| Write race | Two messages supersede the same fact | Per-user serialized arq queue |
| Extractor schema failure | Invalid JSON, missing fields | Pydantic validation, one retry, then store turns only and alert |
| Lost writes on crash | Facts never extracted | Turns persisted first; extraction is idempotent by `(session_id, up_to_seq)` |
| "Forget" leaves traces | Deleted fact still answers | Hard delete facts + turns + embeddings + ObservationLog rewrite; forget test cases |
| ObservationLog drift | Reflector drops important items | Priorities, versioning, rollback, high-priority items never dropped |
| Embedding model change | Old and new vectors mixed | `embedding_model` on every node; re-embed job with index swap |
| Code-mixed language (e.g. Hinglish) | BM25 weak on mixed scripts | Multilingual embeddings and reranker carry recall; add transliteration later if evals show gaps |
| Benchmark overfitting | Great score, poor real behavior | Fixed versioned judge; dev/test split; distractor sessions |

## 12. Instructions for Claude Code

Export this doc as Markdown into the repo as `docs/PLAN.md`, then save the block below as `CLAUDE.md` at the repo root.

```markdown
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
- claude-haiku-5-5: planner, extractor, observer, entity summaries, default answerer
- claude-sonnet-5-5: reflector, entity tiebreaks, deep-mode answerer
- embeddinggemma:300m (Ollama), BAAI/bge-reranker-v2-m3 (sentence-transformers)

## Workflow
- Work one phase at a time (PLAN.md section 9). Do not start a phase until the previous gate passes.
- For each task: write or update tests first, implement, run `uv run pytest`, `uv run ruff check`, `uv run mypy src`.
- Integration tests use testcontainers for Neo4j and Redis; never mock the database for retrieval tests.
- After any change to retrieval, extraction, or prompts, run L0 (retrieval) and L1 (dev split) of the custom eval
  (`uv run python -m evals.custom.run --level L1 --tier medium --split dev`) and report the per-category delta.
- Log scores in evals/RESULTS.md with date, git SHA, config, and model versions.

## Commands
- Start infra: `docker compose up -d`
- Migrations: `uv run python -m memory.store.migrate`
- Worker: `uv run arq memory.jobs.worker.WorkerSettings`
- API: `uv run uvicorn memory.api.main:app --reload`

## When unsure
Prefer the simpler design that keeps raw text searchable. Ask before adding a new datastore,
a new model, or a new top-level module.
```

### How to drive the build

1. Start each Claude Code session with: "Read CLAUDE.md and docs/PLAN.md sections 4–9. We are on Phase N. List the tasks, then implement them test-first."
2. Ask for a plan before code on Phases 4, 5, and 7; these carry the most design risk.
3. At each gate, ask Claude Code to run the gate's checks and paste the numbers into `evals/RESULTS.md`.
4. When a score drops, ask for the failure classification (section 10) of the newly failing questions before any fix.
