"""Ingest a tier through the real write path, once per (tier content, write path) version.

The eval user id embeds a fingerprint of the tier's sessions, the embedding model, the
extractor model and prompt, and the write-path version, so a tier is ingested once and
reused by later runs (the snapshot role in PLAN.md section 10). Changing any of these gives
a new user id and a fresh ingestion.

Two stages, both resumable:
1. Persist every exchange (``persist_exchange``). An interrupted stage is detected by turn
   count and redone from scratch.
2. Extract facts (``run_extraction``), session by session in chronological order, which is
   what the idle trigger does after each conversation. Watermarks make this resume where it
   stopped.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from evals.custom.build_sessions import EvalSession, TierData
from memory.config import Settings
from memory.embed import EmbedderLike
from memory.extract import ExtractionDeps, ExtractionStats, run_extraction
from memory.graphs.write import persist_exchange, turn_id_for
from memory.observe import ReflectorLLM, SummaryLLM, reflect, summarize_entities
from memory.store.neo4j import Neo4jStore

log = logging.getLogger(__name__)

# Bump whenever the write path changes what gets stored.
WRITE_PATH_VERSION = "turns_facts_obs_v1"
INGEST_CONCURRENCY = 4
PROGRESS_EVERY = 20


@dataclass(frozen=True)
class IngestedTier:
    user_id: str
    evidence_turn_ids: dict[str, str]  # fact_id -> turn_id
    turns: int
    extraction: ExtractionStats = field(default_factory=ExtractionStats)
    fact_summary: dict[str, Any] = field(default_factory=dict)


def write_path_id(settings: Settings) -> str:
    ablation = "" if settings.embed_fact_dates else "|no_fact_dates"
    return (
        f"{WRITE_PATH_VERSION}|{settings.embedding_model}|{settings.extractor_model}"
        f"|extract_{settings.extractor_prompt_version}{ablation}"
    )


def eval_user_id(data: TierData, settings: Settings) -> str:
    key = f"{data.fingerprint}|{write_path_id(settings)}"
    return f"eval-{data.tier}-{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def session_id_for(user_id: str, session: EvalSession) -> str:
    return f"{user_id}/{session.key}"


async def _persist_session(
    store: Neo4jStore, embedder: EmbedderLike, user_id: str, session: EvalSession
) -> None:
    session_id = session_id_for(user_id, session)
    for i in range(0, len(session.turns), 2):
        user, assistant = session.turns[i], session.turns[i + 1]
        await persist_exchange(
            store,
            embedder,
            user_id=user_id,
            session_id=session_id,
            user_text=user.text,
            assistant_text=assistant.text,
            user_at=user.created_at,
            assistant_at=assistant.created_at,
        )


async def _persist(store: Neo4jStore, embedder: EmbedderLike, user_id: str, data: TierData) -> int:
    expected = sum(len(s.turns) for s in data.sessions)
    existing = await store.count_turns(user_id)
    if existing == expected:
        log.info("reusing persisted turns for %s (%d)", user_id, expected)
        return expected
    if existing:
        log.warning("%s has %d/%d turns; re-ingesting", user_id, existing, expected)
        await store.delete_user_data(user_id)
    log.info("persisting %s: %d sessions, %d turns", user_id, len(data.sessions), expected)
    sem = asyncio.Semaphore(INGEST_CONCURRENCY)

    async def one(session: EvalSession) -> None:
        async with sem:
            await _persist_session(store, embedder, user_id, session)

    await asyncio.gather(*(one(s) for s in data.sessions))
    stored = await store.count_turns(user_id)
    if stored != expected:
        raise RuntimeError(f"ingested {stored} turns for {user_id}, expected {expected}")
    return expected


async def _extract(
    deps: ExtractionDeps, user_id: str, reflector: ReflectorLLM | None
) -> tuple[ExtractionStats, int]:
    """Extract every pending session in order; reflect whenever the log outgrows its limit
    (what the worker's follow-up jobs do). Returns stats and the number of reflections."""
    pending = await deps.store.pending_extraction(user_id)
    total = ExtractionStats()
    reflections = 0
    if not pending:
        log.info("facts already extracted for %s", user_id)
        return total, reflections
    log.info("extracting facts for %s: %d sessions pending", user_id, len(pending))
    for n, session in enumerate(pending, start=1):  # chronological by last activity
        total.add(await run_extraction(deps, user_id, session_ids=[session.session_id]))
        if reflector is not None and await reflect(deps.store, reflector, deps.settings, user_id):
            reflections += 1
        if n % PROGRESS_EVERY == 0 or n == len(pending):
            log.info(
                "  %d/%d sessions: %d LLM calls, %d facts created, %d merged",
                n, len(pending), total.llm_calls, total.facts_created, total.facts_merged,
            )  # fmt: skip
    if total.failed_sessions:
        raise RuntimeError(f"extraction failed for {len(total.failed_sessions)} sessions; rerun")
    return total, reflections


async def ingest_tier(
    store: Neo4jStore, embedder: EmbedderLike, data: TierData, settings: Settings,
    extraction: ExtractionDeps,
    *,
    reflector: ReflectorLLM | None = None,
    summarizer: SummaryLLM | None = None,
) -> IngestedTier:  # fmt: skip
    user_id = eval_user_id(data, settings)
    turns = await _persist(store, embedder, user_id, data)
    stats, reflections = await _extract(extraction, user_id, reflector)
    # Apply the current profile limits to a reused ingestion too (no-op under the limit).
    if reflector is not None and await reflect(store, reflector, settings, user_id):
        reflections += 1
    summaries = 0
    if summarizer is not None and stats.touched_entities:
        summaries = await summarize_entities(
            store, summarizer, settings, user_id, sorted(stats.touched_entities)
        )
    log.info("profile: %d reflections; %d entity summaries", reflections, summaries)
    by_key = {s.key: s for s in data.sessions}
    evidence = {
        fact_id: turn_id_for(session_id_for(user_id, by_key[key]), seq)
        for fact_id, (key, seq) in data.evidence_locations().items()
    }
    return IngestedTier(
        user_id=user_id,
        evidence_turn_ids=evidence,
        turns=turns,
        extraction=stats,
        fact_summary=await store.fact_summary(user_id),
    )
