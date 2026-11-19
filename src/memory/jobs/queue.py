"""Enqueueing extraction (PLAN.md section 5, steps 1-3) and the job bodies.

Two triggers, both serialized per user:
- threshold: once a user's unextracted turns reach EXTRACT_TOKEN_THRESHOLD, enqueue
  ``extract_job`` with job id ``extract:<user>``; arq refuses a second job with the same id
  while one is queued or running, so a user never has two extractions in flight.
- idle: otherwise enqueue ``idle_extract_job`` deferred by EXTRACT_IDLE_SECONDS, keyed by the
  session's last seq. When it fires it only acts if no newer turn arrived since (the newer
  turn scheduled its own idle job).
A Redis lock per user backs this up: a job that cannot take the lock retries later.
"""

import logging
from typing import Any, Literal

from arq import Retry
from arq.connections import ArqRedis

from memory.config import Settings
from memory.consolidate import consolidate_user
from memory.extract import ExtractionDeps, ExtractionStats, run_extraction
from memory.observe import reflect, summarize_entities
from memory.store.neo4j import Neo4jStore

log = logging.getLogger(__name__)

EXTRACT_JOB = "extract_job"
IDLE_EXTRACT_JOB = "idle_extract_job"
REFLECT_JOB = "reflect_job"
ENTITY_SUMMARY_JOB = "entity_summary_job"
CONSOLIDATION_JOB = "consolidation_job"
LOCK_RETRY_SECONDS = 5


def extract_job_id(user_id: str) -> str:
    return f"extract:{user_id}"


def followup_job_id(user_id: str) -> str:
    return f"extract:{user_id}:followup"


def idle_job_id(session_id: str, last_seq: int) -> str:
    return f"idle:{session_id}:{last_seq}"


def reflect_job_id(user_id: str) -> str:
    return f"reflect:{user_id}"


def summary_job_id(entity_id: str) -> str:
    return f"summary:{entity_id}"


def lock_key(user_id: str) -> str:
    return f"lock:extract:{user_id}"


async def unextracted_tokens(store: Neo4jStore, user_id: str, settings: Settings) -> int:
    pending = await store.pending_extraction(user_id)
    return round(sum(p.new_chars for p in pending) / settings.chars_per_token)


async def enqueue_after_persist(
    redis: ArqRedis, store: Neo4jStore, settings: Settings, user_id: str, session_id: str
) -> Literal["now", "idle", "none"]:
    """Schedule extraction after an exchange was persisted. Returns which trigger fired."""
    pending = await store.pending_extraction(user_id)
    tokens = sum(p.new_chars for p in pending) / settings.chars_per_token
    if tokens >= settings.extract_token_threshold:
        await redis.enqueue_job(EXTRACT_JOB, user_id, _job_id=extract_job_id(user_id))
        return "now"
    session = next((p for p in pending if p.session_id == session_id), None)
    if session is None:
        return "none"
    await redis.enqueue_job(
        IDLE_EXTRACT_JOB,
        user_id,
        session_id,
        session.last_seq,
        _job_id=idle_job_id(session_id, session.last_seq),
        _defer_by=settings.extract_idle_seconds,
    )
    return "idle"


async def _locked_run(
    ctx: dict[str, Any], user_id: str, session_ids: list[str] | None
) -> ExtractionStats:
    deps: ExtractionDeps = ctx["deps"]
    redis: ArqRedis = ctx["redis"]
    lock = redis.lock(lock_key(user_id), timeout=deps.settings.extract_lock_ttl_seconds)
    if not await lock.acquire(blocking=False):
        raise Retry(defer=LOCK_RETRY_SECONDS)
    try:
        stats = await run_extraction(deps, user_id, session_ids=session_ids)
    finally:
        await lock.release()
    await _schedule_followups(redis, deps, user_id, stats)
    return stats


async def _schedule_followups(
    redis: ArqRedis, deps: ExtractionDeps, user_id: str, stats: ExtractionStats
) -> None:
    """Section 5 step 12: Reflector when the log is too long; debounced entity summaries."""
    current = await deps.store.get_observation_log(user_id)
    if current is not None and current.token_count > deps.settings.obslog_reflect_at:
        await redis.enqueue_job(REFLECT_JOB, user_id, _job_id=reflect_job_id(user_id))
    for entity_id in sorted(stats.touched_entities):
        # One deferred job per entity id: arq ignores duplicates while it is pending, which
        # gives at most one summary per entity per ENTITY_SUMMARY_DEBOUNCE_SECONDS.
        await redis.enqueue_job(
            ENTITY_SUMMARY_JOB,
            user_id,
            entity_id,
            _job_id=summary_job_id(entity_id),
            _defer_by=deps.settings.entity_summary_debounce_seconds,
        )


async def consolidation_job(ctx: dict[str, Any]) -> dict[str, Any]:
    """Nightly: consolidate every user, each under the same lock as extraction (merging
    entities while an extraction writes facts about them would race). A user whose
    extraction is running is skipped until the next night."""
    deps: ExtractionDeps = ctx["deps"]
    redis: ArqRedis = ctx["redis"]
    done = skipped = merged = 0
    for user_id in await deps.store.list_user_ids():
        lock = redis.lock(lock_key(user_id), timeout=deps.settings.extract_lock_ttl_seconds)
        if not await lock.acquire(blocking=False):
            skipped += 1
            continue
        try:
            report = await consolidate_user(
                deps.store, deps.embedder, deps.settings, user_id, tiebreaker=deps.tiebreaker
            )
        except Exception:
            log.exception("consolidation failed for %s", user_id)
            continue
        finally:
            await lock.release()
        done += 1
        merged += report.merged
    return {"users": done, "skipped": skipped, "merged": merged}


async def reflect_job(ctx: dict[str, Any], user_id: str) -> dict[str, Any]:
    deps: ExtractionDeps = ctx["deps"]
    result = await reflect(deps.store, ctx["reflector"], deps.settings, user_id)
    return {"reflected": result is not None}


async def entity_summary_job(ctx: dict[str, Any], user_id: str, entity_id: str) -> dict[str, Any]:
    deps: ExtractionDeps = ctx["deps"]
    written = await summarize_entities(
        deps.store, ctx["summarizer"], deps.settings, user_id, [entity_id]
    )
    return {"summaries": written}


async def extract_job(ctx: dict[str, Any], user_id: str) -> dict[str, Any]:
    deps: ExtractionDeps = ctx["deps"]
    stats = await _locked_run(ctx, user_id, None)
    # Turns that arrived while this job ran: keep going if they already reach the threshold.
    if await unextracted_tokens(deps.store, user_id, deps.settings) >= (
        deps.settings.extract_token_threshold
    ):
        redis: ArqRedis = ctx["redis"]
        await redis.enqueue_job(EXTRACT_JOB, user_id, _job_id=followup_job_id(user_id))
    log.info("extracted for %s: %s", user_id, stats)
    return {"facts_created": stats.facts_created, "llm_calls": stats.llm_calls}


async def idle_extract_job(
    ctx: dict[str, Any], user_id: str, session_id: str, last_seq: int
) -> dict[str, Any]:
    deps: ExtractionDeps = ctx["deps"]
    pending = await deps.store.pending_extraction(user_id)
    session = next((p for p in pending if p.session_id == session_id), None)
    if session is None or session.last_seq != last_seq:
        return {"skipped": True}  # already extracted, or a newer turn owns the idle timer
    stats = await _locked_run(ctx, user_id, [session_id])
    return {"facts_created": stats.facts_created, "llm_calls": stats.llm_calls}
