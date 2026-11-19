"""Phase 4 write path against real Neo4j and Redis (only the extractor LLM is faked)."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from arq import Retry
from arq.connections import ArqRedis
from arq.jobs import Job, JobStatus
from neo4j import AsyncDriver

from memory.config import Settings, get_settings
from memory.extract import ExtractionDeps, Extractor, run_extraction
from memory.graphs.write import persist_exchange, turn_id_for
from memory.jobs.queue import (
    enqueue_after_persist,
    extract_job,
    extract_job_id,
    idle_extract_job,
    idle_job_id,
    lock_key,
)
from memory.models import FactType, NewFact
from memory.retrieve import Retriever
from memory.store.migrate import apply_migrations, discover
from memory.store.neo4j import Neo4jStore, VersionConflictError, self_entity_id
from tests.fakes import FakeEmbedder, FakeExtractorLLM

pytestmark = pytest.mark.integration

T0 = datetime(2026, 6, 1, 18, 0, tzinfo=UTC)


def uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def settings(**overrides: Any) -> Settings:
    return get_settings().model_copy(update=overrides)


@pytest.fixture
async def store(neo4j_driver: AsyncDriver) -> AsyncIterator[Neo4jStore]:
    s = get_settings()
    await apply_migrations(
        neo4j_driver,
        discover(s.migrations_dir),
        database="neo4j",
        index_timeout_seconds=s.migration_index_timeout_seconds,
    )
    yield Neo4jStore(neo4j_driver, database="neo4j", settings=s)


def deps(
    store: Neo4jStore, llm: FakeExtractorLLM | None = None, **overrides: Any
) -> ExtractionDeps:
    s = settings(**overrides)
    return ExtractionDeps(
        store=store,
        embedder=FakeEmbedder(),
        extractor=Extractor(llm or FakeExtractorLLM(), s),  # type: ignore[arg-type]
        settings=s,
    )


async def say(
    store: Neo4jStore, user: str, session: str, pairs: list[tuple[str, str]], day: int = 0
) -> None:
    for i, (u, a) in enumerate(pairs):
        at = T0 + timedelta(days=day, minutes=2 * i)
        await persist_exchange(
            store, FakeEmbedder(), user_id=user, session_id=session,
            user_text=u, assistant_text=a, user_at=at, assistant_at=at + timedelta(seconds=20),
        )  # fmt: skip


async def test_extraction_writes_facts_with_provenance_and_watermark(store: Neo4jStore) -> None:
    user, session = uid("u"), uid("s")
    await say(store, user, session, [("I joined JHEX in Pune.", "Congrats!")])
    llm = FakeExtractorLLM()

    stats = await run_extraction(deps(store, llm), user)

    assert (stats.llm_calls, stats.facts_created, stats.entities_created) == (1, 1, 2)
    assert (await store.get_session(user, session)).extracted_through_seq == 1  # type: ignore[union-attr]
    [hit] = await store.fulltext_search_facts(user, "JHEX", 10)
    fact = hit.item
    assert fact.source_turn_ids == [turn_id_for(session, 0)]
    assert fact.subject_entity_id == self_entity_id(user)
    assert fact.text_for_embedding.endswith("(mentioned 1 June 2026)")
    assert fact.extractor_version == f"extract_{get_settings().extractor_prompt_version}"
    assert [e.name for e in await store.find_entities_by_names(user, ["jhex"])] == ["JHEX"]


async def test_rerun_is_noop_and_new_turns_extract_with_context(store: Neo4jStore) -> None:
    user, session = uid("u"), uid("s")
    d = deps(store, extract_batch_max_turns=2, extract_batch_overlap=2)
    await say(store, user, session, [("I live in Pune.", "Nice."), ("I use Neovim.", "Cool.")])
    first = await run_extraction(d, user)
    assert (first.batches, first.facts_created) == (2, 2)
    assert (await run_extraction(d, user)).batches == 0

    await say(store, user, session, [("I switched to VS Code.", "Okay.")], day=1)
    llm: FakeExtractorLLM = d.extractor.llm  # type: ignore[assignment]
    llm.calls.clear()
    third = await run_extraction(d, user)
    assert third.facts_created == 1
    prompt = str(llm.calls[0][-1].content)
    context, new = prompt.split("NEW turns to extract from:")
    assert "I use Neovim." in context and "I use Neovim." not in new
    assert "I switched to VS Code." in new


async def test_restated_fact_merges_instead_of_duplicating(store: Neo4jStore) -> None:
    user = uid("u")
    await say(store, user, f"{user}/a", [("I live in Pune.", "Noted.")])
    await say(store, user, f"{user}/b", [("I live in Pune.", "Yes, you said.")], day=3)

    stats = await run_extraction(deps(store), user)

    assert (stats.facts_created, stats.facts_merged) == (1, 1)
    [hit] = await store.fulltext_search_facts(user, "Pune", 10)
    assert len(hit.item.mention_dates) == 2
    assert sorted(hit.item.source_turn_ids) == sorted(
        [turn_id_for(f"{user}/a", 0), turn_id_for(f"{user}/b", 0)]
    )


async def test_greeting_only_batch_skips_the_llm(store: Neo4jStore) -> None:
    user, session = uid("u"), uid("s")
    await say(store, user, session, [("hi", "Hello!")])
    llm = FakeExtractorLLM()
    stats = await run_extraction(deps(store, llm), user)
    assert (stats.gated_batches, stats.llm_calls, llm.calls) == (1, 0, [])
    assert (await store.get_session(user, session)).extracted_through_seq == 1  # type: ignore[union-attr]


async def test_failed_extraction_keeps_watermark_after_one_retry(store: Neo4jStore) -> None:
    user, session = uid("u"), uid("s")
    await say(store, user, session, [("I live in Pune.", "Noted.")])
    llm = FakeExtractorLLM(fail=True)
    stats = await run_extraction(deps(store, llm), user)
    assert stats.failed_sessions == [session]
    assert len(llm.calls) == 2  # first call + EXTRACT_SCHEMA_RETRIES
    assert (await store.get_session(user, session)).extracted_through_seq == -1  # type: ignore[union-attr]
    assert await store.fulltext_search_turns(user, "Pune", 5)  # raw turn still searchable


async def test_stale_watermark_commit_writes_nothing(store: Neo4jStore) -> None:
    user, session = uid("u"), uid("s")
    await say(store, user, session, [("I live in Pune.", "Noted.")])
    fact = NewFact(
        fact_id=uid("f"), text="The user lives in Pune.", text_for_embedding="x",
        fact_type=FactType.state, attributed_to="user", mentioned_at=T0, confidence=0.9,
        extractor_version="test", derived_from_turn_ids=[turn_id_for(session, 0)],
    )  # fmt: skip
    with pytest.raises(VersionConflictError):
        await store.commit_extraction(
            user, session, entities=[], facts=[fact], expected_watermark=5, up_to_seq=1
        )
    assert await store.get_facts(user, [fact.fact_id]) == []


async def test_temporal_fact_search_is_scored_and_isolated(store: Neo4jStore) -> None:
    alice, bob = uid("alice"), uid("bob")
    for user in (alice, bob):
        session = f"{user}/s"
        await say(store, user, session, [("event", "ok"), ("another", "ok")])
        await store.insert_facts(
            user,
            [
                NewFact(
                    fact_id=uid("f"),
                    text=f"{name} happened",
                    text_for_embedding=name,
                    fact_type=FactType.event,
                    attributed_to="user",
                    mentioned_at=T0,
                    start=start,
                    end=end,
                    time_precision="day",
                    confidence=0.9,
                    extractor_version="test",
                    derived_from_turn_ids=[turn_id_for(session, 0)],
                )
                for name, start, end in [
                    (
                        "july-week",
                        datetime(2026, 7, 10, tzinfo=UTC),
                        datetime(2026, 7, 17, tzinfo=UTC),
                    ),
                    (
                        "whole-july",
                        datetime(2026, 7, 1, tzinfo=UTC),
                        datetime(2026, 8, 1, tzinfo=UTC),
                    ),
                    ("march", datetime(2026, 3, 5, tzinfo=UTC), None),
                ]
            ],
        )
    hits = await store.temporal_search_facts(
        alice, datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC), 10,
        scan_limit=100,
    )  # fmt: skip
    assert [h.item.text for h in hits] == ["whole-july happened", "july-week happened"]
    assert {h.item.user_id for h in hits} == {alice}


async def test_retrieval_includes_facts_and_skips_their_source_turns(store: Neo4jStore) -> None:
    user = uid("u")
    await say(store, user, f"{user}/jobs", [("I joined JHEX Consulting.", "Congrats!")])
    await run_extraction(deps(store), user)
    retriever = Retriever(store, FakeEmbedder(), settings())

    memory = await retriever.retrieve(user, "Where did I join? JHEX Consulting")

    assert memory.fact_ids
    assert memory.covered_turn_ids == [turn_id_for(f"{user}/jobs", 0)]
    assert turn_id_for(f"{user}/jobs", 0) not in memory.hit_turn_ids
    assert memory.text.startswith("## Facts from earlier conversations")


# --- Queue ---------------------------------------------------------------------------------


async def test_idle_trigger_defers_and_threshold_enqueues_once(
    store: Neo4jStore, arq_redis: ArqRedis
) -> None:
    user, session = uid("u"), uid("s")
    await say(store, user, session, [("I live in Pune.", "Noted.")])

    idle = await enqueue_after_persist(arq_redis, store, settings(), user, session)
    assert idle == "idle"
    assert await Job(idle_job_id(session, 1), arq_redis).status() == JobStatus.deferred

    low = settings(extract_token_threshold=1)
    assert await enqueue_after_persist(arq_redis, store, low, user, session) == "now"
    assert await Job(extract_job_id(user), arq_redis).status() == JobStatus.queued
    assert await enqueue_after_persist(arq_redis, store, low, user, session) == "now"
    assert await arq_redis.zcard("arq:queue") == 2  # the idle job + one extract job, not two


async def test_idle_job_runs_only_for_the_latest_turn(
    store: Neo4jStore, arq_redis: ArqRedis
) -> None:
    user, session = uid("u"), uid("s")
    await say(store, user, session, [("I live in Pune.", "Noted.")])
    ctx = {"deps": deps(store), "redis": arq_redis}

    stale = await idle_extract_job(ctx, user, session, 0)  # a newer turn exists (seq 1)
    assert stale == {"skipped": True}
    done = await idle_extract_job(ctx, user, session, 1)
    assert done["facts_created"] == 1
    assert await idle_extract_job(ctx, user, session, 1) == {"skipped": True}  # caught up


async def test_extract_job_retries_while_user_is_locked(
    store: Neo4jStore, arq_redis: ArqRedis
) -> None:
    user, session = uid("u"), uid("s")
    await say(store, user, session, [("I live in Pune.", "Noted.")])
    ctx = {"deps": deps(store), "redis": arq_redis}
    await arq_redis.set(lock_key(user), "other-worker")
    with pytest.raises(Retry):
        await extract_job(ctx, user)
    await arq_redis.delete(lock_key(user))
    assert (await extract_job(ctx, user))["facts_created"] == 1
