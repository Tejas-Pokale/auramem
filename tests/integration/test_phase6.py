"""Phase 6 against real Neo4j: ObservationLog append, Reflector versions and rollback,
entity summaries, and the profile in the answering prompt (only LLMs are faked)."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import BaseMessage
from neo4j import AsyncDriver

from memory.client import MemoryClient
from memory.config import get_settings
from memory.graphs.write import persist_exchange, turn_id_for
from memory.models import (
    EntitySummaryOutput,
    EntityType,
    FactType,
    NewEntity,
    NewFact,
    ReflectionOutput,
)
from memory.observe import reflect, summarize_entities
from memory.store.migrate import apply_migrations, discover
from memory.store.neo4j import Neo4jStore
from tests.fakes import FakeAnswerer, FakeEmbedder

pytestmark = pytest.mark.integration

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
LINES = [
    "- [2026-01-05] (high) Prefers concise bullet answers with the recommendation first.",
    "- [2026-02-10] (medium) Training for a half marathon.",
    "- [2026-03-04] (low) Tried a new ramen place.",
]


def uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


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


async def user_with_session(store: Neo4jStore) -> tuple[str, str]:
    user, session = uid("u"), uid("s")
    await persist_exchange(
        store, FakeEmbedder(), user_id=user, session_id=session, user_text="hello there",
        assistant_text="hi", user_at=T0, assistant_at=T0 + timedelta(seconds=5),
    )  # fmt: skip
    return user, session


async def append(store: Neo4jStore, user: str, session: str, lines: list[str], up_to: int) -> None:
    current = await store.get_session(user, session)
    assert current is not None
    await store.commit_extraction(
        user, session, entities=[], facts=[], observations=lines,
        expected_watermark=current.extracted_through_seq, up_to_seq=up_to,
    )  # fmt: skip


class FakeReflector:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def __call__(self, messages: list[BaseMessage]) -> ReflectionOutput:
        self.calls += 1
        return ReflectionOutput(text=self.text)


async def test_extraction_commit_appends_observations(store: Neo4jStore) -> None:
    user, session = await user_with_session(store)
    await append(store, user, session, LINES[:2], up_to=0)
    await append(store, user, session, LINES[2:], up_to=1)
    log = await store.get_observation_log(user)
    assert log is not None
    assert log.text.splitlines() == LINES
    assert log.version == 2
    assert log.token_count == round(len(log.text) / get_settings().chars_per_token)


async def test_reflect_condenses_snapshots_and_rolls_back(store: Neo4jStore) -> None:
    user, session = await user_with_session(store)
    await append(store, user, session, LINES, up_to=1)
    original = await store.get_observation_log(user)
    assert original is not None
    condensed = LINES[0] + "\n" + LINES[1]
    settings = get_settings().model_copy(update={"obslog_reflect_at": 10})

    result = await reflect(store, FakeReflector(condensed), settings, user)

    assert result is not None and result.after_tokens < result.before_tokens
    log = await store.get_observation_log(user)
    assert log is not None and log.text == condensed and log.version == original.version + 1
    versions = await store.observation_log_versions(user)
    assert [v["version"] for v in versions] == [original.version]

    restored = await store.rollback_observation_log(user, original.version)
    assert restored.text == original.text
    assert len(await store.observation_log_versions(user)) == 2


async def test_reflect_rejects_rewrites_that_drop_high_priority(store: Neo4jStore) -> None:
    user, session = await user_with_session(store)
    await append(store, user, session, LINES, up_to=1)
    settings = get_settings().model_copy(update={"obslog_reflect_at": 10})
    assert await reflect(store, FakeReflector(LINES[1]), settings, user) is None
    log = await store.get_observation_log(user)
    assert log is not None and log.text.splitlines() == LINES


async def test_reflect_is_skipped_under_the_limit(store: Neo4jStore) -> None:
    user, session = await user_with_session(store)
    await append(store, user, session, LINES, up_to=1)
    fake = FakeReflector("x")
    assert await reflect(store, fake, get_settings(), user) is None
    assert fake.calls == 0


async def test_entity_summaries_only_for_entities_with_enough_facts(store: Neo4jStore) -> None:
    user, session = await user_with_session(store)
    busy, quiet = uid("e"), uid("e")
    await store.insert_entities(
        user,
        [NewEntity(entity_id=e, name=n, entity_type=EntityType.person)
         for e, n in [(busy, "Priya"), (quiet, "Theo")]],
    )  # fmt: skip
    facts = [
        NewFact(
            fact_id=uid("f"),
            text=f"Priya fact {i}",
            text_for_embedding="x",
            fact_type=FactType.event,
            attributed_to="user",
            mentioned_at=T0,
            confidence=0.9,
            extractor_version="test",
            derived_from_turn_ids=[turn_id_for(session, 0)],
            mentions_entity_ids=[busy] if i < 3 else [quiet],
        )
        for i in range(4)
    ]
    await store.insert_facts(user, facts)

    class FakeSummarizer:
        async def __call__(self, messages: list[BaseMessage]) -> EntitySummaryOutput:
            return EntitySummaryOutput(summary="Priya is the user's best friend.")

    written = await summarize_entities(store, FakeSummarizer(), get_settings(), user, [busy, quiet])
    assert written == 1
    by_id = {e.entity_id: e for e in await store.get_entities(user, [busy, quiet])}
    assert by_id[busy].summary == "Priya is the user's best friend."
    assert by_id[quiet].summary is None


async def test_profile_is_in_the_answer_prompt_and_wiped_with_the_user(store: Neo4jStore) -> None:
    user, session = await user_with_session(store)
    await append(store, user, session, LINES, up_to=1)
    answerer = FakeAnswerer("ok")
    client = MemoryClient(store, FakeEmbedder(), answerer, get_settings())

    reply = await client.respond(user, f"{user}/new", "Plan my week.")

    system = answerer.prompts[0][0]
    assert "Prefers concise bullet answers" in str(system.content)
    assert reply.profile_tokens > 0
    settings = get_settings().model_copy(update={"obslog_reflect_at": 10})
    await reflect(store, FakeReflector(LINES[0] + "\n" + LINES[1]), settings, user)
    counts = await store.delete_user_data(user)
    assert counts["ObservationLog"] == 1 and counts["ObservationLogVersion"] == 1
