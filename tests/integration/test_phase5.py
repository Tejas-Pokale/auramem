"""Phase 5 against real Neo4j: supersession, entity resolution, S5/S7, labels.

Only the LLMs are faked: a rule-based extractor that emits slotted facts, and a fixed
entity tiebreak.
"""

import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from neo4j import AsyncDriver

from memory.config import Settings, get_settings
from memory.entities import EntityResolver
from memory.extract import ExtractionDeps, Extractor, run_extraction
from memory.graphs.write import persist_exchange
from memory.models import (
    EntityMention,
    EntityMergeDecision,
    EntityType,
    ExtractedFact,
    ExtractionResult,
    FactType,
    NewEntity,
)
from memory.retrieve import Retriever
from memory.retrieve.signals import entity_signal
from memory.store.migrate import apply_migrations, discover
from memory.store.neo4j import Neo4jStore
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

T0 = datetime(2026, 1, 10, 18, 0, tzinfo=UTC)
_TURN = re.compile(r"^\[seq (\d+)\] .*? \| user \(typed\): (.+)$")
RULES: list[tuple[re.Pattern[str], dict[str, Any]]] = [
    (re.compile(r"I work at (\w+)"), {"attribute_key": "employer", "fact_type": FactType.state}),
    (re.compile(r"I am (vegetarian)"), {"attribute_key": "diet", "fact_type": FactType.state}),
    (re.compile(r"I ate (\w+) once"), {"attribute_key": "diet", "fact_type": FactType.event}),
]


class SlotExtractor:
    """Fake extractor: one slotted fact per typed user turn matching a rule above."""

    async def __call__(self, messages: list[BaseMessage]) -> ExtractionResult:
        new = str(messages[-1].content).split("NEW turns to extract from:", 1)[1]
        facts = []
        for line in new.strip().splitlines():
            m = _TURN.match(line)
            if not m:
                continue
            seq, text = int(m.group(1)), m.group(2)
            for pattern, extra in RULES:
                if found := pattern.search(text):
                    name = found.group(1)
                    facts.append(
                        ExtractedFact(
                            text=f"The user said: {text}",
                            attributed_to="user",
                            subject="user",
                            entities=[
                                EntityMention(
                                    surface=name,
                                    canonical_guess=name,
                                    entity_type=EntityType.organization,
                                )
                            ],
                            aliases=[name.lower(), text.lower()],
                            confidence=0.9,
                            source_turn_seqs=[seq],
                            **extra,
                        )
                    )
        return ExtractionResult(facts=facts, observations=[])


class FixedTiebreak:
    def __init__(self, same: bool) -> None:
        self.same = same
        self.calls = 0

    async def __call__(self, messages: list[BaseMessage]) -> EntityMergeDecision:
        self.calls += 1
        return EntityMergeDecision(same_entity=self.same, reason="fixed")


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


def deps(store: Neo4jStore, **overrides: Any) -> ExtractionDeps:
    s: Settings = get_settings().model_copy(update=overrides)
    return ExtractionDeps(
        store=store, embedder=FakeEmbedder(),
        extractor=Extractor(SlotExtractor(), s), settings=s,  # type: ignore[arg-type]
    )  # fmt: skip


async def say(store: Neo4jStore, user: str, session: str, text: str, day: int) -> None:
    at = T0 + timedelta(days=day)
    await persist_exchange(
        store, FakeEmbedder(), user_id=user, session_id=session, user_text=text,
        assistant_text="Noted.", user_at=at, assistant_at=at + timedelta(seconds=20),
    )  # fmt: skip


async def slot_facts(store: Neo4jStore, user: str, word: str) -> dict[str, Any]:
    hits = await store.fulltext_search_facts(user, word, 20, include_hypothetical=True)
    return {h.item.text: h.item for h in hits}


async def test_job_change_supersedes_and_keeps_history(store: Neo4jStore) -> None:
    user = uid("u")
    await say(store, user, f"{user}/a", "I work at Acme.", day=0)
    await say(store, user, f"{user}/b", "I work at Globex.", day=200)

    stats = await run_extraction(deps(store), user)

    assert stats.facts_superseded == 1
    facts = await slot_facts(store, user, "work")
    old, new = facts["The user said: I work at Acme."], facts["The user said: I work at Globex."]
    assert old.attribute_key == new.attribute_key == "works_at"  # "employer" normalized
    assert (old.is_latest, new.is_latest) == (False, True)
    assert old.superseded_by_ids == [new.fact_id]
    assert new.supersedes_ids == [old.fact_id]
    assert old.superseded_on == new.mentioned_at
    latest = await store.vector_search_facts(
        user, (await FakeEmbedder().embed_query(["work"])).vectors[0], 20, latest_only=True
    )
    assert old.fact_id not in {h.item.fact_id for h in latest}


async def test_event_does_not_supersede_a_state(store: Neo4jStore) -> None:
    user = uid("u")
    await say(store, user, f"{user}/a", "I am vegetarian.", day=0)
    await say(store, user, f"{user}/b", "I ate chicken once at a wedding.", day=30)
    stats = await run_extraction(deps(store), user)
    assert stats.facts_superseded == 0
    facts = await slot_facts(store, user, "vegetarian")
    assert facts["The user said: I am vegetarian."].is_latest


async def test_backfilled_older_value_is_stored_as_history(store: Neo4jStore) -> None:
    user = uid("u")
    await say(store, user, f"{user}/now", "I work at Globex.", day=200)
    await run_extraction(deps(store), user)
    await say(store, user, f"{user}/then", "I work at Acme.", day=0)  # told late, happened earlier

    stats = await run_extraction(deps(store), user)

    assert (stats.facts_superseded, stats.facts_stored_as_history) == (0, 1)
    facts = await slot_facts(store, user, "work")
    assert facts["The user said: I work at Globex."].is_latest
    assert not facts["The user said: I work at Acme."].is_latest


async def test_near_duplicate_entity_routes_through_tiebreak(store: Neo4jStore) -> None:
    user = uid("u")
    name = "Northwind Traders"
    vec = (await FakeEmbedder().embed_documents([name])).vectors[0]
    await store.insert_entities(
        user,
        [NewEntity(entity_id=uid("e"), name=name, entity_type=EntityType.organization,
                   embedding=vec, embedding_model="fake-bow")],
    )  # fmt: skip
    mention = EntityMention(
        surface="Northwind Traders Ltd", canonical_guess="Northwind Traders Ltd",
        entity_type=EntityType.organization,
    )  # fmt: skip
    s = get_settings()

    yes = FixedTiebreak(same=True)
    merged = await EntityResolver(store, FakeEmbedder(), s, yes).resolve(user, [mention])
    assert yes.calls == 1 and merged.merged == 1 and not merged.new
    assert "Northwind Traders Ltd" in next(iter(merged.aliases.values()))

    no = FixedTiebreak(same=False)
    separate = await EntityResolver(store, FakeEmbedder(), s, no).resolve(user, [mention])
    assert no.calls == 1 and len(separate.new) == 1


async def test_exact_alias_match_needs_no_tiebreak(store: Neo4jStore) -> None:
    user = uid("u")
    await store.insert_entities(
        user, [NewEntity(entity_id=uid("e"), name="Globex", entity_type=EntityType.organization,
                         aliases=["Globex Corp"])],
    )  # fmt: skip
    tb = FixedTiebreak(same=False)
    mention = EntityMention(
        surface="globex corp", canonical_guess="Globex Corp", entity_type=EntityType.organization
    )
    res = await EntityResolver(store, FakeEmbedder(), get_settings(), tb).resolve(user, [mention])
    assert tb.calls == 0 and not res.new


async def test_s5_s7_and_labels_in_retrieval(store: Neo4jStore) -> None:
    alice, bob = uid("alice"), uid("bob")
    for user in (alice, bob):
        await say(store, user, f"{user}/a", "I work at Acme.", day=0)
        await say(store, user, f"{user}/b", "I work at Globex.", day=200)
        await run_extraction(deps(store), user)

    s5 = await entity_signal(store, alice, names=["Globex"], k=10)
    [hit] = s5["S5_entity_facts"]
    assert hit.item.user_id == alice and "Globex" in hit.item.text

    globex = (await slot_facts(store, alice, "Globex"))["The user said: I work at Globex."]
    expanded = await store.expand_facts(alice, [globex.fact_id], 10)
    assert [h.item.text for h in expanded] == ["The user said: I work at Acme."]
    assert await store.expand_facts(bob, [globex.fact_id], 10) == []  # other user's seed

    memory = await Retriever(store, FakeEmbedder(), get_settings()).retrieve(
        alice, "Where do I work at Globex now?"
    )
    assert "[latest] The user said: I work at Globex." in memory.text
    assert "[superseded on" in memory.text and "I work at Acme." in memory.text


@pytest.mark.parametrize("partners", [0, 1])
async def test_partners_are_capped(store: Neo4jStore, partners: int) -> None:
    user = uid("u")
    for day, name in enumerate(["Acme", "Globex", "Initech", "Umbrella"]):
        await say(store, user, f"{user}/{name}", f"I work at {name}.", day=day * 30)
        await run_extraction(deps(store), user, session_ids=[f"{user}/{name}"])
    s = get_settings().model_copy(update={"max_facts": 1, "max_fact_partners": partners})
    memory = await Retriever(store, FakeEmbedder(), s).retrieve(user, "Where do I work?")
    assert len(memory.fact_ids) == 1 + partners
