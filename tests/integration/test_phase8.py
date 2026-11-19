"""Phase 8 against real Neo4j: forget (preview, confirm, cancel, chat flow) and export.

Only the LLMs and the cross-encoder are faked.
"""

import json
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from langchain_core.messages import BaseMessage
from neo4j import AsyncDriver

from memory.client import MemoryClient
from memory.config import get_settings
from memory.consolidate import consolidate_user
from memory.extract import run_extraction
from memory.forget import Forgetter
from memory.graphs.write import turn_id_for
from memory.jobs.reembed import reembed
from memory.models import (
    EntitySummaryOutput,
    EntityType,
    FactType,
    NewEntity,
    NewFact,
    ReflectionOutput,
)
from memory.rerank import Reranker
from memory.store.migrate import apply_migrations, discover
from memory.store.neo4j import Neo4jStore
from tests.fakes import FakeAnswerer, FakeEmbedder
from tests.integration.test_phase3_retrieval import KeywordCrossEncoder
from tests.integration.test_phase5 import T0, FixedTiebreak, deps, say, uid

pytestmark = pytest.mark.integration

NOW = T0 + timedelta(days=400)
PROFILE = [
    "- [2026-01-10] (high) Works at Acme as an engineer.",
    "- [2026-02-10] (medium) Training for a half marathon.",
]


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


class ProfileWithout:
    """Fake forget rewrite: drops profile lines that mention the word."""

    def __init__(self, word: str) -> None:
        self.word = word
        self.calls = 0

    async def __call__(self, messages: list[BaseMessage]) -> ReflectionOutput:
        self.calls += 1
        log = str(messages[-1].content).split("Current profile:\n", 1)[1]
        kept = [line for line in log.splitlines() if self.word.lower() not in line.lower()]
        return ReflectionOutput(text="\n".join(kept))


class Summarizer:
    async def __call__(self, messages: list[BaseMessage]) -> EntitySummaryOutput:
        return EntitySummaryOutput(summary="A company.")


async def seeded(store: Neo4jStore) -> str:
    user = uid("u")
    for n, name in enumerate(["Acme", "Globex"]):
        await say(store, user, f"{user}/s{n}", f"I work at {name}.", day=n * 30)
        await run_extraction(deps(store), user, session_ids=[f"{user}/s{n}"])
    session = await store.get_session(user, f"{user}/s1")
    assert session is not None
    await store.commit_extraction(
        user, f"{user}/s1", entities=[], facts=[], observations=PROFILE,
        expected_watermark=session.extracted_through_seq, up_to_seq=1,
    )  # fmt: skip
    log = await store.get_observation_log(user)
    assert log is not None  # one rewrite, so a saved version holds the old text
    await store.replace_observation_log(
        user, log.text + "\n- [2026-03-01] (low) Likes tea.", 30,
        expected_version=log.version, reason="test",
    )  # fmt: skip
    return user


def forgetter(store: Neo4jStore, profile: ProfileWithout | None = None) -> Forgetter:
    s = get_settings()
    return Forgetter(
        store, FakeEmbedder(), Reranker(s, KeywordCrossEncoder()), s,
        profile_llm=profile or ProfileWithout("Acme"), summarizer=Summarizer(),
    )  # fmt: skip


async def texts(store: Neo4jStore, user: str) -> str:
    return json.dumps(await store.export_user(user))


async def test_preview_deletes_nothing_and_confirm_deletes_everything(store: Neo4jStore) -> None:
    user, other = await seeded(store), await seeded(store)
    f = forgetter(store)
    before = await texts(store, user)

    preview = await f.preview(user, "Acme", now=NOW)
    assert preview.preview_id is not None
    shown = " ".join(i.text for i in preview.items)
    assert "Acme" in shown and "Globex" not in shown
    assert {i.kind for i in preview.items} == {"fact", "turn"}
    assert await texts(store, user) == before  # a preview never deletes

    report = await f.confirm(user, preview.preview_id, now=NOW)
    assert report is not None and report.facts >= 1 and report.turns >= 1
    assert report.profile_rewritten
    after = await texts(store, user)
    assert "Acme" not in after  # turns, facts, entity, profile, and its old versions
    assert "Globex" in after and "half marathon" in after
    exported = await store.export_user(user)
    assert exported["observation_log_versions"] == []
    [event] = exported["deletion_events"]
    assert event["facts"] == report.facts and "Acme" not in json.dumps(event)
    assert "Acme" in await texts(store, other)  # other users untouched
    assert await f.confirm(user, preview.preview_id, now=NOW) is None  # used up


async def test_cancel_and_expiry_keep_everything(store: Neo4jStore) -> None:
    user = await seeded(store)
    f = forgetter(store)
    before = await texts(store, user)
    preview = await f.preview(user, "Acme", now=NOW)
    assert preview.preview_id is not None
    assert await f.cancel(user, preview.preview_id)
    assert await f.confirm(user, preview.preview_id, now=NOW) is None

    preview = await f.preview(user, "Acme", now=NOW)
    assert preview.preview_id is not None
    late = NOW + timedelta(minutes=get_settings().forget_preview_ttl_minutes + 1)
    assert await f.confirm(user, preview.preview_id, now=late) is None
    assert await texts(store, user) == before


async def test_nothing_found_needs_no_confirmation(store: Neo4jStore) -> None:
    user = await seeded(store)
    preview = await forgetter(store).preview(user, "Zanzibar", now=NOW)
    assert preview.preview_id is None and preview.items == []
    assert "couldn't find" in preview.render()


async def test_chat_flow_shows_preview_then_deletes_on_yes(store: Neo4jStore) -> None:
    user = await seeded(store)
    s = get_settings()
    client = MemoryClient(
        store, FakeEmbedder(), FakeAnswerer("normal answer"), s,
        reranker=Reranker(s, KeywordCrossEncoder()),
        forget_llms=(ProfileWithout("Acme"), Summarizer()),
    )  # fmt: skip
    session = f"{user}/chat"
    request = "Please forget that I work at Acme."
    preview = await client.respond(user, session, request, now=NOW)
    assert preview.mode == "forget" and "Should I permanently delete" in preview.text
    assert "Acme" in await texts(store, user)  # nothing deleted yet
    await client.remember(user, session, request, preview.text, asked_at=NOW, replied_at=NOW)

    done = await client.respond(user, session, "yes", now=NOW + timedelta(minutes=1))
    assert done.mode == "forget" and done.text.startswith("Done.")
    assert "Acme" not in await texts(store, user)  # incl. the request and preview turns

    normal = await client.respond(user, session, "How is my training going?", now=NOW)
    assert normal.text == "normal answer" and normal.mode != "forget"


async def test_chat_no_keeps_everything(store: Neo4jStore) -> None:
    user = await seeded(store)
    s = get_settings()
    client = MemoryClient(
        store, FakeEmbedder(), FakeAnswerer(), s, reranker=Reranker(s, KeywordCrossEncoder()),
        forget_llms=(ProfileWithout("Acme"), Summarizer()),
    )  # fmt: skip
    await client.respond(user, f"{user}/chat", "Forget about Acme", now=NOW)
    reply = await client.respond(user, f"{user}/chat", "no, keep it", now=NOW)
    assert reply.text == "Okay, I kept everything."
    assert "Acme" in await texts(store, user)


async def test_export_has_everything_but_embeddings(store: Neo4jStore) -> None:
    user = await seeded(store)
    exported = await store.export_user(user)
    assert exported["user"]["user_id"] == user
    assert len(exported["turns"]) == 4 and len(exported["facts"]) == 2
    assert exported["entities"] and exported["observation_log"]
    assert '"embedding"' not in json.dumps(exported)
    assert exported["turns"][0]["created_at"].startswith("20")  # JSON-ready dates


async def test_consolidation_merges_duplicates_and_repairs_latest(store: Neo4jStore) -> None:
    user = uid("u")
    await say(store, user, f"{user}/s0", "hello", day=0)
    a, b, other = uid("e"), uid("e"), uid("e")
    await store.insert_entities(
        user,
        [
            NewEntity(entity_id=a, name="Acme Corp", entity_type=EntityType.organization),
            NewEntity(entity_id=b, name="Acme Corp.", entity_type=EntityType.organization),
            NewEntity(entity_id=other, name="Initech", entity_type=EntityType.organization),
        ],
    )

    def fact(n: int, text: str, entity: str, **extra: object) -> NewFact:
        return NewFact(
            fact_id=uid("f"), text=text, text_for_embedding=text, fact_type=FactType.state,
            attributed_to="user", mentioned_at=T0 + timedelta(days=n), confidence=0.9,
            extractor_version="test", derived_from_turn_ids=[turn_id_for(f"{user}/s0", 0)],
            mentions_entity_ids=[entity], **extra,
        )  # fmt: skip

    await store.insert_facts(
        user,
        [
            fact(0, "The user works at Acme Corp.", a),
            fact(1, "Acme Corp. hired the user.", b),
            fact(2, "The user's laptop is a ThinkPad.", other, attribute_key="laptop",
                 is_latest=False),  # nothing supersedes it: the flag is wrong
        ],
    )  # fmt: skip

    report = await consolidate_user(
        store, FakeEmbedder(), get_settings(), user, tiebreaker=FixedTiebreak(same=True)
    )

    assert report.merged == 1 and report.latest_fixed == 1
    names = [e.name for e in await store.list_entities(user)]
    assert sorted(names) == ["Acme Corp", "Initech"] or sorted(names) == ["Acme Corp.", "Initech"]
    [kept] = [e for e in await store.list_entities(user) if e.name.startswith("Acme")]
    assert len(await store.facts_mentioning(user, [kept.entity_id], 10)) == 2
    assert {"acme corp", "acme corp."} <= {n.lower() for n in [kept.name, *kept.aliases]}
    laptop = await store.fulltext_search_facts(user, "ThinkPad", 5, latest_only=True)
    assert len(laptop) == 1  # flags_text agrees with the repaired is_latest
    again = await consolidate_user(store, FakeEmbedder(), get_settings(), user)
    assert again.merged == 0 and again.latest_fixed == 0  # idempotent


class NextModelEmbedder(FakeEmbedder):
    """A "new" embedding model: same vectors, so other tests sharing the container see no change."""

    model = "fake-bow-v2"


async def test_reembed_stages_then_swaps_every_label(store: Neo4jStore) -> None:
    user = await seeded(store)
    settings = get_settings()
    staged = await reembed(
        store, NextModelEmbedder(), settings, model="fake-bow-v2", batch=7, swap=False
    )
    assert staged.staged["Turn"] >= 4 and staged.staged["Fact"] >= 2
    exported = await store.export_user(user)
    assert {t["embedding_model"] for t in exported["turns"]} == {"fake-bow"}  # not swapped yet

    done = await reembed(store, NextModelEmbedder(), settings, model="fake-bow-v2", batch=7)
    assert done.staged == {"Turn": 0, "Fact": 0, "Entity": 0}  # resumable: nothing left
    assert done.swapped["Turn"] >= 4
    exported = await store.export_user(user)
    for key in ("turns", "facts", "entities"):
        assert {n["embedding_model"] for n in exported[key]} == {"fake-bow-v2"}, key
    hits = await store.fulltext_search_facts(user, "Globex", 5)
    vector = (await NextModelEmbedder().embed_query(["I work at Globex"])).vectors[0]
    assert hits and (await store.vector_search_facts(user, vector, 3))  # indexes still serve


async def test_reembed_refuses_other_dimensions(store: Neo4jStore) -> None:
    await seeded(store)
    with pytest.raises(ValueError, match="dimensions"):
        await reembed(store, FakeEmbedder(dims=16), get_settings(), model="tiny", batch=50)


async def test_forget_keeps_unrelated_facts_from_the_same_message(store: Neo4jStore) -> None:
    user = uid("u")
    await say(store, user, f"{user}/s0", "I moved to Leeds and I bake bread at Crumbs.", day=0)
    source = [turn_id_for(f"{user}/s0", 0)]

    def fact(text: str) -> NewFact:
        return NewFact(
            fact_id=uid("f"), text=text, text_for_embedding=text, fact_type=FactType.state,
            attributed_to="user", mentioned_at=T0, confidence=0.9, extractor_version="test",
            derived_from_turn_ids=source,
        )  # fmt: skip

    await store.insert_facts(
        user, [fact("The user lives in Leeds."), fact("The user works at Crumbs.")]
    )
    f = forgetter(store, ProfileWithout("Crumbs"))
    preview = await f.preview(user, "Crumbs", now=NOW)
    assert preview.preview_id is not None
    assert [i.text for i in preview.items if i.kind == "fact"] == ["The user works at Crumbs."]
    await f.confirm(user, preview.preview_id, now=NOW)
    left = [x.text for x in await store.list_facts(user)]
    assert left == ["The user lives in Leeds."]  # kept, though its source message is gone
    assert "Crumbs" not in await texts(store, user)
