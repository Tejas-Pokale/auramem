"""The public auramem API against real Neo4j (models and the cross-encoder are faked)."""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from neo4j import AsyncDriver, AsyncGraphDatabase

from auramem import AsyncAuraMem, AuraMem
from memory.client import MemoryClient
from memory.config import Settings, get_settings
from memory.rerank import Reranker
from memory.store.migrate import apply_migrations, discover
from memory.store.neo4j import Neo4jStore
from tests.fakes import FakeAnswerer, FakeEmbedder
from tests.integration.conftest import NEO4J_TEST_PASSWORD
from tests.integration.test_phase3_retrieval import KeywordCrossEncoder
from tests.integration.test_phase5 import deps, uid
from tests.integration.test_phase8 import ProfileWithout, Summarizer

pytestmark = pytest.mark.integration


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


def mem_for(store: Neo4jStore, settings: Settings | None = None, **kw: Any) -> AsyncAuraMem:
    s = settings or get_settings()
    client = MemoryClient(
        store, FakeEmbedder(), FakeAnswerer("noted"), s,
        reranker=Reranker(s, KeywordCrossEncoder()),
        forget_llms=(ProfileWithout("Acme"), Summarizer()),
    )  # fmt: skip
    return AsyncAuraMem.from_client(client, **kw)


async def test_chat_remembers_and_keeps_one_session(store: Neo4jStore) -> None:
    mem, user = mem_for(store), uid("u")
    first = await mem.chat(user, "Hi there, I like tea.")
    second = await mem.chat(user, "What do I like?")
    assert first.text == second.text == "noted"
    turns = (await mem.export(user))["turns"]
    assert len(turns) == 4 and len({t["session_id"] for t in turns}) == 1


async def test_recall_for_your_own_llm(store: Neo4jStore) -> None:
    mem, user = mem_for(store), uid("u")
    await mem.remember(user, "I work at Acme.", "Nice!")
    recall = await mem.recall(user, "Where do I work at Acme?", session_id="another")
    assert "I work at Acme." in recall.context
    prompt = recall.as_prompt()
    assert "<memory>" in prompt and "Today is" in prompt


async def test_flush_extracts_facts_inline(store: Neo4jStore) -> None:
    mem, user = mem_for(store, extraction="inline", extraction_deps=deps(store)), uid("u")
    await mem.remember(user, "I work at Acme.", "Nice!")
    assert (await mem.export(user))["facts"] == []  # below the threshold: nothing yet
    await mem.flush(user)
    facts = (await mem.export(user))["facts"]
    assert [f["text"] for f in facts] == ["The user said: I work at Acme."]


async def test_threshold_triggers_background_extraction(store: Neo4jStore) -> None:
    s = get_settings().model_copy(update={"extract_token_threshold": 1})
    mem = mem_for(store, s, extraction="inline", extraction_deps=deps(store))
    user = uid("u")
    await mem.remember(user, "I work at Globex.", "Nice!")
    await mem.close()  # waits for the background extraction
    assert len((await store.export_user(user))["facts"]) == 1


async def test_forget_export_and_delete(store: Neo4jStore) -> None:
    mem, user = mem_for(store), uid("u")
    await mem.remember(user, "I work at Acme.", "Nice!")
    await mem.remember(user, "I like tea.", "Good.")
    preview = await mem.forget(user, "Acme")
    assert preview.preview_id is not None and "Acme" in preview.render()
    assert "Acme" in json.dumps(await mem.export(user))  # nothing deleted before confirming
    report = await mem.confirm_forget(user, preview.preview_id)
    assert report is not None and report.turns >= 1
    exported = json.dumps(await mem.export(user))
    assert "Acme" not in exported and "tea" in exported
    counts = await mem.delete_user(user)
    assert counts["Turn"] >= 2 and (await mem.export(user))["turns"] == []


def test_blocking_api_from_plain_code(neo4j_container: Any) -> None:
    async def build() -> AsyncAuraMem:
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", NEO4J_TEST_PASSWORD)
        )
        return mem_for(Neo4jStore(driver, database="neo4j", settings=get_settings()))

    mem = AuraMem.from_factory(build)
    user = uid("u")
    reply = mem.chat(user, "I like hiking.")
    assert reply.text == "noted"
    assert "hiking" in mem.recall(user, "What do I like, hiking?", session_id="other").context
    assert mem.profile(user) == ""
    mem.close()


async def test_new_user_gets_a_fixed_welcome_without_llm(store: Neo4jStore) -> None:
    mem, user = mem_for(store), uid("u")
    text = await mem.greet(user, session_id="first")
    assert "remember our conversations" in text
    client = mem._client
    assert client is not None and isinstance(client.answerer, FakeAnswerer)
    assert client.answerer.prompts == []  # no LLM call for an empty memory
    [turn] = await store.get_session_turns(user, "first")
    assert (turn.role, turn.text, turn.seq) == ("assistant", text, 0)


async def test_returning_user_continues_from_the_last_conversation(store: Neo4jStore) -> None:
    mem, user = mem_for(store), uid("u")
    await mem.remember(user, "I have a job interview on Friday.", "Good luck!", session_id="old")
    session = await store.get_session(user, "old")
    assert session is not None
    await store.commit_extraction(
        user, "old", entities=[], facts=[], observations=["- [2026-10-20] (high) Name: Sam."],
        expected_watermark=session.extracted_through_seq, up_to_seq=1,
    )  # fmt: skip
    client = mem._client
    assert client is not None and isinstance(client.answerer, FakeAnswerer)

    greeting = await mem.greet(user, session_id="new")
    system, prompt = client.answerer.prompts[-1][0], client.answerer.prompts[-1][-1]
    assert "Name: Sam." in str(system.content)  # the profile is in the cached prefix
    assert "job interview on Friday" in str(prompt.content)  # the end of the last conversation
    assert greeting == "noted"

    await mem.chat(user, "It went well!", session_id="new")
    history = client.answerer.prompts[-1]
    assert any(m.type == "ai" and m.content == "noted" for m in history[1:-1])  # greeting seen
