"""Phase 7 against real Neo4j: grep/timeline queries, deep tools, uncapped deep selection, and
the read graph's quick/deep branch (only the LLMs are faked)."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from neo4j import AsyncDriver

from memory.client import MemoryClient
from memory.config import get_settings
from memory.deep import DeepTools
from memory.extract import run_extraction
from memory.models import AnswerOutput, DeepStep, DeepTool, DeepToolCall
from memory.retrieve import Retriever
from memory.store.migrate import apply_migrations, discover
from memory.store.neo4j import Neo4jStore
from tests.fakes import FakeAnswerer, FakeEmbedder
from tests.integration.test_phase5 import deps, say, uid

pytestmark = pytest.mark.integration

JOBS = ["Acme", "Globex", "Initech"]


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


async def seeded(store: Neo4jStore) -> str:
    user = uid("u")
    for n, name in enumerate(JOBS):
        await say(store, user, f"{user}/{name}", f"I work at {name}.", day=n * 30)
        await run_extraction(deps(store), user, session_ids=[f"{user}/{name}"])
    return user


async def test_grep_and_timeline_are_per_user(store: Neo4jStore) -> None:
    alice, bob = await seeded(store), await seeded(store)
    turns = await store.grep_turns(alice, "WORK AT glob")
    assert [h.item.text for h in turns] == ["I work at Globex."]
    assert all(h.item.user_id == alice for h in turns)
    facts = await store.grep_facts(alice, "initech")
    assert facts and all(h.item.user_id == alice for h in facts)
    assert await store.grep_turns(alice, "   ") == []

    [entity] = await store.find_entities_by_names(alice, ["Globex"])
    line = await store.timeline(alice, [entity.entity_id])
    assert [h.item.text for h in line] == ["The user said: I work at Globex."]
    assert await store.timeline(bob, [entity.entity_id]) == []  # another user's entity


async def test_deep_tools_run_each_search(store: Neo4jStore) -> None:
    user = await seeded(store)
    tools = DeepTools(store, FakeEmbedder(), None, get_settings())
    calls: list[tuple[DeepTool, str]] = [
        ("grep", "Initech"), ("timeline", "Acme"), ("search_facts", "work"),
        ("search_turns", "work"),
    ]  # fmt: skip
    for tool, query in calls:
        found = await tools.run(user, DeepToolCall(tool=tool, query=query), "Where do I work?")
        assert found, tool
        assert all(x.item.user_id == user for x in found)


async def test_deep_selection_has_no_fact_cap(store: Neo4jStore) -> None:
    user = await seeded(store)
    s = get_settings().model_copy(update={"max_facts": 1, "max_fact_partners": 0})
    retriever = Retriever(store, FakeEmbedder(), s)
    quick = await retriever.retrieve(user, "Where do I work?", mode="quick")
    deep = await retriever.retrieve(user, "Where do I work?", mode="deep")
    assert len(quick.fact_ids) == 1 and not quick.deep
    assert len(deep.fact_ids) == len(JOBS) and deep.deep


class OneSearchThenAnswer:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, messages: list[BaseMessage]) -> tuple[DeepStep, dict[str, int]]:
        self.calls += 1
        if self.calls == 1:
            call = DeepToolCall(tool="grep", query="Initech")
            return DeepStep(reasoning="check", tool_calls=[call]), {}
        return DeepStep(reasoning="found", answer="You work at Initech."), {}


class NeverCalled:
    async def __call__(self, messages: list[BaseMessage]) -> tuple[AnswerOutput, dict[str, int]]:
        raise AssertionError("no forced answer expected")


@pytest.mark.parametrize("mode", ["deep", "quick"])
async def test_read_graph_branches_on_mode(store: Neo4jStore, mode: Any) -> None:
    user = await seeded(store)
    answerer = FakeAnswerer("quick answer")
    client = MemoryClient(
        store, FakeEmbedder(), answerer, get_settings(),
        deep_llms=(OneSearchThenAnswer(), NeverCalled()),
    )  # fmt: skip
    reply = await client.respond(user, f"{user}/new", "Where do I work?", mode=mode)
    if mode == "deep":
        assert reply.text == "You work at Initech." and reply.mode == "deep"
        assert reply.deep is not None and reply.deep.stop_reason == "answered"
        assert reply.deep.tool_calls == ['grep("Initech")']
        assert answerer.prompts == []
    else:
        assert reply.text == "quick answer" and reply.mode == "quick" and reply.deep is None
