"""Phase 2 read path against real Neo4j: persist -> retrieve -> answer -> /chat."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from neo4j import AsyncDriver

from memory.api.main import create_app
from memory.client import MemoryClient
from memory.config import get_settings
from memory.graphs.read import NO_MEMORY
from memory.graphs.write import persist_exchange, turn_id_for
from memory.store.migrate import apply_migrations, discover
from memory.store.neo4j import Neo4jStore
from tests.fakes import FakeAnswerer, FakeEmbedder

pytestmark = pytest.mark.integration

T0 = datetime(2026, 5, 1, 18, 0, tzinfo=UTC)

CONVERSATIONS = [
    # (session suffix, day offset, [(user, assistant), ...])
    (
        "jobs",
        0,
        [
            (
                "What's a good way to prepare for system design interviews?",
                "Practice capacity estimates.",
            ),
            ("I joined Meditab Software as a data engineering intern.", "Congratulations!"),
            ("Most of the work there is SQL.", "SQL depth will help later."),
        ],
    ),
    (
        "gpu",
        5,
        [
            ("I'm considering RunPod for rented GPU instances.", "RunPod is a reasonable start."),
            ("Which GPUs have 24 GB of memory?", "An RTX 4090 or an A5000."),
        ],
    ),
    (
        "cooking",
        9,
        [
            ("How long should dal simmer?", "About 25 minutes."),
            ("Is jeera rice hard?", "Not at all."),
        ],
    ),
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


async def seed(store: Neo4jStore, embedder: FakeEmbedder, user_id: str) -> None:
    for suffix, day, exchanges in CONVERSATIONS:
        session_id = f"{user_id}/{suffix}"
        for i, (user, assistant) in enumerate(exchanges):
            at = T0 + timedelta(days=day, minutes=2 * i)
            await persist_exchange(
                store,
                embedder,
                user_id=user_id,
                session_id=session_id,
                user_text=user,
                assistant_text=assistant,
                user_at=at,
                assistant_at=at + timedelta(seconds=20),
            )


async def test_persist_exchange_assigns_sequential_ids_and_embeds_documents(
    store: Neo4jStore,
) -> None:
    embedder = FakeEmbedder()
    user = uid("u")
    await seed(store, embedder, user)
    turns = await store.get_session_turns(user, f"{user}/jobs")
    assert [t.turn_id for t in turns] == [turn_id_for(f"{user}/jobs", i) for i in range(6)]
    assert [t.role for t in turns] == ["user", "assistant"] * 3
    assert all(t.embedding_model == "fake-bow" for t in turns)
    assert {kind for kind, _ in embedder.calls} == {"document"}


async def test_retrieval_finds_turn_with_neighbors_and_isolates_users(store: Neo4jStore) -> None:
    embedder = FakeEmbedder()
    alice, bob = uid("alice"), uid("bob")
    await seed(store, embedder, alice)
    await seed(store, embedder, bob)
    client = MemoryClient(store, embedder, FakeAnswerer(), get_settings())

    memory = await client.retriever.retrieve(alice, "Which GPU rental provider was I considering?")

    runpod = turn_id_for(f"{alice}/gpu", 0)
    assert runpod in memory.hit_turn_ids
    assert turn_id_for(f"{alice}/gpu", 1) in memory.turn_ids  # +1 neighbor
    assert "RunPod" in memory.text
    assert all(t.startswith(alice) for t in memory.turn_ids)
    assert "### Conversation on 2026-05-06" in memory.text
    assert 0 < memory.token_estimate <= get_settings().memory_token_budget


async def test_current_session_is_excluded(store: Neo4jStore) -> None:
    embedder = FakeEmbedder()
    user = uid("u")
    await seed(store, embedder, user)
    client = MemoryClient(store, embedder, FakeAnswerer(), get_settings())
    memory = await client.retriever.retrieve(
        user, "RunPod GPU instances", current_session_id=f"{user}/gpu"
    )
    assert not any(t.startswith(f"{user}/gpu:") for t in memory.turn_ids)


async def test_budget_limits_memory(store: Neo4jStore) -> None:
    embedder = FakeEmbedder()
    user = uid("u")
    await seed(store, embedder, user)
    client = MemoryClient(store, embedder, FakeAnswerer(), get_settings())
    small = await client.retriever.retrieve(user, "Meditab SQL RunPod dal", budget=60)
    large = await client.retriever.retrieve(user, "Meditab SQL RunPod dal")
    assert 0 < len(small.turn_ids) < len(large.turn_ids)
    assert small.token_estimate <= 60


async def test_respond_builds_prompt_with_memory_date_and_history(store: Neo4jStore) -> None:
    embedder, answerer = FakeEmbedder(), FakeAnswerer("You were considering RunPod.")
    user = uid("u")
    await seed(store, embedder, user)
    client = MemoryClient(store, embedder, answerer, get_settings())
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)

    reply = await client.respond(
        user, f"{user}/new", "Which GPU provider was I considering?", now=now
    )

    assert reply.text == "You were considering RunPod."
    assert reply.prompt_version == f"answer_{get_settings().answer_prompt_version}"
    system, final = answerer.prompts[0][0], answerer.prompts[0][-1]
    assert "long-term memory" in str(system.content)
    assert "RunPod" in str(final.content)
    assert "Saturday, 24 October 2026" in str(final.content)
    assert str(final.content).rstrip().endswith("Which GPU provider was I considering?")


async def test_no_memory_placeholder_for_new_user(store: Neo4jStore) -> None:
    answerer = FakeAnswerer()
    client = MemoryClient(store, FakeEmbedder(), answerer, get_settings())
    user = uid("new")
    reply = await client.respond(user, f"{user}/s", "What's my name?")
    assert reply.memory.text == ""
    assert NO_MEMORY in str(answerer.prompts[0][-1].content)


async def test_chat_persists_exchange_and_feeds_next_turn(store: Neo4jStore) -> None:
    embedder, answerer = FakeEmbedder(), FakeAnswerer("Noted!")
    client = MemoryClient(store, embedder, answerer, get_settings())
    user = uid("u")
    session = f"{user}/s1"

    await client.chat(user, session, "My favourite editor is Neovim.")
    await client.chat(user, session, "And what did I just say?")

    turns = await store.get_session_turns(user, session)
    assert [t.text for t in turns] == [
        "My favourite editor is Neovim.",
        "Noted!",
        "And what did I just say?",
        "Noted!",
    ]
    # The second call saw the first exchange as in-session history, not as memory.
    second_prompt = answerer.prompts[1]
    assert any("Neovim" in str(m.content) for m in second_prompt[1:-1])
    assert "Neovim" not in str(second_prompt[-1].content)


async def test_api_chat_and_health(store: Neo4jStore) -> None:
    client = MemoryClient(store, FakeEmbedder(), FakeAnswerer("hello back"), get_settings())
    user = uid("api")
    transport = httpx.ASGITransport(app=create_app(client))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        assert (await http.get("/health")).json() == {"status": "ok"}
        r = await http.post(
            "/chat", json={"user_id": user, "session_id": f"{user}/s", "message": "hi"}
        )
        assert r.status_code == 200
        assert r.json()["reply"] == "hello back"
        assert r.json()["prompt_version"] == f"answer_{get_settings().answer_prompt_version}"
        bad = await http.post("/chat", json={"user_id": "", "session_id": "s", "message": "hi"})
        assert bad.status_code == 422
    # The background write ran after the response: the exchange is stored.
    stored = await store.get_session_turns(user, f"{user}/s")
    assert [t.text for t in stored] == ["hi", "hello back"]
