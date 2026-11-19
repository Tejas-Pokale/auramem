"""Phase 1 gate: per-user isolation of vector and BM25 search, plus store CRUD guarantees.

Both users get identical texts and identical vectors, so if the in-index user filter ever
leaked, the other user's rows would tie for the top results and show up.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from neo4j import AsyncDriver

from memory.config import get_settings
from memory.models import EntityType, FactType, NewEntity, NewFact, NewTurn
from memory.store.migrate import apply_migrations, discover
from memory.store.neo4j import (
    ImmutableTurnError,
    InvalidReferenceError,
    Neo4jStore,
    OwnershipError,
    VersionConflictError,
    user_key,
)

pytestmark = pytest.mark.integration

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
DIMS = get_settings().embedding_dims
MODEL = "test-embedder"
TEXTS = [
    "I moved to Pune last March for the JHEX job",
    "My sister Asha is vegetarian",
    "I switched my GPU rentals from RunPod to Vast.ai",
]


def axis(i: int, scale: float = 1.0) -> list[float]:
    """A distinct direction per text, with a small shared component."""
    v = [0.01] * DIMS
    v[i] = scale
    return v


def uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def store(neo4j_driver: AsyncDriver) -> AsyncIterator[Neo4jStore]:
    settings = get_settings()
    await apply_migrations(
        neo4j_driver,
        discover(settings.migrations_dir),
        database="neo4j",
        index_timeout_seconds=settings.migration_index_timeout_seconds,
    )
    yield Neo4jStore(neo4j_driver, database="neo4j", settings=settings)


async def seed_user(store: Neo4jStore, user_id: str) -> dict[str, Any]:
    """One session with three turns, three entities, and three facts, all with shared texts."""
    session_id = uid("s")
    await store.upsert_session(user_id, session_id, started_at=T0)
    turns = await store.upsert_turns(
        user_id,
        session_id,
        [
            NewTurn(
                turn_id=uid("t"),
                role="user",
                text=text,
                created_at=T0 + timedelta(minutes=i),
                seq=i,
                embedding=axis(i),
                embedding_model=MODEL,
            )
            for i, text in enumerate(TEXTS)
        ],
    )
    entities = await store.insert_entities(
        user_id,
        [
            NewEntity(
                entity_id=uid("e"),
                name=name,
                entity_type=etype,
                aliases=aliases,
                embedding=axis(10 + i),
                embedding_model=MODEL,
            )
            for i, (name, etype, aliases) in enumerate(
                [
                    ("Pune", EntityType.location, ["Poona"]),
                    ("Asha", EntityType.person, ["my sister"]),
                    ("Vast.ai", EntityType.organization, ["vast"]),
                ]
            )
        ],
    )
    facts = await store.insert_facts(
        user_id,
        [
            NewFact(
                fact_id=uid("f"),
                text=f"Fact: {text}",
                text_for_embedding=f"Fact: {text} (around October 2026)",
                fact_type=FactType.state,
                attributed_to="user",
                mentioned_at=T0,
                aliases=[f"alias {i}"],
                confidence=0.9,
                extractor_version="test",
                embedding=axis(20 + i),
                embedding_model=MODEL,
                derived_from_turn_ids=[turns[i].turn_id],
                mentions_entity_ids=[entities[i].entity_id],
            )
            for i, text in enumerate(TEXTS)
        ],
    )
    return {"session_id": session_id, "turns": turns, "entities": entities, "facts": facts}


@pytest.fixture
async def two_users(store: Neo4jStore) -> dict[str, dict[str, Any]]:
    alice, bob = uid("alice"), uid("bob")
    return {alice: await seed_user(store, alice), bob: await seed_user(store, bob)}


def owners(hits: list[Any]) -> set[str]:
    return {h.item.user_id for h in hits}


# --- Gate: isolation -------------------------------------------------------------------


async def test_vector_search_is_isolated(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    for user_id in two_users:
        for k in (1, len(TEXTS), 50):
            turns = await store.vector_search_turns(user_id, axis(0), k)
            facts = await store.vector_search_facts(user_id, axis(20), k)
            entities = await store.vector_search_entities(user_id, axis(10), k)
            for hits in (turns, facts, entities):
                assert owners(hits) == {user_id}
                assert len(hits) == min(k, len(TEXTS))
        # The best match is the user's own copy of the queried item.
        top = await store.vector_search_turns(user_id, axis(0), 1)
        assert top[0].item.text == TEXTS[0]


async def test_fulltext_search_is_isolated(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    for user_id in two_users:
        turns = await store.fulltext_search_turns(user_id, "Pune JHEX sister RunPod", 50)
        facts = await store.fulltext_search_facts(user_id, "fact alias", 50)
        entities = await store.fulltext_search_entities(user_id, "Poona Asha vast", 50)
        for hits in (turns, facts, entities):
            assert owners(hits) == {user_id}
            assert len(hits) == len(TEXTS)
        top = await store.fulltext_search_turns(user_id, "Pune", 1)
        assert [h.item.text for h in top] == [TEXTS[0]]


async def test_fulltext_query_injection_cannot_cross_users(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    alice, bob = list(two_users)
    attacks = [
        f"user_key:{user_key(bob)}",
        f"Pune OR user_key:{user_key(bob)}",
        "*:*",
        f'") OR (user_key:{user_key(bob)}',
        "text:Pune",
    ]
    for attack in attacks:
        for hits in (
            await store.fulltext_search_turns(alice, attack, 50),
            await store.fulltext_search_facts(alice, attack, 50),
            await store.fulltext_search_entities(alice, attack, 50),
        ):
            assert bob not in owners(hits), attack


async def test_similar_user_ids_stay_separate(store: Neo4jStore) -> None:
    base = uid("carol")
    ids = [base, base + "2", base.upper(), f"{base} x"]
    for user_id in ids:
        await seed_user(store, user_id)
    for user_id in ids:
        assert owners(await store.vector_search_turns(user_id, axis(0), 50)) == {user_id}
        assert owners(await store.fulltext_search_turns(user_id, "Pune", 50)) == {user_id}


async def test_empty_or_symbol_only_query_returns_nothing(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    alice = next(iter(two_users))
    assert await store.fulltext_search_turns(alice, "  ?? && ", 10) == []


# --- Ownership guards ------------------------------------------------------------------


async def test_cannot_claim_or_write_into_another_users_session(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    alice, bob = list(two_users)
    alice_session = two_users[alice]["session_id"]
    with pytest.raises(OwnershipError):
        await store.upsert_session(bob, alice_session, started_at=T0)
    with pytest.raises(OwnershipError):
        await store.upsert_turns(
            bob,
            alice_session,
            [NewTurn(turn_id=uid("t"), role="user", text="x", created_at=T0, seq=99)],
        )
    assert await store.get_session(bob, alice_session) is None
    assert len(await store.get_session_turns(alice, alice_session)) == len(TEXTS)


async def test_turn_id_owned_by_another_user_is_rejected(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    alice, bob = list(two_users)
    alice_turn = two_users[alice]["turns"][0]
    with pytest.raises(OwnershipError):
        await store.upsert_turns(
            bob,
            two_users[bob]["session_id"],
            [NewTurn(turn_id=alice_turn.turn_id, role="user", text="x", created_at=T0, seq=50)],
        )
    [unchanged] = await store.get_turns(alice, [alice_turn.turn_id])
    assert unchanged == alice_turn


async def test_facts_cannot_reference_another_users_turns_or_entities(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    alice, bob = list(two_users)
    base = {
        "text": "x",
        "text_for_embedding": "x",
        "fact_type": FactType.event,
        "attributed_to": "user",
        "mentioned_at": T0,
        "confidence": 0.9,
        "extractor_version": "test",
    }
    bad = [
        NewFact(
            fact_id=uid("f"), derived_from_turn_ids=[two_users[bob]["turns"][0].turn_id], **base
        ),
        NewFact(
            fact_id=uid("f"),
            derived_from_turn_ids=[two_users[alice]["turns"][0].turn_id],
            mentions_entity_ids=[two_users[bob]["entities"][0].entity_id],
            **base,
        ),
        NewFact(
            fact_id=uid("f"),
            derived_from_turn_ids=[two_users[alice]["turns"][0].turn_id],
            subject_entity_id=two_users[bob]["entities"][0].entity_id,
            **base,
        ),
    ]
    for fact in bad:
        with pytest.raises(InvalidReferenceError):
            await store.insert_facts(alice, [fact])
        assert await store.get_facts(alice, [fact.fact_id]) == []


async def test_reads_by_id_never_return_other_users_rows(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    alice, bob = list(two_users)
    bob_data = two_users[bob]
    assert await store.get_turns(alice, [t.turn_id for t in bob_data["turns"]]) == []
    assert await store.get_facts(alice, [f.fact_id for f in bob_data["facts"]]) == []
    assert await store.get_entities(alice, [e.entity_id for e in bob_data["entities"]]) == []


# --- Turns -----------------------------------------------------------------------------


async def test_turn_upsert_is_idempotent_and_immutable(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    alice = next(iter(two_users))
    data = two_users[alice]
    original = data["turns"][0]
    same = NewTurn(
        turn_id=original.turn_id,
        role=original.role,
        text=original.text,
        created_at=original.created_at,
        seq=original.seq,
    )
    assert await store.upsert_turns(alice, data["session_id"], [same]) == [original]

    changed = same.model_copy(update={"text": "rewritten history"})
    with pytest.raises(ImmutableTurnError):
        await store.upsert_turns(alice, data["session_id"], [changed])
    [stored] = await store.get_turns(alice, [original.turn_id])
    assert stored.text == original.text


async def test_next_chain_links_out_of_order_inserts(store: Neo4jStore) -> None:
    user, session = uid("dave"), uid("s")
    await store.upsert_session(user, session, started_at=T0)

    def turn(seq: int) -> NewTurn:
        return NewTurn(
            turn_id=f"{session}-{seq}",
            role="user" if seq % 2 == 0 else "assistant",
            text=f"turn {seq}",
            created_at=T0 + timedelta(seconds=seq),
            seq=seq,
        )

    await store.upsert_turns(user, session, [turn(0), turn(2)])
    await store.upsert_turns(user, session, [turn(1), turn(3)])

    window = await store.get_turn_window(user, f"{session}-1", neighbors=1)
    assert [t.seq for t in window] == [0, 1, 2]
    window = await store.get_turn_window(user, f"{session}-3", neighbors=2)
    assert [t.seq for t in window] == [1, 2, 3]
    assert [t.seq for t in await store.get_session_turns(user, session)] == [0, 1, 2, 3]


# --- Facts -----------------------------------------------------------------------------


async def test_fact_flags_filter_search(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    alice = next(iter(two_users))
    data = two_users[alice]
    old, new = data["facts"][0], data["facts"][1]
    hypo = (
        await store.insert_facts(
            alice,
            [
                NewFact(
                    fact_id=uid("f"),
                    text="If the user moved to Berlin they would learn German",
                    text_for_embedding="If the user moved to Berlin they would learn German",
                    fact_type=FactType.plan,
                    attributed_to="user",
                    mentioned_at=T0,
                    confidence=0.8,
                    is_hypothetical=True,
                    extractor_version="test",
                    embedding=axis(30),
                    embedding_model=MODEL,
                    derived_from_turn_ids=[data["turns"][0].turn_id],
                )
            ],
        )
    )[0]

    def ids(hits: list[Any]) -> set[str]:
        return {h.item.fact_id for h in hits}

    assert hypo.fact_id not in ids(await store.vector_search_facts(alice, axis(30), 50))
    assert hypo.fact_id not in ids(await store.fulltext_search_facts(alice, "Berlin", 50))
    assert hypo.fact_id in ids(
        await store.vector_search_facts(alice, axis(30), 50, include_hypothetical=True)
    )
    assert hypo.fact_id in ids(
        await store.fulltext_search_facts(alice, "Berlin", 50, include_hypothetical=True)
    )

    assert await store.mark_superseded(alice, new.fact_id, [old.fact_id]) == 1
    [old_now] = await store.get_facts(alice, [old.fact_id])
    assert old_now.is_latest is False
    # History stays reachable by default; latest_only hides it.
    assert old.fact_id in ids(await store.vector_search_facts(alice, axis(20), 50))
    assert old.fact_id not in ids(
        await store.vector_search_facts(alice, axis(20), 50, latest_only=True)
    )
    assert old.fact_id not in ids(
        await store.fulltext_search_facts(alice, "fact", 50, latest_only=True)
    )


async def test_mark_superseded_ignores_other_users_facts(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    alice, bob = list(two_users)
    bob_fact = two_users[bob]["facts"][0]
    n = await store.mark_superseded(alice, two_users[alice]["facts"][1].fact_id, [bob_fact.fact_id])
    assert n == 0
    [still] = await store.get_facts(bob, [bob_fact.fact_id])
    assert still.is_latest is True


# --- ObservationLog ----------------------------------------------------------------------


async def test_observation_log_versioning(store: Neo4jStore) -> None:
    user = uid("erin")
    assert await store.get_observation_log(user) is None

    v1 = await store.save_observation_log(
        user, "- (2026-10-01) [high] lives in Pune", 12, expected_version=0
    )
    assert v1.version == 1
    v2 = await store.save_observation_log(user, v1.text + "\n- more", 15, expected_version=1)
    assert v2.version == 2

    with pytest.raises(VersionConflictError):
        await store.save_observation_log(user, "stale write", 2, expected_version=1)
    log = await store.get_observation_log(user)
    assert log is not None
    assert log.text == v2.text


# --- Deletes -----------------------------------------------------------------------------


async def test_delete_by_ids_is_scoped_and_relinks_turns(
    store: Neo4jStore, two_users: dict[str, dict[str, Any]]
) -> None:
    alice, bob = list(two_users)
    a, b = two_users[alice], two_users[bob]
    middle = a["turns"][1]

    # Alice cannot delete Bob's rows.
    counts = await store.delete_by_ids(
        alice,
        fact_ids=[f.fact_id for f in b["facts"]],
        turn_ids=[t.turn_id for t in b["turns"]],
        entity_ids=[e.entity_id for e in b["entities"]],
    )
    assert (counts.facts, counts.turns, counts.entities) == (0, 0, 0)
    assert len(await store.get_session_turns(bob, b["session_id"])) == len(TEXTS)

    counts = await store.delete_by_ids(
        alice, fact_ids=[a["facts"][1].fact_id], turn_ids=[middle.turn_id]
    )
    assert (counts.facts, counts.turns, counts.entities) == (1, 1, 0)
    assert await store.get_turns(alice, [middle.turn_id]) == []
    assert middle.text not in {
        h.item.text for h in await store.fulltext_search_turns(alice, "sister", 50)
    }

    window = await store.get_turn_window(alice, a["turns"][0].turn_id, neighbors=1)
    assert [t.seq for t in window] == [0, 2]
