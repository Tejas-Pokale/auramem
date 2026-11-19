"""Phase 3 read path against real Neo4j: planner signals, time filter, rerank, threshold."""

import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from neo4j import AsyncDriver

from memory.config import Settings, get_settings
from memory.graphs.write import persist_exchange, turn_id_for
from memory.models import QueryPlan, TimeRange, Turn
from memory.plan import Planner, PlanResult
from memory.rerank import Reranker
from memory.retrieve import Retriever
from memory.store.migrate import apply_migrations, discover
from memory.store.neo4j import Neo4jStore
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

T0 = datetime(2026, 5, 1, 18, 0, tzinfo=UTC)
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
SESSIONS = [
    ("may", 0, "I'm considering RunPod for rented GPU instances.", "RunPod is a fine start."),
    ("june", 40, "I started exploring Vast.ai as an alternative GPU provider.", "Good option."),
    ("july", 75, "My TCS Prime result is still pending and I'm anxious.", "That's stressful."),
    ("cooking", 90, "How long should dal simmer?", "About 25 minutes."),
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


@pytest.fixture
async def user(store: Neo4jStore) -> str:
    user_id = uid("u")
    embedder = FakeEmbedder()
    for name, day, user_text, reply in SESSIONS:
        at = T0 + timedelta(days=day)
        await persist_exchange(
            store,
            embedder,
            user_id=user_id,
            session_id=f"{user_id}/{name}",
            user_text=user_text,
            assistant_text=reply,
            user_at=at,
            assistant_at=at + timedelta(seconds=30),
        )
    return user_id


class FixedPlanner(Planner):
    def __init__(self, plan: QueryPlan) -> None:
        super().__init__(get_settings())
        self.fixed = plan

    async def plan(
        self, message: str, *, now: datetime, recent_turns: Sequence[Turn] = ()
    ) -> PlanResult:
        return PlanResult(self.fixed, "rules", 0.1)


class KeywordCrossEncoder:
    """Score = share of query words found in the document; records every query."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        out = []
        for q, d in sentences:
            self.queries.append(q)
            words = {w.strip("?.,!").lower() for w in q.split()}
            doc = d.lower()
            out.append(sum(w in doc for w in words) / max(1, len(words)))
        return out


def retriever(
    store: Neo4jStore, *, planner: Planner | None = None, rerank: bool = True, **overrides: Any
) -> tuple[Retriever, KeywordCrossEncoder]:
    s: Settings = get_settings().model_copy(update=overrides)
    model = KeywordCrossEncoder()
    reranker = Reranker(s, model) if rerank else None
    return Retriever(store, FakeEmbedder(), s, planner=planner, reranker=reranker), model


async def test_vector_search_time_filter_runs_in_index(store: Neo4jStore, user: str) -> None:
    query = (await FakeEmbedder().embed_query(["GPU provider"])).vectors[0]
    june = T0 + timedelta(days=40)
    hits = await store.vector_search_turns(
        user, query, 50, created_from=june - timedelta(days=1), created_to=june + timedelta(days=1)
    )
    assert {h.item.session_id for h in hits} == {f"{user}/june"}
    open_ended = await store.vector_search_turns(user, query, 50, created_from=june)
    assert {h.item.session_id for h in open_ended} == {
        f"{user}/june",
        f"{user}/july",
        f"{user}/cooking",
    }


async def test_rerank_uses_original_message_not_rewrites(store: Neo4jStore, user: str) -> None:
    plan = QueryPlan(depth="quick", is_temporal=False, rewrites=["RunPod rented GPU instances"])
    r, model = retriever(store, planner=FixedPlanner(plan))
    message = "Which compute vendor did I look at?"
    memory = await r.retrieve(user, message, now=NOW, previous_assistant="Earlier reply text")
    assert set(model.queries) == {message}
    assert memory.plan == plan
    assert "planner" in memory.timings_ms and "rerank" in memory.timings_ms


async def test_rewrites_add_candidates(store: Neo4jStore, user: str) -> None:
    message = "Which compute vendor did I look at?"
    without, _ = retriever(store, rerank=False)
    plan = QueryPlan(depth="quick", is_temporal=False, rewrites=["RunPod rented GPU instances"])
    with_plan, _ = retriever(store, planner=FixedPlanner(plan), rerank=False)
    target = turn_id_for(f"{user}/may", 0)
    base = await without.retrieve(user, message, now=NOW)
    planned = await with_plan.retrieve(user, message, now=NOW)
    assert planned.candidates >= base.candidates
    assert target in planned.hit_turn_ids


async def test_time_range_signal_promotes_turns_in_range(store: Neo4jStore, user: str) -> None:
    july = T0 + timedelta(days=75)
    plan = QueryPlan(
        depth="quick",
        is_temporal=True,
        time_range=TimeRange(start=july - timedelta(days=5), end=july + timedelta(days=5)),
    )
    # No word overlaps any turn, so only the time-range signal separates the candidates.
    r, _ = retriever(store, planner=FixedPlanner(plan), rerank=False, max_turns=2)
    memory = await r.retrieve(user, "What happened then?", now=NOW)
    assert {t.rsplit(":", 1)[0] for t in memory.hit_turn_ids} == {f"{user}/july"}


async def test_threshold_drops_low_scores_and_can_inject_nothing(
    store: Neo4jStore, user: str
) -> None:
    r, _ = retriever(store, rerank_min_score=0.99)
    memory = await r.retrieve(user, "Which GPU rental provider did I consider?", now=NOW)
    assert memory.text == ""
    assert memory.top_rerank_score is not None

    r, _ = retriever(store, rerank_min_score=0.3)
    memory = await r.retrieve(user, "Which GPU rental provider did I consider?", now=NOW)
    assert memory.text and "RunPod" in memory.text
    assert "dal" not in memory.text


async def test_depth_none_injects_only_when_memory_clearly_matches(
    store: Neo4jStore, user: str
) -> None:
    rules = Planner(get_settings())  # real rules: impersonal question -> depth "none"
    r, _ = retriever(store, planner=rules, none_depth_min_score=0.9)
    memory = await r.retrieve(user, "What is the difference between TCP and UDP?", now=NOW)
    assert memory.plan is not None and memory.plan.depth == "none"
    assert memory.text == ""

    r, _ = retriever(store, planner=rules, none_depth_min_score=0.1)
    memory = await r.retrieve(user, "How long should dal simmer?", now=NOW)
    assert memory.plan is not None and memory.plan.depth == "none"
    assert "dal" in memory.text
