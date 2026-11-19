import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from langchain_core.messages import BaseMessage

from memory.config import Settings
from memory.models import QueryPlan, TimeRange
from memory.plan import Planner
from memory.rerank import Reranker

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=IST)


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


class FakePlanLLM:
    def __init__(self, plan: QueryPlan | None = None, delay: float = 0.0, fail: bool = False):
        self.plan = plan or QueryPlan(depth="quick", is_temporal=True, rewrites=["trip dates"])
        self.delay = delay
        self.fail = fail
        self.calls: list[list[BaseMessage]] = []

    async def __call__(self, messages: list[BaseMessage]) -> QueryPlan:
        self.calls.append(messages)
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        return self.plan


async def test_rules_only_when_resolvable() -> None:
    llm = FakePlanLLM()
    result = await Planner(settings(), llm).plan("What did I do in July 2026?", now=NOW)
    assert result.source == "rules"
    assert llm.calls == []
    assert result.plan.time_range is not None


async def test_llm_fallback_for_event_relative_time() -> None:
    llm = FakePlanLLM()
    result = await Planner(settings(), llm).plan("What did I do after my trip?", now=NOW)
    assert result.source == "llm"
    assert result.plan.rewrites == ["trip dates"]
    prompt = str(llm.calls[0][-1].content)
    assert "after my trip" in prompt
    assert "Saturday, 24 October 2026" in prompt


async def test_llm_plan_keeps_rule_fields_it_left_empty() -> None:
    llm = FakePlanLLM(QueryPlan(depth="deep", is_temporal=False))
    result = await Planner(settings(), llm).plan(
        "Where did Asha go after my exam in March 2026?", now=NOW
    )
    assert result.source == "llm"
    assert result.plan.depth == "deep"
    assert result.plan.is_temporal  # from rules
    assert "Asha" in result.plan.entities  # from rules


async def test_llm_time_range_wins_when_given() -> None:
    rng = TimeRange(start=NOW - timedelta(days=10), end=NOW)
    llm = FakePlanLLM(QueryPlan(depth="quick", is_temporal=True, time_range=rng))
    result = await Planner(settings(), llm).plan("What did I do after my trip?", now=NOW)
    assert result.plan.time_range == rng


@pytest.mark.parametrize("llm", [FakePlanLLM(delay=1.0), FakePlanLLM(fail=True)])
async def test_timeout_or_error_falls_back_to_rules(llm: FakePlanLLM) -> None:
    planner = Planner(settings(planner_llm_timeout_seconds=0.05), llm)
    result = await planner.plan("What did I do after my trip?", now=NOW)
    assert result.source == "rules_fallback"
    assert result.plan.is_temporal
    assert result.elapsed_ms < 900


async def test_no_llm_configured_uses_rules() -> None:
    result = await Planner(settings(), None).plan("What did I do after my trip?", now=NOW)
    assert result.source == "rules"


class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the document."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.calls.append(sentences)
        return [len(set(q.lower().split()) & set(d.lower().split())) / 10 for q, d in sentences]


async def test_rerank_orders_by_score_and_keeps_ties_stable() -> None:
    model = FakeCrossEncoder()
    reranker = Reranker(settings(), model)
    docs = ["nothing here", "gpu rental runpod", "gpu notes", "also nothing"]
    ranked = await reranker.rerank("which gpu rental", docs, lambda d: d)
    assert [r.item for r in ranked] == [
        "gpu rental runpod",
        "gpu notes",
        "nothing here",
        "also nothing",
    ]
    assert ranked[0].score == pytest.approx(0.2)
    assert all(q == "which gpu rental" for q, _ in model.calls[0])


async def test_rerank_empty_skips_model() -> None:
    model = FakeCrossEncoder()
    assert await Reranker(settings(), model).rerank("q", [], str) == []
    assert model.calls == []


async def test_pairs_are_scored_in_length_order_but_returned_in_input_order() -> None:
    model = FakeCrossEncoder()
    reranker = Reranker(settings(), model)
    docs = ["gpu " * 30, "gpu", "gpu " * 5]
    ranked = await reranker.rerank("gpu", docs, lambda d: d)
    sent = [d for _, d in model.calls[0]]
    assert sent == sorted(docs, key=len)  # shortest first, so batches pad less
    assert {r.item for r in ranked} == set(docs) and all(r.score == 0.1 for r in ranked)


async def test_concurrent_requests_share_model_calls_with_their_own_results() -> None:
    class Slow(FakeCrossEncoder):
        def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> list[float]:
            time.sleep(0.05)  # the GPU is busy; later requests queue up meanwhile
            return super().predict(sentences, **kwargs)

    model = Slow()
    reranker = Reranker(settings(), model)
    queries = [f"word{i}" for i in range(6)]
    docs = [f"word{i} doc" for i in range(6)]
    results = await asyncio.gather(*(reranker.rerank(q, docs, lambda d: d) for q in queries))
    assert len(model.calls) < len(queries)  # requests were combined
    for q, ranked in zip(queries, results, strict=True):
        assert ranked[0].item == f"{q} doc" and ranked[0].score == pytest.approx(0.1)
