"""Deep-mode loop with fake tools and a fake model: answers, stopping rules, backstops."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from memory.config import Settings, get_settings
from memory.deep import FALLBACK_ANSWER, DeepRunner, call_key
from memory.models import AnswerOutput, DeepStep, DeepTool, DeepToolCall, MemoryContext, Turn
from memory.rerank import Scored
from memory.retrieve import Candidate

NOW = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)


def turn(n: int, text: str = "note") -> Turn:
    return Turn(
        turn_id=f"t{n}",
        user_id="u",
        session_id="s",
        role="user",
        text=f"{text} {n}",
        created_at=NOW,
        seq=n,
        source="typed",
    )


def search(query: str, tool: DeepTool = "search_turns") -> DeepStep:
    return DeepStep(reasoning="looking", tool_calls=[DeepToolCall(tool=tool, query=query)])


class Model:
    """Plays back steps (searching forever after them); records every prompt it saw."""

    def __init__(self, steps: list[DeepStep], *, delay: float = 0.0) -> None:
        self.steps = steps
        self.delay = delay
        self.prompts: list[list[BaseMessage]] = []
        self.calls = 0

    async def __call__(self, messages: list[BaseMessage]) -> tuple[DeepStep, dict[str, int]]:
        self.prompts.append(messages)
        await asyncio.sleep(self.delay)
        step = self.steps[self.calls] if self.calls < len(self.steps) else search(f"q{self.calls}")
        self.calls += 1
        return step, {"input_tokens": 10, "output_tokens": 2}


class Final:
    def __init__(self, text: str = "You haven't told me that.") -> None:
        self.text = text
        self.prompts: list[list[BaseMessage]] = []

    async def __call__(self, messages: list[BaseMessage]) -> tuple[AnswerOutput, dict[str, int]]:
        self.prompts.append(messages)
        return AnswerOutput(answer=self.text), {"input_tokens": 5, "output_tokens": 1}


class Tools:
    """``results(step_index)`` gives (item, score) pairs for each search."""

    def __init__(self, results: Callable[[int], list[tuple[Candidate, float]]]) -> None:
        self.results = results
        self.calls: list[DeepToolCall] = []

    async def run(self, user_id: str, call: DeepToolCall, question: str) -> list[Scored[Candidate]]:
        self.calls.append(call)
        return [Scored(item=i, score=s) for i, s in self.results(len(self.calls))]


def runner(tools: Tools, model: Model, final: Final, **overrides: Any) -> DeepRunner:
    settings: Settings = get_settings().model_copy(update=overrides)
    return DeepRunner(tools, model, final, settings)


async def run(r: DeepRunner, memory: MemoryContext | None = None) -> Any:
    return await r.run(
        user_id="u", question="What did I name my cat?", memory=memory or MemoryContext.empty(),
        now=NOW,
    )  # fmt: skip


async def test_answer_without_searching() -> None:
    model = Model([DeepStep(reasoning="enough", answer="Blue.")])
    result = await run(runner(Tools(lambda n: []), model, Final()))
    assert result.answer == "Blue."
    assert (result.trace.stop_reason, result.trace.steps) == ("answered", 1)


async def test_never_said_stops_after_patience_with_forced_answer() -> None:
    final = Final()
    result = await run(runner(Tools(lambda n: []), Model([]), final, deep_patience=2))
    assert result.answer == "You haven't told me that."
    assert result.trace.stop_reason == "no_progress"
    assert result.trace.steps == 2 and result.trace.new_items == [0, 0]
    assert "Stop searching now" in str(final.prompts[0][-1].content)


async def test_progress_resets_patience() -> None:
    def results(n: int) -> list[tuple[Candidate, float]]:
        return [(turn(n), 0.9)] if n <= 3 else []  # three searches find something new

    result = await run(runner(Tools(results), Model([]), Final(), deep_patience=2))
    assert result.trace.new_items == [1, 1, 1, 0, 0]
    assert result.trace.stop_reason == "no_progress"


async def test_already_seen_and_irrelevant_items_are_not_progress() -> None:
    seen = MemoryContext.empty().model_copy(update={"turn_ids": ["t1"]})

    def results(n: int) -> list[tuple[Candidate, float]]:
        return [(turn(1), 0.9), (turn(100 + n), 0.001)]  # seen, or below DEEP_MIN_RELEVANCE

    result = await run(
        runner(Tools(results), Model([]), Final(), deep_patience=2, deep_min_relevance=0.01),
        seen,
    )
    assert result.trace.new_items == [0, 0]


async def test_repeated_search_is_skipped_and_counts_as_no_progress() -> None:
    tools = Tools(lambda n: [(turn(n), 0.9)])
    model = Model([search("cat"), search("  CAT "), search("cat")])
    result = await run(runner(tools, model, Final(), deep_patience=2))
    assert len(tools.calls) == 1  # the two repeats never ran
    assert result.trace.new_items == [1, 0, 0]
    assert "already searched" in str(model.prompts[2][-1].content)


async def test_results_show_only_new_items_and_the_newest_message_is_cached() -> None:
    model = Model([search("cat"), DeepStep(reasoning="done", answer="None found.")])
    await run(runner(Tools(lambda n: [(turn(7, "my cat Blue"), 0.9)]), model, Final()))
    last = model.prompts[1]
    results = last[-1]
    assert isinstance(results, HumanMessage)
    assert "my cat Blue 7" in str(results.content)
    assert isinstance(results.content, list)
    block = results.content[0]
    assert isinstance(block, dict) and block["cache_control"] == {"type": "ephemeral"}
    earlier = [m for m in last[1:-1] if isinstance(m, HumanMessage)]
    assert all(isinstance(m.content, str) for m in earlier)  # one breakpoint only


async def test_timeout_is_an_outage_backstop() -> None:
    result = await run(
        runner(
            Tools(lambda n: []), Model([], delay=0.2), Final("partial"), deep_timeout_seconds=0.05
        )
    )
    assert (result.trace.stop_reason, result.answer) == ("timeout", "partial")


async def test_context_window_forces_an_answer() -> None:
    big: list[tuple[Candidate, float]] = [(turn(n, "x" * 4000), 0.9) for n in range(50)]
    result = await run(
        runner(Tools(lambda n: big), Model([]), Final(), deep_context_window_tokens=8000)
    )
    assert result.trace.stop_reason == "context_full"


async def test_failed_forced_answer_falls_back() -> None:
    class Broken(Final):
        async def __call__(
            self, messages: list[BaseMessage]
        ) -> tuple[AnswerOutput, dict[str, int]]:
            raise RuntimeError("api down")

    result = await run(runner(Tools(lambda n: []), Model([]), Broken()))
    assert (result.answer, result.trace.stop_reason) == (FALLBACK_ANSWER, "error")


def test_call_key_ignores_case_and_spacing() -> None:
    a = DeepToolCall(tool="grep", query="Blue  Cat")
    b = DeepToolCall(tool="grep", query=" blue cat ")
    assert call_key(a) == call_key(b)
    assert call_key(a) != call_key(DeepToolCall(tool="search_turns", query="blue cat"))
