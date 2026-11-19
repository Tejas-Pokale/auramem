"""Deep mode (PLAN.md section 6 step 8, Phase 7): an agentic memory search loop.

The deep-mode model starts from the uncapped retrieval context and, step by step, either
answers or runs searches: ``search_facts``, ``search_turns``, ``grep``, ``timeline``. Nothing
is capped, neither the context nor the number of searches (user decision 2026-10-25). Every
result is reranked against the ORIGINAL question, and only items not shown before are added.

Stopping rules (memory is finite, so the loop always ends):
1. The model answers.
2. DEEP_PATIENCE steps in a row add no new item scoring at least DEEP_MIN_RELEVANCE. A
   repeated search runs nothing and counts as such a step. The model must then answer from
   what it has ("you haven't told me" when there is nothing).
3. Backstops that never trigger in normal operation: DEEP_TIMEOUT_SECONDS for outages, and the
   model's own context window (DEEP_CONTEXT_WINDOW_TOKENS).
"""

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from memory.config import Settings
from memory.embed import EmbedderLike
from memory.graphs.prompting import format_today, system_message
from memory.models import (
    AnswerOutput,
    DeepStep,
    DeepToolCall,
    Fact,
    Hit,
    MemoryContext,
    TimeRange,
    Turn,
)
from memory.models.context import DeepTrace
from memory.prompts import load_prompt
from memory.rerank import Reranker, Scored
from memory.retrieve import Candidate, candidate_key, rerank_text
from memory.retrieve.assemble import estimate_tokens, fact_line
from memory.retrieve.fusion import reciprocal_rank_fusion
from memory.retrieve.signals import (
    fact_signals,
    resolve_entity_ids,
    temporal_fact_signal,
    temporal_turn_signal,
    turn_signals,
)
from memory.store.neo4j import Neo4jStore

log = logging.getLogger(__name__)

Usage = dict[str, int]
NO_MEMORY = "(no relevant past conversations found)"
_ROLE = {"user": "User", "assistant": "Assistant"}
FALLBACK_ANSWER = "Sorry, I couldn't finish searching my memory just now. Please ask me again."


class StepLLM(Protocol):
    async def __call__(self, messages: list[BaseMessage]) -> tuple[DeepStep, Usage]: ...


class FinalLLM(Protocol):
    async def __call__(self, messages: list[BaseMessage]) -> tuple[AnswerOutput, Usage]: ...


class ToolRunner(Protocol):
    async def run(
        self, user_id: str, call: DeepToolCall, question: str
    ) -> list[Scored[Candidate]]: ...


def render_item(item: Candidate) -> str:
    if isinstance(item, Fact):
        return fact_line(item)
    return f"- [{item.created_at:%Y-%m-%d}] {_ROLE[item.role]}: {item.text}"


def call_key(call: DeepToolCall) -> tuple[str, ...]:
    """Identity of a search, so a repeat (up to case and spacing) is not run again."""
    return (
        call.tool,
        " ".join(call.query.lower().split()),
        call.start.isoformat() if call.start else "",
        call.end.isoformat() if call.end else "",
        (call.entity or "").strip().lower(),
    )


def describe(call: DeepToolCall) -> str:
    parts = [f'"{call.query}"']
    if call.start or call.end:
        start = f"{call.start:%Y-%m-%d}" if call.start else "..."
        end = f"{call.end:%Y-%m-%d}" if call.end else "..."
        parts.append(f"{start} to {end}")
    if call.entity:
        parts.append(f"about {call.entity}")
    return f"{call.tool}({', '.join(parts)})"


class DeepTools:
    """The four deep-mode searches; each returns its hits reranked against the question."""

    def __init__(
        self,
        store: Neo4jStore,
        embedder: EmbedderLike,
        reranker: Reranker | None,
        settings: Settings,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings

    async def run(self, user_id: str, call: DeepToolCall, question: str) -> list[Scored[Candidate]]:
        items = await self._items(user_id, call)
        if self._reranker is None:
            return [Scored(item=i, score=1.0) for i in items]
        scored = await self._reranker.rerank(question, items, rerank_text)
        if call.tool == "timeline":  # keep date order; scores still decide progress
            order = {candidate_key(i): n for n, i in enumerate(items)}
            scored.sort(key=lambda x: order[candidate_key(x.item)])
        return scored

    async def _items(self, user_id: str, call: DeepToolCall) -> list[Candidate]:
        s = self._settings
        k = s.deep_signal_top_k
        span = TimeRange(start=call.start, end=call.end) if call.start or call.end else None
        if call.tool == "grep":
            turns, facts = await asyncio.gather(
                self._store.grep_turns(user_id, call.query),
                self._store.grep_facts(user_id, call.query),
            )
            return [h.item for h in facts] + [h.item for h in turns]
        if call.tool == "timeline":
            ids = await resolve_entity_ids(self._store, user_id, [call.query])
            hits = await self._store.timeline(user_id, ids, start=call.start, end=call.end)
            return [h.item for h in hits]
        vector = (await self._embedder.embed_query([call.query])).vectors[0]
        signals: dict[str, Sequence[Hit[Fact]] | Sequence[Hit[Turn]]] = {}
        if call.tool == "search_facts":
            signals.update(
                await fact_signals(
                    self._store, user_id, query_vector=vector, query_text=call.query, k=k
                )
            )
            if span is not None:
                signals.update(
                    await temporal_fact_signal(
                        self._store, user_id, time_range=span, k=k,
                        scan_limit=s.temporal_fact_scan_limit,
                    )
                )  # fmt: skip
            if call.entity:
                ids = await resolve_entity_ids(self._store, user_id, [call.entity])
                signals["entity"] = await self._store.facts_mentioning(user_id, ids, k)
        else:  # search_turns
            signals.update(
                await turn_signals(
                    self._store, user_id, query_vector=vector, query_text=call.query, k=k
                )
            )
            if span is not None:
                signals.update(
                    await temporal_turn_signal(
                        self._store, user_id, query_vector=vector, time_range=span, k=k
                    )
                )
        ranked = {
            name: [(candidate_key(h.item), h.item) for h in hits] for name, hits in signals.items()
        }
        fused = reciprocal_rank_fusion(ranked, k=s.rrf_k)
        return [f.item for f in fused]


@dataclass
class DeepResult:
    answer: str
    trace: DeepTrace
    usage: Usage = field(default_factory=dict)


def _add_usage(total: Usage, more: Usage) -> None:
    for key, value in more.items():
        total[key] = total.get(key, 0) + value


def _cached(message: str) -> HumanMessage:
    """The newest message carries a cache breakpoint, so each step re-reads the conversation
    so far from the prompt cache instead of paying for it again (cost, not a cap)."""
    block = {"type": "text", "text": message, "cache_control": {"type": "ephemeral"}}
    return HumanMessage(content=[block])


@dataclass
class _Session:
    """State of one deep-mode question: the conversation so far and what has been shown."""

    user_id: str
    question: str
    settings: Settings
    system: BaseMessage
    past: list[BaseMessage]
    opening: str
    seen: set[str]
    used_tokens: int
    started: float = field(default_factory=time.perf_counter)
    exchanges: list[tuple[str, str]] = field(default_factory=list)  # (step json, results)
    searched: set[tuple[str, ...]] = field(default_factory=set)
    trace: DeepTrace = field(default_factory=DeepTrace)
    usage: Usage = field(default_factory=dict)
    context_full: bool = False

    def messages(self, extra: str | None = None) -> list[BaseMessage]:
        texts = [self.opening]
        for step_json, results in self.exchanges:
            texts += [step_json, results]
        if extra is not None:
            texts.append(extra)
        out: list[BaseMessage] = [self.system, *self.past]
        for n, text in enumerate(texts):
            if n % 2:  # the model's own steps
                out.append(AIMessage(text))
            else:  # opening, search results, final instruction
                out.append(_cached(text) if n == len(texts) - 1 else HumanMessage(text))
        return out

    def take(self, scored: Sequence[Scored[Candidate]]) -> tuple[list[str], int]:
        """Lines for results not shown before, and how many of them are relevant. Stops adding
        (and flags ``context_full``) only when the model's context window would overflow."""
        s = self.settings
        lines: list[str] = []
        relevant = 0
        for x in scored:
            key = candidate_key(x.item)
            if key in self.seen:
                continue
            line = render_item(x.item)
            cost = estimate_tokens(line + "\n", chars_per_token=s.chars_per_token)
            if self.used_tokens + cost + s.answer_max_tokens > s.deep_context_window_tokens:
                self.context_full = True
                break
            self.used_tokens += cost
            self.seen.add(key)
            lines.append(line)
            relevant += x.score >= s.deep_min_relevance
        return lines, relevant

    def record(self, step: DeepStep | None, results: str) -> None:
        step_json = step.model_dump_json(exclude_none=True) if step is not None else "{}"
        self.exchanges.append((step_json, results))
        self.used_tokens += estimate_tokens(
            step_json + results, chars_per_token=self.settings.chars_per_token
        )

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000


class DeepRunner:
    def __init__(
        self, tools: ToolRunner, step_llm: StepLLM, final_llm: FinalLLM, settings: Settings
    ) -> None:
        self._tools = tools
        self._step = step_llm
        self._final = final_llm
        self._settings = settings
        self.model = settings.deep_answer_model
        self.prompt_version = f"deep_{settings.deep_prompt_version}"

    async def run(
        self,
        *,
        user_id: str,
        question: str,
        memory: MemoryContext,
        now: datetime,
        history: Sequence[Turn] = (),
        profile: str = "",
    ) -> DeepResult:
        session = self._session(
            user_id=user_id,
            question=question,
            memory=memory,
            now=now,
            history=history,
            profile=profile,
        )
        try:
            async with asyncio.timeout(self._settings.deep_timeout_seconds):
                return await self._loop(session)
        except TimeoutError:
            log.warning("deep mode: timed out for %s after %d steps", user_id, session.trace.steps)
            return await self._finish(session, "timeout", "the search took too long")

    def _session(
        self,
        *,
        user_id: str,
        question: str,
        memory: MemoryContext,
        now: datetime,
        history: Sequence[Turn],
        profile: str,
    ) -> _Session:
        s = self._settings
        version = s.deep_prompt_version
        notes = ""
        if memory.guidance and memory.text:
            bullets = "\n".join(f"- {g}" for g in memory.guidance)
            notes = f"\n<retrieval_notes>\n{bullets}\n</retrieval_notes>\n"
        opening = load_prompt("deep", version, "user").text.format(
            memory=memory.text or NO_MEMORY, notes=notes, today=format_today(now), message=question
        )
        past: list[BaseMessage] = [
            HumanMessage(t.text) if t.role == "user" else AIMessage(t.text) for t in history
        ]
        prefix = [load_prompt("deep", version, "system").text, profile, opening]
        prefix += [t.text for t in history]
        return _Session(
            user_id=user_id,
            question=question,
            settings=s,
            system=system_message(version, profile, name="deep"),
            past=past,
            opening=opening,
            seen={f"fact:{i}" for i in memory.fact_ids} | {f"turn:{i}" for i in memory.turn_ids},
            used_tokens=estimate_tokens("".join(prefix), chars_per_token=s.chars_per_token),
        )

    async def _loop(self, session: _Session) -> DeepResult:
        no_progress = 0
        while True:
            session.trace.steps += 1
            try:
                step, more = await self._step(session.messages())
                _add_usage(session.usage, more)
            except Exception:
                log.warning("deep mode: step failed for %s", session.user_id, exc_info=True)
                step = None
            if step is not None and step.answer and not step.tool_calls:
                session.trace.stop_reason = "answered"
                session.trace.elapsed_ms = session.elapsed_ms()
                return DeepResult(answer=step.answer, trace=session.trace, usage=session.usage)
            relevant = await self._search(session, step)
            session.trace.new_items.append(relevant)
            if session.context_full:
                return await self._finish(
                    session, "context_full", "the conversation reached the model's context limit"
                )
            no_progress = 0 if relevant else no_progress + 1
            if no_progress >= self._settings.deep_patience:
                return await self._finish(
                    session, "no_progress", "the last searches found nothing new and relevant"
                )

    async def _search(self, session: _Session, step: DeepStep | None) -> int:
        """Run the step's searches; returns how many new relevant items they added."""
        if step is None:
            session.record(None, "(that step failed; search again or answer)")
            return 0
        if not step.tool_calls:
            session.record(step, "(no searches and no answer: search or answer)")
            return 0
        blocks: list[str] = []
        relevant = 0
        for call in step.tool_calls:
            session.trace.tool_calls.append(describe(call))
            key = call_key(call)
            if key in session.searched:
                blocks.append(f"### {describe(call)}: already searched, skipped")
                continue
            session.searched.add(key)
            scored = await self._tools.run(session.user_id, call, session.question)
            lines, found = session.take(scored)
            relevant += found
            header = f"### {describe(call)}: {len(lines)} new results"
            blocks.append("\n".join([header, *lines]))
            if session.context_full:
                break
        session.record(step, "\n\n".join(blocks))
        return relevant

    async def _finish(self, session: _Session, reason: str, why: str) -> DeepResult:
        """The model must answer now, from what it has found."""
        session.trace.stop_reason = reason
        instruction = load_prompt("deep", self._settings.deep_prompt_version, "final").text
        try:
            output, more = await self._final(session.messages(instruction.format(reason=why)))
            _add_usage(session.usage, more)
            answer = output.answer
        except Exception:
            log.exception("deep mode: forced answer failed for %s", session.user_id)
            session.trace.stop_reason = "error"
            answer = FALLBACK_ANSWER
        session.trace.elapsed_ms = session.elapsed_ms()
        return DeepResult(answer=answer, trace=session.trace, usage=session.usage)
