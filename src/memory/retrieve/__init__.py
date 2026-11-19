"""Multi-signal retrieval: signals S1-S7, RRF fusion, rerank, budgeted assembly (PLAN.md 6).

Flow:
1. In parallel: the query planner, and baseline signals for the raw message (plus the
   previous assistant turn, to resolve pronouns): S1 + S3 over facts, S2 + S4 over turns.
   Retrieval never waits for the plan.
2. With the plan: S1-S4 again for each rewrite; S6 (facts) and S6t (turns) for a time range.
   S5 adds facts about the entities the plan names.
3. RRF over every signal (keys ``fact:<id>`` / ``turn:<id>``); S7 expands one hop from the top
   GRAPH_SEED_FACTS facts and everything is fused again. Current-session turns are dropped
   and the top RERANK_CANDIDATES kept.
4. Cross-encoder rerank of facts and turns together against the ORIGINAL message. If the
   plan says depth="none" and nothing scores above NONE_DEPTH_MIN_SCORE, inject nothing.
5. Up to MAX_FACTS facts (plus the other side of any supersession among them), then up to
   MAX_TURNS turns (each +-TURN_NEIGHBORS) whose source is not already covered by a selected
   fact, packed into one token budget.

Deep mode (planner depth "deep", or forced by the caller) uses the same pipeline but selects
without any fact, turn, or token cap (``deep_limits``); the deep loop then searches further.
"""

import asyncio
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

from memory.config import Settings
from memory.embed import EmbedderLike
from memory.models import Fact, Hit, MemoryContext, QueryPlan, Turn
from memory.plan import Planner, PlanResult
from memory.rerank import Reranker
from memory.retrieve.assemble import (
    estimate_tokens,
    render_context,
    select_facts,
    select_windows,
)
from memory.retrieve.fusion import Fused, reciprocal_rank_fusion
from memory.retrieve.signals import (
    entity_signal,
    fact_signals,
    graph_signal,
    temporal_fact_signal,
    temporal_turn_signal,
    turn_signals,
)
from memory.store.neo4j import Neo4jStore

Candidate = Turn | Fact
Mode = Literal["auto", "quick", "deep"]
UNBOUNDED = 10**9  # "no cap" for deep mode (user decision 2026-10-25)


def deep_limits(settings: Settings) -> Settings:
    """Selection settings for deep mode: every reranked candidate, uncut, no token budget."""
    return settings.model_copy(
        update={
            "max_facts": UNBOUNDED,
            "max_fact_partners": UNBOUNDED,
            "max_turns": UNBOUNDED,
            "max_excerpt_turns": UNBOUNDED,
            "memory_token_budget": UNBOUNDED,
            "excerpt_assistant_max_chars": UNBOUNDED,
        }
    )


_ROLE = {"user": "User", "assistant": "Assistant"}
_FACT_SOURCE = {"user": "user", "assistant": "assistant"}


_FACT_SIGNALS = ("S1_", "S3_", "S5_", "S6_", "S7_")
_BM25_SIGNALS = ("S3_", "S4_")


def _enabled(signal: str, s: Settings) -> bool:
    """Ablation switches (evals): drop whole signal families before fusion."""
    if not s.use_fact_index and signal.startswith(_FACT_SIGNALS):
        return False
    return s.use_bm25 or not signal.startswith(_BM25_SIGNALS)


def candidate_key(item: Candidate) -> str:
    return f"turn:{item.turn_id}" if isinstance(item, Turn) else f"fact:{item.fact_id}"


def _ranked(hits: Sequence[Hit[Turn]] | Sequence[Hit[Fact]]) -> list[tuple[str, Candidate]]:
    return [(candidate_key(h.item), h.item) for h in hits]


def rerank_text(item: Candidate) -> str:
    """What the cross-encoder sees: a date, who said it, and the text."""
    if isinstance(item, Turn):
        return f"[{item.created_at:%Y-%m-%d}] {_ROLE[item.role]}: {item.text}"
    when = item.start or item.mentioned_at
    source = _FACT_SOURCE.get(item.attributed_to, "third party")
    return f"[{when:%Y-%m-%d}] Fact ({source}): {item.text}"


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    def since(self, name: str, started: float) -> None:
        self.ms[name] = (time.perf_counter() - started) * 1000


class Retriever:
    def __init__(
        self,
        store: Neo4jStore,
        embedder: EmbedderLike,
        settings: Settings,
        *,
        planner: Planner | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings
        self._deep = deep_limits(settings)
        self.planner = planner
        self._reranker = reranker

    async def retrieve(
        self,
        user_id: str,
        message: str,
        *,
        now: datetime | None = None,
        recent_turns: Sequence[Turn] = (),
        previous_assistant: str | None = None,
        current_session_id: str | None = None,
        budget: int | None = None,
        mode: Mode = "auto",
    ) -> MemoryContext:
        s = self._settings
        timer = _Timer()
        started = time.perf_counter()
        query = f"{message}\n{previous_assistant}" if previous_assistant else message

        async def baseline() -> tuple[list[float], dict[str, Any]]:
            t = time.perf_counter()
            vector = (await self._embedder.embed_query([query])).vectors[0]
            turns, facts = await asyncio.gather(
                turn_signals(
                    self._store, user_id, query_vector=vector, query_text=query, k=s.signal_top_k
                ),
                fact_signals(
                    self._store, user_id, query_vector=vector, query_text=query, k=s.signal_top_k
                ),
            )
            timer.since("baseline", t)
            return vector, {**turns, **facts}

        async def plan() -> PlanResult | None:
            if self.planner is None or now is None:
                return None
            result = await self.planner.plan(message, now=now, recent_turns=recent_turns)
            timer.ms["planner"] = result.elapsed_ms
            return result

        (vector, signals), plan_result = await asyncio.gather(baseline(), plan())
        query_plan = plan_result.plan if plan_result else None
        deep = mode == "deep" or (
            mode == "auto" and query_plan is not None and query_plan.depth == "deep"
        )
        if query_plan is not None:
            t = time.perf_counter()
            signals.update(await self._plan_signals(user_id, vector, query_plan))
            timer.since("plan_signals", t)

        t = time.perf_counter()
        fused = await self._fuse_with_graph(user_id, signals)
        timer.since("fusion_graph", t)
        candidates: list[Candidate] = [
            f.item
            for f in fused
            if not (isinstance(f.item, Turn) and f.item.session_id == current_session_id)
        ]

        top_score: float | None = None
        ranked: list[Candidate]
        if self._reranker is not None:
            t = time.perf_counter()
            scored = await self._reranker.rerank(
                message, candidates[: s.rerank_candidates], rerank_text
            )
            timer.since("rerank", t)
            top_score = scored[0].score if scored else None
            skip = (
                not deep
                and query_plan is not None
                and query_plan.depth == "none"
                and (top_score is None or top_score < s.none_depth_min_score)
            )
            ranked = [] if skip else [x.item for x in scored if x.score >= s.rerank_min_score]
        else:
            ranked = candidates

        memory = await self._select(
            user_id, ranked, self._deep if deep else s, budget, candidates=len(fused)
        )
        timer.since("total", started)
        return memory.model_copy(
            update={
                "plan": query_plan,
                "plan_source": plan_result.source if plan_result else None,
                "top_rerank_score": top_score,
                "timings_ms": timer.ms,
                "deep": deep,
            }
        )

    async def _select(
        self,
        user_id: str,
        ranked: Sequence[Candidate],
        lim: Settings,
        budget: int | None,
        *,
        candidates: int,
    ) -> MemoryContext:
        """Facts (plus supersession partners), then turn windows not covered by them, within
        the limits in ``lim`` (quick-mode settings, or ``deep_limits``)."""
        s = self._settings
        total_budget = lim.memory_token_budget if budget is None else budget
        facts = select_facts(
            (c for c in ranked if isinstance(c, Fact)),
            budget=total_budget,
            max_facts=lim.max_facts,
            chars_per_token=s.chars_per_token,
        )
        facts = await self._with_partners(user_id, facts, total_budget, lim)
        covered = {tid for f in facts for tid in f.source_turn_ids}
        turns = [c for c in ranked if isinstance(c, Turn) and c.turn_id not in covered]
        turns = turns[: lim.max_turns]
        hits = {t.turn_id for t in turns}  # matched turns are never cut, only their neighbors

        packed: list[Turn] = []
        if turns:
            windows = await self._store.get_turn_windows(
                user_id, [t.turn_id for t in turns], neighbors=s.turn_neighbors
            )
            facts_tokens = estimate_tokens(
                render_context(facts, []), chars_per_token=s.chars_per_token
            )
            packed = select_windows(
                (windows[t.turn_id] for t in turns if t.turn_id in windows),
                budget=max(0, total_budget - facts_tokens),
                chars_per_token=s.chars_per_token,
                max_turns=lim.max_excerpt_turns,
                assistant_max_chars=lim.excerpt_assistant_max_chars,
                uncut=hits,
            )
        packed_ids = {t.turn_id for t in packed}
        text = render_context(
            facts, packed, assistant_max_chars=lim.excerpt_assistant_max_chars, uncut=hits
        )
        return MemoryContext(
            text=text,
            token_estimate=estimate_tokens(text, chars_per_token=s.chars_per_token),
            hit_turn_ids=[t.turn_id for t in turns if t.turn_id in packed_ids],
            turn_ids=sorted(packed_ids),
            fact_ids=[f.fact_id for f in facts],
            covered_turn_ids=sorted(covered),
            candidates=candidates,
        )

    async def _fuse_with_graph(
        self, user_id: str, signals: dict[str, Any]
    ) -> list[Fused[Candidate]]:
        """RRF, then S7: expand from the best facts so far and fuse again with the expansion."""
        s = self._settings

        def fuse() -> list[Fused[Candidate]]:
            ranked = {name: _ranked(hits) for name, hits in signals.items() if _enabled(name, s)}
            return reciprocal_rank_fusion(ranked, k=s.rrf_k)

        fused = fuse()
        seeds = [f.item.fact_id for f in fused if isinstance(f.item, Fact)][: s.graph_seed_facts]
        if not seeds or not s.use_graph:
            return fused
        signals.update(
            await graph_signal(self._store, user_id, seed_fact_ids=seeds, k=s.signal_top_k)
        )
        return fuse()

    async def _with_partners(
        self, user_id: str, facts: list[Fact], budget: int, lim: Settings
    ) -> list[Fact]:
        """Add the other side of every supersession or conflict among the selected facts, up to
        MAX_FACT_PARTNERS (best-ranked facts' partners first), budget allowing.

        A superseded value is only useful next to its replacement (and vice versa), so
        "before / now" pairs arrive together.
        """
        chosen = {f.fact_id for f in facts}
        partner_ids = list(
            dict.fromkeys(
                pid
                for f in facts
                for pid in (*f.superseded_by_ids, *f.supersedes_ids, *f.conflicts_with_ids)
                if pid not in chosen
            )
        )[: lim.max_fact_partners]
        if not partner_ids:
            return facts
        order = {pid: i for i, pid in enumerate(partner_ids)}
        partners = sorted(
            await self._store.get_facts(user_id, partner_ids), key=lambda f: order[f.fact_id]
        )
        return select_facts(
            [*facts, *partners],
            budget=budget,
            max_facts=lim.max_facts + len(partners),
            chars_per_token=lim.chars_per_token,
        )

    async def _plan_signals(
        self, user_id: str, vector: list[float], plan: QueryPlan
    ) -> dict[str, Any]:
        s = self._settings
        rewrites = plan.rewrites[: s.planner_max_rewrites]
        tasks: list[Any] = []
        if rewrites:
            embedded = await self._embedder.embed_query(list(rewrites))
            for i, (text, v) in enumerate(zip(rewrites, embedded.vectors, strict=True)):
                suffix = f"_rewrite{i}"
                tasks.append(
                    turn_signals(
                        self._store,
                        user_id,
                        query_vector=v,
                        query_text=text,
                        k=s.signal_top_k,
                        suffix=suffix,
                    )
                )
                tasks.append(
                    fact_signals(
                        self._store,
                        user_id,
                        query_vector=v,
                        query_text=text,
                        k=s.signal_top_k,
                        suffix=suffix,
                    )
                )
        if plan.time_range is not None:
            tasks.append(
                temporal_turn_signal(
                    self._store,
                    user_id,
                    query_vector=vector,
                    time_range=plan.time_range,
                    k=s.signal_top_k,
                )
            )
            tasks.append(
                temporal_fact_signal(
                    self._store,
                    user_id,
                    time_range=plan.time_range,
                    k=s.signal_top_k,
                    scan_limit=s.temporal_fact_scan_limit,
                )
            )
        if plan.entities:
            tasks.append(entity_signal(self._store, user_id, names=plan.entities, k=s.signal_top_k))
        out: dict[str, Any] = {}
        for result in await asyncio.gather(*tasks):
            out.update(result)
        return out
