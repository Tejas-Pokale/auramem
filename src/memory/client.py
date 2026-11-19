"""MemoryClient: the only entry point apps use (PLAN.md section 7)."""

from datetime import datetime
from types import TracebackType
from typing import Any, Self

from arq.connections import ArqRedis
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage

from memory.config import Settings
from memory.deep import DeepRunner, DeepTools, FinalLLM, StepLLM
from memory.embed import Embedder, EmbedderLike
from memory.forget import Forgetter
from memory.graphs.llm import chat_model, structured, structured_with_usage
from memory.graphs.prompting import format_today, system_message
from memory.graphs.read import Answerer, ClaudeAnswerer, ReadState, build_read_graph, now_in
from memory.graphs.write import persist_exchange, persist_turn
from memory.jobs.queue import enqueue_after_persist
from memory.models import (
    AnswerOutput,
    DeepStep,
    ForgetPreview,
    ForgetReport,
    QueryPlan,
    Reply,
    Turn,
)
from memory.obs import prompt_label
from memory.observe import ReflectorLLM, SummaryLLM, claude_reflector, claude_summarizer
from memory.plan import Planner
from memory.prompts import load_prompt
from memory.rerank import Reranker
from memory.retrieve import Mode, Retriever
from memory.store.neo4j import Neo4jStore


class MemoryClient:
    def __init__(
        self,
        store: Neo4jStore,
        embedder: EmbedderLike,
        answerer: Answerer,
        settings: Settings,
        *,
        planner: Planner | None = None,
        reranker: Reranker | None = None,
        use_profile: bool = True,
        deep_llms: tuple[StepLLM, FinalLLM] | None = None,
        forget_llms: tuple[ReflectorLLM, SummaryLLM] | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker
        self.answerer = answerer
        self.queue: ArqRedis | None = None
        self.retriever = Retriever(store, embedder, settings, planner=planner, reranker=reranker)
        deep = None
        if deep_llms is not None:
            tools = DeepTools(store, embedder, reranker, settings)
            deep = DeepRunner(tools, *deep_llms, settings)
        # Forget needs the reranker to match memories precisely enough to show for approval.
        self.forgetter: Forgetter | None = None
        if forget_llms is not None and reranker is not None:
            profile_llm, summarizer = forget_llms
            self.forgetter = Forgetter(
                store, embedder, reranker, settings, profile_llm=profile_llm, summarizer=summarizer
            )
        self._graph = build_read_graph(
            store, self.retriever, answerer, settings, use_profile=use_profile, deep=deep,
            forgetter=self.forgetter,
        )  # fmt: skip

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        plan_llm = structured(
            chat_model(
                settings.planner_model,
                settings,
                effort=settings.planner_effort,
                max_tokens=settings.planner_max_tokens,
            ),
            QueryPlan,
        )
        return cls(
            Neo4jStore.connect(settings),
            Embedder(settings),
            ClaudeAnswerer(settings, model=settings.answer_model, effort=settings.answer_effort),
            settings,
            planner=Planner(settings, plan_llm),
            reranker=Reranker(settings),
            deep_llms=claude_deep_llms(settings),
            forget_llms=(claude_reflector(settings), claude_summarizer(settings)),
        )

    async def greet(self, user_id: str, session_id: str, *, now: datetime | None = None) -> str:
        """Open a new conversation as a continuation of the last one, and store the opening
        as the conversation's first (assistant) turn so later replies can refer to it.

        Uses the profile and the end of the user's previous conversation. A user with no
        memory yet gets a fixed welcome (no LLM call).
        """
        s = self.settings
        now = now or now_in(s.default_timezone)
        log_ = await self.store.get_observation_log(user_id)
        profile = log_.text if log_ is not None else ""
        last = await self.store.last_conversation(
            user_id, exclude_session=session_id, limit=s.greeting_recent_turns
        )
        if not profile.strip() and not last:
            text = (
                "Hi! I'm your assistant, and I remember our conversations from one session "
                "to the next. What would you like to talk about?"
            )
        else:
            version = s.greeting_prompt_version
            excerpt = "\n".join(
                f"[{t.created_at:%Y-%m-%d %H:%M}] {t.role.capitalize()}: {t.text}" for t in last
            )
            messages: list[BaseMessage] = [
                system_message(version, profile, name="greet"),
                HumanMessage(
                    load_prompt("greet", version, "user").text.format(
                        last_conversation=excerpt or "(none yet)", today=format_today(now)
                    )
                ),
            ]
            with prompt_label(f"greet_{version}"):
                output, _ = await self.answerer(messages)
            text = output.answer
        await persist_turn(
            self.store, self.embedder, user_id=user_id, session_id=session_id,
            role="assistant", text=text, at=now,
        )  # fmt: skip
        return text

    async def respond(
        self,
        user_id: str,
        session_id: str,
        message: str,
        *,
        now: datetime | None = None,
        mode: Mode = "auto",
    ) -> Reply:
        """Read path only: retrieve memory and answer. Persists nothing.

        ``mode``: "auto" lets the planner choose quick or deep; "quick" never runs the deep
        loop; "deep" always does (when deep mode is configured).
        """
        state: ReadState = {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "now": now or now_in(self.settings.default_timezone),
            "mode": mode,
        }
        result = await self._graph.ainvoke(state)
        reply: Reply = result["reply"]
        return reply

    async def remember(
        self,
        user_id: str,
        session_id: str,
        message: str,
        reply_text: str,
        *,
        asked_at: datetime,
        replied_at: datetime,
    ) -> list[Turn]:
        """Write path: persist the exchange, then schedule extraction (when a queue is attached).

        Call after the reply has been sent; nothing here blocks on extraction.
        """
        turns = await persist_exchange(
            self.store,
            self.embedder,
            user_id=user_id,
            session_id=session_id,
            user_text=message,
            assistant_text=reply_text,
            user_at=asked_at,
            assistant_at=replied_at,
        )
        if self.queue is not None:
            await enqueue_after_persist(self.queue, self.store, self.settings, user_id, session_id)
        return turns

    async def forget_preview(
        self, user_id: str, what: str, *, now: datetime | None = None
    ) -> ForgetPreview:
        """Everything memory holds about ``what``. Deletes nothing (see forget_confirm)."""
        return await self._require_forgetter().preview(
            user_id, what, now=now or now_in(self.settings.default_timezone)
        )

    async def forget_confirm(
        self, user_id: str, preview_id: str, *, now: datetime | None = None
    ) -> ForgetReport | None:
        """Hard-delete what the preview showed. None if the preview is unknown or expired."""
        return await self._require_forgetter().confirm(
            user_id, preview_id, now=now or now_in(self.settings.default_timezone)
        )

    async def forget_cancel(self, user_id: str, preview_id: str) -> bool:
        return await self._require_forgetter().cancel(user_id, preview_id)

    async def export(self, user_id: str) -> dict[str, Any]:
        """Everything stored for the user (no embeddings), as JSON-ready data."""
        return await self.store.export_user(user_id)

    def _require_forgetter(self) -> Forgetter:
        if self.forgetter is None:
            raise RuntimeError("forget needs a reranker and forget_llms")
        return self.forgetter

    def attach_queue(self, queue: ArqRedis) -> None:
        self.queue = queue

    async def chat(self, user_id: str, session_id: str, message: str) -> Reply:
        asked_at = now_in(self.settings.default_timezone)
        reply = await self.respond(user_id, session_id, message, now=asked_at)
        await self.remember(
            user_id,
            session_id,
            message,
            reply.text,
            asked_at=asked_at,
            replied_at=now_in(self.settings.default_timezone),
        )
        return reply

    async def close(self) -> None:
        if self.queue is not None:
            await self.queue.aclose()
        await self.store.close()
        if isinstance(self.embedder, Embedder):
            await self.embedder.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


def claude_deep_llms(settings: Settings) -> tuple[StepLLM, FinalLLM]:
    """The deep-mode model, once for search steps and once for the forced final answer."""

    def model() -> ChatAnthropic:
        return chat_model(
            settings.deep_answer_model,
            settings,
            effort=settings.deep_answer_effort,
            max_tokens=settings.answer_max_tokens,
        )

    return (
        structured_with_usage(model(), DeepStep),
        structured_with_usage(model(), AnswerOutput),
    )
