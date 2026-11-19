"""Read path as a LangGraph state graph (PLAN.md section 6).

load the current session and the ObservationLog -> forget check -> retrieval (planner,
signals, rerank, assembly) -> answer, or the deep-mode loop when the question is routed (or
forced) deep. A forget request is answered with a preview and a yes/no question; the next
"yes" in that session deletes, "no" keeps everything (Phase 8). The
ObservationLog goes right after the frozen system prompt with a cache breakpoint, so the
provider's prompt cache reuses that prefix across requests; anything that changes per request
(date, memory block, message) comes later.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Literal, Protocol, TypedDict
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from memory.config import Settings
from memory.deep import DeepRunner
from memory.forget import CANCEL, CONFIRM, Forgetter, forget_request
from memory.graphs.llm import Usage, chat_model, structured_with_usage
from memory.graphs.prompting import format_today, system_message
from memory.models import AnswerOutput, MemoryContext, Reply, Turn
from memory.prompts import load_prompt
from memory.retrieve import Mode, Retriever
from memory.store.neo4j import Neo4jStore

NO_MEMORY = "(no relevant past conversations found)"
__all__ = ["NO_MEMORY", "format_today", "system_message"]  # re-exported for callers


class Answerer(Protocol):
    model: str

    async def __call__(self, messages: list[BaseMessage]) -> tuple[AnswerOutput, Usage]: ...


class ClaudeAnswerer:
    def __init__(self, settings: Settings, *, model: str, effort: str) -> None:
        self.model = model
        self._call = structured_with_usage(
            chat_model(model, settings, effort=effort, max_tokens=settings.answer_max_tokens),
            AnswerOutput,
        )

    async def __call__(self, messages: list[BaseMessage]) -> tuple[AnswerOutput, Usage]:
        return await self._call(messages)


class ReadState(TypedDict, total=False):
    user_id: str
    session_id: str
    message: str
    now: datetime
    history: list[Turn]
    profile: str
    memory: MemoryContext
    reply: Reply
    mode: Mode


def build_messages(
    *,
    version: str,
    history: list[Turn],
    memory: MemoryContext,
    now: datetime,
    message: str,
    profile: str = "",
) -> list[BaseMessage]:
    notes = ""
    if memory.guidance and memory.text:
        bullets = "\n".join(f"- {g}" for g in memory.guidance)
        notes = f"\n<retrieval_notes>\n{bullets}\n</retrieval_notes>\n"
    user = load_prompt("answer", version, "user").text.format(
        memory=memory.text or NO_MEMORY, notes=notes, today=format_today(now), message=message
    )
    past: list[BaseMessage] = [
        HumanMessage(t.text) if t.role == "user" else AIMessage(t.text) for t in history
    ]
    return [system_message(version, profile), *past, HumanMessage(user)]


def build_read_graph(
    store: Neo4jStore,
    retriever: Retriever,
    answerer: Answerer,
    settings: Settings,
    *,
    use_profile: bool = True,
    deep: DeepRunner | None = None,
    forgetter: Forgetter | None = None,
) -> CompiledStateGraph[ReadState, None, ReadState, ReadState]:
    version = settings.answer_prompt_version

    async def load_history(state: ReadState) -> ReadState:
        history, profile = await asyncio.gather(
            store.get_session_turns(state["user_id"], state["session_id"]),
            store.get_observation_log(state["user_id"]),
        )
        text = profile.text if profile is not None and use_profile else ""
        return {"history": history, "profile": text}

    def handled(text: str) -> ReadState:
        reply = Reply(
            text=text,
            model="none",
            prompt_version="forget",
            memory=MemoryContext.empty(),
            mode="forget",
        )
        return {"reply": reply}

    async def check_forget(state: ReadState) -> ReadState:
        """Forget requests and the yes/no that follows a preview never reach the answerer."""
        if forgetter is None:
            return {}
        user, session, message, now = (
            state["user_id"], state["session_id"], state["message"], state["now"]
        )  # fmt: skip
        pending = await store.session_pending_forget(
            user, session,
            not_before=now - timedelta(minutes=settings.forget_preview_ttl_minutes),
        )  # fmt: skip
        if pending is not None and CONFIRM.match(message):
            report = await forgetter.confirm(user, str(pending["preview_id"]), now=now)
            return handled(report.render() if report else "That request expired; ask again.")
        if pending is not None and CANCEL.match(message):
            await forgetter.cancel(user, str(pending["preview_id"]))
            return handled("Okay, I kept everything.")
        if (what := forget_request(message)) is not None:
            preview = await forgetter.preview(
                user, what, now=now, session_id=session, request_text=message
            )
            text = preview.render()
            if preview.preview_id is not None:
                await store.set_pending_preview_text(user, preview.preview_id, text)
            return handled(text)
        return {}

    def after_forget_check(state: ReadState) -> Literal["retrieve", "__end__"]:
        return "__end__" if "reply" in state else "retrieve"

    async def retrieve(state: ReadState) -> ReadState:
        history = state["history"]
        previous = history[-1].text if history and history[-1].role == "assistant" else None
        memory = await retriever.retrieve(
            state["user_id"],
            state["message"],
            now=state["now"],
            recent_turns=history[-settings.planner_recent_turns :],
            previous_assistant=previous,
            current_session_id=state["session_id"],
            mode=state.get("mode", "auto"),
        )
        return {"memory": memory}

    async def answer(state: ReadState) -> ReadState:
        messages = build_messages(
            version=version,
            history=state["history"],
            memory=state["memory"],
            now=state["now"],
            message=state["message"],
            profile=state.get("profile", ""),
        )
        output, usage = await answerer(messages)
        reply = Reply(
            text=output.answer,
            model=answerer.model,
            prompt_version=f"answer_{version}",
            memory=state["memory"],
            profile_tokens=round(len(state.get("profile", "")) / settings.chars_per_token),
            usage=usage,
            mode=_mode(state["memory"]),
        )
        return {"reply": reply}

    async def deep_answer(state: ReadState) -> ReadState:
        assert deep is not None
        result = await deep.run(
            user_id=state["user_id"],
            question=state["message"],
            memory=state["memory"],
            now=state["now"],
            history=state["history"],
            profile=state.get("profile", ""),
        )
        reply = Reply(
            text=result.answer,
            model=deep.model,
            prompt_version=deep.prompt_version,
            memory=state["memory"],
            profile_tokens=round(len(state.get("profile", "")) / settings.chars_per_token),
            usage=result.usage,
            mode="deep",
            deep=result.trace,
        )
        return {"reply": reply}

    def route(state: ReadState) -> Literal["answer", "deep_answer"]:
        return "deep_answer" if deep is not None and state["memory"].deep else "answer"

    graph = StateGraph(ReadState)
    graph.add_node("load_history", load_history)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("deep_answer", deep_answer)
    graph.add_edge(START, "load_history")
    graph.add_node("check_forget", check_forget)
    graph.add_edge("load_history", "check_forget")
    graph.add_conditional_edges("check_forget", after_forget_check)
    graph.add_conditional_edges("retrieve", route)
    graph.add_edge("answer", END)
    graph.add_edge("deep_answer", END)
    return graph.compile()


def _mode(memory: MemoryContext) -> str:
    """How a quick answer was produced: "none" when the planner saw no need for memory."""
    return "none" if memory.plan is not None and memory.plan.depth == "none" else "quick"


def now_in(timezone: str) -> datetime:
    return datetime.now(ZoneInfo(timezone))
