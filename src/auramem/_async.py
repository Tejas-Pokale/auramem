"""AsyncAuraMem: the async public API over the memory engine (``memory``)."""

import asyncio
import logging
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from types import TracebackType
from typing import Any, Literal, Self

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from memory.client import MemoryClient
from memory.config import Settings, get_settings
from memory.entities import ClaudeTiebreaker
from memory.extract import ExtractionDeps, Extractor, ExtractorLLM, run_extraction
from memory.graphs.read import format_today, now_in
from memory.jobs.queue import unextracted_tokens
from memory.models import Fact, ForgetPreview, ForgetReport, Reply
from memory.observe import (
    ReflectorLLM,
    SummaryLLM,
    claude_reflector,
    claude_summarizer,
    reflect,
    summarize_entities,
)
from memory.retrieve import Mode
from memory.store.migrate import apply_migrations, discover

log = logging.getLogger("auramem")

Extraction = Literal["inline", "worker", "off"]


@dataclass(frozen=True)
class Recall:
    """What memory holds for a message, ready to put into any LLM's prompt."""

    context: str  # dated facts and conversation excerpts ("" when nothing relevant)
    profile: str  # the always-on user profile
    today: str
    memory_tokens: int = 0
    mode: str = "quick"

    def as_prompt(self) -> str:
        """A block to place before the user's message in your own LLM call."""
        parts = []
        if self.profile.strip():
            parts.append(f"<user_profile>\n{self.profile}\n</user_profile>")
        parts.append(
            f"<memory>\n{self.context or '(no relevant past conversations found)'}\n</memory>"
        )
        parts.append(f"Today is {self.today}.")
        return "\n\n".join(parts)


@dataclass
class _Extraction:
    deps: ExtractionDeps
    reflector: ReflectorLLM | None
    summarizer: SummaryLLM | None
    locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    tasks: set[asyncio.Task[None]] = field(default_factory=set)


class AsyncAuraMem:
    """Long-term memory for an LLM app.

    ``extraction`` decides when facts and profile notes are extracted from stored messages:

    - ``"inline"`` (default): in this process, in the background, once a user's unextracted
      messages reach EXTRACT_TOKEN_THRESHOLD; ``flush()`` extracts everything now (call it at
      the end of a conversation or before exiting).
    - ``"worker"``: queue jobs for the arq worker (``arq memory.jobs.worker.WorkerSettings``),
      which also runs the idle trigger and nightly consolidation.
    - ``"off"``: store messages only (they are still searchable as raw text).

    Settings come from keyword arguments, then environment variables / ``.env`` (see
    ``memory.config.Settings``), e.g. ``AsyncAuraMem(neo4j_uri="bolt://db:7687")``.
    """

    def __init__(
        self,
        *,
        extraction: Extraction = "inline",
        auto_migrate: bool = True,
        **settings: Any,
    ) -> None:
        self.settings: Settings = Settings(**settings) if settings else get_settings()
        self._extraction_mode = extraction
        self._auto_migrate = auto_migrate
        self._client: MemoryClient | None = None
        self._extraction: _Extraction | None = None
        self._queue: ArqRedis | None = None
        self._run_id = uuid.uuid4().hex[:8]
        self._started = False

    @classmethod
    def from_client(
        cls,
        client: MemoryClient,
        *,
        extraction: Extraction = "off",
        extraction_deps: ExtractionDeps | None = None,
    ) -> Self:
        """Wrap an existing engine client (custom models, tests). Nothing is migrated."""
        mem = cls(extraction=extraction, auto_migrate=False)
        mem.settings = client.settings
        mem._client = client
        if extraction == "inline" and extraction_deps is not None:
            mem._extraction = _Extraction(extraction_deps, None, None)
        mem._started = True
        return mem

    # -- lifecycle --

    async def start(self) -> Self:
        """Connect, apply schema migrations, and set up extraction. Called on first use."""
        if self._started:
            return self
        s = self.settings
        self._client = MemoryClient.from_settings(s)
        if self._auto_migrate:
            await apply_migrations(
                self._client.store.driver,
                discover(s.migrations_dir),
                database=s.neo4j_database,
                index_timeout_seconds=s.migration_index_timeout_seconds,
            )
        if self._extraction_mode == "worker":
            self._queue = await create_pool(RedisSettings.from_dsn(s.redis_url))
            self._client.attach_queue(self._queue)
        elif self._extraction_mode == "inline":
            deps = ExtractionDeps(
                store=self._client.store,
                embedder=self._client.embedder,
                extractor=Extractor(ExtractorLLM(s), s),
                settings=s,
                tiebreaker=ClaudeTiebreaker(s),
            )
            self._extraction = _Extraction(deps, claude_reflector(s), claude_summarizer(s))
        self._started = True
        return self

    async def close(self) -> None:
        """Finish background extraction, then disconnect."""
        if self._extraction is not None and self._extraction.tasks:
            await asyncio.gather(*self._extraction.tasks, return_exceptions=True)
        if self._client is not None:
            await self._client.close()  # also closes an attached queue
        self._started = False

    async def __aenter__(self) -> Self:
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def _ready(self) -> MemoryClient:
        await self.start()
        assert self._client is not None
        return self._client

    def _session(self, user_id: str, session_id: str | None) -> str:
        """One conversation per user per AuraMem instance unless the caller names one."""
        return session_id or f"{user_id}/auramem-{self._run_id}"

    # -- conversation --

    async def chat(
        self,
        user_id: str,
        message: str,
        *,
        session_id: str | None = None,
        mode: Mode = "auto",
    ) -> Reply:
        """Answer ``message`` with the user's memory, then remember the exchange.

        ``mode``: "auto" lets the planner choose quick or deep search, "quick" or "deep" force
        one. A forget request ("forget that I ...") is answered with a preview and a yes/no
        question; the user's "yes" in the same session deletes.
        """
        client = await self._ready()
        session = self._session(user_id, session_id)
        asked_at = now_in(self.settings.default_timezone)
        reply = await client.respond(user_id, session, message, now=asked_at, mode=mode)
        await self.remember(user_id, message, reply.text, session_id=session, asked_at=asked_at)
        return reply

    async def greet(self, user_id: str, *, session_id: str | None = None) -> str:
        """The assistant's opening message for a new conversation: picks up from the user's
        profile and the end of their previous conversation, and is stored as the first turn
        of this one."""
        client = await self._ready()
        return await client.greet(user_id, self._session(user_id, session_id))

    async def remember(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        *,
        session_id: str | None = None,
        asked_at: datetime | None = None,
        replied_at: datetime | None = None,
    ) -> None:
        """Store an exchange answered elsewhere (e.g. by your own LLM after ``recall``)."""
        client = await self._ready()
        session = self._session(user_id, session_id)
        asked = asked_at or now_in(self.settings.default_timezone)
        await client.remember(
            user_id, session, user_message, assistant_message,
            asked_at=asked, replied_at=replied_at or now_in(self.settings.default_timezone),
        )  # fmt: skip
        if self._queue is not None:  # worker mode: client.remember already enqueued
            return
        await self._maybe_extract(user_id)

    async def recall(
        self,
        user_id: str,
        message: str,
        *,
        session_id: str | None = None,
        mode: Literal["quick", "deep"] = "quick",
    ) -> Recall:
        """Memory for ``message`` without answering it: use it with any LLM, then call
        ``remember`` with the exchange. ``mode="deep"`` returns the uncapped context."""
        client = await self._ready()
        now = now_in(self.settings.default_timezone)
        session = self._session(user_id, session_id)
        history = await client.store.get_session_turns(user_id, session)
        previous = history[-1].text if history and history[-1].role == "assistant" else None
        memory = await client.retriever.retrieve(
            user_id, message, now=now, recent_turns=history[-self.settings.planner_recent_turns :],
            previous_assistant=previous, current_session_id=session, mode=mode,
        )  # fmt: skip
        return Recall(
            context=memory.text,
            profile=await self.profile(user_id),
            today=format_today(now),
            memory_tokens=memory.token_estimate,
            mode="deep" if memory.deep else "quick",
        )

    async def profile(self, user_id: str) -> str:
        """The user's always-on profile (dated, prioritized notes)."""
        client = await self._ready()
        log_ = await client.store.get_observation_log(user_id)
        return log_.text if log_ is not None else ""

    async def facts(self, user_id: str, *, history: bool = False) -> list[Fact]:
        """Stored facts, oldest first. By default only current ones: no superseded values
        and no hypotheticals; ``history=True`` returns everything."""
        facts = await (await self._ready()).store.list_facts(user_id)
        if history:
            return facts
        return [f for f in facts if f.is_latest and not f.is_hypothetical]

    async def stats(self, user_id: str) -> dict[str, int]:
        """Counts of what is stored, and how much text still awaits extraction."""
        client = await self._ready()
        store = client.store
        facts = await store.list_facts(user_id)
        pending = await store.pending_extraction(user_id)
        profile = await store.get_observation_log(user_id)
        return {
            "messages": await store.count_turns(user_id),
            "facts": len(facts),
            "current_facts": sum(f.is_latest and not f.is_hypothetical for f in facts),
            "superseded_facts": sum(not f.is_latest for f in facts),
            "conflicting_facts": sum(bool(f.conflicts_with_ids) for f in facts),
            "entities": len(await store.list_entities(user_id)),
            "profile_tokens": profile.token_count if profile is not None else 0,
            "messages_awaiting_extraction": sum(p.new_turns for p in pending),
            "tokens_awaiting_extraction": await unextracted_tokens(store, user_id, self.settings),
        }

    # -- extraction --

    async def flush(self, user_id: str | None = None) -> None:
        """Extract every pending message now (one user, or every user this instance saw)."""
        if self._extraction is None:
            await self._ready()
        if self._extraction is None:
            return
        users = [user_id] if user_id else list(self._extraction.locks)
        await asyncio.gather(*(self._extract(u) for u in users))

    async def _maybe_extract(self, user_id: str) -> None:
        ex = self._extraction
        if ex is None:
            return
        ex.locks.setdefault(user_id, asyncio.Lock())
        tokens = await unextracted_tokens(ex.deps.store, user_id, self.settings)
        if tokens < self.settings.extract_token_threshold:
            return
        self._background(self._extract(user_id))

    def _background(self, work: Awaitable[None]) -> None:
        assert self._extraction is not None
        task: asyncio.Task[None] = asyncio.ensure_future(work)
        self._extraction.tasks.add(task)
        task.add_done_callback(self._extraction.tasks.discard)

    async def _extract(self, user_id: str) -> None:
        ex = self._extraction
        assert ex is not None
        lock = ex.locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            try:
                stats = await run_extraction(ex.deps, user_id)
                if ex.reflector is not None:
                    await reflect(ex.deps.store, ex.reflector, self.settings, user_id)
                if ex.summarizer is not None and stats.touched_entities:
                    await summarize_entities(
                        ex.deps.store, ex.summarizer, self.settings, user_id,
                        sorted(stats.touched_entities),
                    )  # fmt: skip
            except Exception:
                log.exception(
                    "auramem: extraction failed for %s (messages stay searchable)", user_id
                )

    # -- forget, export, delete --

    async def forget(self, user_id: str, what: str) -> ForgetPreview:
        """Everything memory holds about ``what``. Deletes nothing: show ``preview.render()``
        to the user and call ``confirm_forget`` only after they agree."""
        return await (await self._ready()).forget_preview(user_id, what)

    async def confirm_forget(self, user_id: str, preview_id: str) -> ForgetReport | None:
        """Permanently delete what the preview showed. None if unknown or expired."""
        return await (await self._ready()).forget_confirm(user_id, preview_id)

    async def cancel_forget(self, user_id: str, preview_id: str) -> bool:
        return await (await self._ready()).forget_cancel(user_id, preview_id)

    async def export(self, user_id: str) -> dict[str, Any]:
        """Everything stored for the user (no embeddings), as JSON-ready data."""
        return await (await self._ready()).export(user_id)

    async def delete_user(self, user_id: str) -> dict[str, int]:
        """Hard-delete every memory of this user (irreversible). Returns counts per type."""
        return await (await self._ready()).store.delete_user_data(user_id)
