"""AuraMem: a blocking wrapper over AsyncAuraMem for plain (non-async) code.

It runs AsyncAuraMem on a private event loop in a background thread, so background extraction
keeps running between calls, and every method simply blocks until its result is ready.
"""

import asyncio
import threading
from collections.abc import Awaitable, Callable, Coroutine
from types import TracebackType
from typing import Any, Literal, Self

from auramem._async import AsyncAuraMem, Extraction, Recall
from memory.models import Fact, ForgetPreview, ForgetReport, Reply
from memory.retrieve import Mode


class AuraMem:
    """Same API as AsyncAuraMem, without ``await``. Use ``with AuraMem() as mem:`` or call
    ``close()`` at the end, so pending extraction finishes."""

    def __init__(
        self, *, extraction: Extraction = "inline", auto_migrate: bool = True, **settings: Any
    ) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._async = self._run(self._make(extraction, auto_migrate, settings))

    @staticmethod
    async def _make(
        extraction: Extraction, auto_migrate: bool, settings: dict[str, Any]
    ) -> AsyncAuraMem:
        return AsyncAuraMem(extraction=extraction, auto_migrate=auto_migrate, **settings)

    @classmethod
    def from_factory(cls, factory: Callable[[], Awaitable[AsyncAuraMem]]) -> Self:
        """Blocking access to a custom AsyncAuraMem (own models, tests). The factory runs on
        this wrapper's event loop, where async drivers must be created."""
        self = cls.__new__(cls)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._async = self._run(_call(factory))
        return self

    def _run[T](self, work: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(work, self._loop).result()

    @property
    def settings(self) -> Any:
        return self._async.settings

    def start(self) -> Self:
        self._run(self._async.start())
        return self

    def close(self) -> None:
        try:
            self._run(self._async.close())
            self._run(_shutdown_loop())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)
            if not self._loop.is_running():
                self._loop.close()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def chat(
        self, user_id: str, message: str, *, session_id: str | None = None, mode: Mode = "auto"
    ) -> Reply:
        return self._run(self._async.chat(user_id, message, session_id=session_id, mode=mode))

    def greet(self, user_id: str, *, session_id: str | None = None) -> str:
        return self._run(self._async.greet(user_id, session_id=session_id))

    def remember(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        *,
        session_id: str | None = None,
    ) -> None:
        self._run(
            self._async.remember(user_id, user_message, assistant_message, session_id=session_id)
        )

    def recall(
        self,
        user_id: str,
        message: str,
        *,
        session_id: str | None = None,
        mode: Literal["quick", "deep"] = "quick",
    ) -> Recall:
        return self._run(self._async.recall(user_id, message, session_id=session_id, mode=mode))

    def profile(self, user_id: str) -> str:
        return self._run(self._async.profile(user_id))

    def facts(self, user_id: str, *, history: bool = False) -> list[Fact]:
        return self._run(self._async.facts(user_id, history=history))

    def stats(self, user_id: str) -> dict[str, int]:
        return self._run(self._async.stats(user_id))

    def flush(self, user_id: str | None = None) -> None:
        self._run(self._async.flush(user_id))

    def forget(self, user_id: str, what: str) -> ForgetPreview:
        return self._run(self._async.forget(user_id, what))

    def confirm_forget(self, user_id: str, preview_id: str) -> ForgetReport | None:
        return self._run(self._async.confirm_forget(user_id, preview_id))

    def cancel_forget(self, user_id: str, preview_id: str) -> bool:
        return self._run(self._async.cancel_forget(user_id, preview_id))

    def export(self, user_id: str) -> dict[str, Any]:
        return self._run(self._async.export(user_id))

    def delete_user(self, user_id: str) -> dict[str, int]:
        return self._run(self._async.delete_user(user_id))


async def _shutdown_loop() -> None:
    """Cancel leftover tasks and async generators so the loop can close cleanly."""
    current = asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not current]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.get_running_loop().shutdown_asyncgens()


async def _call(factory: Callable[[], Awaitable[AsyncAuraMem]]) -> AsyncAuraMem:
    return await factory()
