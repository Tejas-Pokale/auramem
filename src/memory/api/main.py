"""FastAPI app: /chat, /forget (preview, confirm, cancel), /export, /health (PLAN.md 8).

uv run uvicorn memory.api.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from memory.client import MemoryClient
from memory.config import get_settings
from memory.graphs.read import now_in
from memory.models import ForgetPreview, ForgetReport
from memory.obs import configure_logging
from memory.retrieve import Mode

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    mode: Mode = "auto"  # "deep" forces the deep-mode search loop, "quick" never runs it


class ChatResponse(BaseModel):
    reply: str
    model: str
    prompt_version: str
    memory_tokens: int
    mode: str


class ForgetRequest(BaseModel):
    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1, description="What to forget, in the user's words")


class ForgetDecision(BaseModel):
    user_id: str = Field(min_length=1)
    preview_id: str = Field(min_length=1)


class ForgetPreviewResponse(BaseModel):
    preview: ForgetPreview
    message: str  # what to show the user before asking for confirmation


def create_app(client: MemoryClient | None = None) -> FastAPI:
    """Build the app. A provided client is used as-is (and not closed by the app)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if client is not None:
            yield
            return
        settings = get_settings()
        configure_logging(settings)
        app.state.client = MemoryClient.from_settings(settings)
        try:
            app.state.client.attach_queue(
                await create_pool(RedisSettings.from_dsn(settings.redis_url))
            )
        except (OSError, RedisError):
            log.warning("Redis unavailable: turns are stored but not extracted", exc_info=True)
        try:
            yield
        finally:
            await app.state.client.close()

    app = FastAPI(title="memory-layer", lifespan=lifespan)
    if client is not None:
        app.state.client = client

    def get_client(request: Request) -> MemoryClient:
        c: MemoryClient = request.app.state.client
        return c

    @app.get("/health")
    async def health(request: Request) -> dict[str, str]:
        c = get_client(request)
        await c.store.count_turns("__health__")
        return {"status": "ok"}

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request, tasks: BackgroundTasks) -> ChatResponse:
        c = get_client(request)
        tz = c.settings.default_timezone
        asked_at = now_in(tz)
        reply = await c.respond(
            body.user_id, body.session_id, body.message, now=asked_at, mode=body.mode
        )
        # The write path runs after the response is sent (section 5).
        tasks.add_task(
            c.remember,
            body.user_id,
            body.session_id,
            body.message,
            reply.text,
            asked_at=asked_at,
            replied_at=now_in(tz),
        )
        return ChatResponse(
            reply=reply.text,
            model=reply.model,
            prompt_version=reply.prompt_version,
            memory_tokens=reply.memory.token_estimate,
            mode=reply.mode,
        )

    @app.post("/forget")
    async def forget(body: ForgetRequest, request: Request) -> ForgetPreviewResponse:
        """Step 1: show what would be deleted. Nothing is deleted here."""
        preview = await get_client(request).forget_preview(body.user_id, body.query)
        return ForgetPreviewResponse(preview=preview, message=preview.render())

    @app.post("/forget/confirm")
    async def forget_confirm(body: ForgetDecision, request: Request) -> ForgetReport:
        """Step 2, after the user said yes: permanently delete what the preview showed."""
        report = await get_client(request).forget_confirm(body.user_id, body.preview_id)
        if report is None:
            raise HTTPException(404, "unknown or expired preview; request a new one")
        return report

    @app.post("/forget/cancel")
    async def forget_cancel(body: ForgetDecision, request: Request) -> dict[str, bool]:
        return {"cancelled": await get_client(request).forget_cancel(body.user_id, body.preview_id)}

    @app.get("/export")
    async def export(user_id: str, request: Request) -> dict[str, Any]:
        return await get_client(request).export(user_id)

    return app


app = create_app()
