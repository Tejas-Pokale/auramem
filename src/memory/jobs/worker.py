"""arq worker for the write path.

uv run arq memory.jobs.worker.WorkerSettings
"""

import sys
from typing import Any, ClassVar

from arq import cron, func
from arq.connections import RedisSettings

from memory.config import get_settings
from memory.embed import Embedder
from memory.entities import ClaudeTiebreaker
from memory.extract import ExtractionDeps, Extractor, ExtractorLLM
from memory.jobs.queue import (
    CONSOLIDATION_JOB,
    ENTITY_SUMMARY_JOB,
    EXTRACT_JOB,
    IDLE_EXTRACT_JOB,
    REFLECT_JOB,
    consolidation_job,
    entity_summary_job,
    extract_job,
    idle_extract_job,
    reflect_job,
)
from memory.obs import configure_logging
from memory.observe import claude_reflector, claude_summarizer
from memory.store.neo4j import Neo4jStore


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings)
    ctx["deps"] = ExtractionDeps(
        store=Neo4jStore.connect(settings),
        embedder=Embedder(settings),
        extractor=Extractor(ExtractorLLM(settings), settings),
        settings=settings,
        tiebreaker=ClaudeTiebreaker(settings),
    )
    ctx["reflector"] = claude_reflector(settings)
    ctx["summarizer"] = claude_summarizer(settings)


async def shutdown(ctx: dict[str, Any]) -> None:
    deps: ExtractionDeps = ctx["deps"]
    await deps.store.close()
    if isinstance(deps.embedder, Embedder):
        await deps.embedder.aclose()


_settings = get_settings()


class WorkerSettings:
    functions: ClassVar[list[Any]] = [
        func(extract_job, name=EXTRACT_JOB, timeout=_settings.worker_job_timeout_seconds),
        func(idle_extract_job, name=IDLE_EXTRACT_JOB, timeout=_settings.worker_job_timeout_seconds),
        func(reflect_job, name=REFLECT_JOB, timeout=_settings.worker_job_timeout_seconds),
        func(
            entity_summary_job,
            name=ENTITY_SUMMARY_JOB,
            timeout=_settings.worker_job_timeout_seconds,
        ),
    ]
    cron_jobs: ClassVar[list[Any]] = [
        cron(
            consolidation_job,
            name=CONSOLIDATION_JOB,
            hour={_settings.consolidation_hour},
            minute={0},
            timeout=_settings.consolidation_timeout_seconds,
            unique=True,
        )
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(_settings.redis_url)
    max_jobs = _settings.worker_max_jobs
    keep_result = 0  # results would block re-enqueueing the same per-user job id
    # Windows' event loop has no signal handlers; Ctrl+C still stops the worker there.
    handle_signals = sys.platform != "win32"
