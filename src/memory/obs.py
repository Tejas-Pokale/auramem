"""Observability (PLAN.md section 8, Phase 8): structured JSON logs and a per-call LLM trace.

Every LLM call (``memory.graphs.llm``) emits one ``llm_call`` record on the ``memory.trace``
logger: model, call type (output schema), prompt id, token counts including prompt-cache reads
and writes, latency, outcome, and the cost when prices are configured (LLM_PRICES, USD per
million tokens). Set TRACE_FILE to also append the records to a JSONL file, and
LOG_FORMAT=json to make every log line JSON.
"""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memory.config import Settings, get_settings

TRACE = logging.getLogger("memory.trace")
_prompt: ContextVar[str | None] = ContextVar("memory_prompt", default=None)

# Output schema -> the settings field holding its prompt version (the default prompt id).
_PROMPTS = {
    "ExtractionResult": ("extract", "extractor_prompt_version"),
    "AnswerOutput": ("answer", "answer_prompt_version"),
    "DeepStep": ("deep", "deep_prompt_version"),
    "QueryPlan": ("plan", "planner_prompt_version"),
    "ReflectionOutput": ("reflect", "reflector_prompt_version"),
    "EntitySummaryOutput": ("entity_summary", "entity_summary_prompt_version"),
    "EntityMergeDecision": ("entity_tiebreak", "entity_tiebreak_prompt_version"),
}


@contextmanager
def prompt_label(prompt_id: str) -> Iterator[None]:
    """Name the prompt for LLM calls made inside the block (when the schema is ambiguous)."""
    token = _prompt.set(prompt_id)
    try:
        yield
    finally:
        _prompt.reset(token)


def prompt_id_for(schema: str, settings: Settings) -> str | None:
    if (label := _prompt.get()) is not None:
        return label
    if schema not in _PROMPTS:
        return None
    name, field = _PROMPTS[schema]
    return f"{name}_{getattr(settings, field)}"


def cost_usd(model: str, usage: dict[str, int], settings: Settings) -> float | None:
    prices = settings.llm_prices.get(model)
    if prices is None:
        return None
    per_token = {k: v / 1_000_000 for k, v in prices.items()}
    cached = usage.get("cache_read_tokens", 0) + usage.get("cache_write_tokens", 0)
    fresh = max(0, usage.get("input_tokens", 0) - cached)
    return round(
        fresh * per_token.get("input", 0.0)
        + usage.get("cache_read_tokens", 0) * per_token.get("cache_read", 0.0)
        + usage.get("cache_write_tokens", 0) * per_token.get("cache_write", 0.0)
        + usage.get("output_tokens", 0) * per_token.get("output", 0.0),
        6,
    )


def trace_llm_call(
    *,
    model: str,
    schema: str,
    usage: dict[str, int],
    started: float,
    error: BaseException | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    record: dict[str, Any] = {
        "event": "llm_call",
        "at": datetime.now(UTC).isoformat(),
        "model": model,
        "call": schema,
        "prompt": prompt_id_for(schema, s),
        **usage,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "ok": error is None,
        "error": type(error).__name__ if error is not None else None,
        "cost_usd": cost_usd(model, usage, s),
    }
    line = json.dumps(record)
    TRACE.info(line)
    if s.trace_file:
        path = Path(s.trace_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return record


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "at": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        message = record.getMessage()
        try:  # trace records are JSON already: merge them instead of nesting a string
            parsed = json.loads(message) if message.startswith("{") else None
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            payload.update(parsed)
        else:
            payload["message"] = message
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(settings: Settings | None = None) -> None:
    """Root logging for the API and the worker: JSON lines when LOG_FORMAT=json."""
    s = settings or get_settings()
    handler = logging.StreamHandler()
    if s.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(s.log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
