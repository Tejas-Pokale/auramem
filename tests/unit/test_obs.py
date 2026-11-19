"""Tracing: every LLM call emits one llm_call record with tokens, latency, prompt and cost."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import BaseMessage

from memory.config import get_settings
from memory.graphs.llm import _traced
from memory.models import AnswerOutput
from memory.obs import JsonFormatter, cost_usd, prompt_id_for, prompt_label

USAGE = {"input_tokens": 1000, "output_tokens": 200, "cache_read_tokens": 600,
         "cache_write_tokens": 0}  # fmt: skip


class Runnable:
    def __init__(self, parsed: Any, parsing_error: Exception | None = None) -> None:
        self.parsed = parsed
        self.parsing_error = parsing_error

    async def ainvoke(self, messages: list[BaseMessage]) -> dict[str, Any]:
        raw = SimpleNamespace(
            usage_metadata={
                "input_tokens": 1000, "output_tokens": 200,
                "input_token_details": {"cache_read": 600},
            }
        )  # fmt: skip
        return {"raw": raw, "parsed": self.parsed, "parsing_error": self.parsing_error}


def test_cost_needs_configured_prices() -> None:
    s = get_settings()
    assert cost_usd("m", USAGE, s) is None
    priced = s.model_copy(
        update={"llm_prices": {"m": {"input": 1.0, "output": 5.0, "cache_read": 0.1}}}
    )
    # 400 fresh input + 600 cache reads + 200 output, per million tokens
    assert cost_usd("m", USAGE, priced) == pytest.approx((400 * 1 + 600 * 0.1 + 200 * 5) / 1e6)


def test_prompt_id_from_schema_or_label() -> None:
    s = get_settings()
    assert prompt_id_for("AnswerOutput", s) == f"answer_{s.answer_prompt_version}"
    assert prompt_id_for("SomethingElse", s) is None
    with prompt_label("forget_v1"):
        assert prompt_id_for("ReflectionOutput", s) == "forget_v1"
    assert prompt_id_for("ReflectionOutput", s) == f"reflect_{s.reflector_prompt_version}"


async def test_traced_call_logs_one_record(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    trace_file = tmp_path / "trace.jsonl"
    monkeypatch.setenv("TRACE_FILE", str(trace_file))
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)  # later tests must not inherit TRACE_FILE
    with caplog.at_level(logging.INFO, logger="memory.trace"):
        parsed, usage = await _traced(
            cast(Any, Runnable({"answer": "hi"})),
            "claude-test",
            AnswerOutput,
            [],
        )
    assert parsed.answer == "hi" and usage["cache_read_tokens"] == 600
    [record] = [json.loads(r.getMessage()) for r in caplog.records]
    assert record["event"] == "llm_call" and record["model"] == "claude-test"
    assert record["call"] == "AnswerOutput" and record["ok"] is True
    assert record["input_tokens"] == 1000 and record["latency_ms"] >= 0
    assert json.loads(trace_file.read_text(encoding="utf-8").strip())["model"] == "claude-test"


async def test_failed_parse_is_traced_and_raised(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="memory.trace"), pytest.raises(ValueError):
        await _traced(
            cast(Any, Runnable(None, ValueError("bad json"))),
            "claude-test",
            AnswerOutput,
            [],
        )
    [record] = [json.loads(r.getMessage()) for r in caplog.records]
    assert record["ok"] is False and record["error"] == "ValueError"
    assert record["input_tokens"] == 1000  # tokens were still spent


def test_json_formatter_merges_trace_records() -> None:
    record = logging.LogRecord(
        "memory.trace", logging.INFO, "", 0, '{"event": "llm_call"}', None, None
    )
    out = json.loads(JsonFormatter().format(record))
    assert out["event"] == "llm_call" and out["level"] == "INFO"
    plain = logging.LogRecord("x", logging.WARNING, "", 0, "hello %s", ("you",), None)
    assert json.loads(JsonFormatter().format(plain))["message"] == "hello you"
