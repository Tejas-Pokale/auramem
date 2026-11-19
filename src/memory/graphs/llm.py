"""Chat model construction and structured calls.

Every LLM call goes through ``structured`` so it returns a validated Pydantic model.
Structured output uses the API's native JSON-schema format (``method="json_schema"``):
Sonnet 5.5 rejects the forced tool choice LangChain's default method relies on.
Haiku 5.5 and Sonnet 5.5 also reject non-default temperature, so only effort is set.
Every call emits an ``llm_call`` trace record (model, call type, prompt id, tokens, latency,
cost when prices are configured); see ``memory.obs``.
"""

import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from memory.config import Settings
from memory.obs import trace_llm_call


def chat_model(model: str, settings: Settings, *, effort: str, max_tokens: int) -> ChatAnthropic:
    return ChatAnthropic(
        model=model,
        api_key=settings.anthropic_api_key,
        max_tokens=max_tokens,
        effort=effort,
        max_retries=settings.llm_max_retries,
    )


def structured[M: BaseModel](llm: ChatAnthropic, schema: type[M]) -> "StructuredCall[M]":
    return StructuredCall(llm, schema)


Usage = dict[str, int]


async def _traced[M: BaseModel](
    runnable: Runnable[list[BaseMessage], object],
    model: str,
    schema: type[M],
    messages: list[BaseMessage],
) -> tuple[M, Usage]:
    """One LLM call: parse the output, read token usage, and emit an ``llm_call`` trace."""
    started = time.perf_counter()
    usage: Usage = {}
    try:
        result = await runnable.ainvoke(messages)
        assert isinstance(result, dict)
        meta = getattr(result["raw"], "usage_metadata", None) or {}
        details = meta.get("input_token_details") or {}
        usage = {
            "input_tokens": int(meta.get("input_tokens") or 0),
            "output_tokens": int(meta.get("output_tokens") or 0),
            "cache_read_tokens": int(details.get("cache_read") or 0),
            "cache_write_tokens": int(details.get("cache_creation") or 0),
        }
        if result.get("parsing_error") is not None:
            raise ValueError(f"structured output failed to parse: {result['parsing_error']}")
        parsed = schema.model_validate(result["parsed"])
    except BaseException as e:
        trace_llm_call(model=model, schema=schema.__name__, usage=usage, started=started, error=e)
        raise
    trace_llm_call(model=model, schema=schema.__name__, usage=usage, started=started)
    return parsed, usage


class StructuredCall[M: BaseModel]:
    """Returns the validated Pydantic output; every call is traced (``memory.obs``)."""

    def __init__(self, llm: ChatAnthropic, schema: type[M]) -> None:
        self._runnable = llm.with_structured_output(schema, method="json_schema", include_raw=True)
        self._model = llm.model
        self._schema = schema

    async def __call__(self, messages: list[BaseMessage]) -> M:
        parsed, _ = await _traced(self._runnable, self._model, self._schema, messages)
        return parsed


def structured_with_usage[M: BaseModel](
    llm: ChatAnthropic, schema: type[M]
) -> "StructuredUsageCall[M]":
    return StructuredUsageCall(llm, schema)


class StructuredUsageCall[M: BaseModel]:
    """Like StructuredCall, but also returns token usage (including prompt-cache reads)."""

    def __init__(self, llm: ChatAnthropic, schema: type[M]) -> None:
        self._runnable = llm.with_structured_output(schema, method="json_schema", include_raw=True)
        self._model = llm.model
        self._schema = schema

    async def __call__(self, messages: list[BaseMessage]) -> tuple[M, Usage]:
        return await _traced(self._runnable, self._model, self._schema, messages)
