import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from memory.config import Settings
from memory.embed import DOCUMENT_PROMPT, QUERY_PROMPT, Embedder, EmbeddingError

DIMS = 4


def make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "embedding_dims": DIMS,
        "embedding_batch_size": 2,
        "ollama_url": "http://ollama.test",
    }
    return Settings(_env_file=None, **(defaults | overrides))


class FakeOllama:
    """Records /api/embed requests and returns one deterministic vector per input."""

    def __init__(self, dims: int = DIMS, model: str = "embeddinggemma:300m") -> None:
        self.requests: list[dict[str, Any]] = []
        self.dims = dims
        self.model = model

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        body = json.loads(request.content)
        self.requests.append(body)
        vectors = [[float(len(text))] + [0.0] * (self.dims - 1) for text in body["input"]]
        return httpx.Response(200, json={"model": self.model, "embeddings": vectors})


def make_embedder(fake: FakeOllama, **overrides: Any) -> Embedder:
    settings = make_settings(**overrides)
    client = httpx.AsyncClient(
        base_url=settings.ollama_url, transport=httpx.MockTransport(fake.handler)
    )
    return Embedder(settings, client=client)


def test_prompts_match_claude_md() -> None:
    assert QUERY_PROMPT == "task: search result | query: "
    assert DOCUMENT_PROMPT == "title: none | text: "


@pytest.mark.parametrize(
    ("method", "prefix"),
    [("embed_query", QUERY_PROMPT), ("embed_documents", DOCUMENT_PROMPT)],
)
async def test_prepends_prompt(method: str, prefix: str) -> None:
    fake = FakeOllama()
    embedder = make_embedder(fake)
    call: Callable[[list[str]], Any] = getattr(embedder, method)

    result = await call(["where do I work?"])

    assert fake.requests[0]["input"] == [prefix + "where do I work?"]
    assert fake.requests[0]["model"] == "embeddinggemma:300m"
    assert result.model == "embeddinggemma:300m"
    assert len(result.vectors) == 1


async def test_batches_and_preserves_order() -> None:
    fake = FakeOllama()
    embedder = make_embedder(fake, embedding_batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    result = await embedder.embed_documents(texts)

    assert [len(r["input"]) for r in fake.requests] == [2, 2, 1]
    expected_lengths = [len(DOCUMENT_PROMPT + t) for t in texts]
    assert [v[0] for v in result.vectors] == expected_lengths


async def test_empty_input_makes_no_request() -> None:
    fake = FakeOllama()
    result = await make_embedder(fake).embed_query([])
    assert result.vectors == []
    assert fake.requests == []


async def test_wrong_dimensions_raise() -> None:
    fake = FakeOllama(dims=DIMS + 1)
    with pytest.raises(EmbeddingError, match="dimensions"):
        await make_embedder(fake).embed_documents(["x"])


async def test_count_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "m", "embeddings": []})

    settings = make_settings()
    client = httpx.AsyncClient(base_url=settings.ollama_url, transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="returned 0 vectors for 1"):
        await Embedder(settings, client=client).embed_documents(["x"])


async def test_http_error_raises_embedding_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    settings = make_settings()
    client = httpx.AsyncClient(base_url=settings.ollama_url, transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="404"):
        await Embedder(settings, client=client).embed_query(["x"])
