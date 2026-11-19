"""EmbeddingGemma via Ollama, with task prompts prepended by us (CLAUDE.md).

Ollama does not apply EmbeddingGemma's prompt templates, so every text is prefixed here:
queries with ``QUERY_PROMPT`` and stored documents with ``DOCUMENT_PROMPT``. Mixing the
two up silently degrades retrieval, so callers choose by method, never by flag.
"""

from collections.abc import Sequence
from types import TracebackType
from typing import Protocol, Self

import httpx
from pydantic import BaseModel

from memory.config import Settings

QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "


class EmbeddingError(RuntimeError):
    pass


class Embeddings(BaseModel):
    vectors: list[list[float]]
    model: str


class _OllamaEmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]


class EmbedderLike(Protocol):
    async def embed_query(self, texts: Sequence[str]) -> Embeddings: ...

    async def embed_documents(self, texts: Sequence[str]) -> Embeddings: ...


class Embedder:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._model = settings.embedding_model
        self._dims = settings.embedding_dims
        self._batch_size = settings.embedding_batch_size
        self._client = client or httpx.AsyncClient(
            base_url=settings.ollama_url, timeout=settings.ollama_timeout_seconds
        )

    async def embed_query(self, texts: Sequence[str]) -> Embeddings:
        return await self._embed([QUERY_PROMPT + t for t in texts])

    async def embed_documents(self, texts: Sequence[str]) -> Embeddings:
        return await self._embed([DOCUMENT_PROMPT + t for t in texts])

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _embed(self, inputs: list[str]) -> Embeddings:
        vectors: list[list[float]] = []
        model = self._model
        for start in range(0, len(inputs), self._batch_size):
            batch = inputs[start : start + self._batch_size]
            result = await self._request(batch)
            if len(result.embeddings) != len(batch):
                raise EmbeddingError(
                    f"Ollama returned {len(result.embeddings)} vectors for {len(batch)} inputs"
                )
            for vector in result.embeddings:
                if len(vector) != self._dims:
                    raise EmbeddingError(
                        f"expected {self._dims} dimensions from {result.model}, got {len(vector)}"
                    )
            vectors.extend(result.embeddings)
            model = result.model
        return Embeddings(vectors=vectors, model=model)

    async def _request(self, batch: list[str]) -> _OllamaEmbedResponse:
        try:
            response = await self._client.post(
                "/api/embed", json={"model": self._model, "input": batch}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(
                f"Ollama /api/embed failed with {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.HTTPError as e:
            raise EmbeddingError(f"Ollama /api/embed request failed: {e!r}") from e
        return _OllamaEmbedResponse.model_validate(response.json())
