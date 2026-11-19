"""Smoke test against a real Ollama with embeddinggemma:300m.

Skipped (with a reason) when Ollama is not reachable, e.g. in CI. The prompt contract
itself is covered by tests/unit/test_embed.py.
"""

import math

import httpx
import pytest

from memory.config import get_settings
from memory.embed import Embedder

pytestmark = pytest.mark.ollama


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.fixture
async def embedder() -> Embedder:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=2) as client:
            (await client.get("/api/version")).raise_for_status()
    except httpx.HTTPError as e:
        pytest.skip(f"Ollama not reachable at {settings.ollama_url}: {e!r}")
    return Embedder(settings)


async def test_query_retrieves_matching_document(embedder: Embedder) -> None:
    async with embedder:
        docs = await embedder.embed_documents(
            [
                "I started a new job at JHEX as an ML engineer in October.",
                "My sister Asha has been vegetarian since college.",
                "I rent GPUs on Vast.ai now instead of RunPod.",
            ]
        )
        query = await embedder.embed_query(["Where do I work now?"])

    assert docs.model == get_settings().embedding_model
    assert all(len(v) == get_settings().embedding_dims for v in docs.vectors)
    scores = [_cosine(query.vectors[0], d) for d in docs.vectors]
    assert scores.index(max(scores)) == 0
