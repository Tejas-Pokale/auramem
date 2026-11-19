"""Deterministic stand-ins for the embedding model and the answering LLM.

The database is never faked (CLAUDE.md); only model calls are.
"""

import hashlib
import math
import re
from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from memory.config import get_settings
from memory.embed import Embeddings
from memory.models import (
    AnswerOutput,
    EntityMention,
    EntityType,
    ExtractedFact,
    ExtractionResult,
    FactType,
)

_WORD = re.compile(r"[a-z0-9]+")


def bag_of_words_vector(text: str, dims: int) -> list[float]:
    """Hashed bag of words, L2-normalized: shared words mean high cosine similarity."""
    v = [0.0] * dims
    for word in _WORD.findall(text.lower()):
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v] if any(v) else [1.0 / math.sqrt(dims)] * dims


class FakeEmbedder:
    model = "fake-bow"

    def __init__(self, dims: int | None = None) -> None:
        self.dims = dims or get_settings().embedding_dims
        self.calls: list[tuple[str, list[str]]] = []

    async def embed_query(self, texts: Sequence[str]) -> Embeddings:
        self.calls.append(("query", list(texts)))
        return Embeddings(
            vectors=[bag_of_words_vector(t, self.dims) for t in texts], model=self.model
        )

    async def embed_documents(self, texts: Sequence[str]) -> Embeddings:
        self.calls.append(("document", list(texts)))
        return Embeddings(
            vectors=[bag_of_words_vector(t, self.dims) for t in texts], model=self.model
        )


class FakeAnswerer:
    """Records the prompt it was given and echoes a fixed answer."""

    model = "fake-answerer"

    def __init__(self, answer: str = "fake answer") -> None:
        self.answer = answer
        self.prompts: list[list[BaseMessage]] = []

    async def __call__(self, messages: list[BaseMessage]) -> tuple[AnswerOutput, dict[str, int]]:
        self.prompts.append(messages)
        return AnswerOutput(answer=self.answer), {}


_NEW_TURN = re.compile(r"^\[seq (\d+)\] .*? \| user \(typed\): (.+)$")


class FakeExtractorLLM:
    """One user-attributed fact per typed user turn in the NEW section; records calls.

    ``fail`` makes every call raise a validation-style error (to test retries/failures).
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[list[BaseMessage]] = []
        self.fail = fail

    async def __call__(self, messages: list[BaseMessage]) -> ExtractionResult:
        self.calls.append(messages)
        if self.fail:
            raise ValueError("schema mismatch")
        prompt = str(messages[-1].content)
        new_section = prompt.split("NEW turns to extract from:", 1)[1]
        facts = []
        for line in new_section.strip().splitlines():
            if m := _NEW_TURN.match(line):
                seq, text = int(m.group(1)), m.group(2)
                names = [w.strip(".,!?") for w in text.split()[1:] if w[:1].isupper()]
                facts.append(
                    ExtractedFact(
                        text=f"The user said: {text}",
                        fact_type=FactType.state,
                        attributed_to="user",
                        subject="user",
                        entities=[
                            EntityMention(
                                surface=n, canonical_guess=n, entity_type=EntityType.organization
                            )
                            for n in names
                        ],
                        aliases=[text.lower()],
                        confidence=0.9,
                        source_turn_seqs=[seq],
                    )
                )
        return ExtractionResult(facts=facts, observations=[])
