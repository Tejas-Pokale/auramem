"""Write path (PLAN.md section 5).

Phase 2 implements step 1 only: persist both turns of an exchange, embedded with the
EmbeddingGemma document prompt. Extraction (arq) arrives in Phase 4.
"""

import logging
from datetime import datetime
from typing import Literal

from memory.embed import EmbedderLike, EmbeddingError
from memory.models import NewTurn, Turn, TurnSource
from memory.store.neo4j import Neo4jStore

log = logging.getLogger(__name__)


def turn_id_for(session_id: str, seq: int) -> str:
    """Deterministic, so re-persisting an exchange after a crash is a no-op."""
    return f"{session_id}:{seq}"


async def persist_turn(
    store: Neo4jStore,
    embedder: EmbedderLike,
    *,
    user_id: str,
    session_id: str,
    role: Literal["user", "assistant"],
    text: str,
    at: datetime,
) -> Turn:
    """Save one turn on its own (e.g. the assistant's opening message of a conversation)."""
    await store.upsert_session(user_id, session_id, started_at=at)
    seq = await store.next_turn_seq(user_id, session_id)
    try:
        embedded = await embedder.embed_documents([text])
        vector: list[float] | None = embedded.vectors[0]
        model: str | None = embedded.model
    except EmbeddingError:
        log.exception("embedding failed; storing the turn without a vector")
        vector, model = None, None
    turn = NewTurn(
        turn_id=turn_id_for(session_id, seq),
        role=role,
        text=text,
        created_at=at,
        seq=seq,
        source="typed",
        embedding=vector,
        embedding_model=model,
    )
    [stored] = await store.upsert_turns(user_id, session_id, [turn])
    return stored


async def persist_exchange(
    store: Neo4jStore,
    embedder: EmbedderLike,
    *,
    user_id: str,
    session_id: str,
    user_text: str,
    assistant_text: str,
    user_at: datetime,
    assistant_at: datetime,
    user_source: TurnSource = "typed",
) -> list[Turn]:
    """Save the user and assistant turns (raw text first; embeddings when available).

    If embedding fails the turns are still stored without vectors: they stay findable by
    BM25, and the raw text is never lost (section 5, step 1).
    """
    await store.upsert_session(user_id, session_id, started_at=user_at)
    seq = await store.next_turn_seq(user_id, session_id)
    texts = [user_text, assistant_text]
    try:
        embedded = await embedder.embed_documents(texts)
        vectors: list[list[float] | None] = list(embedded.vectors)
        model: str | None = embedded.model
    except EmbeddingError:
        log.exception("embedding failed; storing turns without vectors")
        vectors, model = [None, None], None

    turns = [
        NewTurn(
            turn_id=turn_id_for(session_id, seq + i),
            role=role,
            text=text,
            created_at=at,
            seq=seq + i,
            source=source,
            embedding=vector,
            embedding_model=model if vector is not None else None,
        )
        for i, (role, text, at, source, vector) in enumerate(
            [
                ("user", user_text, user_at, user_source, vectors[0]),
                ("assistant", assistant_text, assistant_at, "typed", vectors[1]),
            ]
        )
    ]
    return await store.upsert_turns(user_id, session_id, turns)
