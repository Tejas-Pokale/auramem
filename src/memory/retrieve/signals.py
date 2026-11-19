"""Retrieval signals (PLAN.md section 6, step 3). Each returns up to ``k`` ranked hits.

Facts: S1 (vector), S3 (BM25), S6 (event time overlapping the planner's range).
Turns: S2 (vector), S4 (BM25), S6t (S2 restricted in-index to turns created in the range).
Entities: S5 (facts mentioning entities named in the message). Graph: S7 (one hop from
the top facts via shared entities and SUPERSEDES).
"""

import asyncio
from collections.abc import Sequence

from memory.models import Fact, Hit, TimeRange, Turn
from memory.store.neo4j import Neo4jStore

S2_VECTOR_TURNS = "S2_vector_turns"
S4_BM25_TURNS = "S4_bm25_turns"
S6T_TEMPORAL_TURNS = "S6t_temporal_turns"


async def turn_signals(
    store: Neo4jStore,
    user_id: str,
    *,
    query_vector: Sequence[float],
    query_text: str,
    k: int,
    suffix: str = "",
) -> dict[str, list[Hit[Turn]]]:
    vector, bm25 = await asyncio.gather(
        store.vector_search_turns(user_id, query_vector, k),
        store.fulltext_search_turns(user_id, query_text, k),
    )
    return {S2_VECTOR_TURNS + suffix: vector, S4_BM25_TURNS + suffix: bm25}


async def temporal_turn_signal(
    store: Neo4jStore,
    user_id: str,
    *,
    query_vector: Sequence[float],
    time_range: TimeRange,
    k: int,
) -> dict[str, list[Hit[Turn]]]:
    hits = await store.vector_search_turns(
        user_id,
        query_vector,
        k,
        created_from=time_range.start,
        created_to=time_range.end,
    )
    return {S6T_TEMPORAL_TURNS: hits}


S1_VECTOR_FACTS = "S1_vector_facts"
S3_BM25_FACTS = "S3_bm25_facts"
S6_TEMPORAL_FACTS = "S6_temporal_facts"


async def fact_signals(
    store: Neo4jStore,
    user_id: str,
    *,
    query_vector: Sequence[float],
    query_text: str,
    k: int,
    suffix: str = "",
) -> dict[str, list[Hit[Fact]]]:
    """S1 + S3. ``is_latest`` is not filtered so history stays reachable; hypotheticals are."""
    vector, bm25 = await asyncio.gather(
        store.vector_search_facts(user_id, query_vector, k),
        store.fulltext_search_facts(user_id, query_text, k),
    )
    return {S1_VECTOR_FACTS + suffix: vector, S3_BM25_FACTS + suffix: bm25}


async def temporal_fact_signal(
    store: Neo4jStore, user_id: str, *, time_range: TimeRange, k: int, scan_limit: int
) -> dict[str, list[Hit[Fact]]]:
    """S6: facts whose event time overlaps the planner's range, scored by overlap."""
    hits = await store.temporal_search_facts(
        user_id, time_range.start, time_range.end, k, scan_limit=scan_limit
    )
    return {S6_TEMPORAL_FACTS: hits}


S5_ENTITY_FACTS = "S5_entity_facts"
S7_GRAPH_FACTS = "S7_graph_facts"


async def entity_signal(
    store: Neo4jStore, user_id: str, *, names: Sequence[str], k: int
) -> dict[str, list[Hit[Fact]]]:
    """S5: facts that mention (or are about) entities named in the message.

    Names resolve to the user's entities by exact name/alias, then by the best full-text
    entity match for names that did not match exactly.
    """
    if not names:
        return {S5_ENTITY_FACTS: []}
    ids = await resolve_entity_ids(store, user_id, names)
    return {S5_ENTITY_FACTS: await store.facts_mentioning(user_id, ids, k)}


async def resolve_entity_ids(store: Neo4jStore, user_id: str, names: Sequence[str]) -> list[str]:
    """The user's entities for ``names``: exact name/alias match, else the best full-text match."""
    exact = await store.find_entities_by_names(user_id, names)
    matched = {n.lower() for e in exact for n in [e.name, *e.aliases]}
    ids = [e.entity_id for e in exact]
    for name in names:
        if name.lower() not in matched:
            hits = await store.fulltext_search_entities(user_id, name, 1)
            ids += [h.item.entity_id for h in hits]
    return ids


async def graph_signal(
    store: Neo4jStore, user_id: str, *, seed_fact_ids: Sequence[str], k: int
) -> dict[str, list[Hit[Fact]]]:
    """S7: facts one hop from the top facts (shared entities, SUPERSEDES either way)."""
    return {S7_GRAPH_FACTS: await store.expand_facts(user_id, seed_fact_ids, k)}
