"""Nightly consolidation (PLAN.md section 5 "Background jobs", Phase 8).

Per user:
1. Entity resolution across the whole graph, to catch merges the write path missed (it only
   compares a new mention with existing entities). Same scoring, bands, and LLM tiebreak as
   at write time. The less-mentioned entity is merged into the more-mentioned one; facts are
   not changed apart from the entity they point to.
2. ``is_latest`` repair: a fact is latest exactly when nothing SUPERSEDES it (flags only).
3. Reports, without changes: slots where several facts are still latest, and orphaned facts
   (no source turn left).
"""

import logging
from dataclasses import dataclass, field

from memory.config import Settings
from memory.embed import EmbedderLike
from memory.entities import EntityResolver, Tiebreaker
from memory.store.neo4j import Neo4jStore

log = logging.getLogger(__name__)


@dataclass
class ConsolidationReport:
    user_id: str
    entities_checked: int = 0
    merged: int = 0
    tiebreak_merges: int = 0
    latest_fixed: int = 0
    slot_clashes: int = 0
    orphan_facts: int = 0
    merges: list[tuple[str, str]] = field(default_factory=list)  # (kept name, merged name)


async def consolidate_user(
    store: Neo4jStore,
    embedder: EmbedderLike,
    settings: Settings,
    user_id: str,
    *,
    tiebreaker: Tiebreaker | None = None,
) -> ConsolidationReport:
    report = ConsolidationReport(user_id=user_id)
    resolver = EntityResolver(store, embedder, settings, tiebreaker)
    gone: set[str] = set()
    entities = await store.list_entities(user_id)  # most mentioned first
    rank = {e.entity_id: n for n, e in enumerate(entities)}
    for entity in reversed(entities):  # fold rarely mentioned entities into frequent ones
        if entity.entity_id in gone:
            continue
        report.entities_checked += 1
        match = await resolver.duplicate_of(user_id, entity, exclude=gone)
        if match is None:
            continue
        other, how = match
        keep, drop = (
            (other, entity) if rank.get(other.entity_id, 0) <= rank[entity.entity_id] else
            (entity, other)
        )  # fmt: skip
        if await store.merge_entities(user_id, keep=keep.entity_id, drop=drop.entity_id):
            gone.add(drop.entity_id)
            report.merged += 1
            report.tiebreak_merges += how == "tiebreak"
            report.merges.append((keep.name, drop.name))
    report.latest_fixed = await store.repair_latest_flags(user_id)
    report.slot_clashes = await store.slot_clashes(user_id)
    report.orphan_facts = await store.orphan_facts(user_id)
    log.info(
        "consolidated %s: %d entities checked, %d merged (%d by tiebreak), %d latest flags "
        "fixed, %d slot clashes, %d orphan facts",
        user_id, report.entities_checked, report.merged, report.tiebreak_merges,
        report.latest_fixed, report.slot_clashes, report.orphan_facts,
    )  # fmt: skip
    return report
