"""Entity resolution (PLAN.md section 5, step 7).

1. Exact match on a normalized name or alias of the user's existing entities.
2. Otherwise score candidates found by vector and full-text search over the user's entities:
   score = w_string * string similarity (best of name and aliases)
         + w_embedding * cosine similarity of the names
         + w_cooccurrence * share of the batch's other entities already co-mentioned with it.
   score >= ENTITY_MERGE merges; ENTITY_LLM_BAND_LOW <= score < ENTITY_MERGE asks the LLM
   tiebreak (Sonnet 5.5); lower creates a new entity.
On a merge the new surface form becomes an alias, so it matches exactly next time.
"""

import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from memory.config import Settings
from memory.embed import EmbedderLike
from memory.graphs.llm import chat_model, structured
from memory.models import Entity, EntityMention, EntityMergeDecision, EntityType, NewEntity
from memory.prompts import load_prompt
from memory.store.neo4j import Neo4jStore

log = logging.getLogger(__name__)

Route = Literal["merge", "tiebreak", "new"]


@dataclass
class Resolution:
    """Mention key -> entity id, plus entities to create and aliases to add on merges."""

    ids: dict[str, str] = field(default_factory=dict)
    new: list[NewEntity] = field(default_factory=list)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    merged: int = 0
    tiebreaks: int = 0

    def entity_id(self, mention: EntityMention) -> str:
        return self.ids[mention_key(mention)]


def mention_key(mention: EntityMention) -> str:
    return mention.canonical_guess.strip().lower() or mention.surface.strip().lower()


def name_key(name: str) -> str:
    return name.strip().lower()


def _normalized(name: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", name.lower()).split())


def string_similarity(a: str, b: str) -> float:
    """Normalized edit similarity in [0, 1] (case, spacing and punctuation ignored)."""
    a, b = _normalized(a), _normalized(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def entity_score(
    name: str, candidate: Entity, *, cosine: float, cooccurrence: float, settings: Settings
) -> float:
    s = settings
    best = max(string_similarity(name, n) for n in [candidate.name, *candidate.aliases])
    return (
        s.entity_weight_string * best
        + s.entity_weight_embedding * max(0.0, cosine)
        + s.entity_weight_cooccurrence * cooccurrence
    )


def route(score: float, settings: Settings) -> Route:
    if score >= settings.entity_merge:
        return "merge"
    if score >= settings.entity_llm_band_low:
        return "tiebreak"
    return "new"


def _compatible(a: EntityType, b: EntityType) -> bool:
    return a == b or EntityType.other in (a, b)


class Tiebreaker(Protocol):
    async def __call__(self, messages: list[BaseMessage]) -> EntityMergeDecision: ...


class ClaudeTiebreaker:
    def __init__(self, settings: Settings) -> None:
        self._call = structured(
            chat_model(
                settings.entity_tiebreak_model,
                settings,
                effort=settings.entity_tiebreak_effort,
                max_tokens=settings.entity_tiebreak_max_tokens,
            ),
            EntityMergeDecision,
        )

    async def __call__(self, messages: list[BaseMessage]) -> EntityMergeDecision:
        return await self._call(messages)


@dataclass
class EntityResolver:
    store: Neo4jStore
    embedder: EmbedderLike
    settings: Settings
    tiebreaker: Tiebreaker | None = None

    async def resolve(
        self,
        user_id: str,
        mentions: Sequence[EntityMention],
        contexts: dict[str, str] | None = None,
    ) -> Resolution:
        """Map every mention to an entity id; ``contexts`` gives a fact sentence per mention."""
        contexts = contexts or {}
        names = [n for m in mentions for n in (m.surface, m.canonical_guess)]
        known = _index(await self.store.find_entities_by_names(user_id, names))
        out = Resolution()
        unresolved: dict[str, EntityMention] = {}
        for m in mentions:
            key = mention_key(m)
            if key in out.ids:
                continue
            existing = known.get(key) or known.get(name_key(m.surface))
            if existing:
                out.ids[key] = existing
            else:
                unresolved.setdefault(key, m)

        pending: dict[str, NewEntity] = {}
        for key, m in unresolved.items():
            if key in out.ids:
                continue
            match = await self._score_and_route(user_id, m, set(out.ids.values()), contexts, out)
            if match is not None:
                out.ids[key] = match
                out.merged += 1
                surface_forms = {m.surface.strip(), m.canonical_guess.strip()} - {""}
                out.aliases.setdefault(match, []).extend(sorted(surface_forms))
                continue
            aliases = [m.surface] if name_key(m.surface) != key else []
            pending[key] = NewEntity(
                entity_id=f"ent-{uuid.uuid4().hex}",
                name=m.canonical_guess.strip() or m.surface.strip(),
                entity_type=m.entity_type,
                aliases=aliases,
            )
            out.ids[key] = pending[key].entity_id
        for m in mentions:  # surface forms resolve too ("my sister" -> the Asha entity)
            out.ids.setdefault(name_key(m.surface), out.ids[mention_key(m)])
        out.new = list(pending.values())
        return out

    async def _score_and_route(
        self,
        user_id: str,
        mention: EntityMention,
        batch_entity_ids: set[str],
        contexts: dict[str, str],
        out: Resolution,
    ) -> str | None:
        s = self.settings
        name = mention.canonical_guess.strip() or mention.surface.strip()
        candidates = await self._candidates(user_id, name, mention.entity_type)
        if not candidates:
            return None
        co = await self.store.co_mentioned(user_id, [c.entity_id for c, _ in candidates])
        others = batch_entity_ids - {c.entity_id for c, _ in candidates}
        best: tuple[float, Entity] | None = None
        for candidate, cosine in candidates:
            shared = co.get(candidate.entity_id, set()) & others
            cooc = len(shared) / len(others) if others else 0.0
            score = entity_score(name, candidate, cosine=cosine, cooccurrence=cooc, settings=s)
            if best is None or score > best[0]:
                best = (score, candidate)
        assert best is not None
        score, candidate = best
        decision = route(score, s)
        if decision == "merge":
            return candidate.entity_id
        if decision == "tiebreak" and self.tiebreaker is not None:
            out.tiebreaks += 1
            if await self._tiebreak(user_id, mention, candidate, contexts):
                return candidate.entity_id
        return None

    async def duplicate_of(
        self, user_id: str, entity: Entity, *, exclude: set[str]
    ) -> tuple[Entity, str] | None:
        """For nightly consolidation: an existing entity that ``entity`` duplicates, with how
        it was decided ("merge" or "tiebreak"); same scoring and bands as at write time, with
        co-occurrence measured against the entities ``entity`` already appears with."""
        s = self.settings
        candidates = [
            (c, cos)
            for c, cos in await self._candidates(user_id, entity.name, entity.entity_type)
            if c.entity_id != entity.entity_id and c.entity_id not in exclude
        ]
        if not candidates:
            return None
        ids = [entity.entity_id, *(c.entity_id for c, _ in candidates)]
        co = await self.store.co_mentioned(user_id, ids)
        own = co.get(entity.entity_id, set())
        scored = []
        for candidate, cosine in candidates:
            shared = (co.get(candidate.entity_id, set()) & own) - {candidate.entity_id}
            cooc = len(shared) / len(own) if own else 0.0
            names = [entity.name, *entity.aliases]
            score = max(
                entity_score(n, candidate, cosine=cosine, cooccurrence=cooc, settings=s)
                for n in names
            )
            scored.append((score, candidate))
        score, candidate = max(scored, key=lambda x: x[0])
        decision = route(score, s)
        if decision == "merge":
            return candidate, "merge"
        if decision == "tiebreak" and self.tiebreaker is not None:
            mention = EntityMention(
                surface=entity.name, canonical_guess=entity.name, entity_type=entity.entity_type
            )
            context = {mention_key(mention): entity.summary or "(none)"}
            if await self._tiebreak(user_id, mention, candidate, context):
                return candidate, "tiebreak"
        return None

    async def _candidates(
        self, user_id: str, name: str, entity_type: EntityType
    ) -> list[tuple[Entity, float]]:
        """(entity, cosine of names) for vector and full-text candidates of a compatible type."""
        k = self.settings.entity_candidates
        vector = (await self.embedder.embed_documents([name])).vectors[0]
        by_vector = await self.store.vector_search_entities(user_id, vector, k)
        by_text = await self.store.fulltext_search_entities(user_id, name, k)
        found: dict[str, tuple[Entity, float | None]] = {}
        for hit in by_vector:
            # Neo4j reports cosine similarity rescaled to [0, 1] as (1 + cos) / 2.
            found[hit.item.entity_id] = (hit.item, 2 * hit.score - 1)
        for hit in by_text:
            found.setdefault(hit.item.entity_id, (hit.item, None))
        missing = [e for e, cos in found.values() if cos is None]
        if missing:
            vectors = (await self.embedder.embed_documents([e.name for e in missing])).vectors
            for e, v in zip(missing, vectors, strict=True):
                found[e.entity_id] = (e, _cosine(vector, v))
        return [
            (e, cos or 0.0)
            for e, cos in found.values()
            if _compatible(e.entity_type, entity_type) and not e.entity_id.endswith(":self")
        ]

    async def _tiebreak(
        self, user_id: str, mention: EntityMention, candidate: Entity, contexts: dict[str, str]
    ) -> bool:
        assert self.tiebreaker is not None
        facts = await self.store.facts_mentioning(
            user_id, [candidate.entity_id], self.settings.entity_tiebreak_facts
        )
        version = self.settings.entity_tiebreak_prompt_version
        user = load_prompt("entity_tiebreak", version, "user").text.format(
            mention=mention.surface,
            mention_type=mention.entity_type.value,
            mention_context=contexts.get(mention_key(mention), "(none)"),
            candidate=candidate.name,
            candidate_type=candidate.entity_type.value,
            aliases=", ".join(candidate.aliases) or "(none)",
            summary=candidate.summary or "(none)",
            facts="\n".join(f"- {h.item.text}" for h in facts) or "(none)",
        )
        messages: list[BaseMessage] = [
            SystemMessage(load_prompt("entity_tiebreak", version, "system").text),
            HumanMessage(user),
        ]
        try:
            decision = await self.tiebreaker(messages)
        except Exception:  # the tiebreak is an optimization; a new entity is the safe default
            log.warning("entity tiebreak failed; creating a new entity", exc_info=True)
            return False
        return decision.same_entity


def _index(entities: Sequence[Entity]) -> dict[str, str]:
    index: dict[str, str] = {}
    for e in entities:
        for name in [e.name, *e.aliases]:
            index.setdefault(name_key(name), e.entity_id)
    return index


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
