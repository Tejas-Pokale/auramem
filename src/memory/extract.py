"""Fact extraction: write path steps 2-10 (PLAN.md section 5).

``run_extraction`` processes every session with turns past its extraction watermark, one
batch at a time (at most EXTRACT_BATCH_MAX_TURNS new turns, with EXTRACT_BATCH_OVERLAP
earlier turns as context). Each batch is gated, sent to the extractor (Haiku 5.5,
structured ``ExtractionResult``), post-processed into ``NewFact``s, embedded with its time
attached, linked to entities, and committed together with the new watermark in one
transaction. A crash or a failed extraction leaves the watermark unchanged, so the batch is
simply retried; the raw turns are already stored and searchable either way.
"""

import asyncio
import calendar
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from typing import Literal, Protocol

import anthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from memory.config import Settings
from memory.embed import EmbedderLike
from memory.entities import EntityResolver, Resolution, Tiebreaker, mention_key
from memory.graphs.llm import chat_model, structured
from memory.models import (
    EntityMention,
    EntityType,
    ExtractedFact,
    ExtractionResult,
    Fact,
    FactType,
    NewEntity,
    NewFact,
    Turn,
)
from memory.observe import format_observations
from memory.prompts import load_prompt
from memory.store.neo4j import Neo4jStore, VersionConflictError, self_entity_id

log = logging.getLogger(__name__)

_ACK = re.compile(
    r"^\W*(hi|hello|hey|hii+|yo|oyy|ok|okay|k|kk|cool|nice|great|thanks|thank you|thx|ty|"
    r"got it|sure|yes|yeah|yep|no|nope|hmm+|lol|bye|good (morning|night|evening)|noted|"
    r"understood|makes sense|good to know)\W*$",
    re.IGNORECASE,
)
_USER_SUBJECTS = {"user", "the user", "me", "i", "myself"}
PASTED_SOURCE = "third_party:pasted"

# Generic synonyms for common single-valued slots, so supersession can match facts that the
# extractor keyed differently. Applied after snake_case normalization.
SLOT_SYNONYMS = {
    "employer": "works_at",
    "company": "works_at",
    "current_company": "works_at",
    "workplace": "works_at",
    "current_employer": "works_at",
    "job": "job_title",
    "role": "job_title",
    "position": "job_title",
    "designation": "job_title",
    "pay": "salary",
    "package": "salary",
    "ctc": "salary",
    "compensation": "salary",
    "bond": "employment_bond",
    "location": "lives_in",
    "city": "lives_in",
    "current_city": "lives_in",
    "residence": "lives_in",
}
SUPERSEDING_TYPES = frozenset(
    {FactType.state, FactType.trait, FactType.preference, FactType.relationship}
)


def normalize_slot(key: str | None) -> str | None:
    """snake_case the extractor's attribute_key and map common synonyms to one name."""
    if not key:
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    return SLOT_SYNONYMS.get(slug, slug) or None


class _Slotted(Protocol):
    @property
    def fact_type(self) -> FactType: ...
    @property
    def subject_entity_id(self) -> str | None: ...
    @property
    def attribute_key(self) -> str | None: ...
    @property
    def is_hypothetical(self) -> bool: ...
    @property
    def start(self) -> datetime | None: ...
    @property
    def mentioned_at(self) -> datetime: ...
    @property
    def relation_to_existing(self) -> str: ...


def supersession(
    new: _Slotted, old: _Slotted
) -> Literal["new_wins", "old_wins", "conflict"] | None:
    """Which of two facts about the same single-valued attribute is current (PLAN.md 5.9).

    Only same subject + same slot, never hypotheticals, and only between states, traits,
    preferences and relationships: an event never supersedes a state ("ate chicken once"
    does not end "is vegetarian"). Newer = later start when both have one, else later mention.
    A fact the extractor marked as a contradiction (a different value for something that did
    not change) is linked as a conflict instead: both stay current and both are shown.
    """
    same_slot = (
        bool(new.attribute_key)
        and new.attribute_key == old.attribute_key
        and new.subject_entity_id is not None
        and new.subject_entity_id == old.subject_entity_id
        and not (new.is_hypothetical or old.is_hypothetical)
    )
    if not same_slot:
        return None
    if new.relation_to_existing == "contradiction":
        return "conflict"
    if new.fact_type not in SUPERSEDING_TYPES or old.fact_type not in SUPERSEDING_TYPES:
        return None
    if new.start is not None and old.start is not None and new.start != old.start:
        return "new_wins" if new.start > old.start else "old_wins"
    if new.mentioned_at == old.mentioned_at:
        return None
    return "new_wins" if new.mentioned_at > old.mentioned_at else "old_wins"


class ExtractionFailedError(RuntimeError):
    pass


def _is_transient(e: anthropic.APIError) -> bool:
    if isinstance(e, anthropic.APIConnectionError | anthropic.RateLimitError):
        return True
    return isinstance(e, anthropic.APIStatusError) and e.status_code >= _SERVER_ERROR


_SERVER_ERROR = 500


# --- Gate and formatting -------------------------------------------------------------------


def is_gated(turn: Turn, *, max_tokens: int) -> bool:
    """Short greetings/acks carry nothing to extract (they stay searchable as turns)."""
    return len(turn.text.split()) <= max_tokens and bool(_ACK.match(turn.text))


def _stamp(at: datetime) -> str:
    return f"{at:%Y-%m-%d %H:%M} {at.tzname() or 'UTC'} ({at:%A})"


def format_turns(turns: Sequence[Turn]) -> str:
    return "\n".join(
        f"[seq {t.seq}] {_stamp(t.created_at)} | {t.role} ({t.source}): {t.text}" for t in turns
    )


# --- Extractor ---------------------------------------------------------------------------


def _related_line(f: Fact) -> str:
    """Existing fact as shown to the extractor: its slot key, so new facts reuse it."""
    slot = f"[slot {f.attribute_key}] " if f.attribute_key else ""
    status = "" if f.is_latest else "(no longer current) "
    return f"- {slot}{status}{f.text}"


class ExtractorLLM:
    """Structured Haiku call; ``__call__`` is the seam tests replace."""

    def __init__(self, settings: Settings) -> None:
        self._call = structured(
            chat_model(
                settings.extractor_model,
                settings,
                effort=settings.extractor_effort,
                max_tokens=settings.extractor_max_tokens,
            ),
            ExtractionResult,
        )

    async def __call__(self, messages: list[BaseMessage]) -> ExtractionResult:
        return await self._call(messages)


@dataclass
class Extractor:
    llm: ExtractorLLM
    settings: Settings

    @property
    def version(self) -> str:
        return f"extract_{self.settings.extractor_prompt_version}"

    def messages(
        self,
        *,
        session_start: datetime,
        context: Sequence[Turn],
        new: Sequence[Turn],
        observations: str,
        related: Sequence[Fact],
    ) -> list[BaseMessage]:
        version = self.settings.extractor_prompt_version
        user = load_prompt("extract", version, "user").text.format(
            session_start=_stamp(session_start),
            observations=observations or "(none yet)",
            related="\n".join(_related_line(f) for f in related) or "(none)",
            context=format_turns(context) or "(none)",
            turns=format_turns(new),
        )
        return [SystemMessage(load_prompt("extract", version, "system").text), HumanMessage(user)]

    async def extract(self, messages: list[BaseMessage]) -> ExtractionResult:
        """Call the extractor with two independent retry budgets.

        - Schema/validation errors: EXTRACT_SCHEMA_RETRIES retries (PLAN.md section 5).
        - Transient API errors (connection, 429, 5xx): EXTRACT_TRANSIENT_RETRIES retries with
          exponential backoff, on top of the SDK's own retries.
        Anything else, or an exhausted budget, raises ExtractionFailedError: the batch is
        left unextracted (turns stay searchable) and retried by a later job.
        """
        s = self.settings
        schema_failures = transient_failures = 0
        while True:
            try:
                return await self.llm(messages)
            except (ValidationError, ValueError) as e:
                schema_failures += 1
                log.warning("extraction output invalid (%d): %s", schema_failures, e)
                if schema_failures > s.extract_schema_retries:
                    raise ExtractionFailedError("extraction output failed validation") from e
            except anthropic.APIError as e:
                if not _is_transient(e) or transient_failures >= s.extract_transient_retries:
                    raise ExtractionFailedError(f"extraction API call failed: {e}") from e
                delay = s.extract_retry_backoff_seconds * 2**transient_failures
                transient_failures += 1
                log.warning("transient extraction error (%s); retrying in %.1fs", e, delay)
                await asyncio.sleep(delay)


# --- Post-processing ----------------------------------------------------------------------


def _aware(value: datetime | None, tz: tzinfo | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=tz)


def time_phrase(
    start: datetime | None, precision: str, status: str | None, mentioned_at: datetime
) -> str:
    """Readable time for ``text_for_embedding`` (section 5, step 6)."""
    if start is None:
        m = mentioned_at
        return f"mentioned {m.day} {calendar.month_name[m.month]} {m.year}"
    day = f"{start.day} {calendar.month_name[start.month]} {start.year}"
    month = f"{calendar.month_name[start.month]} {start.year}"
    phrase = {
        "exact": f"on {day} at {start:%H:%M}",
        "day": f"on {day}",
        "week": f"in the week of {day}",
        "month": f"in {month}",
        "year": f"in {start.year}",
    }.get(precision, f"around {month}")
    return f"planned {phrase}" if status == "planned" else phrase


def subject_mentions(result: ExtractionResult) -> list[EntityMention]:
    """Every entity the facts refer to, including non-user subjects and third parties."""
    out: list[EntityMention] = []
    for f in result.facts:
        out.extend(f.entities)
        for name in (f.subject, f.third_party_name):
            if name and name.strip().lower() not in _USER_SUBJECTS:
                key = name.strip().lower()
                known = next(
                    (m for m in f.entities if key in (mention_key(m), m.surface.strip().lower())),
                    None,
                )
                out.append(
                    known
                    or EntityMention(
                        surface=name, canonical_guess=name, entity_type=EntityType.other
                    )
                )
    return out


def mention_contexts(result: ExtractionResult) -> dict[str, str]:
    """First fact sentence each entity appears in (context for the entity tiebreak)."""
    out: dict[str, str] = {}
    for f in result.facts:
        for m in f.entities:
            out.setdefault(mention_key(m), f.text)
    return out


def _entity_id_for(name: str, resolution: Resolution) -> str | None:
    return resolution.ids.get(name.strip().lower())


@dataclass
class Postprocessed:
    facts: list[NewFact]
    dropped_low_confidence: int = 0
    dropped_no_source: int = 0


def postprocess(
    result: ExtractionResult,
    *,
    user_id: str,
    new_turns: Sequence[Turn],
    resolution: Resolution,
    settings: Settings,
    extractor_version: str,
) -> Postprocessed:
    """ExtractedFact -> NewFact: filters, attribution, ids, and time-augmented text."""
    by_seq = {t.seq: t for t in new_turns}
    tz = new_turns[-1].created_at.tzinfo
    out = Postprocessed(facts=[])
    for f in result.facts:
        if f.confidence < settings.min_fact_confidence:
            out.dropped_low_confidence += 1
            continue
        sources = [by_seq[s] for s in dict.fromkeys(f.source_turn_seqs) if s in by_seq]
        if not sources:
            out.dropped_no_source += 1
            continue

        subject_is_user = f.subject.strip().lower() in _USER_SUBJECTS
        subject_id = (
            self_entity_id(user_id) if subject_is_user else _entity_id_for(f.subject, resolution)
        )
        out.facts.append(
            _to_new_fact(
                f,
                sources=sources,
                subject_id=subject_id,
                attributed_to=_attribution(f, sources, resolution, subject_id, subject_is_user),
                mention_ids=[resolution.entity_id(m) for m in f.entities],
                tz=tz,
                extractor_version=extractor_version,
                time_in_embedding=settings.embed_fact_dates,
            )
        )
    return out


def _attribution(
    f: ExtractedFact,
    sources: Sequence[Turn],
    resolution: Resolution,
    subject_id: str | None,
    subject_is_user: bool,
) -> str:
    # Prompt-injection guard (section 5, step 5): pasted/tool content cannot create facts
    # attributed to the user unless the user also typed them.
    typed_by_user = any(t.role == "user" and t.source == "typed" for t in sources)
    if f.attributed_to == "assistant":
        return "assistant"
    if f.attributed_to == "user" and typed_by_user:
        return "user"
    if f.attributed_to == "user":
        return PASTED_SOURCE
    name = f.third_party_name or ("" if subject_is_user else f.subject)
    third = _entity_id_for(name, resolution) if name else None
    return f"third_party:{third or subject_id or 'unknown'}"


def _to_new_fact(
    f: ExtractedFact,
    *,
    sources: Sequence[Turn],
    subject_id: str | None,
    attributed_to: str,
    mention_ids: list[str],
    tz: tzinfo | None,
    extractor_version: str,
    time_in_embedding: bool = True,
) -> NewFact:
    mentioned_at = max(t.created_at for t in sources)
    start = _aware(f.start, tz)
    end = _aware(f.end, tz)
    if start and end and end < start:
        end = None
    phrase = time_phrase(start, f.time_precision, f.status, mentioned_at)
    if subject_id and subject_id not in mention_ids and not subject_id.endswith(":self"):
        mention_ids = [*mention_ids, subject_id]
    return NewFact(
        fact_id=f"fact-{uuid.uuid4().hex}",
        text=f.text.strip(),
        text_for_embedding=f"{f.text.strip()} ({phrase})" if time_in_embedding else f.text.strip(),
        fact_type=f.fact_type,
        attributed_to=attributed_to,
        subject_entity_id=subject_id,
        attribute_key=normalize_slot(f.attribute_key),
        start=start,
        end=end,
        mentioned_at=mentioned_at,
        time_precision=f.time_precision,
        status=f.status,
        aliases=[a.strip() for a in f.aliases if a.strip()][:4],
        confidence=f.confidence,
        is_hypothetical=f.is_hypothetical,
        importance=f.importance,
        relation_to_existing=f.relation_to_existing,
        extractor_version=extractor_version,
        derived_from_turn_ids=[t.turn_id for t in sources],
        mentions_entity_ids=list(dict.fromkeys(mention_ids)),
    )


# --- Running extraction -------------------------------------------------------------------


@dataclass
class ExtractionStats:
    batches: int = 0
    gated_batches: int = 0
    llm_calls: int = 0
    facts_created: int = 0
    facts_merged: int = 0
    entities_created: int = 0
    dropped_low_confidence: int = 0
    dropped_no_source: int = 0
    observations: int = 0
    forget_intents: int = 0
    entities_merged: int = 0
    entity_tiebreaks: int = 0
    facts_superseded: int = 0
    facts_stored_as_history: int = 0
    fact_conflicts: int = 0
    failed_sessions: list[str] = field(default_factory=list)
    touched_entities: set[str] = field(default_factory=set)

    def add(self, other: "ExtractionStats") -> None:
        for name in self.__dataclass_fields__:
            value = getattr(other, name)
            if isinstance(value, int):
                setattr(self, name, getattr(self, name) + value)
        self.failed_sessions.extend(other.failed_sessions)
        self.touched_entities |= other.touched_entities


@dataclass
class ExtractionDeps:
    store: Neo4jStore
    embedder: EmbedderLike
    extractor: Extractor
    settings: Settings
    tiebreaker: Tiebreaker | None = None  # entity merge tiebreak; None = no LLM tiebreaks

    @property
    def resolver(self) -> EntityResolver:
        return EntityResolver(self.store, self.embedder, self.settings, self.tiebreaker)


async def _attach_embeddings(
    embedder: EmbedderLike, facts: list[NewFact], entities: list[NewEntity]
) -> tuple[list[NewFact], list[NewEntity]]:
    if facts:
        e = await embedder.embed_documents([f.text_for_embedding for f in facts])
        facts = [
            f.model_copy(update={"embedding": v, "embedding_model": e.model})
            for f, v in zip(facts, e.vectors, strict=True)
        ]
    if entities:
        e = await embedder.embed_documents([n.name for n in entities])
        entities = [
            n.model_copy(update={"embedding": v, "embedding_model": e.model})
            for n, v in zip(entities, e.vectors, strict=True)
        ]
    return facts, entities


async def extract_batch(
    deps: ExtractionDeps, user_id: str, session_id: str
) -> ExtractionStats | None:
    """Extract and commit the next batch of one session. None when the session is caught up."""
    s = deps.settings
    session = await deps.store.get_session(user_id, session_id)
    if session is None:
        return None
    turns = await deps.store.get_session_turns(user_id, session_id)
    watermark = session.extracted_through_seq
    new = [t for t in turns if t.seq > watermark][: s.extract_batch_max_turns]
    if not new:
        return None
    earlier = [t for t in turns if t.seq <= watermark]
    context = earlier[-s.extract_batch_overlap :] if earlier else []
    stats = ExtractionStats(batches=1)
    up_to = new[-1].seq

    content = [t for t in new if not is_gated(t, max_tokens=s.gate_max_tokens)]
    facts: list[NewFact] = []
    entities: list[NewEntity] = []
    aliases: dict[str, list[str]] = {}
    observations: list[str] = []
    if not content:
        stats.gated_batches = 1
    else:
        batch_text = "\n".join(t.text for t in content)
        query = await deps.embedder.embed_query([batch_text])
        related = await deps.store.vector_search_facts(
            user_id, query.vectors[0], s.extract_related_facts, include_hypothetical=True
        )
        log_ = await deps.store.get_observation_log(user_id)
        messages = deps.extractor.messages(
            session_start=turns[0].created_at,
            context=context,
            new=new,
            observations=log_.text if log_ else "",
            related=[h.item for h in related],
        )
        result = await deps.extractor.extract(messages)
        stats.llm_calls = 1
        stats.observations = len(result.observations)
        observations = format_observations(result.observations)
        stats.forget_intents = len(result.forget_intents)
        resolution = await deps.resolver.resolve(
            user_id, subject_mentions(result), mention_contexts(result)
        )
        stats.entities_merged = resolution.merged
        stats.entity_tiebreaks = resolution.tiebreaks
        processed = postprocess(
            result,
            user_id=user_id,
            new_turns=new,
            resolution=resolution,
            settings=s,
            extractor_version=deps.extractor.version,
        )
        stats.dropped_low_confidence = processed.dropped_low_confidence
        stats.dropped_no_source = processed.dropped_no_source
        facts, entities = await _attach_embeddings(deps.embedder, processed.facts, resolution.new)
        aliases = resolution.aliases

    commit = await deps.store.commit_extraction(
        user_id,
        session_id,
        entities=entities,
        facts=facts,
        expected_watermark=watermark,
        up_to_seq=up_to,
        entity_aliases=aliases,
        decide=supersession,
        observations=observations,
    )
    stats.facts_created = commit.created
    stats.facts_merged = commit.merged
    stats.entities_created = commit.entities_created
    stats.facts_superseded = commit.superseded
    stats.facts_stored_as_history = commit.stored_as_history
    stats.fact_conflicts = commit.conflicts
    stats.touched_entities = {eid for f in commit.facts for eid in f.mentioned_entity_ids}
    return stats


async def run_extraction(
    deps: ExtractionDeps, user_id: str, *, session_ids: Sequence[str] | None = None
) -> ExtractionStats:
    """Extract every pending batch for the user (or only the given sessions), oldest first."""
    total = ExtractionStats()
    pending = await deps.store.pending_extraction(user_id)
    targets = [p.session_id for p in pending if session_ids is None or p.session_id in session_ids]
    for session_id in targets:
        while True:
            try:
                stats = await extract_batch(deps, user_id, session_id)
            except VersionConflictError:
                log.info("watermark moved for %s; re-reading", session_id)
                continue
            except ExtractionFailedError:
                log.exception("extraction failed for %s; turns stay searchable", session_id)
                total.failed_sessions.append(session_id)
                break
            if stats is None:
                break
            total.add(stats)
    return total
