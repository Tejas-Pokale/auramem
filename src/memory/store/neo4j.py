"""Neo4j data access layer (PLAN.md sections 4 and 7).

Every public method takes ``user_id`` and every query is scoped by it:

* Vector search filters ``user_id`` inside the index (``SEARCH ... WHERE``).
* Full-text search requires ``+user_key:<sha256(user_id)>`` inside the Lucene query
  (full-text indexes have no WHERE). User text is reduced to escaped, lower-cased terms
  confined to the content fields, so it cannot add or remove clauses.
* Both also re-check ``user_id`` after the index, as defense in depth.
* Writes that reference other nodes (sessions, turns, entities) verify ownership and
  roll back on any mismatch.

Raw turns are written once and never modified; facts are ADD-only.
"""

import hashlib
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Literal, Self

from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
    AsyncManagedTransaction,
    NotificationDisabledClassification,
    RoutingControl,
)
from neo4j.time import DateTime as Neo4jDateTime
from pydantic import BaseModel

from memory.config import Settings
from memory.models import (
    Entity,
    Fact,
    Hit,
    NewEntity,
    NewFact,
    NewTurn,
    ObservationLog,
    Session,
    Turn,
    User,
)

# --- Errors ------------------------------------------------------------------------------


class StoreError(RuntimeError):
    pass


class OwnershipError(StoreError):
    """A referenced session or turn id belongs to a different user."""


class InvalidReferenceError(StoreError):
    """A referenced node does not exist for this user."""


class ImmutableTurnError(StoreError):
    """An upsert tried to change a stored raw turn."""


class VersionConflictError(StoreError):
    """The ObservationLog changed since the caller read it."""


@dataclass(frozen=True)
class DeleteCounts:
    facts: int
    turns: int
    entities: int


@dataclass(frozen=True)
class ExtractionCommit:
    facts: list[Fact]
    created: int
    merged: int
    entities_created: int
    superseded: int = 0  # older facts marked not-latest by this batch
    conflicts: int = 0  # contradiction links created by this batch
    stored_as_history: int = 0  # new facts that arrived already superseded (backfilled info)


SupersessionDecision = Callable[[Fact, Fact], Literal["new_wins", "old_wins", "conflict"] | None]


@dataclass(frozen=True)
class PendingSession:
    session_id: str
    watermark: int
    new_turns: int
    new_chars: int
    last_seq: int
    last_turn_at: datetime


def self_entity_id(user_id: str) -> str:
    """Entity standing for the user themself: the subject of facts about the user."""
    return f"{user_id}:self"


def fact_dedup_key(user_id: str, subject_entity_id: str | None, text: str) -> str:
    """Exact-duplicate key: same user, same subject, same text up to case/space/punctuation."""
    normalized = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
    raw = f"{user_id}|{subject_entity_id or ''}|{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _entity_row(e: NewEntity) -> dict[str, Any]:
    return {"props": _props(e, exclude=set()), "aliases_text": " ".join(e.aliases)}


def _overlap_score(fact: Fact, start: datetime | None, end: datetime | None) -> float:
    """Overlap / union of the fact interval and the query range; open bounds clamped."""
    assert fact.start is not None
    f_start = fact.start
    f_end = fact.end if fact.end and fact.end > f_start else f_start + timedelta(days=1)
    q_start = start or f_start
    q_end = end or f_end
    overlap = (min(f_end, q_end) - max(f_start, q_start)).total_seconds()
    union = (max(f_end, q_end) - min(f_start, q_start)).total_seconds()
    return max(0.0, overlap) / union if union > 0 else 0.0


# --- Full-text helpers --------------------------------------------------------------------

_LUCENE_SPECIAL = frozenset('+-&|!(){}[]^"~*?:\\/')

FLAG_LATEST = "latest"
FLAG_SUPERSEDED = "superseded"
FLAG_FACTUAL = "factual"
FLAG_HYPOTHETICAL = "hypothetical"


def user_key(user_id: str) -> str:
    """Single-token tenant key for full-text indexes (the standard analyzer keeps hex whole)."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def lucene_terms(text: str, *, max_terms: int | None = None) -> list[str]:
    """Whitespace-split, lower-cased (so AND/OR/NOT are plain words), escaped, de-duplicated.

    Tokens with no letter or digit are dropped; they would analyze to nothing.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for raw in text.split():
        if not any(ch.isalnum() for ch in raw):
            continue
        term = "".join("\\" + ch if ch in _LUCENE_SPECIAL else ch for ch in raw.lower())
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if max_terms is not None and len(terms) >= max_terms:
            break
    return terms


def fulltext_query(
    user_id: str,
    text: str,
    fields: Sequence[str],
    *,
    max_terms: int,
    require: Sequence[tuple[str, str]] = (),
) -> str | None:
    """Lucene query: the user's key and any required flag tokens, plus a match in any field.

    Returns None when the text has no searchable terms.
    """
    terms = lucene_terms(text, max_terms=max_terms)
    if not terms:
        return None
    joined = " ".join(terms)
    must = [f"+user_key:{user_key(user_id)}", *(f"+{field}:{token}" for field, token in require)]
    should = " ".join(f"{field}:({joined})" for field in fields)
    return f"{' '.join(must)} +({should})"


def fact_flags(*, is_latest: bool, is_hypothetical: bool) -> str:
    latest = FLAG_LATEST if is_latest else FLAG_SUPERSEDED
    factual = FLAG_HYPOTHETICAL if is_hypothetical else FLAG_FACTUAL
    return f"{latest} {factual}"


# --- Conversion helpers -------------------------------------------------------------------


def _native(value: Any) -> Any:
    """Convert neo4j temporal values (recursively) to Python datetimes."""
    if isinstance(value, Neo4jDateTime):
        return value.to_native()
    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_native(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    """Datetimes as ISO 8601 strings (recursively), for export."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items() if k != "embedding"}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _props(model: BaseModel, *, exclude: set[str]) -> dict[str, Any]:
    """Model fields as Neo4j property values; None is omitted (Neo4j has no null props)."""
    data = model.model_dump(exclude=exclude)
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in data.items() if v is not None}


def _projection(var: str, model: type[BaseModel], computed: dict[str, str] | None = None) -> str:
    """Map projection of exactly the model's fields, so embeddings are never read back.

    ``computed`` maps field names to Cypher expressions for fields that are not stored as
    plain properties (defaults, related node ids).
    """
    computed = computed or {}
    parts = [f".{name}" for name in model.model_fields if name not in computed]
    parts += [f"{name}: {expr}" for name, expr in computed.items()]
    return var + " {" + ", ".join(parts) + "}"


def _unique(ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def _in_order[T](rows: list[T], ids: Sequence[str], key: Callable[[T], str]) -> list[T]:
    by_id = {key(r): r for r in rows}
    return [by_id[i] for i in _unique(ids) if i in by_id]


# --- Cypher ------------------------------------------------------------------------------

_USER = _projection("u", User)
_SESSION = _projection(
    "s", Session, {"extracted_through_seq": "coalesce(s.extracted_through_seq, -1)"}
)
_TURN = _projection("t", Turn)
_FACT = _projection(
    "f",
    Fact,
    {
        "mention_dates": "coalesce(f.mention_dates, [f.mentioned_at])",
        "source_turn_ids": "[(f)-[:DERIVED_FROM]->(src:Turn) | src.turn_id]",
        "superseded_on": (
            "head([(sup:Fact)-[:SUPERSEDES]->(f) | coalesce(sup.start, sup.mentioned_at)])"
        ),
        "superseded_by_ids": "[(sup:Fact)-[:SUPERSEDES]->(f) | sup.fact_id]",
        "supersedes_ids": "[(f)-[:SUPERSEDES]->(old:Fact) | old.fact_id]",
        "conflicts_with_ids": "[(f)-[:CONFLICTS_WITH]-(c:Fact) | c.fact_id]",
        "mentioned_entity_ids": "[(f)-[:MENTIONS]->(me:Entity) | me.entity_id]",
    },
)
_ENTITY = _projection("e", Entity)
_LOG = _projection("o", ObservationLog)

_ENSURE_USER = f"""
MERGE (u:User {{user_id: $user_id}})
  ON CREATE SET u.created_at = $now, u.timezone = $timezone
RETURN {_USER} AS user
"""

_GET_USER = f"MATCH (u:User {{user_id: $user_id}}) RETURN {_USER} AS user"

_UPSERT_SESSION = f"""
MERGE (u:User {{user_id: $user_id}})
  ON CREATE SET u.created_at = $now
MERGE (s:Session {{session_id: $session_id}})
  ON CREATE SET s.user_id = $user_id, s.started_at = $started_at, s.title = $title
FOREACH (_ IN CASE WHEN s.user_id = $user_id THEN [1] ELSE [] END |
  MERGE (u)-[:HAS_SESSION]->(s))
RETURN s.user_id = $user_id AS owned, {_SESSION} AS session
"""

_GET_SESSION = f"""
MATCH (s:Session {{session_id: $session_id}}) WHERE s.user_id = $user_id
RETURN {_SESSION} AS session
"""

_END_SESSION = f"""
MATCH (s:Session {{session_id: $session_id}}) WHERE s.user_id = $user_id
SET s.ended_at = $ended_at
RETURN {_SESSION} AS session
"""

_SESSION_OWNER = "MATCH (s:Session {session_id: $session_id}) RETURN s.user_id AS owner"

_FOREIGN_TURN_IDS = """
MATCH (t:Turn) WHERE t.turn_id IN $turn_ids
  AND (t.user_id <> $user_id OR t.session_id <> $session_id)
RETURN t.turn_id AS turn_id
"""

_UPSERT_TURNS = f"""
MATCH (s:Session {{session_id: $session_id}}) WHERE s.user_id = $user_id
UNWIND $rows AS row
MERGE (t:Turn {{turn_id: row.turn_id}})
  ON CREATE SET t += row.props,
    t.user_id = $user_id, t.user_key = $user_key, t.session_id = $session_id
MERGE (t)-[:IN_SESSION]->(s)
RETURN {_TURN} AS turn
"""

# Rebuilds NEXT so it always links consecutive surviving turns by seq, whatever order turns
# were inserted or deleted in.
_RELINK_SESSIONS = """
UNWIND $session_ids AS sid
CALL (sid) {
  MATCH (t:Turn {session_id: sid}) WHERE t.user_id = $user_id
  WITH t ORDER BY t.seq
  WITH collect(t) AS chain
  UNWIND range(0, size(chain) - 2) AS i
  WITH chain[i] AS a, chain[i + 1] AS b
  MERGE (a)-[:NEXT]->(b)
  RETURN count(*) AS linked
}
CALL (sid) {
  MATCH (a:Turn {session_id: sid})-[r:NEXT]->(b:Turn)
  WHERE a.user_id = $user_id
    AND EXISTS { MATCH (m:Turn {session_id: sid}) WHERE a.seq < m.seq < b.seq }
  DELETE r
  RETURN count(*) AS unlinked
}
RETURN sum(linked) AS linked, sum(unlinked) AS unlinked
"""

_COUNT_TURNS = "MATCH (t:Turn) WHERE t.user_id = $user_id RETURN count(t) AS n"

# Labels holding per-user data, in delete order (dependents first).
_USER_LABELS = (
    "Fact", "Turn", "Entity", "Session", "ObservationLog", "ObservationLogVersion",
    "PendingForget", "DeletionEvent", "User",
)  # fmt: skip
_DELETE_USER_LABEL = {
    label: f"MATCH (n:{label}) WHERE n.user_id = $user_id DETACH DELETE n RETURN count(*) AS n"
    for label in _USER_LABELS
}

_GET_TURNS = f"""
MATCH (t:Turn) WHERE t.turn_id IN $turn_ids AND t.user_id = $user_id
RETURN {_TURN} AS turn
"""

_GET_SESSION_TURNS = f"""
MATCH (t:Turn {{session_id: $session_id}}) WHERE t.user_id = $user_id
RETURN {_TURN} AS turn ORDER BY t.seq
"""

# The last turns of the user's most recent conversation other than ``exclude``.
_LAST_CONVERSATION = f"""
MATCH (t:Turn) WHERE t.user_id = $user_id AND t.session_id <> $exclude
WITH t.session_id AS sid, max(t.created_at) AS last
ORDER BY last DESC
LIMIT 1
MATCH (t:Turn {{session_id: sid}}) WHERE t.user_id = $user_id
WITH t ORDER BY t.seq DESC LIMIT $limit
RETURN {_TURN} AS turn ORDER BY t.seq
"""

_GET_TURN_WINDOWS = f"""
UNWIND $turn_ids AS center_id
MATCH (c:Turn {{turn_id: center_id}}) WHERE c.user_id = $user_id
CALL (c) {{
  MATCH (t:Turn {{session_id: c.session_id}}) WHERE t.seq < c.seq AND t.user_id = $user_id
  RETURN t ORDER BY t.seq DESC LIMIT $neighbors
  UNION
  MATCH (t:Turn {{session_id: c.session_id}}) WHERE t.seq > c.seq AND t.user_id = $user_id
  RETURN t ORDER BY t.seq ASC LIMIT $neighbors
  UNION
  RETURN c AS t
}}
WITH center_id, t ORDER BY t.seq
RETURN center_id, collect({_TURN}) AS window
"""

_INSERT_ENTITIES = f"""
UNWIND $rows AS row
CREATE (e:Entity)
SET e = row.props, e.user_id = $user_id, e.user_key = $user_key,
    e.aliases_text = row.aliases_text
RETURN {_ENTITY} AS entity
"""

_GET_ENTITIES = f"""
MATCH (e:Entity) WHERE e.entity_id IN $entity_ids AND e.user_id = $user_id
RETURN {_ENTITY} AS entity
"""

# ADD-only with an exact-duplicate guard: a fact whose dedup_key exists is merged (new
# mention date, max confidence, extra provenance) instead of inserted again.
_INSERT_FACTS = f"""
UNWIND $rows AS row
MERGE (f:Fact {{dedup_key: row.dedup_key}})
  ON CREATE SET f += row.props, f.user_id = $user_id, f.user_key = $user_key,
    f.aliases_text = row.aliases_text, f.flags_text = row.flags_text,
    f.mention_dates = [row.props.mentioned_at]
  ON MATCH SET f.mention_dates = coalesce(f.mention_dates, [f.mentioned_at])
      + row.props.mentioned_at,
    f.confidence = CASE WHEN row.props.confidence > f.confidence
      THEN row.props.confidence ELSE f.confidence END
WITH f, row, f.fact_id = row.props.fact_id AS created
CALL (f, row) {{
  UNWIND row.turn_ids AS tid
  MATCH (t:Turn {{turn_id: tid}}) WHERE t.user_id = $user_id
  MERGE (f)-[:DERIVED_FROM]->(t)
  RETURN count(t) AS turns_linked
}}
CALL (f, row) {{
  UNWIND row.entity_ids AS eid
  MATCH (e:Entity {{entity_id: eid}}) WHERE e.user_id = $user_id
  MERGE (f)-[:MENTIONS]->(e)
  RETURN count(e) AS entities_linked
}}
CALL (row) {{
  OPTIONAL MATCH (se:Entity {{entity_id: row.subject_entity_id}}) WHERE se.user_id = $user_id
  RETURN row.subject_entity_id IS NULL OR se IS NOT NULL AS subject_ok
}}
RETURN {_FACT} AS fact, created, f.user_id = $user_id AS owned, turns_linked,
  entities_linked, subject_ok
"""

_GET_FACTS = f"""
MATCH (f:Fact) WHERE f.fact_id IN $fact_ids AND f.user_id = $user_id
RETURN {_FACT} AS fact
"""

_ADVANCE_WATERMARK = """
MATCH (s:Session {session_id: $session_id}) WHERE s.user_id = $user_id
  AND coalesce(s.extracted_through_seq, -1) = $expected
SET s.extracted_through_seq = $up_to
RETURN s.session_id AS session_id
"""

_ENSURE_SELF_ENTITY = """
MERGE (e:Entity {entity_id: $entity_id})
  ON CREATE SET e.user_id = $user_id, e.user_key = $user_key, e.name = 'user',
    e.entity_type = 'person', e.aliases = ['me', 'myself', 'I'], e.aliases_text = 'me myself I',
    e.mention_count = 0
"""

_FIND_ENTITIES_BY_NAME = f"""
MATCH (e:Entity) WHERE e.user_id = $user_id
  AND (toLower(e.name) IN $names OR any(a IN e.aliases WHERE toLower(a) IN $names))
RETURN {_ENTITY} AS entity
"""

_PENDING_EXTRACTION = """
MATCH (s:Session) WHERE s.user_id = $user_id
WITH s, coalesce(s.extracted_through_seq, -1) AS watermark
MATCH (t:Turn {session_id: s.session_id}) WHERE t.user_id = $user_id AND t.seq > watermark
WITH s, watermark, count(t) AS new_turns, sum(size(t.text)) AS new_chars,
     max(t.seq) AS last_seq, max(t.created_at) AS last_turn_at
RETURN s.session_id AS session_id, watermark, new_turns, new_chars, last_seq, last_turn_at
ORDER BY last_turn_at
"""

# Facts whose [start, end] overlaps [$from, $to); open bounds are allowed. Scoring happens in
# Python; this only narrows candidates via the range index on Fact.start.
_TEMPORAL_FACTS = f"""
MATCH (f:Fact) WHERE f.user_id = $user_id AND f.start IS NOT NULL
  AND f.is_hypothetical = false
  AND ($to IS NULL OR f.start < $to)
  AND ($from IS NULL OR coalesce(f.end, f.start) >= $from)
RETURN {_FACT} AS item
ORDER BY f.start DESC
LIMIT $scan
"""

_FACT_SUMMARY = """
MATCH (f:Fact) WHERE f.user_id = $user_id
RETURN f.fact_type AS fact_type,
       CASE WHEN f.attributed_to STARTS WITH 'third_party' THEN 'third_party'
            ELSE f.attributed_to END AS attribution,
       f.is_hypothetical AS hypothetical,
       f.is_latest AS latest,
       count(f) AS n,
       sum(size(coalesce(f.mention_dates, [f.mentioned_at]))) AS mentions
"""

_TURNS_WITH_FACTS = """
MATCH (f:Fact)-[:DERIVED_FROM]->(t:Turn) WHERE f.user_id = $user_id AND t.user_id = $user_id
RETURN DISTINCT t.turn_id AS turn_id
"""

_SLOT_RIVALS = f"""
MATCH (f:Fact) WHERE f.user_id = $user_id AND f.attribute_key = $attribute_key
  AND f.subject_entity_id = $subject_entity_id AND f.is_latest = true
  AND f.is_hypothetical = false AND f.fact_id <> $fact_id
RETURN {_FACT} AS fact
"""

_APPEND_OBSERVATIONS = """
MERGE (o:ObservationLog {user_id: $user_id})
  ON CREATE SET o.version = 0, o.text = '', o.token_count = 0
WITH o, CASE WHEN o.text = '' THEN $lines ELSE o.text + '\n' + $lines END AS text
SET o.text = text, o.token_count = toInteger(round(size(text) / $chars_per_token)),
    o.version = o.version + 1, o.updated_at = $now
"""

_REPLACE_LOG = f"""
MATCH (o:ObservationLog {{user_id: $user_id}}) WHERE o.version = $expected_version
CREATE (:ObservationLogVersion {{
  user_id: $user_id, version: o.version, text: o.text, token_count: o.token_count,
  replaced_at: $now, replaced_by: $reason
}})
SET o.text = $text, o.token_count = $token_count, o.version = o.version + 1, o.updated_at = $now
RETURN {_projection("o", ObservationLog)} AS log
"""

_LOG_VERSION = """
MATCH (v:ObservationLogVersion {user_id: $user_id, version: $version})
RETURN v.text AS text, v.token_count AS token_count
"""

_LOG_VERSIONS = """
MATCH (v:ObservationLogVersion {user_id: $user_id})
RETURN v.version AS version, v.token_count AS token_count, v.replaced_at AS replaced_at,
       v.replaced_by AS replaced_by
ORDER BY v.version
"""

_ENTITIES_FOR_SUMMARY = f"""
UNWIND $entity_ids AS eid
MATCH (e:Entity {{entity_id: eid}}) WHERE e.user_id = $user_id AND NOT eid ENDS WITH ':self'
CALL (e) {{
  MATCH (f:Fact)-[:MENTIONS]->(e) WHERE f.user_id = $user_id AND f.is_hypothetical = false
  WITH f ORDER BY f.is_latest DESC, f.mentioned_at DESC
  RETURN collect(f.text)[..$max_facts] AS facts, count(f) AS n
}}
WITH e, facts, n WHERE n >= $min_facts
RETURN {_ENTITY} AS entity, facts
"""

_SET_ENTITY_SUMMARY = """
MATCH (e:Entity {entity_id: $entity_id}) WHERE e.user_id = $user_id
SET e.summary = $summary, e.summary_updated_at = $now
"""

_CLEAR_ENTITY_SUMMARIES = """
MATCH (e:Entity) WHERE e.entity_id IN $ids AND e.user_id = $user_id
REMOVE e.summary, e.summary_updated_at
"""

_LINK_CONFLICT = """
MATCH (a:Fact {fact_id: $a}), (b:Fact {fact_id: $b})
WHERE a.user_id = $user_id AND b.user_id = $user_id
MERGE (a)-[:CONFLICTS_WITH]->(b)
"""

_ADD_ENTITY_ALIASES = """
UNWIND $rows AS row
MATCH (e:Entity {entity_id: row.entity_id}) WHERE e.user_id = $user_id
WITH e, [a IN row.aliases WHERE NOT toLower(a) IN
         [x IN coalesce(e.aliases, []) + [e.name] | toLower(x)]] AS fresh
WITH e, coalesce(e.aliases, []) + fresh AS merged
SET e.aliases = merged,
    e.aliases_text = reduce(s = '', a IN merged | s + ' ' + a),
    e.mention_count = coalesce(e.mention_count, 0) + 1
"""

_CO_MENTIONED = """
UNWIND $entity_ids AS eid
MATCH (e:Entity {entity_id: eid}) WHERE e.user_id = $user_id
OPTIONAL MATCH (e)<-[:MENTIONS]-(f:Fact)-[:MENTIONS]->(o:Entity)
  WHERE f.user_id = $user_id AND o <> e
RETURN eid, collect(DISTINCT o.entity_id) AS co
"""

# S5: facts that mention (or are about) the given entities; current values first.
_FACTS_MENTIONING = f"""
MATCH (f:Fact) WHERE f.user_id = $user_id AND f.is_hypothetical = false
  AND (f.subject_entity_id IN $entity_ids
       OR EXISTS {{ MATCH (f)-[:MENTIONS]->(e:Entity) WHERE e.entity_id IN $entity_ids }})
RETURN {_FACT} AS item, CASE WHEN f.is_latest THEN 1.0 ELSE 0.5 END AS score
ORDER BY score DESC, f.mentioned_at DESC
LIMIT $k
"""

# S7: one hop from seed facts via shared entities (not the user's self entity) and
# SUPERSEDES in either direction. Score = number of paths reaching the fact.
_EXPAND_FACTS = f"""
MATCH (s:Fact) WHERE s.fact_id IN $seed_ids AND s.user_id = $user_id
CALL (s) {{
  MATCH (s)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(f:Fact)
  WHERE f.user_id = $user_id AND f.is_hypothetical = false AND NOT e.entity_id ENDS WITH ':self'
  RETURN f
  UNION ALL
  MATCH (s)-[:SUPERSEDES]-(f:Fact) WHERE f.user_id = $user_id
  RETURN f
}}
WITH f, count(*) AS paths WHERE NOT f.fact_id IN $seed_ids
RETURN {_FACT} AS item, toFloat(paths) AS score
ORDER BY score DESC, f.mentioned_at DESC
LIMIT $k
"""

# Deep mode (PLAN.md section 6 step 8). grep: exact, case-insensitive substring over the
# user's raw turns or fact text and aliases; every match is returned (no cap, by design).
_GREP_TURNS = f"""
MATCH (t:Turn) WHERE t.user_id = $user_id AND toLower(t.text) CONTAINS $needle
RETURN {_TURN} AS item, 1.0 AS score
ORDER BY t.created_at, t.seq
"""

_GREP_FACTS = f"""
MATCH (f:Fact) WHERE f.user_id = $user_id AND f.is_hypothetical = false
  AND (toLower(f.text) CONTAINS $needle
       OR toLower(coalesce(f.aliases_text, '')) CONTAINS $needle)
RETURN {_FACT} AS item, 1.0 AS score
ORDER BY coalesce(f.start, f.mentioned_at)
"""

# timeline: every fact about or mentioning the entities, in event order, optionally limited
# to facts whose time overlaps [start, end).
_TIMELINE = f"""
MATCH (f:Fact) WHERE f.user_id = $user_id AND f.is_hypothetical = false
  AND (f.subject_entity_id IN $entity_ids
       OR EXISTS {{ MATCH (f)-[:MENTIONS]->(e:Entity) WHERE e.entity_id IN $entity_ids }})
WITH f, coalesce(f.start, f.mentioned_at) AS t0, coalesce(f.end, f.start, f.mentioned_at) AS t1
WHERE ($start IS NULL OR t1 >= $start) AND ($end IS NULL OR t0 < $end)
RETURN {_FACT} AS item, 1.0 AS score
ORDER BY t0, f.mentioned_at
"""

_MARK_SUPERSEDED = """
MATCH (n:Fact {fact_id: $new_fact_id}) WHERE n.user_id = $user_id
UNWIND $old_fact_ids AS oid
MATCH (o:Fact {fact_id: oid}) WHERE o.user_id = $user_id AND o <> n
SET o.is_latest = false,
    o.flags_text = CASE WHEN o.is_hypothetical
      THEN $superseded_hypothetical ELSE $superseded_factual END
MERGE (n)-[:SUPERSEDES]->(o)
RETURN count(o) AS superseded
"""

_GET_LOG = f"MATCH (o:ObservationLog {{user_id: $user_id}}) RETURN {_LOG} AS log"

_SAVE_LOG = f"""
MERGE (o:ObservationLog {{user_id: $user_id}})
  ON CREATE SET o.version = 0, o.text = '', o.token_count = 0, o.updated_at = $now
WITH o WHERE o.version = $expected_version
SET o.text = $text, o.token_count = $token_count, o.version = o.version + 1,
    o.updated_at = $now
RETURN {_LOG} AS log
"""

_DELETE_FACTS = """
MATCH (f:Fact) WHERE f.fact_id IN $ids AND f.user_id = $user_id
DETACH DELETE f
RETURN count(*) AS deleted
"""

_DELETE_TURNS = """
MATCH (t:Turn) WHERE t.turn_id IN $ids AND t.user_id = $user_id
WITH t, t.session_id AS sid
DETACH DELETE t
RETURN count(*) AS deleted, collect(DISTINCT sid) AS session_ids
"""

_DELETE_ENTITIES = """
MATCH (e:Entity) WHERE e.entity_id IN $ids AND e.user_id = $user_id
DETACH DELETE e
RETURN count(*) AS deleted
"""

# --- Forget and export (Phase 8) ---------------------------------------------------------

_FACTS_DERIVED_FROM = f"""
MATCH (f:Fact)-[:DERIVED_FROM]->(t:Turn)
WHERE t.turn_id IN $turn_ids AND t.user_id = $user_id AND f.user_id = $user_id
WITH DISTINCT f
RETURN {_FACT} AS item
"""

_SAVE_PENDING_FORGET = """
MATCH (p:PendingForget) WHERE p.user_id = $user_id AND p.created_at < $not_before
DETACH DELETE p
WITH count(*) AS expired
CREATE (p:PendingForget {
  preview_id: $preview_id, user_id: $user_id, session_id: $session_id,
  description: $description, fact_ids: $fact_ids, turn_ids: $turn_ids,
  request_text: $request_text, preview_text: $preview_text, created_at: $now
})
"""

_PENDING_FORGET = """
MATCH (p:PendingForget {preview_id: $preview_id})
WHERE p.user_id = $user_id AND p.created_at >= $not_before
RETURN p {.*} AS pending
"""

_SESSION_PENDING_FORGET = """
MATCH (p:PendingForget)
WHERE p.user_id = $user_id AND p.session_id = $session_id AND p.created_at >= $not_before
RETURN p {.*} AS pending
ORDER BY p.created_at DESC
LIMIT 1
"""

_SET_PENDING_PREVIEW_TEXT = """
MATCH (p:PendingForget {preview_id: $preview_id}) WHERE p.user_id = $user_id
SET p.preview_text = $preview_text
"""

_DELETE_PENDING_FORGET = """
MATCH (p:PendingForget {preview_id: $preview_id}) WHERE p.user_id = $user_id
DETACH DELETE p
RETURN count(*) AS n
"""

_SESSION_TURNS_WITH_TEXT = """
MATCH (t:Turn {session_id: $session_id}) WHERE t.user_id = $user_id AND t.text IN $texts
RETURN t.turn_id AS turn_id
"""

# Entities nothing refers to any more (they would still carry the forgotten name).
_DELETE_ORPHAN_ENTITIES = """
MATCH (e:Entity) WHERE e.entity_id IN $ids AND e.user_id = $user_id
  AND NOT e.entity_id ENDS WITH ':self'
  AND NOT EXISTS { MATCH (:Fact)-[:MENTIONS]->(e) }
  AND NOT EXISTS { MATCH (f:Fact) WHERE f.user_id = $user_id AND f.subject_entity_id = e.entity_id }
DETACH DELETE e
RETURN count(*) AS deleted
"""

_PURGE_LOG_VERSIONS = """
MATCH (v:ObservationLogVersion) WHERE v.user_id = $user_id
DETACH DELETE v
RETURN count(*) AS n
"""

_RECORD_DELETION = """
CREATE (:DeletionEvent {
  user_id: $user_id, at: $now, kind: $kind, facts: $facts, turns: $turns, entities: $entities
})
"""

# --- Nightly consolidation (PLAN.md section 5, Phase 8) --------------------------------

_LIST_USER_IDS = "MATCH (u:User) RETURN u.user_id AS user_id ORDER BY user_id"

_LIST_ENTITIES = f"""
MATCH (e:Entity) WHERE e.user_id = $user_id AND NOT e.entity_id ENDS WITH ':self'
RETURN {_ENTITY} AS entity
ORDER BY coalesce(e.mention_count, 0) DESC, e.name
"""

# Move every reference from ``drop`` to ``keep``, keep drop's names as aliases, then delete
# drop. Facts are not modified apart from the entity they point to.
_MERGE_ENTITIES = """
MATCH (keep:Entity {entity_id: $keep}), (drop:Entity {entity_id: $drop})
WHERE keep.user_id = $user_id AND drop.user_id = $user_id AND keep <> drop
CALL (keep, drop) {
  MATCH (f:Fact)-[r:MENTIONS]->(drop) WHERE f.user_id = $user_id
  MERGE (f)-[:MENTIONS]->(keep)
  DELETE r
}
CALL (keep, drop) {
  MATCH (f:Fact) WHERE f.user_id = $user_id AND f.subject_entity_id = drop.entity_id
  SET f.subject_entity_id = keep.entity_id
}
WITH keep, drop, coalesce(keep.aliases, []) AS old,
     [x IN coalesce(keep.aliases, []) + [keep.name] | toLower(x)] AS known
WITH keep, drop, old + [a IN [drop.name] + coalesce(drop.aliases, [])
                        WHERE NOT toLower(a) IN known] AS merged
SET keep.aliases = merged,
    keep.aliases_text = reduce(s = '', a IN merged | s + ' ' + a),
    keep.mention_count = coalesce(keep.mention_count, 0) + coalesce(drop.mention_count, 0)
DETACH DELETE drop
RETURN count(*) AS merged
"""

# is_latest must agree with the SUPERSEDES edges: latest exactly when nothing supersedes it.
_REPAIR_LATEST = """
MATCH (f:Fact) WHERE f.user_id = $user_id AND f.attribute_key IS NOT NULL
WITH f, EXISTS { MATCH (:Fact)-[:SUPERSEDES]->(f) } AS replaced
WHERE f.is_latest = replaced
SET f.is_latest = NOT replaced,
    f.flags_text = CASE
      WHEN replaced AND f.is_hypothetical THEN $superseded_hypothetical
      WHEN replaced THEN $superseded_factual
      WHEN f.is_hypothetical THEN $latest_hypothetical
      ELSE $latest_factual END
RETURN count(f) AS fixed
"""

# Slots where several facts are still latest (and not marked as conflicting): for review.
_SLOT_CLASHES = """
MATCH (f:Fact) WHERE f.user_id = $user_id AND f.attribute_key IS NOT NULL AND f.is_latest
  AND f.is_hypothetical = false AND NOT EXISTS { MATCH (f)-[:CONFLICTS_WITH]-(:Fact) }
WITH f.subject_entity_id AS subject, f.attribute_key AS slot, count(f) AS n
WHERE n > 1
RETURN count(*) AS clashes
"""

_ORPHAN_FACTS = """
MATCH (f:Fact) WHERE f.user_id = $user_id AND NOT EXISTS { MATCH (f)-[:DERIVED_FROM]->(:Turn) }
RETURN count(f) AS orphans
"""

# --- Re-embedding (PLAN.md section 5 "Background jobs", Phase 8) ------------------------
# New vectors go to embedding_next first; the swap then copies them over per label in one
# transaction, so a label's index never mixes models.
EMBEDDED_LABELS: dict[str, tuple[str, str]] = {  # label -> (id property, embedded text)
    "Turn": ("turn_id", "text"),
    "Fact": ("fact_id", "text_for_embedding"),
    "Entity": ("entity_id", "name"),
}
_REEMBED_PENDING = {
    label: f"""
MATCH (n:{label})
WHERE coalesce(n.embedding_next_model, '') <> $model AND coalesce(n.embedding_model, '') <> $model
RETURN n.{key} AS id, n.{text} AS text
LIMIT $batch
"""
    for label, (key, text) in EMBEDDED_LABELS.items()
}
_REEMBED_SET_NEXT = {
    label: f"""
UNWIND $rows AS row
MATCH (n:{label} {{{key}: row.id}})
SET n.embedding_next = row.vector, n.embedding_next_model = $model
"""
    for label, (key, _) in EMBEDDED_LABELS.items()
}
_REEMBED_SWAP = {
    label: f"""
MATCH (n:{label}) WHERE n.embedding_next_model = $model
SET n.embedding = n.embedding_next, n.embedding_model = n.embedding_next_model
REMOVE n.embedding_next, n.embedding_next_model
RETURN count(n) AS swapped
"""
    for label in EMBEDDED_LABELS
}
_REEMBED_REMAINING = {
    label: f"""
MATCH (n:{label}) WHERE coalesce(n.embedding_model, '') <> $model
  AND coalesce(n.embedding_next_model, '') <> $model
RETURN count(n) AS n
"""
    for label in EMBEDDED_LABELS
}

_EXPORT = {
    "sessions": "MATCH (s:Session) WHERE s.user_id = $user_id "
    f"RETURN {_projection('s', Session)} AS item ORDER BY s.started_at",
    "turns": "MATCH (t:Turn) WHERE t.user_id = $user_id "
    f"RETURN {_TURN} AS item ORDER BY t.created_at, t.seq",
    "facts": "MATCH (f:Fact) WHERE f.user_id = $user_id "
    f"RETURN {_FACT} AS item ORDER BY f.mentioned_at",
    "entities": "MATCH (e:Entity) WHERE e.user_id = $user_id "
    f"RETURN {_ENTITY} AS item ORDER BY e.name",
    "observation_log": f"MATCH (o:ObservationLog) WHERE o.user_id = $user_id RETURN {_LOG} AS item",
    "observation_log_versions": "MATCH (v:ObservationLogVersion) WHERE v.user_id = $user_id "
    "RETURN v {.*} AS item ORDER BY v.version",
    "deletion_events": "MATCH (d:DeletionEvent) WHERE d.user_id = $user_id "
    "RETURN d {.*} AS item ORDER BY d.at",
}


def _vector_query(label: str, var: str, index: str, projection: str, where: str) -> str:
    return f"""
CYPHER 25
MATCH ({var}:{label})
  SEARCH {var} IN (
    VECTOR INDEX {index}
    FOR $vector
    WHERE {where}
    LIMIT $k
  ) SCORE AS score
WITH {var}, score WHERE {var}.user_id = $user_id
RETURN {projection} AS item, score
ORDER BY score DESC
"""


def _fulltext_query(label: str, var: str, index: str, projection: str) -> str:
    return f"""
CYPHER 25
MATCH ({var}:{label})
  SEARCH {var} IN (
    FULLTEXT INDEX {index}
    FOR $query
    LIMIT $k
  ) SCORE AS score
WITH {var}, score WHERE {var}.user_id = $user_id
RETURN {projection} AS item, score
ORDER BY score DESC
"""


# Keyed by (has lower bound, has upper bound) on created_at; [created_from, created_to).
_VECTOR_TURNS = {
    (has_from, has_to): _vector_query(
        "Turn",
        "t",
        "turn_vec",
        _TURN,
        " AND ".join(
            ["t.user_id = $user_id"]
            + (["t.created_at >= $created_from"] if has_from else [])
            + (["t.created_at < $created_to"] if has_to else [])
        ),
    )
    for has_from in (False, True)
    for has_to in (False, True)
}
_VECTOR_ENTITIES = _vector_query("Entity", "e", "entity_vec", _ENTITY, "e.user_id = $user_id")
# Keyed by (latest_only, include_hypothetical). In-index filters allow only AND-ed
# equality/range predicates, so each combination is its own query.
_VECTOR_FACTS = {
    (latest_only, include_hypothetical): _vector_query(
        "Fact",
        "f",
        "fact_vec",
        _FACT,
        " AND ".join(
            ["f.user_id = $user_id"]
            + ([] if include_hypothetical else ["f.is_hypothetical = false"])
            + (["f.is_latest = true"] if latest_only else [])
        ),
    )
    for latest_only in (False, True)
    for include_hypothetical in (False, True)
}
_FULLTEXT_TURNS = _fulltext_query("Turn", "t", "turn_text", _TURN)
_FULLTEXT_FACTS = _fulltext_query("Fact", "f", "fact_text", _FACT)
_FULLTEXT_ENTITIES = _fulltext_query("Entity", "e", "entity_text", _ENTITY)

_TURN_FIELDS = ("text",)
_FACT_FIELDS = ("text", "aliases_text")
_ENTITY_FIELDS = ("name", "aliases_text")


# --- Store -------------------------------------------------------------------------------


class Neo4jStore:
    def __init__(
        self,
        driver: AsyncDriver,
        *,
        database: str,
        settings: Settings,
        owns_driver: bool = False,
    ) -> None:
        self._driver = driver
        self._db = database
        self._max_terms = settings.fulltext_max_terms
        self._chars_per_token = settings.chars_per_token
        self._owns_driver = owns_driver

    @classmethod
    def connect(cls, settings: Settings) -> Self:
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
            # "Property does not exist" notices are expected on a young database.
            notifications_disabled_classifications=[
                NotificationDisabledClassification.UNRECOGNIZED
            ],
        )
        return cls(driver, database=settings.neo4j_database, settings=settings, owns_driver=True)

    @property
    def driver(self) -> AsyncDriver:
        """The underlying driver (for schema migrations)."""
        return self._driver

    async def close(self) -> None:
        if self._owns_driver:
            await self._driver.close()

    # -- plumbing --

    # The Cypher argument is positional-only so Cypher parameters may be named "query".

    async def _read(self, cypher: str, /, **params: Any) -> list[dict[str, Any]]:
        result = await self._driver.execute_query(
            cypher, params, routing_=RoutingControl.READ, database_=self._db
        )
        return [_native(record.data()) for record in result.records]

    async def _write(self, cypher: str, /, **params: Any) -> list[dict[str, Any]]:
        result = await self._driver.execute_query(
            cypher, params, routing_=RoutingControl.WRITE, database_=self._db
        )
        return [_native(record.data()) for record in result.records]

    async def _in_write_tx[R](self, work: Callable[[AsyncManagedTransaction], Awaitable[R]]) -> R:
        """Run ``work`` in one write transaction; raising inside it rolls everything back."""
        async with self._driver.session(database=self._db) as session:
            return await session.execute_write(work)

    @staticmethod
    async def _run(
        tx: AsyncManagedTransaction, cypher: str, /, **params: Any
    ) -> list[dict[str, Any]]:
        result = await tx.run(cypher, params)
        return [_native(record.data()) async for record in result]

    # -- users and sessions --

    async def ensure_user(
        self, user_id: str, *, timezone: str | None = None, now: datetime | None = None
    ) -> User:
        [row] = await self._write(
            _ENSURE_USER, user_id=user_id, timezone=timezone, now=now or datetime.now(UTC)
        )
        return User.model_validate(row["user"])

    async def get_user(self, user_id: str) -> User | None:
        rows = await self._read(_GET_USER, user_id=user_id)
        return User.model_validate(rows[0]["user"]) if rows else None

    async def upsert_session(
        self,
        user_id: str,
        session_id: str,
        *,
        started_at: datetime,
        title: str | None = None,
        now: datetime | None = None,
    ) -> Session:
        """Create the session (and user) if new. Raises OwnershipError if another user owns it."""
        [row] = await self._write(
            _UPSERT_SESSION,
            user_id=user_id,
            session_id=session_id,
            started_at=started_at,
            title=title,
            now=now or datetime.now(UTC),
        )
        if not row["owned"]:
            raise OwnershipError(f"session {session_id} belongs to another user")
        return Session.model_validate(row["session"])

    async def get_session(self, user_id: str, session_id: str) -> Session | None:
        rows = await self._read(_GET_SESSION, user_id=user_id, session_id=session_id)
        return Session.model_validate(rows[0]["session"]) if rows else None

    async def end_session(
        self, user_id: str, session_id: str, *, ended_at: datetime
    ) -> Session | None:
        rows = await self._write(
            _END_SESSION, user_id=user_id, session_id=session_id, ended_at=ended_at
        )
        return Session.model_validate(rows[0]["session"]) if rows else None

    # -- turns --

    async def upsert_turns(
        self, user_id: str, session_id: str, turns: Sequence[NewTurn]
    ) -> list[Turn]:
        """Insert raw turns. Re-sending an identical turn is a no-op; changing one is an error."""
        if not turns:
            return []
        rows = [{"turn_id": t.turn_id, "props": _props(t, exclude={"turn_id"})} for t in turns]
        turn_ids = [t.turn_id for t in turns]

        async def work(tx: AsyncManagedTransaction) -> list[Turn]:
            owner = await self._run(tx, _SESSION_OWNER, session_id=session_id)
            if not owner:
                raise InvalidReferenceError(f"session {session_id} does not exist")
            if owner[0]["owner"] != user_id:
                raise OwnershipError(f"session {session_id} belongs to another user")
            foreign = await self._run(
                tx, _FOREIGN_TURN_IDS, turn_ids=turn_ids, user_id=user_id, session_id=session_id
            )
            if foreign:
                ids = [r["turn_id"] for r in foreign]
                raise OwnershipError(f"turn ids already used elsewhere: {ids}")

            stored_rows = await self._run(
                tx,
                _UPSERT_TURNS,
                rows=rows,
                user_id=user_id,
                user_key=user_key(user_id),
                session_id=session_id,
            )
            stored = [Turn.model_validate(r["turn"]) for r in stored_rows]
            for new, old in zip(turns, stored, strict=True):
                if (new.role, new.text, new.seq, new.created_at, new.source) != (
                    old.role,
                    old.text,
                    old.seq,
                    old.created_at,
                    old.source,
                ):
                    raise ImmutableTurnError(
                        f"turn {new.turn_id} already stored with other content"
                    )
            await self._run(tx, _RELINK_SESSIONS, session_ids=[session_id], user_id=user_id)
            return stored

        return await self._in_write_tx(work)

    async def get_turns(self, user_id: str, turn_ids: Sequence[str]) -> list[Turn]:
        rows = await self._read(_GET_TURNS, user_id=user_id, turn_ids=_unique(turn_ids))
        turns = [Turn.model_validate(r["turn"]) for r in rows]
        return _in_order(turns, turn_ids, lambda t: t.turn_id)

    async def last_conversation(
        self, user_id: str, *, exclude_session: str, limit: int
    ) -> list[Turn]:
        """The last ``limit`` turns of the user's most recent other conversation, in order."""
        rows = await self._read(
            _LAST_CONVERSATION, user_id=user_id, exclude=exclude_session, limit=limit
        )
        return [Turn.model_validate(r["turn"]) for r in rows]

    async def get_session_turns(self, user_id: str, session_id: str) -> list[Turn]:
        rows = await self._read(_GET_SESSION_TURNS, user_id=user_id, session_id=session_id)
        return [Turn.model_validate(r["turn"]) for r in rows]

    async def get_turn_window(self, user_id: str, turn_id: str, *, neighbors: int) -> list[Turn]:
        """The turn plus up to ``neighbors`` surviving turns on each side, in seq order."""
        windows = await self.get_turn_windows(user_id, [turn_id], neighbors=neighbors)
        return windows.get(turn_id, [])

    async def get_turn_windows(
        self, user_id: str, turn_ids: Sequence[str], *, neighbors: int
    ) -> dict[str, list[Turn]]:
        """Window around each of this user's turns (±``neighbors`` within its session).

        Keyed by center turn id; ids that are missing or belong to another user are absent.
        """
        rows = await self._read(
            _GET_TURN_WINDOWS, user_id=user_id, turn_ids=_unique(turn_ids), neighbors=neighbors
        )
        return {r["center_id"]: [Turn.model_validate(t) for t in r["window"]] for r in rows}

    async def count_turns(self, user_id: str) -> int:
        [row] = await self._read(_COUNT_TURNS, user_id=user_id)
        return int(row["n"])

    async def next_turn_seq(self, user_id: str, session_id: str) -> int:
        turns = await self.get_session_turns(user_id, session_id)
        return turns[-1].seq + 1 if turns else 0

    # -- entities --

    async def insert_entities(self, user_id: str, entities: Sequence[NewEntity]) -> list[Entity]:
        if not entities:
            return []
        rows = [_entity_row(e) for e in entities]
        result = await self._write(
            _INSERT_ENTITIES, rows=rows, user_id=user_id, user_key=user_key(user_id)
        )
        return [Entity.model_validate(r["entity"]) for r in result]

    async def get_entities(self, user_id: str, entity_ids: Sequence[str]) -> list[Entity]:
        rows = await self._read(_GET_ENTITIES, user_id=user_id, entity_ids=_unique(entity_ids))
        entities = [Entity.model_validate(r["entity"]) for r in rows]
        return _in_order(entities, entity_ids, lambda e: e.entity_id)

    # -- facts --

    async def insert_facts(self, user_id: str, facts: Sequence[NewFact]) -> list[Fact]:
        """ADD-only insert with DERIVED_FROM / MENTIONS edges to this user's turns and entities.

        A fact whose (subject, normalized text) already exists for the user is merged into the
        existing node (exact-duplicate guard) and that node is returned instead.
        Raises InvalidReferenceError (and writes nothing) if any referenced turn or entity
        is missing or belongs to another user.
        """
        if not facts:
            return []

        async def work(tx: AsyncManagedTransaction) -> list[Fact]:
            return [fact for fact, _ in await self._insert_facts_tx(tx, user_id, facts)]

        return await self._in_write_tx(work)

    async def _insert_facts_tx(
        self, tx: AsyncManagedTransaction, user_id: str, facts: Sequence[NewFact]
    ) -> list[tuple[Fact, bool]]:
        """Insert or merge facts inside ``tx``. Returns (fact, created) per input fact."""
        rows: list[dict[str, Any]] = [
            {
                "props": _props(f, exclude={"derived_from_turn_ids", "mentions_entity_ids"}),
                "dedup_key": fact_dedup_key(user_id, f.subject_entity_id, f.text),
                "aliases_text": " ".join(f.aliases),
                "flags_text": fact_flags(is_latest=f.is_latest, is_hypothetical=f.is_hypothetical),
                "turn_ids": _unique(f.derived_from_turn_ids),
                "entity_ids": _unique(f.mentions_entity_ids),
                "subject_entity_id": f.subject_entity_id,
            }
            for f in facts
        ]
        result = await self._run(
            tx, _INSERT_FACTS, rows=rows, user_id=user_id, user_key=user_key(user_id)
        )
        for row, out in zip(rows, result, strict=True):
            fact_id = out["fact"]["fact_id"]
            if not out["owned"]:
                raise OwnershipError(f"dedup key of fact {row['props']['fact_id']} collides")
            if out["turns_linked"] != len(row["turn_ids"]):
                raise InvalidReferenceError(f"fact {fact_id}: unknown turn in {row['turn_ids']}")
            if out["entities_linked"] != len(row["entity_ids"]):
                raise InvalidReferenceError(
                    f"fact {fact_id}: unknown entity in {row['entity_ids']}"
                )
            if not out["subject_ok"]:
                raise InvalidReferenceError(
                    f"fact {fact_id}: unknown subject entity {row['subject_entity_id']}"
                )
        return [(Fact.model_validate(r["fact"]), bool(r["created"])) for r in result]

    async def commit_extraction(
        self,
        user_id: str,
        session_id: str,
        *,
        entities: Sequence[NewEntity],
        facts: Sequence[NewFact],
        expected_watermark: int,
        up_to_seq: int,
        entity_aliases: dict[str, list[str]] | None = None,
        decide: SupersessionDecision | None = None,
        observations: Sequence[str] = (),
        now: datetime | None = None,
    ) -> ExtractionCommit:
        """Write one extraction batch atomically: entities, facts, supersession, watermark.

        ``decide(new, old)`` says which of two facts about the same subject and slot is
        current; losers get is_latest=false and a SUPERSEDES edge from the winner (nothing is
        ever deleted). Raises VersionConflictError if the session's watermark moved since the
        batch was read (another extraction ran), so a batch is never applied twice.
        """

        async def work(tx: AsyncManagedTransaction) -> ExtractionCommit:
            rows = await self._run(
                tx,
                _ADVANCE_WATERMARK,
                user_id=user_id,
                session_id=session_id,
                expected=expected_watermark,
                up_to=up_to_seq,
            )
            if not rows:
                raise VersionConflictError(
                    f"session {session_id} watermark is no longer {expected_watermark}"
                )
            await self._run(
                tx,
                _ENSURE_SELF_ENTITY,
                user_id=user_id,
                user_key=user_key(user_id),
                entity_id=self_entity_id(user_id),
            )
            if entities:
                await self._run(
                    tx,
                    _INSERT_ENTITIES,
                    user_id=user_id,
                    user_key=user_key(user_id),
                    rows=[_entity_row(e) for e in entities],
                )
            if entity_aliases:
                await self._run(
                    tx,
                    _ADD_ENTITY_ALIASES,
                    user_id=user_id,
                    rows=[{"entity_id": k, "aliases": v} for k, v in entity_aliases.items()],
                )
            if observations:
                await self._run(
                    tx,
                    _APPEND_OBSERVATIONS,
                    user_id=user_id,
                    lines="\n".join(observations),
                    chars_per_token=self._chars_per_token,
                    now=now or datetime.now(UTC),
                )
            written = await self._insert_facts_tx(tx, user_id, facts) if facts else []
            created = sum(c for _, c in written)
            superseded = history = conflicts = 0
            if decide is not None:
                new = [f for f, was_created in written if was_created and f.attribute_key]
                for fact in sorted(new, key=lambda f: (f.start or f.mentioned_at, f.mentioned_at)):
                    won, lost, linked = await self._supersede_tx(tx, user_id, fact, decide)
                    superseded += won
                    history += lost
                    conflicts += linked
            return ExtractionCommit(
                facts=[f for f, _ in written],
                created=created,
                merged=len(written) - created,
                entities_created=len(entities),
                superseded=superseded,
                stored_as_history=history,
                conflicts=conflicts,
            )

        return await self._in_write_tx(work)

    async def _supersede_tx(
        self,
        tx: AsyncManagedTransaction,
        user_id: str,
        fact: Fact,
        decide: SupersessionDecision,
    ) -> tuple[int, int, int]:
        """Compare a new fact with the current facts in its slot. Returns (won, lost, conflicts)."""
        rows = await self._run(
            tx,
            _SLOT_RIVALS,
            user_id=user_id,
            attribute_key=fact.attribute_key,
            subject_entity_id=fact.subject_entity_id,
            fact_id=fact.fact_id,
        )
        won = lost = conflicts = 0
        for row in rows:
            rival = Fact.model_validate(row["fact"])
            verdict = decide(fact, rival)
            if verdict is None:
                continue
            if verdict == "conflict":
                await self._run(
                    tx, _LINK_CONFLICT, user_id=user_id, a=fact.fact_id, b=rival.fact_id
                )
                conflicts += 1
                continue
            winner, loser = (fact, rival) if verdict == "new_wins" else (rival, fact)
            await self._run(
                tx,
                _MARK_SUPERSEDED,
                user_id=user_id,
                new_fact_id=winner.fact_id,
                old_fact_ids=[loser.fact_id],
                superseded_factual=fact_flags(is_latest=False, is_hypothetical=False),
                superseded_hypothetical=fact_flags(is_latest=False, is_hypothetical=True),
            )
            if loser is fact:
                lost += 1
                break  # the new fact is history; it supersedes nothing
            won += 1
        return won, lost, conflicts

    async def co_mentioned(self, user_id: str, entity_ids: Sequence[str]) -> dict[str, set[str]]:
        """For each entity, the other entities that appear with it in at least one fact."""
        rows = await self._read(_CO_MENTIONED, user_id=user_id, entity_ids=_unique(entity_ids))
        return {r["eid"]: set(r["co"]) for r in rows}

    async def facts_mentioning(
        self, user_id: str, entity_ids: Sequence[str], k: int
    ) -> list[Hit[Fact]]:
        """S5: facts mentioning or about the given entities, current values first."""
        if not entity_ids:
            return []
        rows = await self._read(
            _FACTS_MENTIONING, user_id=user_id, entity_ids=_unique(entity_ids), k=k
        )
        return [Hit[Fact](item=Fact.model_validate(r["item"]), score=r["score"]) for r in rows]

    async def grep_turns(self, user_id: str, pattern: str) -> list[Hit[Turn]]:
        """Deep mode: every turn of this user containing ``pattern`` (case-insensitive)."""
        needle = pattern.strip().lower()
        if not needle:
            return []
        rows = await self._read(_GREP_TURNS, user_id=user_id, needle=needle)
        return [Hit[Turn](item=Turn.model_validate(r["item"]), score=r["score"]) for r in rows]

    async def grep_facts(self, user_id: str, pattern: str) -> list[Hit[Fact]]:
        """Deep mode: every non-hypothetical fact whose text or aliases contain ``pattern``."""
        needle = pattern.strip().lower()
        if not needle:
            return []
        rows = await self._read(_GREP_FACTS, user_id=user_id, needle=needle)
        return [Hit[Fact](item=Fact.model_validate(r["item"]), score=r["score"]) for r in rows]

    async def timeline(
        self,
        user_id: str,
        entity_ids: Sequence[str],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Hit[Fact]]:
        """Deep mode: every fact about or mentioning the entities, oldest event first."""
        if not entity_ids:
            return []
        rows = await self._read(
            _TIMELINE, user_id=user_id, entity_ids=_unique(entity_ids), start=start, end=end
        )
        return [Hit[Fact](item=Fact.model_validate(r["item"]), score=r["score"]) for r in rows]

    async def expand_facts(self, user_id: str, seed_ids: Sequence[str], k: int) -> list[Hit[Fact]]:
        """S7: facts one hop from the seeds (shared entities, SUPERSEDES either way)."""
        if not seed_ids:
            return []
        rows = await self._read(_EXPAND_FACTS, user_id=user_id, seed_ids=_unique(seed_ids), k=k)
        return [Hit[Fact](item=Fact.model_validate(r["item"]), score=r["score"]) for r in rows]

    async def replace_observation_log(
        self,
        user_id: str,
        text: str,
        token_count: int,
        *,
        expected_version: int,
        reason: str,
        now: datetime | None = None,
    ) -> ObservationLog:
        """Rewrite the log (Reflector), snapshotting the old text as an ObservationLogVersion.

        Raises VersionConflictError if the log changed since ``expected_version``.
        """
        rows = await self._write(
            _REPLACE_LOG,
            user_id=user_id,
            text=text,
            token_count=token_count,
            expected_version=expected_version,
            reason=reason,
            now=now or datetime.now(UTC),
        )
        if not rows:
            raise VersionConflictError(f"observation log moved past version {expected_version}")
        return ObservationLog.model_validate(rows[0]["log"])

    async def observation_log_versions(self, user_id: str) -> list[dict[str, Any]]:
        return await self._read(_LOG_VERSIONS, user_id=user_id)

    async def rollback_observation_log(self, user_id: str, to_version: int) -> ObservationLog:
        """Restore a snapshot as the newest version (the current text is snapshotted first)."""
        rows = await self._read(_LOG_VERSION, user_id=user_id, version=to_version)
        if not rows:
            raise InvalidReferenceError(f"no observation log version {to_version}")
        current = await self.get_observation_log(user_id)
        if current is None:
            raise InvalidReferenceError("user has no observation log")
        return await self.replace_observation_log(
            user_id,
            rows[0]["text"],
            int(rows[0]["token_count"]),
            expected_version=current.version,
            reason=f"rollback to v{to_version}",
        )

    async def entities_for_summary(
        self, user_id: str, entity_ids: Sequence[str], *, min_facts: int, max_facts: int
    ) -> list[tuple[Entity, list[str]]]:
        """Entities with at least ``min_facts`` facts, with their latest fact texts."""
        rows = await self._read(
            _ENTITIES_FOR_SUMMARY,
            user_id=user_id,
            entity_ids=_unique(entity_ids),
            min_facts=min_facts,
            max_facts=max_facts,
        )
        return [(Entity.model_validate(r["entity"]), list(r["facts"])) for r in rows]

    async def clear_entity_summaries(self, user_id: str, entity_ids: Sequence[str]) -> None:
        if entity_ids:
            await self._write(_CLEAR_ENTITY_SUMMARIES, user_id=user_id, ids=_unique(entity_ids))

    async def set_entity_summary(
        self, user_id: str, entity_id: str, summary: str, now: datetime | None = None
    ) -> None:
        await self._write(
            _SET_ENTITY_SUMMARY,
            user_id=user_id,
            entity_id=entity_id,
            summary=summary,
            now=now or datetime.now(UTC),
        )

    async def ensure_self_entity(self, user_id: str) -> str:
        await self._write(
            _ENSURE_SELF_ENTITY,
            user_id=user_id,
            user_key=user_key(user_id),
            entity_id=self_entity_id(user_id),
        )
        return self_entity_id(user_id)

    async def find_entities_by_names(self, user_id: str, names: Sequence[str]) -> list[Entity]:
        """This user's entities whose name or alias matches one of ``names`` (case-insensitive)."""
        keys = _unique(n.strip().lower() for n in names if n.strip())
        if not keys:
            return []
        rows = await self._read(_FIND_ENTITIES_BY_NAME, user_id=user_id, names=keys)
        return [Entity.model_validate(r["entity"]) for r in rows]

    async def pending_extraction(self, user_id: str) -> list[PendingSession]:
        """Sessions with turns after their extraction watermark, oldest activity first."""
        rows = await self._read(_PENDING_EXTRACTION, user_id=user_id)
        return [PendingSession(**r) for r in rows]

    async def fact_summary(self, user_id: str) -> dict[str, Any]:
        """Counts of this user's facts by type, attribution, and hypothetical flag."""
        rows = await self._read(_FACT_SUMMARY, user_id=user_id)
        summary: dict[str, Any] = {
            "facts": 0, "mentions": 0, "hypothetical": 0, "superseded": 0,
            "by_type": {}, "by_attribution": {},
        }  # fmt: skip
        for r in rows:
            n = int(r["n"])
            summary["facts"] += n
            summary["mentions"] += int(r["mentions"])
            summary["hypothetical"] += n if r["hypothetical"] else 0
            summary["superseded"] += 0 if r["latest"] else n
            for key, value in (("by_type", r["fact_type"]), ("by_attribution", r["attribution"])):
                summary[key][value] = summary[key].get(value, 0) + n
        return summary

    async def turns_with_facts(self, user_id: str) -> set[str]:
        """Ids of this user's turns that at least one fact was derived from."""
        rows = await self._read(_TURNS_WITH_FACTS, user_id=user_id)
        return {r["turn_id"] for r in rows}

    async def temporal_search_facts(
        self,
        user_id: str,
        start: datetime | None,
        end: datetime | None,
        k: int,
        *,
        scan_limit: int,
    ) -> list[Hit[Fact]]:
        """S6: facts whose event time overlaps [start, end), scored by overlap / union.

        A fact without ``end`` is treated as lasting one day from ``start``.
        """
        if start is None and end is None:
            return []
        rows = await self._read(
            _TEMPORAL_FACTS, user_id=user_id, **{"from": start, "to": end}, scan=scan_limit
        )
        hits = [
            Hit[Fact](item=fact, score=_overlap_score(fact, start, end))
            for fact in (Fact.model_validate(r["item"]) for r in rows)
        ]
        hits.sort(key=lambda h: (-h.score, h.item.fact_id))
        return hits[:k]

    async def get_facts(self, user_id: str, fact_ids: Sequence[str]) -> list[Fact]:
        rows = await self._read(_GET_FACTS, user_id=user_id, fact_ids=_unique(fact_ids))
        facts = [Fact.model_validate(r["fact"]) for r in rows]
        return _in_order(facts, fact_ids, lambda f: f.fact_id)

    async def mark_superseded(
        self, user_id: str, new_fact_id: str, old_fact_ids: Sequence[str]
    ) -> int:
        """Set is_latest=false on the old facts and link SUPERSEDES. Never deletes.

        Returns how many old facts were marked (facts of other users are ignored).
        """
        [row] = await self._write(
            _MARK_SUPERSEDED,
            user_id=user_id,
            new_fact_id=new_fact_id,
            old_fact_ids=_unique(old_fact_ids),
            superseded_factual=fact_flags(is_latest=False, is_hypothetical=False),
            superseded_hypothetical=fact_flags(is_latest=False, is_hypothetical=True),
        )
        return int(row["superseded"])

    # -- observation log --

    async def get_observation_log(self, user_id: str) -> ObservationLog | None:
        rows = await self._read(_GET_LOG, user_id=user_id)
        return ObservationLog.model_validate(rows[0]["log"]) if rows else None

    async def save_observation_log(
        self,
        user_id: str,
        text: str,
        token_count: int,
        *,
        expected_version: int,
        now: datetime | None = None,
    ) -> ObservationLog:
        """Replace the log if it is still at ``expected_version`` (0 for a new log)."""

        async def work(tx: AsyncManagedTransaction) -> ObservationLog:
            rows = await self._run(
                tx,
                _SAVE_LOG,
                user_id=user_id,
                text=text,
                token_count=token_count,
                expected_version=expected_version,
                now=now or datetime.now(UTC),
            )
            if not rows:
                raise VersionConflictError(
                    f"observation log for user changed since version {expected_version}"
                )
            return ObservationLog.model_validate(rows[0]["log"])

        return await self._in_write_tx(work)

    # -- deletion --

    async def delete_by_ids(
        self,
        user_id: str,
        *,
        fact_ids: Sequence[str] = (),
        turn_ids: Sequence[str] = (),
        entity_ids: Sequence[str] = (),
    ) -> DeleteCounts:
        """Hard-delete this user's nodes (with their embeddings and edges) in one transaction.

        Only for explicit forget requests. NEXT chains are relinked around deleted turns.
        """

        async def work(tx: AsyncManagedTransaction) -> DeleteCounts:
            [facts] = await self._run(tx, _DELETE_FACTS, ids=_unique(fact_ids), user_id=user_id)
            [turns] = await self._run(tx, _DELETE_TURNS, ids=_unique(turn_ids), user_id=user_id)
            [entities] = await self._run(
                tx, _DELETE_ENTITIES, ids=_unique(entity_ids), user_id=user_id
            )
            if turns["session_ids"]:
                await self._run(
                    tx, _RELINK_SESSIONS, session_ids=turns["session_ids"], user_id=user_id
                )
            return DeleteCounts(
                facts=int(facts["deleted"]),
                turns=int(turns["deleted"]),
                entities=int(entities["deleted"]),
            )

        return await self._in_write_tx(work)

    async def facts_derived_from(self, user_id: str, turn_ids: Sequence[str]) -> list[Fact]:
        """Facts extracted from any of these turns."""
        if not turn_ids:
            return []
        rows = await self._read(_FACTS_DERIVED_FROM, user_id=user_id, turn_ids=_unique(turn_ids))
        return [Fact.model_validate(r["item"]) for r in rows]

    async def save_pending_forget(
        self,
        user_id: str,
        *,
        preview_id: str,
        session_id: str | None,
        description: str,
        fact_ids: Sequence[str],
        turn_ids: Sequence[str],
        request_text: str,
        now: datetime,
        not_before: datetime,
    ) -> None:
        """Store a forget preview awaiting confirmation (and drop this user's expired ones)."""
        await self._write(
            _SAVE_PENDING_FORGET, user_id=user_id, preview_id=preview_id, session_id=session_id,
            description=description, fact_ids=list(fact_ids), turn_ids=list(turn_ids),
            request_text=request_text, preview_text="", now=now, not_before=not_before,
        )  # fmt: skip

    async def set_pending_preview_text(self, user_id: str, preview_id: str, text: str) -> None:
        await self._write(
            _SET_PENDING_PREVIEW_TEXT, user_id=user_id, preview_id=preview_id, preview_text=text
        )

    async def pending_forget(
        self, user_id: str, preview_id: str, *, not_before: datetime
    ) -> dict[str, Any] | None:
        rows = await self._read(
            _PENDING_FORGET, user_id=user_id, preview_id=preview_id, not_before=not_before
        )
        return dict(rows[0]["pending"]) if rows else None

    async def session_pending_forget(
        self, user_id: str, session_id: str, *, not_before: datetime
    ) -> dict[str, Any] | None:
        rows = await self._read(
            _SESSION_PENDING_FORGET, user_id=user_id, session_id=session_id,
            not_before=not_before,
        )  # fmt: skip
        return dict(rows[0]["pending"]) if rows else None

    async def delete_pending_forget(self, user_id: str, preview_id: str) -> bool:
        [row] = await self._write(_DELETE_PENDING_FORGET, user_id=user_id, preview_id=preview_id)
        return int(row["n"]) > 0

    async def session_turns_with_text(
        self, user_id: str, session_id: str, texts: Sequence[str]
    ) -> list[str]:
        rows = await self._read(
            _SESSION_TURNS_WITH_TEXT, user_id=user_id, session_id=session_id,
            texts=[t for t in texts if t],
        )  # fmt: skip
        return [str(r["turn_id"]) for r in rows]

    async def delete_orphan_entities(self, user_id: str, entity_ids: Sequence[str]) -> int:
        if not entity_ids:
            return 0
        [row] = await self._write(_DELETE_ORPHAN_ENTITIES, user_id=user_id, ids=_unique(entity_ids))
        return int(row["deleted"])

    async def purge_observation_log_versions(self, user_id: str) -> int:
        """Drop every saved profile version (they may hold forgotten content)."""
        [row] = await self._write(_PURGE_LOG_VERSIONS, user_id=user_id)
        return int(row["n"])

    async def record_deletion(
        self, user_id: str, *, kind: str, facts: int, turns: int, entities: int, now: datetime
    ) -> None:
        """Audit trail without content: only when and how much was deleted."""
        await self._write(
            _RECORD_DELETION, user_id=user_id, kind=kind, facts=facts, turns=turns,
            entities=entities, now=now,
        )  # fmt: skip

    async def reembed_pending(self, label: str, model: str, batch: int) -> list[tuple[str, str]]:
        """Up to ``batch`` (id, text) of ``label`` nodes not yet embedded with ``model``.

        Spans all users: an embedding model change applies to the whole database.
        """
        rows = await self._read(_REEMBED_PENDING[label], model=model, batch=batch)
        return [(str(r["id"]), str(r["text"] or "")) for r in rows]

    async def set_next_embeddings(
        self, label: str, model: str, rows: Sequence[tuple[str, Sequence[float]]]
    ) -> None:
        await self._write(
            _REEMBED_SET_NEXT[label],
            model=model,
            rows=[{"id": i, "vector": list(v)} for i, v in rows],
        )

    async def reembed_remaining(self, label: str, model: str) -> int:
        [row] = await self._read(_REEMBED_REMAINING[label], model=model)
        return int(row["n"])

    async def swap_embeddings(self, label: str, model: str) -> int:
        """Copy every staged vector of ``label`` into ``embedding`` in one transaction."""
        [row] = await self._write(_REEMBED_SWAP[label], model=model)
        return int(row["swapped"])

    async def list_user_ids(self) -> list[str]:
        """Every user (for nightly jobs that run per user)."""
        return [str(r["user_id"]) for r in await self._read(_LIST_USER_IDS)]

    async def list_entities(self, user_id: str) -> list[Entity]:
        """The user's entities (not the self entity), most mentioned first."""
        rows = await self._read(_LIST_ENTITIES, user_id=user_id)
        return [Entity.model_validate(r["entity"]) for r in rows]

    async def merge_entities(self, user_id: str, *, keep: str, drop: str) -> bool:
        rows = await self._write(_MERGE_ENTITIES, user_id=user_id, keep=keep, drop=drop)
        return bool(rows) and int(rows[0]["merged"]) > 0

    async def repair_latest_flags(self, user_id: str) -> int:
        """Make is_latest agree with SUPERSEDES; returns how many facts were fixed."""
        [row] = await self._write(
            _REPAIR_LATEST,
            user_id=user_id,
            latest_factual=fact_flags(is_latest=True, is_hypothetical=False),
            latest_hypothetical=fact_flags(is_latest=True, is_hypothetical=True),
            superseded_factual=fact_flags(is_latest=False, is_hypothetical=False),
            superseded_hypothetical=fact_flags(is_latest=False, is_hypothetical=True),
        )
        return int(row["fixed"])

    async def slot_clashes(self, user_id: str) -> int:
        [row] = await self._read(_SLOT_CLASHES, user_id=user_id)
        return int(row["clashes"])

    async def orphan_facts(self, user_id: str) -> int:
        [row] = await self._read(_ORPHAN_FACTS, user_id=user_id)
        return int(row["orphans"])

    async def list_facts(self, user_id: str) -> list[Fact]:
        """Every fact of the user (current and superseded), oldest mention first."""
        rows = await self._read(_EXPORT["facts"], user_id=user_id)
        return [Fact.model_validate(r["item"]) for r in rows]

    async def export_user(self, user_id: str) -> dict[str, Any]:
        """Everything stored for the user, without embeddings."""
        out: dict[str, Any] = {"user": None}
        user = await self.get_user(user_id)
        out["user"] = user.model_dump(mode="json") if user else None
        for key, query in _EXPORT.items():
            rows = await self._read(query, user_id=user_id)
            out[key] = [_jsonable(r["item"]) for r in rows]
        return out

    async def delete_user_data(self, user_id: str) -> dict[str, int]:
        """Hard-delete every node belonging to the user, in one transaction. Returns counts."""

        async def work(tx: AsyncManagedTransaction) -> dict[str, int]:
            counts: dict[str, int] = {}
            for label, query in _DELETE_USER_LABEL.items():
                [row] = await self._run(tx, query, user_id=user_id)
                counts[label] = int(row["n"])
            return counts

        return await self._in_write_tx(work)

    # -- search --

    async def vector_search_turns(
        self,
        user_id: str,
        vector: Sequence[float],
        k: int,
        *,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[Hit[Turn]]:
        """Nearest turns; optional [created_from, created_to) filter runs inside the index."""
        query = _VECTOR_TURNS[(created_from is not None, created_to is not None)]
        rows = await self._read(
            query,
            user_id=user_id,
            vector=list(vector),
            k=k,
            created_from=created_from,
            created_to=created_to,
        )
        return [Hit[Turn](item=Turn.model_validate(r["item"]), score=r["score"]) for r in rows]

    async def vector_search_facts(
        self,
        user_id: str,
        vector: Sequence[float],
        k: int,
        *,
        latest_only: bool = False,
        include_hypothetical: bool = False,
    ) -> list[Hit[Fact]]:
        query = _VECTOR_FACTS[(latest_only, include_hypothetical)]
        rows = await self._read(query, user_id=user_id, vector=list(vector), k=k)
        return [Hit[Fact](item=Fact.model_validate(r["item"]), score=r["score"]) for r in rows]

    async def vector_search_entities(
        self, user_id: str, vector: Sequence[float], k: int
    ) -> list[Hit[Entity]]:
        rows = await self._read(_VECTOR_ENTITIES, user_id=user_id, vector=list(vector), k=k)
        return [Hit[Entity](item=Entity.model_validate(r["item"]), score=r["score"]) for r in rows]

    async def fulltext_search_turns(self, user_id: str, text: str, k: int) -> list[Hit[Turn]]:
        query = fulltext_query(user_id, text, _TURN_FIELDS, max_terms=self._max_terms)
        if query is None:
            return []
        rows = await self._read(_FULLTEXT_TURNS, user_id=user_id, query=query, k=k)
        return [Hit[Turn](item=Turn.model_validate(r["item"]), score=r["score"]) for r in rows]

    async def fulltext_search_facts(
        self,
        user_id: str,
        text: str,
        k: int,
        *,
        latest_only: bool = False,
        include_hypothetical: bool = False,
    ) -> list[Hit[Fact]]:
        require = [("flags_text", FLAG_LATEST)] if latest_only else []
        if not include_hypothetical:
            require.append(("flags_text", FLAG_FACTUAL))
        query = fulltext_query(
            user_id, text, _FACT_FIELDS, max_terms=self._max_terms, require=require
        )
        if query is None:
            return []
        rows = await self._read(_FULLTEXT_FACTS, user_id=user_id, query=query, k=k)
        return [Hit[Fact](item=Fact.model_validate(r["item"]), score=r["score"]) for r in rows]

    async def fulltext_search_entities(self, user_id: str, text: str, k: int) -> list[Hit[Entity]]:
        query = fulltext_query(user_id, text, _ENTITY_FIELDS, max_terms=self._max_terms)
        if query is None:
            return []
        rows = await self._read(_FULLTEXT_ENTITIES, user_id=user_id, query=query, k=k)
        return [Hit[Entity](item=Entity.model_validate(r["item"]), score=r["score"]) for r in rows]
