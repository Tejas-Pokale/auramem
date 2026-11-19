"""ObservationLog and entity summaries (PLAN.md section 5 steps 10 and 12, Phase 6).

The ObservationLog is a compact, always-on profile per user: dated, priority-tagged bullets
appended by every extraction batch and placed right after the system prompt on every read (a
stable prefix the provider's prompt cache can reuse). When it grows past OBSLOG_REFLECT_AT
tokens the Reflector (Sonnet 5.5) condenses it to about OBSLOG_TARGET tokens. The previous
version is snapshotted first, and a rewrite that loses high-priority bullets is rejected, so
the log can always be rolled back.

Entity summaries are short neutral profiles (Haiku 5.5) of entities with enough facts, used
by the entity tiebreak and available to retrieval.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from memory.config import Settings
from memory.graphs.llm import chat_model, structured
from memory.models import EntitySummaryOutput, Observation, ReflectionOutput
from memory.prompts import load_prompt
from memory.store.neo4j import Neo4jStore, VersionConflictError

log = logging.getLogger(__name__)

_HIGH = re.compile(r"\(high\)", re.IGNORECASE)
_BULLET = re.compile(r"^- \[(\d{4}-\d{2}-\d{2})\] \((high|medium|low)\) (.+)$")


def format_observations(observations: Sequence[Observation]) -> list[str]:
    """One bullet per observation: ``- [YYYY-MM-DD] (priority) text``."""
    return [
        f"- [{o.date:%Y-%m-%d}] ({o.priority}) {' '.join(o.text.split())}"
        for o in observations
        if o.text.strip()
    ]


def high_priority_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if _HIGH.search(line)]


def estimate_tokens(text: str, settings: Settings) -> int:
    return round(len(text) / settings.chars_per_token)


class ReflectorLLM(Protocol):
    async def __call__(self, messages: list[BaseMessage]) -> ReflectionOutput: ...


class SummaryLLM(Protocol):
    async def __call__(self, messages: list[BaseMessage]) -> EntitySummaryOutput: ...


def claude_reflector(settings: Settings) -> ReflectorLLM:
    return structured(
        chat_model(
            settings.reflector_model,
            settings,
            effort=settings.reflector_effort,
            max_tokens=settings.reflector_max_tokens,
        ),
        ReflectionOutput,
    )


def claude_summarizer(settings: Settings) -> SummaryLLM:
    return structured(
        chat_model(
            settings.entity_summary_model,
            settings,
            effort=settings.entity_summary_effort,
            max_tokens=settings.entity_summary_max_tokens,
        ),
        EntitySummaryOutput,
    )


@dataclass(frozen=True)
class ReflectResult:
    before_tokens: int
    after_tokens: int
    version: int


_KEY_TERM = re.compile(r"\$?\d[\d,.:/%]*\d|\$?\d|\b[A-Z][\w'.-]{2,}")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _key_terms(lines: Sequence[str]) -> set[str]:
    """Names and numbers in the bullets' text (dates excluded: merging may change them)."""
    terms: set[str] = set()
    for line in lines:
        m = _BULLET.match(line)
        text = _DATE.sub(" ", m.group(3) if m else line)
        terms |= {t.rstrip(".,;:").lower() for t in _KEY_TERM.findall(text)}
    return {t for t in terms if t}


def _kept_high_priority(before: str, after: str, *, min_coverage: float) -> bool:
    """High-priority substance survives: at least ``min_coverage`` of the names and numbers
    in high-priority bullets still appear. Bullets may be merged, reworded, and re-dated
    (the Reflector is told to date a merged bullet with its newest information)."""
    terms = _key_terms(high_priority_lines(before))
    if not terms:
        return True
    text = after.lower()
    return sum(t in text for t in terms) / len(terms) >= min_coverage


async def reflect(
    store: Neo4jStore, llm: ReflectorLLM, settings: Settings, user_id: str, *, force: bool = False
) -> ReflectResult | None:
    """Condense the user's log if it is over OBSLOG_REFLECT_AT tokens. None when not needed."""
    current = await store.get_observation_log(user_id)
    if current is None:
        return None
    # Measured now, not the stored count, which may predate a chars_per_token change.
    tokens = estimate_tokens(current.text, settings)
    if tokens <= settings.obslog_reflect_at and not force:
        return None
    rewritten = await _rewrite(llm, settings, current.text, tokens)
    if not _kept_high_priority(
        current.text, rewritten, min_coverage=settings.reflector_min_high_coverage
    ):
        log.warning("reflector dropped high-priority items for %s; keeping the old log", user_id)
        return None
    try:
        saved = await store.replace_observation_log(
            user_id,
            rewritten,
            estimate_tokens(rewritten, settings),
            expected_version=current.version,
            reason=f"reflector_{settings.reflector_prompt_version}",
        )
    except VersionConflictError:
        log.info("observation log changed during reflection for %s; will retry later", user_id)
        return None
    return ReflectResult(tokens, saved.token_count, saved.version)


async def _rewrite(llm: ReflectorLLM, settings: Settings, text: str, tokens: int) -> str:
    version = settings.reflector_prompt_version
    user = load_prompt("reflect", version, "user").text.format(
        log=text,
        tokens=tokens,
        target=settings.obslog_target,
    )
    messages: list[BaseMessage] = [
        SystemMessage(load_prompt("reflect", version, "system").text),
        HumanMessage(user),
    ]
    output = await llm(messages)
    lines = [line.rstrip() for line in output.text.splitlines() if line.strip()]
    return "\n".join(lines)


async def summarize_entities(
    store: Neo4jStore,
    llm: SummaryLLM,
    settings: Settings,
    user_id: str,
    entity_ids: Sequence[str],
) -> int:
    """(Re)write summaries for the given entities that have enough facts. Returns count."""
    targets = await store.entities_for_summary(
        user_id,
        entity_ids,
        min_facts=settings.entity_summary_min_facts,
        max_facts=settings.entity_summary_facts,
    )
    version = settings.entity_summary_prompt_version
    written = 0
    for entity, facts in targets:
        user = load_prompt("entity_summary", version, "user").text.format(
            name=entity.name,
            entity_type=entity.entity_type.value,
            aliases=", ".join(entity.aliases) or "(none)",
            facts="\n".join(f"- {f}" for f in facts),
        )
        messages: list[BaseMessage] = [
            SystemMessage(load_prompt("entity_summary", version, "system").text),
            HumanMessage(user),
        ]
        try:
            output = await llm(messages)
        except Exception:  # a missing summary is harmless; never fail the batch for it
            log.warning("entity summary failed for %s", entity.entity_id, exc_info=True)
            continue
        await store.set_entity_summary(user_id, entity.entity_id, output.summary.strip())
        written += 1
    return written
