"""Generate a fully synthetic memory benchmark: one invented person, ~500 conversations, ~200
verified questions.

    uv run python -m evals.synth.generate            # all stages, resumable
    uv run python -m evals.synth.generate --stage questions

Stages (each writes to evals/synth/data/ and is skipped if its output exists):
  plan       Sonnet: persona, attribute histories, dated events, planned contradictions.
  ledger     Sonnet, one call per month in parallel: every fact that will be said, with ids.
  sessions   Haiku: render each session of ledger facts as a natural chat; every turn lists the
             fact ids it states, and each fact is checked to appear with its key values.
  distractors Haiku: general Q&A sessions with no personal content, up to the target count.
  questions  Sonnet: draft questions per test type from the ledger (over-generated).
  verify     Sonnet: check each draft against the whole ledger; keep the target mix.
The ledger (what was said) is the ground truth for questions, not the plan.
"""

import argparse
import asyncio
import calendar
import csv
import itertools
import json
import logging
import random
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

import anthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from evals.custom.generate_distractors import PROMPT as DISTRACTOR_PROMPT
from evals.custom.generate_distractors import DistractorSession
from evals.synth.models import (
    AttributesPart,
    ConflictsPart,
    DistractorTopics,
    EventsPart,
    LedgerFact,
    LifePlan,
    MonthDraft,
    PersonaPart,
    QuestionBatch,
    RenderedSession,
    VerificationBatch,
)
from memory.config import Settings, get_settings
from memory.graphs.llm import chat_model, structured

log = logging.getLogger("evals.synth")

DATA = Path(__file__).parent / "data"
PLAN = DATA / "plan.json"
LEDGER = DATA / "ledger.jsonl"
SESSIONS = DATA / "sessions.jsonl"
TOPICS = DATA / "distractor_topics.json"
DISTRACTORS = DATA / "distractors.jsonl"
DRAFTS = DATA / "questions_draft.jsonl"
QUESTIONS = DATA / "questions.csv"
MANIFEST = DATA / "manifest.json"

STRONG = "claude-sonnet-5-5"
FAST = "claude-haiku-5-5"
SEED = 500
MONTHS = [f"{y}-{m:02d}" for y, m in [(2024, 11), (2024, 12)] + [(2025, i) for i in range(1, 13)]
          + [(2026, i) for i in range(1, 11)]]  # fmt: skip
TARGET_CONVERSATIONS = 500
ASKED_AT = "2026-11-01"
CONCURRENCY = 6
RETRIES = 4
TARGETS = {
    "single_fact": 45, "multi_fact": 30, "knowledge_update": 30, "temporal": 30,
    "conflict": 10, "abstention": 20, "assistant_recall": 15, "preference_application": 10,
    "sensitive_history": 5, "inference": 5,
}  # fmt: skip
OVERGENERATE = 1.25
DEV_SHARE = 0.3  # per test type; tune on dev only (CLAUDE.md), report on test
PAIR = 2
MIN_NAME_LEN = 2

M = TypeVar("M", bound=BaseModel)


# --- LLM plumbing ---------------------------------------------------------------------------


async def ask[M: BaseModel](
    settings: Settings, model: str, schema: type[M], system: str, user: str, *,
    effort: str = "medium", max_tokens: int = 16000,
) -> M:  # fmt: skip
    call = structured(chat_model(model, settings, effort=effort, max_tokens=max_tokens), schema)
    for attempt in range(1, RETRIES + 1):
        try:
            return await call([SystemMessage(system), HumanMessage(user)])
        except (ValidationError, ValueError, OutputParserException, anthropic.APIError) as e:
            if attempt == RETRIES:
                raise
            log.warning("%s call failed (%s); retry %d", schema.__name__, e, attempt)
            await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")


async def bounded[T](items: Sequence[Any], fn: Callable[[Any], Awaitable[T]]) -> list[T]:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(item: Any) -> T:
        async with sem:
            return await fn(item)

    return list(await asyncio.gather(*(one(i) for i in items)))


def write_jsonl(path: Path, rows: Sequence[BaseModel | dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            data = row.model_dump() if isinstance(row, BaseModel) else row
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


# --- Stage A: plan ----------------------------------------------------------------------------

PLAN_SYSTEM = """You design a realistic, internally consistent life for a fictional person, used to
test an AI assistant's long-term memory. Everything must be invented (no real private people).
Be concrete: names, places, companies, numbers, dates."""

PERIOD = "between 2024-11-01 and 2026-10-31"

PERSONA_USER = f"""Invent one fictional adult whose life we follow {PERIOD} while they chat
with an AI assistant. Give: name, preferred name, birth year, hometown, background, personality,
and 8-14 recurring people (family, partner or dating, friends, manager, colleagues, a doctor, a
pet), each with a relation and one or two facts. Also give 15 stable preferences (food, tools,
how they like answers formatted, schedule habits, dislikes) and 6 sensitive matters with dates in
the period (a health scare, a family member's illness, money stress, a breakup or grief)."""

ATTRIBUTES_USER = """Persona (JSON):
{persona}

List 25-30 single-valued attributes of this person's life, each with a dated value history
({period}; the first value may start earlier). At least 14 must CHANGE during the period, some
twice: employer, job title, salary, city/home, monthly rent, phone, laptop, relationship status,
commute, workout routine, running personal best, diet, manager's name, current course, savings
goal. Others stay stable (blood type, hometown, number of siblings...). Values must be concrete
(names, numbers with units)."""

EVENTS_USER = """Persona (JSON):
{persona}

Attribute histories (JSON):
{attributes}

List 35-40 dated events between {start} and {end}, spread across every month: trips with
destinations, interviews, an injury and recovery, weddings, purchases with prices, exams, races
with times, family news, hobbies started and dropped. Events must agree with the attribute
histories (a job change happens on its date). Give concrete names, places, and numbers."""

CONFLICTS_USER = """Persona (JSON):
{persona}

Attribute histories (JSON):
{attributes}

Plan 10 contradictions: the person states two different values for the same thing in two
different months ({period}) without it being a real change (misremembered dates or scores, two
quoted prices, different counts). Use a short snake_case topic slug, both values, both months
(YYYY-MM), and why they differ."""


async def stage_plan(settings: Settings) -> LifePlan:
    if PLAN.exists():
        return LifePlan.model_validate_json(PLAN.read_text(encoding="utf-8"))
    log.info("stage plan: persona")
    persona = await ask(settings, STRONG, PersonaPart, PLAN_SYSTEM, PERSONA_USER)
    persona_json = persona.persona.model_dump_json()
    log.info("stage plan: attributes")
    attributes = await ask(
        settings, STRONG, AttributesPart, PLAN_SYSTEM,
        ATTRIBUTES_USER.format(persona=persona_json, period=PERIOD),
    )  # fmt: skip
    attributes_json = attributes.model_dump_json()
    log.info("stage plan: events and conflicts")
    first, second, conflicts = await asyncio.gather(
        ask(settings, STRONG, EventsPart, PLAN_SYSTEM, EVENTS_USER.format(
            persona=persona_json, attributes=attributes_json, start="2024-11-01", end="2025-10-31"
        )),
        ask(settings, STRONG, EventsPart, PLAN_SYSTEM, EVENTS_USER.format(
            persona=persona_json, attributes=attributes_json, start="2025-11-01", end="2026-10-31"
        )),
        ask(settings, STRONG, ConflictsPart, PLAN_SYSTEM, CONFLICTS_USER.format(
            persona=persona_json, attributes=attributes_json, period=PERIOD
        )),
    )  # fmt: skip
    plan = LifePlan(
        persona=persona.persona,
        attributes=attributes.attributes,
        events=sorted(first.events + second.events, key=lambda e: e.date),
        conflicts=conflicts.conflicts,
        preferences=persona.preferences,
        sensitive_topics=persona.sensitive_topics,
    )
    PLAN.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return plan


# --- Stage B: ledger --------------------------------------------------------------------------

MONTH_SYSTEM = """You write the exact facts a fictional user (and their AI assistant) state in
chat sessions during one month. Facts must be consistent with the life plan: only say what is
true at that time, announce attribute changes in the month they happen (or shortly after),
and tell events on or after their date."""

MONTH_USER = """Life plan (JSON):
{plan}

Month: {month}
Attribute values that change this month: {changes}
Events this month: {events}
Planned contradictions said this month: {conflicts}

Write the facts said during {n_sessions} chat sessions this month
(sessions numbered 1..{n_sessions}, 1-4 facts each, about {n_facts} facts in total). Mix of kinds:
- update: announcing a changed attribute value (use the plan's attribute key and new value).
- event: telling something that happened (event_date set), with names/places/numbers.
- personal: other details of their life, opinions, plans, the people around them.
- preference: one of their preferences, stated naturally (not every month).
- third_party: facts about another person in their life.
- assistant_advice (speaker assistant): concrete advice or a recommendation the assistant gives
  for the user's situation in that session (1-3 per month).
- hypothetical: an "if I ..." / "maybe someday" statement (0-1 per month).
- conflict: the planned contradiction for this month, stated as plain fact, using the same
  attribute key as the earlier version (set attribute to the conflict topic slug).
- sensitive: a sensitive matter from the plan, if one happens this month.
Also restate a few current attribute values in passing (kind personal, with attribute/value).
Every statement must be a single natural sentence that contains the key value verbatim."""


def _month_context(plan: LifePlan, month: str) -> dict[str, Any]:
    changes = [
        f"{a.key} -> {v.value} (from {v.from_date})"
        for a in plan.attributes
        for v in a.history
        if v.from_date.startswith(month)
    ]
    events = [f"{e.date}: {e.title} - {e.details}" for e in plan.events if e.date.startswith(month)]
    conflicts = [
        f"{c.topic}: first says '{c.first_value}'"
        if c.first_month == month
        else f"{c.topic}: now says '{c.second_value}' (earlier said '{c.first_value}')"
        for c in plan.conflicts
        if month in (c.first_month, c.second_month)
    ]
    return {
        "changes": "; ".join(changes) or "none",
        "events": "; ".join(events) or "none",
        "conflicts": "; ".join(conflicts) or "none",
    }


async def stage_ledger(settings: Settings, plan: LifePlan) -> list[LedgerFact]:
    if LEDGER.exists():
        return [LedgerFact.model_validate(r) for r in read_jsonl(LEDGER)]
    rng = random.Random(SEED)
    persona_sessions = TARGET_CONVERSATIONS - 150  # the rest are distractors
    per_month = [persona_sessions // len(MONTHS)] * len(MONTHS)
    for i in rng.sample(range(len(MONTHS)), persona_sessions - sum(per_month)):
        per_month[i] += 1
    plan_json = plan.model_dump_json()
    log.info("stage ledger: %d months, %d persona sessions", len(MONTHS), persona_sessions)

    async def month_facts(i: int) -> list[LedgerFact]:
        month, n = MONTHS[i], per_month[i]
        user = MONTH_USER.format(
            plan=plan_json, month=month, n_sessions=n, n_facts=round(n * 2.4),
            **_month_context(plan, month),
        )  # fmt: skip
        draft = await ask(settings, STRONG, MonthDraft, MONTH_SYSTEM, user)
        last_day = calendar.monthrange(int(month[:4]), int(month[5:]))[1]
        facts = [
            f.model_copy(update={"told_day": min(f.told_day, last_day)})
            for f in draft.facts
            if 1 <= f.session <= n
        ]
        return [
            LedgerFact(**f.model_dump(), fact_id=f"F{month.replace('-', '')}-{k:03d}", month=month)
            for k, f in enumerate(facts, start=1)
        ]

    ledger = [f for month in await bounded(range(len(MONTHS)), month_facts) for f in month]
    write_jsonl(LEDGER, ledger)
    log.info("ledger: %d facts", len(ledger))
    return ledger


# --- Stage C: conversations -----------------------------------------------------------------

RENDER_SYSTEM = """You write one realistic chat session between a user and an AI assistant.
The listed facts must each be stated, by the given speaker, in a natural way. Weave them into
an ordinary conversation (questions, follow-ups, reactions). The assistant never invents facts
about the user. Keep every name, number, amount, and date in a fact verbatim."""

RENDER_USER = """Date: {day}
User's first name: {name}
People in the user's life (their relation to the user never changes; do not alter it):
{people}

Facts to state in this session (speaker, id, statement):
{facts}

Write {exchanges} exchanges (user then assistant, alternating, starting with the user). For every
turn list the fact ids it states (an id may appear in exactly one turn, spoken by its speaker).
Turns that state no fact have an empty list.{feedback}"""

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,'/-]*")
GENERATOR_VERSION = 2

# Relation words that may directly precede a person's name ("my sister Rosa").
RELATION_SYNONYMS = {
    "mother": {"mother", "mom", "mum", "mama", "mami", "ma"},
    "father": {"father", "dad", "papa", "papá"},
    "sister": {"sister"},
    "brother": {"brother"},
    "aunt": {"aunt", "tía", "tia", "auntie"},
    "uncle": {"uncle", "tío", "tio"},
    "grandmother": {"grandmother", "grandma", "abuela"},
    "grandfather": {"grandfather", "grandpa", "abuelo"},
    "cousin": {"cousin"},
    "friend": {"friend", "bestie"},
    "manager": {"manager", "boss"},
    "colleague": {"colleague", "coworker", "co-worker", "teammate"},
    "neighbor": {"neighbor", "neighbour"},
    "landlord": {"landlord"},
    "doctor": {"doctor", "dr", "dr."},
    "partner": {"partner", "boyfriend", "girlfriend", "fiancé", "fiance", "husband", "wife"},
    "ex": {"ex", "ex-boyfriend", "ex-girlfriend", "ex-partner"},
    "dog": {"dog", "pup", "puppy"},
    "cat": {"cat", "kitten"},
}
_ALL_RELATION_WORDS = {w for words in RELATION_SYNONYMS.values() for w in words}


def allowed_relation_words(relation: str) -> set[str]:
    """Relation words compatible with a persona relation like 'tía (aunt)' or 'best friend'."""
    text = relation.lower()
    allowed: set[str] = set()
    for key, words in RELATION_SYNONYMS.items():
        if key in text or any(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
            allowed |= words
    if "dating" in text or "boyfriend" in text or "girlfriend" in text:
        allowed |= RELATION_SYNONYMS["partner"]
    if "ex-" in text or text.startswith("ex"):
        allowed |= RELATION_SYNONYMS["ex"] | RELATION_SYNONYMS["partner"]
    return allowed


def relation_problems(session: RenderedSession, plan: LifePlan) -> list[str]:
    """'my sister Rosa' when Rosa is the user's mother, and similar relation drift."""
    problems = []
    for person in plan.persona.people:
        first = person.name.split()[0].strip(".,")
        if first.lower() in ("dr", "dr."):
            first = person.name.split()[1]
        allowed = allowed_relation_words(person.relation)
        pattern = re.compile(rf"\b([\w'.-]+)\s+(?:named\s+)?{re.escape(first)}\b", re.IGNORECASE)
        for turn in session.turns:
            for m in pattern.finditer(turn.text):
                word = m.group(1).lower().removesuffix("'s")
                if word in _ALL_RELATION_WORDS and word not in allowed:
                    problems.append(
                        f"{person.name} is the user's {person.relation}, not their '{word}'"
                    )
    return sorted(set(problems))


def _key_tokens(fact: LedgerFact) -> list[str]:
    """Values that must appear verbatim: numbers and capitalized words of the value."""
    source = fact.value or fact.statement
    tokens = [t.strip(".,") for t in _TOKEN.findall(source)]
    return [
        t
        for t in tokens
        if any(c.isdigit() for c in t) or (t[:1].isupper() and len(t) > MIN_NAME_LEN)
    ]


def check_render(session: RenderedSession, facts: Sequence[LedgerFact]) -> list[str]:
    """Problems with a rendered session (empty list = valid)."""
    problems = []
    roles = [t.role for t in session.turns]
    if not roles or roles[0] != "user" or any(a == b for a, b in itertools.pairwise(roles)):
        problems.append("turns must alternate user/assistant starting with the user")
    if len(roles) % 2:
        problems.append("end with an assistant turn")
    placed = {fid: t for t in session.turns for fid in t.fact_ids}
    for f in facts:
        turn = placed.get(f.fact_id)
        if turn is None:
            problems.append(f"{f.fact_id} is not stated")
        elif turn.role != f.speaker:
            problems.append(f"{f.fact_id} must be said by the {f.speaker}")
        else:
            missing = [t for t in _key_tokens(f) if t.lower() not in turn.text.lower()]
            if missing:
                problems.append(f"{f.fact_id} turn lacks {missing}")
    return problems


def group_sessions(ledger: Sequence[LedgerFact]) -> dict[str, list[LedgerFact]]:
    groups: dict[str, list[LedgerFact]] = defaultdict(list)
    for f in ledger:
        groups[f"{f.month}-s{f.session:02d}"].append(f)
    return dict(sorted(groups.items()))


async def stage_sessions(
    settings: Settings, plan: LifePlan, ledger: Sequence[LedgerFact]
) -> list[dict[str, Any]]:
    if SESSIONS.exists():
        return read_jsonl(SESSIONS)
    groups = group_sessions(ledger)
    log.info("stage sessions: rendering %d sessions", len(groups))
    sessions = await _render_many(settings, plan, list(groups.items()))
    sessions = [s for s in sessions if not s["structural_problems"]]
    write_jsonl(SESSIONS, sessions)
    unverified = sum(len(s["unverified_fact_ids"]) for s in sessions)
    log.info("sessions: %d rendered, %d facts unverified", len(sessions), unverified)
    return sessions


async def _render_many(
    settings: Settings, plan: LifePlan, items: list[tuple[str, list[LedgerFact]]]
) -> list[dict[str, Any]]:
    rng = random.Random(SEED + 1)

    async def render(item: tuple[str, list[LedgerFact]]) -> dict[str, Any]:
        key, facts = item
        day = min(f.told_day for f in facts)
        exchanges = max(2, min(6, len(facts) + rng.randint(0, 2)))
        listing = "\n".join(f"- {f.speaker}, {f.fact_id}: {f.statement}" for f in facts)
        feedback = ""
        problems: list[str] = []
        session: RenderedSession | None = None
        for _ in range(3):
            user = RENDER_USER.format(
                day=f"{key[:7]}-{day:02d}", name=plan.persona.preferred_name, facts=listing,
                exchanges=exchanges, feedback=feedback, people=cast(plan),
            )  # fmt: skip
            session = await ask(settings, FAST, RenderedSession, RENDER_SYSTEM, user, effort="low")
            problems = check_render(session, facts) + relation_problems(session, plan)
            if not problems:
                break
            feedback = "\n\nFix these problems from your last attempt:\n- " + "\n- ".join(problems)
        assert session is not None
        bad = {p.split()[0] for p in problems if p.startswith("F")}
        if any(" is the user's " in p for p in problems):
            bad |= {f.fact_id for f in facts}  # a session with relation drift can't be evidence
        return {
            "session_key": key,
            "date": f"{key[:7]}-{day:02d}",
            "turns": [t.model_dump() for t in session.turns],
            "generator_version": GENERATOR_VERSION,
            "unverified_fact_ids": sorted(bad),
            "structural_problems": [p for p in problems if not p.startswith("F")],
        }

    return await bounded(items, render)


def cast(plan: LifePlan) -> str:
    return "\n".join(f"- {p.name}: {p.relation}" for p in plan.persona.people)


async def stage_repair(
    settings: Settings,
    plan: LifePlan,
    ledger: Sequence[LedgerFact],
    sessions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Re-render sessions from an older generator that fail the v2 checks (relation drift)."""
    if all(s.get("generator_version") == GENERATOR_VERSION for s in sessions):
        return sessions
    groups = group_sessions(ledger)
    failing = {
        s["session_key"]
        for s in sessions
        if relation_problems(RenderedSession.model_validate({"turns": s["turns"]}), plan)
    }
    log.info("stage repair: %d of %d sessions fail the relation check", len(failing), len(sessions))
    rendered = await _render_many(settings, plan, [(k, groups[k]) for k in sorted(failing)])
    by_key = {s["session_key"]: s for s in rendered}
    repaired = []
    for s in sessions:
        fixed = by_key.get(s["session_key"], {**s, "generator_version": GENERATOR_VERSION})
        if not fixed.get("structural_problems"):
            repaired.append(fixed)
    write_jsonl(SESSIONS, repaired)
    still = sum(
        1
        for s in repaired
        if relation_problems(RenderedSession.model_validate({"turns": s["turns"]}), plan)
    )
    log.info(
        "repair: %d re-rendered, %d still drift (their facts are unverified)", len(by_key), still
    )
    return repaired


# --- Stage C2: distractors ------------------------------------------------------------------


async def stage_distractors(
    settings: Settings, plan: LifePlan, n_persona_sessions: int
) -> list[dict[str, Any]]:
    if DISTRACTORS.exists():
        return read_jsonl(DISTRACTORS)
    need = TARGET_CONVERSATIONS - n_persona_sessions
    if not TOPICS.exists():
        topics = await ask(
            settings, STRONG, DistractorTopics,
            "You list diverse, impersonal conversation topics for an AI assistant.",
            f"List {need + 20} distinct general topics people ask an AI assistant about: "
            "software, science, cooking, history, finance concepts, health concepts, travel "
            "logistics in general, language, math, productivity. No personal situations.",
            effort="low",
        )  # fmt: skip
        TOPICS.write_text(json.dumps(topics.topics, indent=1), encoding="utf-8")
    topics_list: list[str] = json.loads(TOPICS.read_text(encoding="utf-8"))
    names = {plan.persona.full_name, plan.persona.preferred_name} | {
        p.name for p in plan.persona.people
    }
    blocked = re.compile("|".join(re.escape(n) for n in names if n), re.IGNORECASE)
    log.info("stage distractors: generating %d", need)

    async def one(topic: str) -> dict[str, Any] | None:
        result = await ask(
            settings, FAST, DistractorSession, "You write realistic assistant chats.",
            DISTRACTOR_PROMPT.format(topic=topic, exchanges=3), effort="low", max_tokens=4096,
        )  # fmt: skip
        turns = [t.model_dump() for t in result.turns]
        if len(turns) % 2:
            turns = turns[:-1]
        if not turns or any(blocked.search(t["text"]) for t in turns):
            return None
        return {"topic": topic, "turns": turns}

    made = [d for d in await bounded(topics_list, one) if d is not None][:need]
    rng = random.Random(SEED + 2)
    days = sorted(rng.randint(0, 729) for _ in made)
    start = date(2024, 11, 1).toordinal()
    rows = [
        {"distractor_id": f"D{i:03d}", "date": date.fromordinal(start + d).isoformat(), **m}
        for i, (m, d) in enumerate(zip(made, days, strict=True), start=1)
    ]
    write_jsonl(DISTRACTORS, rows)
    log.info("distractors: %d", len(rows))
    return rows


# --- Stage D: questions -----------------------------------------------------------------------

QUESTION_SYSTEM = """You write evaluation questions for an AI assistant with long-term memory of
its chats with one user. Each question is asked by the user on 2026-11-01, in the first person
("Where do I ..."), and must be answerable only from the listed facts (what was said in past
chats). The expected answer is short and complete, with the key values from the facts.
Use exactly the fact ids given as evidence."""

TYPE_GUIDE = {
    "single_fact": "One question per fact, about the specific detail it states.",
    "multi_fact": "One question per group that needs ALL facts of the group (list or combine).",
    "knowledge_update": "The facts are successive values of one attribute (oldest first). Ask "
    "about the CURRENT value (most of the time) or explicitly about the PREVIOUS value. The "
    "expected answer gives the latest value for 'current' questions.",
    "temporal": "Ask when something happened, which of two things happened first, how long "
    "between them, or what happened in a given month. Use the event dates.",
    "conflict": "The two facts state different values for the same thing. Ask for the value; "
    "the expected answer must mention BOTH values and that they differ.",
    "assistant_recall": "Ask what the assistant advised or recommended; the answer is that advice.",
    "preference_application": "Ask the assistant to DO a task where the preference should shape "
    "the result without restating it; the expected answer describes how the preference applies.",
    "sensitive_history": "Ask what the user said about this; the expected answer presents it as "
    "past and does not claim it is still true.",
    "inference": "Ask a question whose answer follows from combining the facts (a reason, an "
    "implication); the expected answer states the reasoning.",
}

ABSTAIN_USER = """The user's life is summarized below with the topics their chats covered.
Write {n} plausible personal questions the user might ask whose answers were NEVER stated in
the chats (things a memory system must say it does not know): specific numbers, names, dates,
or current statuses that were not mentioned. Set must_abstain true, evidence_fact_ids empty,
and expected_answer explaining that this was never mentioned (mention the closest thing that
was said, if any).

Persona: {persona}
Topics covered (attribute keys): {attributes}
Sample of what was said:
{sample}"""


def _fact_line(f: LedgerFact) -> str:
    extra = f" [attribute {f.attribute}={f.value}]" if f.attribute else ""
    when = f" [event {f.event_date}]" if f.event_date else ""
    told = f"{f.month}-{f.told_day:02d}"
    return f"{f.fact_id} ({told}, {f.speaker}, {f.kind}){extra}{when}: {f.statement}"


_STATED_DATE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\.?\s+\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(/\d{2,4})?\b",
    re.IGNORECASE,
)


def date_is_stated(f: LedgerFact) -> bool:
    """The event date can be read from the conversation: written out, or the session date."""
    if not f.event_date:
        return False
    return f.event_date == f"{f.month}-{f.told_day:02d}" or bool(_STATED_DATE.search(f.statement))


def _usable(f: LedgerFact, unverified: set[str]) -> bool:
    return f.fact_id not in unverified


def select_candidates(
    ledger: Sequence[LedgerFact], unverified: set[str], rng: random.Random
) -> dict[str, list[list[LedgerFact]]]:
    """Groups of evidence facts per test type (each group becomes one question)."""
    facts = [f for f in ledger if _usable(f, unverified)]
    by_attr: dict[str, list[LedgerFact]] = defaultdict(list)
    for f in facts:
        if f.attribute and f.kind in ("update", "personal"):
            by_attr[f.attribute].append(f)
    chains = {k: v for k, v in by_attr.items() if len({x.value for x in v}) > 1}
    changed = {f.fact_id for chain in chains.values() for f in chain}
    stable = [
        f for f in facts
        if f.kind in ("personal", "event", "third_party") and f.fact_id not in changed
    ]  # fmt: skip
    sessions: dict[str, list[LedgerFact]] = defaultdict(list)
    for f in stable:
        sessions[f"{f.month}-{f.session}"].append(f)
    conflicts: dict[str, list[LedgerFact]] = defaultdict(list)
    for f in facts:
        if f.kind == "conflict" and f.attribute:
            conflicts[f.attribute].append(f)
    for f in facts:
        if f.attribute in conflicts and f.kind != "conflict" and f not in conflicts[f.attribute]:
            conflicts[f.attribute].insert(0, f)
    dated = [f for f in facts if f.kind == "event" and date_is_stated(f)]

    def pick(pool: Sequence[Any], n: int) -> list[Any]:
        return rng.sample(list(pool), min(n, len(pool)))

    def want(t: str) -> int:
        return round(TARGETS[t] * OVERGENERATE)

    temporal = [[f] for f in pick(dated, want("temporal") // 2)]
    while len(temporal) < want("temporal"):
        temporal.append(sorted(pick(dated, PAIR), key=lambda f: f.event_date or ""))
    return {
        "single_fact": [[f] for f in pick(stable, want("single_fact"))],
        "multi_fact": pick([g for g in sessions.values() if len(g) >= PAIR], want("multi_fact")),
        "knowledge_update": [
            sorted(c, key=lambda f: (f.month, f.told_day))
            for c in pick(list(chains.values()), want("knowledge_update"))
        ],
        "temporal": temporal,
        "conflict": [g for g in conflicts.values() if len(g) >= PAIR][: want("conflict")],
        "assistant_recall": [
            [f]
            for f in pick(
                [f for f in facts if f.kind == "assistant_advice"], want("assistant_recall")
            )
        ],
        "preference_application": [
            [f]
            for f in pick(
                [f for f in facts if f.kind == "preference"], want("preference_application")
            )
        ],
        "sensitive_history": [
            [f]
            for f in pick([f for f in facts if f.kind == "sensitive"], want("sensitive_history"))
        ],
        "inference": [pick(stable, 3) for _ in range(want("inference"))],
    }


async def stage_questions(
    settings: Settings,
    plan: LifePlan,
    ledger: Sequence[LedgerFact],
    sessions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if DRAFTS.exists():
        return read_jsonl(DRAFTS)
    rng = random.Random(SEED + 3)
    unverified = {fid for s in sessions for fid in s["unverified_fact_ids"]}
    rendered = {fid for s in sessions for t in s["turns"] for fid in t["fact_ids"]}
    ledger = [f for f in ledger if f.fact_id in rendered]
    candidates = select_candidates(ledger, unverified, rng)
    jobs: list[tuple[str, list[list[LedgerFact]]]] = []
    for test_type, groups in candidates.items():
        for i in range(0, len(groups), 8):
            jobs.append((test_type, groups[i : i + 8]))
    log.info("stage questions: %d batches", len(jobs))

    async def draft(job: tuple[str, list[list[LedgerFact]]]) -> list[dict[str, Any]]:
        test_type, groups = job
        listing = "\n\n".join(
            f"Group {i + 1}:\n" + "\n".join(_fact_line(f) for f in g) for i, g in enumerate(groups)
        )
        user = (
            f"Test type: {test_type}. {TYPE_GUIDE[test_type]}\n"
            f"Write exactly one question per group ({len(groups)} groups).\n\n{listing}"
        )
        batch = await ask(settings, STRONG, QuestionBatch, QUESTION_SYSTEM, user)
        return [{"test_type": test_type, **q.model_dump()} for q in batch.questions]

    drafts = [q for batch in await bounded(jobs, draft) for q in batch]
    sample = "\n".join(_fact_line(f) for f in rng.sample(ledger, min(80, len(ledger))))
    abstain = await ask(
        settings, STRONG, QuestionBatch, QUESTION_SYSTEM,
        ABSTAIN_USER.format(
            n=round(TARGETS["abstention"] * 1.5), persona=plan.persona.model_dump_json(),
            attributes=", ".join(a.key for a in plan.attributes), sample=sample,
        ),
    )  # fmt: skip
    drafts += [
        {"test_type": "abstention", **q.model_dump(), "must_abstain": True, "evidence_fact_ids": []}
        for q in abstain.questions
    ]
    known = {f.fact_id for f in ledger}
    drafts = [d for d in drafts if set(d["evidence_fact_ids"]) <= known]
    write_jsonl(DRAFTS, drafts)
    log.info("drafted %d questions", len(drafts))
    return drafts


VERIFY_SYSTEM = """You audit evaluation questions for a long-term memory benchmark. The full log
of facts said in the user's past chats is given. For each question decide keep=true only if:
- for normal questions: the expected answer is fully supported by the cited facts AND is still
  correct given the whole log (no later fact changes it, unless the question is explicitly
  about the past), and the question is unambiguous;
- for must_abstain questions: nothing in the log answers it;
- for conflict questions: the log really contains both values;
- for temporal questions: every date in the expected answer is either written in a cited
  statement or equals the date that statement was told (shown in parentheses). Reject the
  question if the answer needs a date that was never said.
If the expected answer is only slightly off, keep it and give corrected_answer."""


async def stage_verify(
    settings: Settings, ledger: Sequence[LedgerFact], drafts: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    if QUESTIONS.exists():
        return list(csv.DictReader(QUESTIONS.open(encoding="utf-8")))
    log_text = "\n".join(_fact_line(f) for f in ledger)
    batches = [list(enumerate(drafts))[i : i + 20] for i in range(0, len(drafts), 20)]
    log.info("stage verify: %d batches", len(batches))

    async def verify(batch: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, Any]]:
        listing = "\n\n".join(
            f"[{i}] type={q['test_type']} must_abstain={q['must_abstain']}\n"
            f"Q: {q['question']}\nExpected: {q['expected_answer']}\n"
            f"Evidence: {', '.join(q['evidence_fact_ids']) or 'none'}"
            for i, q in batch
        )
        result = await ask(
            settings, STRONG, VerificationBatch, VERIFY_SYSTEM,
            f"Fact log:\n{log_text}\n\nQuestions:\n{listing}",
        )  # fmt: skip
        return [(v.index, v) for v in result.verdicts]

    verdicts = dict(v for batch in await bounded(batches, verify) for v in batch)
    kept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, q in enumerate(drafts):
        v = verdicts.get(i)
        if v is None or not v.keep:
            continue
        fixed = {**q, "expected_answer": v.corrected_answer} if v.corrected_answer else q
        kept[fixed["test_type"]].append(fixed)
    rng = random.Random(SEED + 4)
    final: list[dict[str, Any]] = []
    for test_type, n in TARGETS.items():
        pool = kept.get(test_type, [])
        final += rng.sample(pool, min(n, len(pool)))
    short = sum(TARGETS.values()) - len(final)
    if short > 0:  # top up from any type with spare verified questions
        spare = [q for t, pool in kept.items() for q in pool if q not in final]
        final += rng.sample(spare, min(short, len(spare)))
    rows = [
        {
            "question_id": f"S{n:03d}",
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "test_type": q["test_type"],
            "evidence_fact_ids": ";".join(q["evidence_fact_ids"]),
            "asked_at": ASKED_AT,
            "must_abstain": str(q["must_abstain"]).lower(),
            "split": "test",
        }
        for n, q in enumerate(final, start=1)
    ]
    assign_splits(rows)
    with QUESTIONS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log.info(
        "verified: kept %d of %d drafts; final %d",
        sum(map(len, kept.values())),
        len(drafts),
        len(rows),
    )
    return rows


def assign_splits(rows: list[dict[str, Any]]) -> None:
    """Fixed dev/test split, stratified by test type: DEV_SHARE of each type goes to dev."""
    rng = random.Random(SEED + 5)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["test_type"]].append(row)
    for test_type in sorted(by_type):
        group = sorted(by_type[test_type], key=lambda r: str(r["question_id"]))
        dev = {id(r) for r in rng.sample(group, round(len(group) * DEV_SHARE))}
        for row in group:
            row["split"] = "dev" if id(row) in dev else "test"


def stage_split() -> None:
    """Re-assign the split column of the existing questions file (nothing else changes)."""
    with QUESTIONS.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assign_splits(rows)
    with QUESTIONS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log.info("split: %d dev, %d test", sum(r["split"] == "dev" for r in rows),
             sum(r["split"] == "test" for r in rows))  # fmt: skip


# --- main ---------------------------------------------------------------------------------------


async def main_async(stage: str | None) -> None:
    if stage == "split":
        stage_split()
        return
    settings = get_settings()
    DATA.mkdir(parents=True, exist_ok=True)
    plan = await stage_plan(settings)
    if stage == "plan":
        return
    ledger = await stage_ledger(settings, plan)
    if stage == "ledger":
        return
    sessions = await stage_sessions(settings, plan, ledger)
    sessions = await stage_repair(settings, plan, ledger, sessions)
    distractors = await stage_distractors(settings, plan, len(sessions))
    if stage in ("sessions", "distractors"):
        return
    drafts = await stage_questions(settings, plan, ledger, sessions)
    questions = await stage_verify(settings, ledger, drafts)
    by_type: dict[str, int] = defaultdict(int)
    for q in questions:
        by_type[q["test_type"]] += 1
    MANIFEST.write_text(
        json.dumps(
            {
                "persona": plan.persona.full_name,
                "months": [MONTHS[0], MONTHS[-1]],
                "ledger_facts": len(ledger),
                "persona_sessions": len(sessions),
                "distractor_sessions": len(distractors),
                "conversations": len(sessions) + len(distractors),
                "turns": sum(len(s["turns"]) for s in sessions)
                + sum(len(d["turns"]) for d in distractors),
                "unverified_facts": sum(len(s["unverified_fact_ids"]) for s in sessions),
                "questions": len(questions),
                "questions_by_type": dict(sorted(by_type.items())),
                "models": {"strong": STRONG, "fast": FAST},
                "seed": SEED,
                "generator_version": GENERATOR_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(MANIFEST.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        choices=["plan", "ledger", "sessions", "distractors", "questions", "split", "all"],
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main_async(None if args.stage in (None, "all") else args.stage))


if __name__ == "__main__":
    main()
