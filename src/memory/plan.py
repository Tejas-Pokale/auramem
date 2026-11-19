"""Query planner (PLAN.md section 6, step 2): rules first, Haiku 5.5 fallback.

The rules produce a complete ``QueryPlan`` in well under a millisecond: time range (with
``dateparser`` for explicit dates), temporal flag, depth, entities, and generic guidance.
The LLM is only called when the rules detect something they cannot resolve, mainly time
anchored to a personal event ("two weeks before my trip"), and is bounded by a timeout
that falls back to the rules plan. Measured Haiku planning calls take 1.3-4.6 s, so an
LLM-on-every-message planner could not meet the 500 ms p95 target.

Time ranges are [start, end) in the timezone of ``now``. Weeks start on Monday. A month or
weekday without a year means the most recent one that is not in the future.
"""

import asyncio
import calendar
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal, Protocol, cast

import dateparser
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from memory.config import Settings
from memory.models import QueryPlan, TimeRange, Turn
from memory.prompts import load_prompt

log = logging.getLogger(__name__)

# --- Vocabulary --------------------------------------------------------------------------

_MONTH_NAMES = [
    ("january", "jan"),
    ("february", "feb"),
    ("march", "mar"),
    ("april", "apr"),
    ("may",),
    ("june", "jun"),
    ("july", "jul"),
    ("august", "aug"),
    ("september", "sep", "sept"),
    ("october", "oct"),
    ("november", "nov"),
    ("december", "dec"),
]
_MONTHS = {name: i for i, names in enumerate(_MONTH_NAMES, start=1) for name in names}
_WEEKDAYS = {
    d: i
    for i, d in enumerate(
        ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    )
}
_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "couple of": 2, "few": 3,
}  # fmt: skip
_QUARTERS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "last": 4}
_MONTHS_PER_YEAR = 12
_MONTHS_PER_QUARTER = 3
_SATURDAY = 5

_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_RE = "|".join(_WEEKDAYS)
_NUM_RE = r"\d+|" + "|".join(sorted(_NUMBERS, key=len, reverse=True))
_UNIT_RE = r"days?|weeks?|months?|years?"
_YEAR_RE = r"(?:19|20)\d{2}"

Unit = Literal["day", "week", "month", "year"]


@dataclass(frozen=True)
class Period:
    start: datetime | None
    end: datetime | None
    label: str


# --- Calendar helpers --------------------------------------------------------------------


def _midnight(day: date, now: datetime) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=now.tzinfo)


def _label(day: date) -> str:
    return f"{day.day} {calendar.month_name[day.month]} {day.year}"


def _span(start: date, end: date, now: datetime, label: str) -> Period:
    return Period(_midnight(start, now), _midnight(end, now), label)


def _day(day: date, now: datetime) -> Period:
    return _span(day, day + timedelta(days=1), now, _label(day))


def _week(day: date, now: datetime) -> Period:
    monday = day - timedelta(days=day.weekday())
    return _span(monday, monday + timedelta(days=7), now, f"the week of {_label(monday)}")


def _add_months(day: date, months: int) -> date:
    year, month0 = divmod(day.year * _MONTHS_PER_YEAR + day.month - 1 + months, _MONTHS_PER_YEAR)
    month = month0 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _month(year: int, month: int, now: datetime) -> Period:
    first = date(year, month, 1)
    return _span(first, _add_months(first, 1), now, f"{calendar.month_name[month]} {year}")


def _year(year: int, now: datetime) -> Period:
    return _span(date(year, 1, 1), date(year + 1, 1, 1), now, str(year))


def _quarter(year: int, q: int, now: datetime) -> Period:
    first = date(year, _MONTHS_PER_QUARTER * (q - 1) + 1, 1)
    return _span(first, _add_months(first, _MONTHS_PER_QUARTER), now, f"Q{q} {year}")


def _shift(day: date, n: int, unit: Unit) -> date:
    if unit == "day":
        return day + timedelta(days=n)
    if unit == "week":
        return day + timedelta(weeks=n)
    return _add_months(day, n * (1 if unit == "month" else _MONTHS_PER_YEAR))


def _period_of(day: date, unit: Unit, now: datetime) -> Period:
    if unit == "day":
        return _day(day, now)
    if unit == "week":
        return _week(day, now)
    return _month(day.year, day.month, now) if unit == "month" else _year(day.year, now)


def _unit(word: str) -> Unit:
    return cast(Unit, word.rstrip("s"))


def _num(word: str) -> int:
    return int(word) if word.isdigit() else _NUMBERS[word]


def _until_tomorrow(start: date, now: datetime, label: str) -> Period:
    return _span(start, now.date() + timedelta(days=1), now, label)


# --- Single time expressions: an ordered table of rules ---------------------------------

_Rule = Callable[[str, datetime], Period | None]


def _iso_date(t: str, now: datetime) -> Period | None:
    if m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t):
        try:
            return _day(date(int(m.group(1)), int(m.group(2)), int(m.group(3))), now)
        except ValueError:
            return None
    return None


_HUMAN_DATES = [
    re.compile(rf"\b(?:{_MONTH_RE})\.? \d{{1,2}}(?:st|nd|rd|th)?,? {_YEAR_RE}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)? (?:of )?(?:{_MONTH_RE}),? {_YEAR_RE}\b"),
    re.compile(rf"\b\d{{1,2}}[/.]\d{{1,2}}[/.]{_YEAR_RE}\b"),
]


def _human_date(t: str, now: datetime) -> Period | None:
    """dateparser handles the many human formats; DMY is the Indian convention."""
    for pattern in _HUMAN_DATES:
        if m := pattern.search(t):
            parsed = dateparser.parse(
                m.group(0),
                languages=["en"],
                settings={"DATE_ORDER": "DMY", "PREFER_DATES_FROM": "past"},
            )
            if parsed:
                return _day(parsed.date(), now)
    return None


def _quarter_rule(t: str, now: datetime) -> Period | None:
    m = re.search(rf"\b(?:q([1-4])|(first|second|third|fourth|last) quarter of) ({_YEAR_RE})\b", t)
    if not m:
        return None
    q = int(m.group(1)) if m.group(1) else _QUARTERS[m.group(2)]
    return _quarter(int(m.group(3)), q, now)


def _month_year(t: str, now: datetime) -> Period | None:
    if m := re.search(rf"\b({_MONTH_RE}),? ({_YEAR_RE})\b", t):
        return _month(int(m.group(2)), _MONTHS[m.group(1)], now)
    return None


def _most_recent_month(month: int, now: datetime) -> Period:
    today = now.date()
    return _month(today.year if month <= today.month else today.year - 1, month, now)


def _month_in(t: str, now: datetime) -> Period | None:
    """'in March' -> the most recent March. 'may' needs a preposition and no number after."""
    m = re.search(
        rf"\b(?:in|during|of|from|since|until|by|for) (?:the month of )?({_MONTH_RE})\b(?! \d)", t
    )
    return _most_recent_month(_MONTHS[m.group(1)], now) if m else None


def _past_n(t: str, now: datetime) -> Period | None:
    m = re.search(
        rf"\b(?:in |over |during )?(?:the )?(?:past|last|previous) ({_NUM_RE}) ({_UNIT_RE})\b", t
    )
    if not m:
        return None
    start = _shift(now.date(), -_num(m.group(1)), _unit(m.group(2)))
    return _until_tomorrow(start, now, f"the past {m.group(1)} {m.group(2)}")


def _n_ago(t: str, now: datetime) -> Period | None:
    if m := re.search(rf"\b({_NUM_RE}) ({_UNIT_RE}) ago\b", t):
        unit = _unit(m.group(2))
        return _period_of(_shift(now.date(), -_num(m.group(1)), unit), unit, now)
    return None


def _earlier_this_year(t: str, now: datetime) -> Period | None:
    if re.search(r"\bearlier this year\b", t):
        return _until_tomorrow(date(now.year, 1, 1), now, f"earlier in {now.year}")
    return None


def _weekend(t: str, now: datetime) -> Period | None:
    m = re.search(r"\b(last|this|next) weekend\b", t)
    if not m:
        return None
    today = now.date()
    if m.group(1) == "last":
        saturday = today - timedelta(days=(today.weekday() - _SATURDAY) % 7 or 7)
    else:
        saturday = today + timedelta(days=(_SATURDAY - today.weekday()) % 7)
        if m.group(1) == "next":
            saturday += timedelta(days=7)
    return _span(saturday, saturday + timedelta(days=2), now, f"the weekend of {_label(saturday)}")


def _relative_unit(t: str, now: datetime) -> Period | None:
    if m := re.search(r"\b(last|this|next) (week|month|year)\b", t):
        unit = _unit(m.group(2))
        offset = {"last": -1, "this": 0, "next": 1}[m.group(1)]
        return _period_of(_shift(now.date(), offset, unit), unit, now)
    return None


def _relative_day(t: str, now: datetime) -> Period | None:
    if m := re.search(r"\b(yesterday|today|tomorrow)\b", t):
        offset = {"yesterday": -1, "today": 0, "tomorrow": 1}[m.group(1)]
        return _day(now.date() + timedelta(days=offset), now)
    return None


def _weekday(t: str, now: datetime) -> Period | None:
    m = re.search(rf"\b(last|this|next|on|past) ({_WEEKDAY_RE})\b", t)
    if not m:
        return None
    today, target = now.date(), _WEEKDAYS[m.group(2)]
    back = (today.weekday() - target) % 7
    if m.group(1) in ("last", "past"):
        day = today - timedelta(days=back or 7)  # strictly before today
    elif m.group(1) == "on":
        day = today - timedelta(days=back)  # most recent, today included
    elif m.group(1) == "this":
        day = today + timedelta(days=target - today.weekday())  # this week's
    else:
        day = today + timedelta(days=(target - today.weekday()) % 7 or 7)
    return _day(day, now)


def _year_in(t: str, now: datetime) -> Period | None:
    m = re.search(rf"\b(?:in|during|of|from|for|since|before|after|until|by) ({_YEAR_RE})\b", t)
    return _year(int(m.group(1)), now) if m else None


def _bare_month(t: str, now: datetime) -> Period | None:
    m = re.fullmatch(rf"\s*({_MONTH_RE})\s*", t)
    return _most_recent_month(_MONTHS[m.group(1)], now) if m else None


def _bare_year(t: str, now: datetime) -> Period | None:
    m = re.fullmatch(rf"\s*({_YEAR_RE})\s*", t)
    return _year(int(m.group(1)), now) if m else None


_POINT_RULES: tuple[_Rule, ...] = (
    _iso_date,
    _human_date,
    _quarter_rule,
    _month_year,
    _month_in,
    _past_n,
    _n_ago,
    _earlier_this_year,
    _weekend,
    _relative_unit,
    _relative_day,
    _weekday,
    _year_in,
)
# Fragments inside "between X and Y" / "since X" may be just a month or a year.
_FRAGMENT_RULES = (*_POINT_RULES, _bare_month, _bare_year)


def _point(text: str, now: datetime, *, fragment: bool = False) -> Period | None:
    """The first time expression found in ``text`` (rules in priority order), or None."""
    for rule in _FRAGMENT_RULES if fragment else _POINT_RULES:
        if (period := rule(text, now)) is not None:
            return period
    return None


# --- Compound expressions and cues -------------------------------------------------------

_RECENT = re.compile(r"(?<!most )\b(recently|lately)\b")
_END = r"(?=[?.!,;]|$| and | or | but )"
_BETWEEN = re.compile(rf"\bbetween (.+?) and (.+?){_END}")
_BOUNDED = re.compile(rf"\b(since|before|after|until|till) (.+?){_END}")
_PERSONAL_ANCHOR = re.compile(r"(?:my|our|i|we)\b")
_EVENT_RELATIVE = re.compile(
    r"\b(?:before|after|since|until|following|prior to|around|during) (?:my|the|our|i|we)\b"
    r"(?! (?:week|month|year|weekend|day)s?\b)"
    r"|\bwhen (?:i|we)\b"
)
_ORDERING = re.compile(
    r"\bbefore or after\b|\bwhich\b.{0,60}\bfirst\b|\bin what order\b|\bwhat order\b"
    r"|\bfirst or\b|\bearliest\b"
)
_TEMPORAL_CUE = re.compile(
    r"\bwhen (?:did|was|were|do|does|is|are|will)\b|\bhow long\b|\bhow recently\b"
    r"|\bhow many (?:days|weeks|months|years)\b|\bago\b|\btimeline\b|\blatest\b"
    r"|\bmost recent(?:ly)?\b|\blast time\b"
)
_DEEP = re.compile(
    r"\bhow many\b|\bhow often\b|\bhow long\b|\bhow much time\b|\bin what order\b"
    r"|\bbefore or after\b|\blist (?:all|every)\b|\ball the\b|\bevery time\b|\beach time\b"
    r"|\bin total\b|\btimeline\b|\bwhich\b.{0,60}\bfirst\b|\bfirst or\b"
    # Several separate asks in one message ("..., and who ..."), or two things asked for at
    # once ("what X and Y did I ...").
    r"|\band (?:what|who|whom|where|when|which|why|how)\b"
    r"|\b(?:what|which|how much|how many)\b[^?.]{0,40}\band\b[^?.]{0,40}"
    r"\b(?:did|do|does|have|has)\b"
    # List questions: a plural object of "which/what ... did/have I ..." or "what are some".
    r"|\b(?:which|what)\s+(?:[\w-]+\s+){0,6}[a-z]+s\s+(?:have|did|do|had)\s+i\b"
    r"|\bwhat are (?:some|the|all)\b"
    # Overviews, comparisons, and earlier values.
    r"|\bwhat do (?:i|you) know about\b|\beverything\b|\bsummari[sz]e\b|\boverview\b"
    r"|\bcompare\b|\bused to\b|\bpreviously\b|\bbefore i\b"
)
_PERSONAL = re.compile(
    r"\b(?:i|i'm|i've|i'd|i'll|me|my|mine|myself|we|our|us)\b"
    r"|\byou (?:said|told|suggested|recommended|advised|mentioned)\b"
    r"|\byour (?:advice|suggestion|recommendation)\b"
)
_ENTITY_STOPWORDS = {
    "i", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "did", "do",
    "does", "is", "are", "was", "were", "can", "could", "should", "would", "will", "write",
    "suggest", "draft", "tell", "give", "list", "please", "the", "a", "an", "my", "and", "or",
    "last", "next", "this", "yes", "no", "ok", "hi", "hello", "hey", "thanks", "q1", "q2",
    "q3", "q4",
} | set(_MONTHS) | set(_WEEKDAYS)  # fmt: skip


def _between(t: str, now: datetime) -> Period | None:
    m = _BETWEEN.search(t)
    if not m:
        return None
    end = _point(m.group(2), now, fragment=True)
    start_text = m.group(1)
    if end is not None and end.start is not None and not re.search(_YEAR_RE, start_text):
        start_text = f"{start_text} {end.start.year}"  # "between March and May 2026"
    start = _point(start_text, now, fragment=True)
    if start is None or end is None:
        return None
    return Period(start.start, end.end, f"{start.label} to {end.label}")


def _bounded(t: str, now: datetime) -> Period | None:
    for m in _BOUNDED.finditer(t):
        keyword, rest = m.group(1), m.group(2)
        if t[: m.start()].endswith("before or ") or (
            keyword == "before" and t[m.end(1) :].startswith(" or after")
        ):
            continue  # an ordering question, not a bound
        if _PERSONAL_ANCHOR.match(rest):
            continue  # anchored to a personal event ("after my exam in March"), not a date
        p = _point(rest, now, fragment=True)
        if p is None:
            continue
        if keyword == "since":
            return Period(
                p.start, _midnight(now.date() + timedelta(days=1), now), f"since {p.label}"
            )
        if keyword == "after":
            return Period(p.end, None, f"after {p.label}")
        return Period(None, p.start, f"before {p.label}")
    return None


def parse_time(text: str, now: datetime, *, recent_days: int = 30) -> tuple[Period | None, bool]:
    """Resolve the message's time expression. Returns (period, event_relative)."""
    t = text.lower()
    period = _between(t, now) or _bounded(t, now)
    if period is None and not _ORDERING.search(t) and _EVENT_RELATIVE.search(t):
        return None, True
    if period is None:
        period = _point(t, now)
    if period is None and _RECENT.search(t):
        start = now.date() - timedelta(days=recent_days)
        period = _until_tomorrow(start, now, "the past few weeks")
    return period, False


def _entities(text: str) -> list[str]:
    found: list[str] = []
    for run in re.findall(r"\b[A-Z][\w.+-]*(?:\s+[A-Z][\w.+-]*)*", text):
        words = run.split()
        while words and words[0].lower() in _ENTITY_STOPWORDS:
            words.pop(0)
        name = " ".join(words).rstrip(".,")
        if name and name.lower() not in _ENTITY_STOPWORDS and name not in found:
            found.append(name)
    return found


@dataclass(frozen=True)
class RulePlan:
    plan: QueryPlan
    needs_llm: bool
    reason: str | None = None


def rule_plan(message: str, now: datetime, *, recent_days: int = 30) -> RulePlan:
    t = message.lower()
    period, event_relative = parse_time(message, now, recent_days=recent_days)
    ordering = bool(_ORDERING.search(t))
    is_temporal = period is not None or event_relative or ordering or bool(_TEMPORAL_CUE.search(t))

    if not _PERSONAL.search(t):
        depth: Literal["none", "quick", "deep"] = "none"
    elif _DEEP.search(t):
        depth = "deep"
    else:
        depth = "quick"

    guidance: list[str] = []
    if period is not None:
        guidance.append(
            f"The question refers to {period.label}; focus on conversations from that period "
            "and use their dates."
        )
    elif is_temporal:
        guidance.append("The answer depends on dates or order: compare the conversation dates.")
    if depth == "deep":
        guidance.append(
            "The answer may need several conversations: collect every relevant item first."
        )

    plan = QueryPlan(
        depth=depth,
        is_temporal=is_temporal,
        time_range=TimeRange(start=period.start, end=period.end) if period else None,
        entities=_entities(message),
        guidance=guidance,
    )
    return RulePlan(
        plan=plan,
        needs_llm=event_relative,
        reason="time relative to a personal event" if event_relative else None,
    )


# --- Planner with LLM fallback ------------------------------------------------------------


class PlanLLM(Protocol):
    async def __call__(self, messages: list[BaseMessage]) -> QueryPlan: ...


@dataclass(frozen=True)
class PlanResult:
    plan: QueryPlan
    source: Literal["rules", "llm", "rules_fallback"]
    elapsed_ms: float


class Planner:
    def __init__(self, settings: Settings, llm: PlanLLM | None = None) -> None:
        self._settings = settings
        self._llm = llm

    async def plan(
        self, message: str, *, now: datetime, recent_turns: Sequence[Turn] = ()
    ) -> PlanResult:
        started = time.perf_counter()
        s = self._settings
        rules = rule_plan(message, now, recent_days=s.planner_recent_days)
        if not rules.needs_llm or self._llm is None:
            return PlanResult(rules.plan, "rules", (time.perf_counter() - started) * 1000)
        try:
            llm_plan = await asyncio.wait_for(
                self._llm(self._messages(message, now, recent_turns, rules)),
                timeout=s.planner_llm_timeout_seconds,
            )
        except Exception:  # timeout, API or validation error: the rules plan is still usable
            log.warning("planner LLM fallback failed; using rules plan", exc_info=True)
            return PlanResult(rules.plan, "rules_fallback", (time.perf_counter() - started) * 1000)
        merged = llm_plan.model_copy(
            update={
                "time_range": llm_plan.time_range or rules.plan.time_range,
                "is_temporal": llm_plan.is_temporal or rules.plan.is_temporal,
                "entities": llm_plan.entities or rules.plan.entities,
                "guidance": llm_plan.guidance or rules.plan.guidance,
            }
        )
        return PlanResult(merged, "llm", (time.perf_counter() - started) * 1000)

    def _messages(
        self, message: str, now: datetime, recent: Sequence[Turn], rules: RulePlan
    ) -> list[BaseMessage]:
        version = self._settings.planner_prompt_version
        history = "\n".join(f"{t.role}: {t.text}" for t in recent) or "(none)"
        user = load_prompt("plan", version, "user").text.format(
            today=f"{now:%A, %d %B %Y} ({now.tzname() or 'UTC'})",
            history=history,
            rules=rules.plan.model_dump_json(),
            message=message,
        )
        return [SystemMessage(load_prompt("plan", version, "system").text), HumanMessage(user)]
