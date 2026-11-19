"""Budgeted context assembly (PLAN.md section 6, step 6).

Facts are taken best-first into the budget and rendered chronologically; selected turns are
expanded with their neighbors, packed into what is left in rank order (up to a turn cap), then
rendered grouped by conversation under date headers, oldest first. Long assistant turns added
as context are cut in the rendering only (turns that search matched are kept whole); stored
turns are never changed.
"""

from collections.abc import Collection, Iterable, Sequence
from datetime import datetime
from itertools import groupby

from memory.models import Fact, Turn

_ROLE_LABEL = {"user": "User", "assistant": "Assistant"}
_GAP = "..."
_CUT = "…"


def estimate_tokens(text: str, *, chars_per_token: float) -> int:
    return round(len(text) / chars_per_token)


def _clip(text: str, max_chars: int | None) -> str:
    """Cut at a word boundary to at most ``max_chars`` (plus the cut mark)."""
    if max_chars is None or len(text) <= max_chars:
        return text
    head = text[:max_chars]
    space = head.rfind(" ")
    return (head[:space] if space > max_chars // 2 else head).rstrip(" ,;:") + _CUT


def _line(
    turn: Turn, assistant_max_chars: int | None = None, uncut: Collection[str] = frozenset()
) -> str:
    clip = turn.role == "assistant" and turn.turn_id not in uncut
    text = _clip(turn.text, assistant_max_chars) if clip else turn.text
    return f"{_ROLE_LABEL[turn.role]}: {text}"


def _header(turn: Turn) -> str:
    return f"### Conversation on {turn.created_at:%Y-%m-%d (%A)}"


def select_windows(
    windows: Iterable[Sequence[Turn]],
    *,
    budget: int,
    chars_per_token: float,
    max_turns: int | None = None,
    assistant_max_chars: int | None = None,
    uncut: Collection[str] = frozenset(),
) -> list[Turn]:
    """Take windows best-first while they fit the budget and the turn cap; skip ones that
    don't and keep trying smaller ones.

    Turns already taken by an earlier window are not counted again.
    """
    taken: dict[str, Turn] = {}
    sessions: set[str] = set()
    used = 0
    for window in windows:
        new = [t for t in window if t.turn_id not in taken]
        if not new or (max_turns is not None and len(taken) + len(new) > max_turns):
            continue
        cost = sum(
            estimate_tokens(
                _line(t, assistant_max_chars, uncut) + "\n", chars_per_token=chars_per_token
            )
            for t in new
        )
        new_sessions = {t.session_id for t in new} - sessions
        cost += sum(
            estimate_tokens(_header(t) + "\n\n", chars_per_token=chars_per_token)
            for t in new
            if t.session_id in new_sessions
        )
        if used + cost > budget:
            continue
        used += cost
        sessions |= new_sessions
        for t in new:
            taken[t.turn_id] = t
    return list(taken.values())


def render_memory(
    turns: Iterable[Turn],
    *,
    assistant_max_chars: int | None = None,
    uncut: Collection[str] = frozenset(),
) -> str:
    """Group turns by conversation, oldest conversation first, turns in order within each."""
    ordered = sorted(turns, key=lambda t: (t.session_id, t.seq))
    sessions = [list(group) for _, group in groupby(ordered, key=lambda t: t.session_id)]
    sessions.sort(key=lambda s: (s[0].created_at, s[0].session_id))
    blocks: list[str] = []
    for session in sessions:
        lines = [_header(session[0])]
        for prev, cur in zip([None, *session], session, strict=False):
            if prev is not None and cur.seq - prev.seq > 1:
                lines.append(_GAP)
            lines.append(_line(cur, assistant_max_chars, uncut))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# --- Facts ------------------------------------------------------------------------------

FACTS_HEADER = "## Facts from earlier conversations"
EXCERPTS_HEADER = "## Conversation excerpts"


def _day(at: datetime) -> str:
    return f"{at:%Y-%m-%d}"


def _at_precision(at: datetime, precision: str) -> str:
    if precision == "year":
        return str(at.year)
    if precision in ("month", "unknown"):
        return f"{at:%Y-%m}"
    return _day(at)


def _when(fact: Fact) -> str:
    """The fact's time at its precision, including the end of a range when there is one."""
    if fact.start is None:
        return ""
    start = _at_precision(fact.start, fact.time_precision)
    end = _at_precision(fact.end, fact.time_precision) if fact.end else start
    return start if end == start else f"{start} to {end}"


def _source(fact: Fact) -> str | None:
    """Only non-user sources are marked; an unmarked fact was said by the user."""
    if fact.attributed_to == "user":
        return None
    if fact.attributed_to == "assistant":
        return "said by the assistant"
    return "from pasted or third-party content"


def _label(fact: Fact) -> str:
    """[latest] / [superseded on ...] for single-valued attributes (PLAN.md section 6.6)."""
    if fact.conflicts_with_ids:
        return "[conflicting] "
    if not fact.attribute_key:
        return ""
    if fact.is_latest:
        return "[latest] "
    if fact.superseded_on is not None:
        return f"[superseded on {_day(fact.superseded_on)}] "
    return "[superseded] "


def fact_line(fact: Fact) -> str:
    """``- [label] text (when; told DATE; source)``. The told date is shown only when it is not
    the same day as ``when``, and the source only when it is not the user."""
    when = _when(fact)
    told = _day(fact.mentioned_at)
    details = [when] if when else []
    if when != told:
        details.append(f"told {told}")
    if source := _source(fact):
        details.append(source)
    return f"- {_label(fact)}{fact.text} ({'; '.join(details)})"


def fact_sort_key(fact: Fact) -> tuple[datetime, str]:
    return (fact.start or fact.mentioned_at, fact.fact_id)


def select_facts(
    facts: Iterable[Fact], *, budget: int, max_facts: int, chars_per_token: float
) -> list[Fact]:
    """Take facts best-first while they fit the budget (header cost included once)."""
    chosen: list[Fact] = []
    used = estimate_tokens(FACTS_HEADER + "\n", chars_per_token=chars_per_token)
    for fact in facts:
        if len(chosen) >= max_facts:
            break
        cost = estimate_tokens(fact_line(fact) + "\n", chars_per_token=chars_per_token)
        if used + cost > budget:
            continue
        chosen.append(fact)
        used += cost
    return chosen


def render_facts(facts: Iterable[Fact]) -> str:
    ordered = sorted(facts, key=fact_sort_key)
    if not ordered:
        return ""
    return "\n".join([FACTS_HEADER, *(fact_line(f) for f in ordered)])


def render_context(
    facts: Iterable[Fact],
    turns: Iterable[Turn],
    *,
    assistant_max_chars: int | None = None,
    uncut: Collection[str] = frozenset(),
) -> str:
    """Facts (chronological) then conversation excerpts; empty sections are omitted."""
    facts_text = render_facts(facts)
    turns_text = render_memory(turns, assistant_max_chars=assistant_max_chars, uncut=uncut)
    if facts_text and turns_text:
        return f"{facts_text}\n\n{EXCERPTS_HEADER}\n{turns_text}"
    return facts_text or turns_text
