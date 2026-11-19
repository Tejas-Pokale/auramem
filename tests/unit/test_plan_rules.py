"""Rule-based planner: time expressions (Phase 3 gate: 30+ cases), depth, LLM fallback triggers.

Reference time: Saturday 24 October 2026, 12:00 IST. Ranges are [start, end); weeks start on
Monday; a month or weekday without a year means the most recent one that is not in the future.
"""

from datetime import datetime, timedelta, timezone

import pytest

from memory.plan import rule_plan

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=IST)


def d(y: int, m: int, day: int) -> datetime:
    return datetime(y, m, day, tzinfo=IST)


TOMORROW = d(2026, 10, 25)

RANGES: list[tuple[str, datetime | None, datetime | None]] = [
    ("What did I do yesterday?", d(2026, 10, 23), d(2026, 10, 24)),
    ("What did I say today?", d(2026, 10, 24), d(2026, 10, 25)),
    ("What's planned for tomorrow?", d(2026, 10, 25), d(2026, 10, 26)),
    ("What did I work on last week?", d(2026, 10, 12), d(2026, 10, 19)),
    ("Anything I mentioned this week?", d(2026, 10, 19), d(2026, 10, 26)),
    ("What am I doing next week?", d(2026, 10, 26), d(2026, 11, 2)),
    ("What did I do last month?", d(2026, 9, 1), d(2026, 10, 1)),
    ("What happened this month?", d(2026, 10, 1), d(2026, 11, 1)),
    ("Where did I work last year?", d(2025, 1, 1), d(2026, 1, 1)),
    ("What have I learned this year?", d(2026, 1, 1), d(2027, 1, 1)),
    ("What did I tell you last Tuesday?", d(2026, 10, 20), d(2026, 10, 21)),
    ("What did I say on Monday?", d(2026, 10, 19), d(2026, 10, 20)),
    ("What did I mention last Saturday?", d(2026, 10, 17), d(2026, 10, 18)),
    ("What did I ask 3 days ago?", d(2026, 10, 21), d(2026, 10, 22)),
    ("What was I doing two weeks ago?", d(2026, 10, 5), d(2026, 10, 12)),
    ("What did I say a month ago?", d(2026, 9, 1), d(2026, 10, 1)),
    ("Where was I two years ago?", d(2024, 1, 1), d(2025, 1, 1)),
    ("What did I mention in the past three days?", d(2026, 10, 21), TOMORROW),
    ("What came up over the last two weeks?", d(2026, 10, 10), TOMORROW),
    ("What did I explore in the last 6 months?", d(2026, 4, 24), TOMORROW),
    ("What did I do in March?", d(2026, 3, 1), d(2026, 4, 1)),
    ("What did I mention in December?", d(2025, 12, 1), d(2026, 1, 1)),
    ("What happened in October?", d(2026, 10, 1), d(2026, 11, 1)),
    ("What was I worried about in July 2026?", d(2026, 7, 1), d(2026, 8, 1)),
    ("What happened on June 6, 2026?", d(2026, 6, 6), d(2026, 6, 7)),
    ("Where was I on 6 June 2026?", d(2026, 6, 6), d(2026, 6, 7)),
    ("What was planned for 2026-01-27?", d(2026, 1, 27), d(2026, 1, 28)),
    ("What did I study in 2025?", d(2025, 1, 1), d(2026, 1, 1)),
    ("What did I work on in Q1 2026?", d(2026, 1, 1), d(2026, 4, 1)),
    ("What did I do in the first quarter of 2026?", d(2026, 1, 1), d(2026, 4, 1)),
    ("What have I done since January?", d(2026, 1, 1), TOMORROW),
    ("Where did I work before March 2026?", None, d(2026, 3, 1)),
    ("What did I do after June 2026?", d(2026, 7, 1), None),
    ("What did I do between March and May 2026?", d(2026, 3, 1), d(2026, 6, 1)),
    ("What have I tried earlier this year?", d(2026, 1, 1), TOMORROW),
    ("What have I been worried about recently?", d(2026, 9, 24), TOMORROW),
    ("What did I do last weekend?", d(2026, 10, 17), d(2026, 10, 19)),
    ("LAST TUESDAY I said something, what was it?", d(2026, 10, 20), d(2026, 10, 21)),
]


@pytest.mark.parametrize(("message", "start", "end"), RANGES)
def test_time_ranges(message: str, start: datetime | None, end: datetime | None) -> None:
    result = rule_plan(message, NOW)
    assert result.plan.is_temporal
    assert result.plan.time_range is not None, message
    assert (result.plan.time_range.start, result.plan.time_range.end) == (start, end)
    assert not result.needs_llm


def test_at_least_thirty_time_expressions() -> None:
    assert len(RANGES) >= 30


@pytest.mark.parametrize(
    "message",
    [
        "What did I do two weeks before my trip?",
        "What did I say the week after my exam?",
        "What was I worried about when I joined JHEX?",
        "Where did I work after my internship?",
    ],
)
def test_event_relative_time_needs_llm(message: str) -> None:
    result = rule_plan(message, NOW)
    assert result.plan.is_temporal
    assert result.plan.time_range is None
    assert result.needs_llm


@pytest.mark.parametrize(
    "message",
    [
        "When did I attend the TCS Prime interview?",
        "Which internship did I do first, the Flutter one or the React Native one?",
        "Did I join Meditab before or after I was selected for Optum?",
        "How long was my diploma?",
    ],
)
def test_temporal_questions_without_a_range(message: str) -> None:
    result = rule_plan(message, NOW)
    assert result.plan.is_temporal
    assert result.plan.time_range is None
    assert not result.needs_llm


@pytest.mark.parametrize(
    "message",
    ["What is my laptop model?", "Which vector database did I use?", "Write a LinkedIn headline."],
)
def test_non_temporal(message: str) -> None:
    result = rule_plan(message, NOW)
    assert not result.plan.is_temporal
    assert result.plan.time_range is None


@pytest.mark.parametrize(
    ("message", "depth"),
    [
        ("hi", "none"),
        ("thanks!", "none"),
        ("oyy", "none"),
        ("What is the difference between TCP and UDP?", "none"),
        ("What is my preferred name?", "quick"),
        ("Write a LinkedIn headline for me.", "quick"),
        ("What did you tell me about ATS?", "quick"),
        ("How many internships have I done?", "deep"),
        ("Which internship did I do first, Flutter or React Native?", "deep"),
        ("Did I join Meditab before or after Optum?", "deep"),
        ("List all the cloud tools I've explored.", "deep"),
        # Phase 7 routing: several asks, lists, overviews, earlier values.
        ("Where did I go on holiday, and who came with me?", "deep"),
        ("What rent and deposit did I pay for the flat?", "deep"),
        ("Which board games have I bought this year?", "deep"),
        ("What are some hobbies I picked up?", "deep"),
        ("What do you know about my brother?", "deep"),
        ("Which city did I live in before I moved here?", "deep"),
        ("Which car was I driving previously?", "deep"),
        ("What tool did I use to deploy it?", "quick"),
        ("Which board game did I buy last?", "quick"),
        ("What is my favourite pasta sauce?", "quick"),
    ],
)
def test_depth(message: str, depth: str) -> None:
    assert rule_plan(message, NOW).plan.depth == depth


def test_guidance_mentions_resolved_range() -> None:
    plan = rule_plan("What was I worried about in July 2026?", NOW).plan
    assert any("July 2026" in g for g in plan.guidance)


def test_entities_are_capitalized_names() -> None:
    plan = rule_plan("Did I join Meditab before or after Optum?", NOW).plan
    assert {"Meditab", "Optum"} <= set(plan.entities)
    assert "Did" not in plan.entities
