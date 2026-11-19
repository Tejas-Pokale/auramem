from datetime import UTC, datetime, timedelta

import pytest

from memory.config import get_settings
from memory.models import Turn
from memory.retrieve import _enabled
from memory.retrieve.assemble import estimate_tokens, render_memory, select_windows
from memory.retrieve.fusion import reciprocal_rank_fusion

T0 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)


def turn(session: str, seq: int, text: str = "", day: int = 0) -> Turn:
    return Turn(
        turn_id=f"{session}:{seq}",
        user_id="u",
        session_id=session,
        role="user" if seq % 2 == 0 else "assistant",
        text=text or f"{session} turn {seq}",
        created_at=T0 + timedelta(days=day, minutes=seq),
        seq=seq,
        source="typed",
    )


class TestRRF:
    def test_scores_sum_reciprocal_ranks(self) -> None:
        fused = reciprocal_rank_fusion(
            {"vec": [("a", 1), ("b", 2)], "bm25": [("b", 2), ("c", 3)]}, k=60
        )
        scores = {f.key: f.score for f in fused}
        assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["c"] == pytest.approx(1 / 62)
        assert [f.key for f in fused] == ["b", "a", "c"]

    def test_records_rank_per_signal(self) -> None:
        fused = reciprocal_rank_fusion({"vec": [("a", 1)], "bm25": [("x", 0), ("a", 1)]}, k=60)
        a = next(f for f in fused if f.key == "a")
        assert a.ranks == {"vec": 1, "bm25": 2}

    def test_ties_break_on_best_rank_then_key(self) -> None:
        fused = reciprocal_rank_fusion(
            {"s1": [("b", 0), ("a", 0)], "s2": [("a", 0), ("b", 0)]}, k=60
        )
        assert [f.key for f in fused] == ["a", "b"]

    def test_duplicate_keys_in_one_list_count_once(self) -> None:
        fused = reciprocal_rank_fusion({"s": [("a", 0), ("a", 0)]}, k=60)
        assert len(fused) == 1
        assert fused[0].score == pytest.approx(1 / 61)

    def test_empty(self) -> None:
        assert reciprocal_rank_fusion({"s": []}, k=60) == []


class TestSelectWindows:
    def test_turn_cap_counts_neighbors_and_tries_smaller_windows(self) -> None:
        big = [turn("s1", i) for i in range(4)]
        small = [turn("s2", 0), turn("s2", 1)]
        other = [turn("s3", 0), turn("s3", 1)]
        selected = select_windows(
            [small, big, other], budget=10_000, chars_per_token=4, max_turns=4
        )
        assert [t.turn_id for t in selected] == ["s2:0", "s2:1", "s3:0", "s3:1"]

    def test_long_assistant_turns_are_cut_in_rendering_only(self) -> None:
        long = turn("s", 1, "word " * 100)
        text = render_memory([turn("s", 0, "hi"), long], assistant_max_chars=50)
        assistant_line = text.splitlines()[-1]
        assert assistant_line.endswith("\u2026") and len(assistant_line) <= len("Assistant: ") + 51
        assert long.text == "word " * 100  # the stored turn is unchanged
        assert "hi" in text  # user turns are never cut

    def test_matched_assistant_turns_are_kept_whole(self) -> None:
        long = turn("s", 1, "word " * 100)
        text = render_memory([long], assistant_max_chars=50, uncut={long.turn_id})
        assert text.splitlines()[-1] == f"Assistant: {long.text}"

    def test_keeps_rank_order_and_dedupes_overlaps(self) -> None:
        w1 = [turn("s1", 0), turn("s1", 1), turn("s1", 2)]
        w2 = [turn("s1", 1), turn("s1", 2), turn("s1", 3)]
        selected = select_windows([w1, w2], budget=10_000, chars_per_token=4)
        assert [t.turn_id for t in selected] == ["s1:0", "s1:1", "s1:2", "s1:3"]

    def test_skips_windows_that_do_not_fit_but_tries_smaller_ones(self) -> None:
        big = [turn("s1", 0, "x" * 4000)]
        small = [turn("s2", 0, "short")]
        selected = select_windows([big, small], budget=100, chars_per_token=4)
        assert [t.turn_id for t in selected] == ["s2:0"]

    def test_zero_budget_selects_nothing(self) -> None:
        assert select_windows([[turn("s1", 0)]], budget=0, chars_per_token=4) == []


class TestRenderMemory:
    def test_groups_by_session_oldest_first_with_dates(self) -> None:
        turns = [
            turn("late", 1, "later reply", day=10),
            turn("early", 0, "first", day=0),
            turn("late", 0, "later question", day=10),
        ]
        text = render_memory(turns)
        assert text.index("2026-03-01") < text.index("2026-03-11")
        assert text.index("later question") < text.index("later reply")
        assert "User: first" in text
        assert "Assistant: later reply" in text

    def test_marks_gaps_inside_a_session(self) -> None:
        text = render_memory([turn("s", 0, "a"), turn("s", 5, "b")])
        assert "..." in text

    def test_empty(self) -> None:
        assert render_memory([]) == ""


def test_estimate_tokens() -> None:
    assert estimate_tokens("abcd" * 10, chars_per_token=4) == 10
    assert estimate_tokens("", chars_per_token=4) == 0


def test_ablation_switches_drop_signal_families() -> None:
    names = [
        "S1_vector_facts", "S2_vector_turns", "S3_bm25_facts_rewrite0", "S4_bm25_turns",
        "S5_entity_facts", "S6_temporal_facts", "S6t_temporal_turns", "S7_graph_facts",
    ]  # fmt: skip
    s = get_settings()
    assert all(_enabled(n, s) for n in names)
    no_facts = s.model_copy(update={"use_fact_index": False})
    assert [n for n in names if _enabled(n, no_facts)] == [
        "S2_vector_turns", "S4_bm25_turns", "S6t_temporal_turns",
    ]  # fmt: skip
    no_bm25 = s.model_copy(update={"use_bm25": False})
    assert [n for n in names if not _enabled(n, no_bm25)] == [
        "S3_bm25_facts_rewrite0", "S4_bm25_turns",
    ]  # fmt: skip
