"""Eval harness logic on synthetic data (the real corpus is private and not in CI)."""

import csv
import json
from pathlib import Path

import pytest
from evals.custom.build_sessions import ACKS, distractor_sessions, fact_sessions
from evals.custom.run import Record, classify_failure, summarize

CORPUS_COLS = [
    "fact_id",
    "source_fact_id",
    "role",
    "text",
    "session_key",
    "session_date",
    "approximate",
    "reply_to",
]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.csv"
    rows = [
        ["1", "1", "user", "I live in Pune.", "home", "2026-01-05", "true", ""],
        ["2", "2", "user", "Should I learn Rust?", "home", "2026-01-05", "true", ""],
        ["A1", "", "assistant", "Yes, start with ownership.", "home", "2026-01-05", "true", "2"],
        ["3", "3", "user", "I moved to Mumbai.", "move", "2026-06-01", "true", ""],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CORPUS_COLS)
        w.writerows(rows)
    return path


def test_fact_sessions_pair_facts_with_replies(corpus: Path) -> None:
    sessions = fact_sessions(corpus, seed=1)
    home, move = sessions
    assert [t.role for t in home.turns] == ["user", "assistant", "user", "assistant"]
    assert home.turns[0].fact_id == "1"
    assert home.turns[1].text in ACKS and home.turns[1].fact_id is None
    assert home.turns[3].text == "Yes, start with ownership."
    assert home.turns[3].fact_id == "A1"
    assert move.turns[0].created_at.date().isoformat() == "2026-06-01"
    stamps = [t.created_at for t in home.turns]
    assert stamps == sorted(stamps)


def test_fact_sessions_are_deterministic(corpus: Path) -> None:
    a, b = fact_sessions(corpus, seed=7), fact_sessions(corpus, seed=7)
    assert [(t.text, t.created_at) for s in a for t in s.turns] == [
        (t.text, t.created_at) for s in b for t in s.turns
    ]


def test_distractor_sessions_drop_dangling_user_turn(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    turns = [
        {"role": "user", "text": "q1"},
        {"role": "assistant", "text": "a1"},
        {"role": "user", "text": "dangling"},
    ]
    path.write_text(json.dumps({"distractor_id": "D001", "turns": turns}) + "\n", encoding="utf-8")
    [session] = distractor_sessions(path)
    assert [t.text for t in session.turns] == ["q1", "a1"]


def record(qid: str, run: int, **kw: object) -> Record:
    base: dict[str, object] = {
        "question_id": qid,
        "run": run,
        "test_type": "single_fact",
        "must_abstain": False,
        "evidence_fact_ids": ["1"],
        "missing_evidence": [],
        "memory_tokens": 100,
        "retrieval_ms": 50.0,
    }
    return Record(**(base | kw))  # type: ignore[arg-type]


class TestClassifyFailure:
    def test_correct_has_no_failure(self) -> None:
        assert classify_failure(record("Q1", 0, verdict="CORRECT")) is None

    def test_wrong_abstention_is_fabrication(self) -> None:
        r = record("Q1", 0, verdict="WRONG", must_abstain=True, evidence_fact_ids=[])
        assert classify_failure(r) == "fabrication"

    def test_wrong_with_missing_evidence_is_retrieval_miss(self) -> None:
        r = record("Q1", 0, verdict="WRONG", missing_evidence=["1"])
        assert classify_failure(r) == "retrieval_miss"

    def test_wrong_with_evidence_is_reasoning_error(self) -> None:
        assert classify_failure(record("Q1", 0, verdict="WRONG")) == "reasoning_error"


def test_summarize_averages_runs_and_excludes_abstention_from_recall() -> None:
    records = [
        record("Q1", 0, verdict="CORRECT"),
        record("Q1", 1, verdict="WRONG", failure="reasoning_error"),
        record("Q2", 0, verdict="CORRECT", missing_evidence=["1"]),
        record("Q2", 1, verdict="CORRECT", missing_evidence=["1"]),
        record(
            "Q3",
            0,
            verdict="WRONG",
            must_abstain=True,
            evidence_fact_ids=[],
            test_type="abstention",
            failure="fabrication",
        ),
        record(
            "Q3",
            1,
            verdict="CORRECT",
            must_abstain=True,
            evidence_fact_ids=[],
            test_type="abstention",
        ),
    ]
    s = summarize(records, runs=2)
    assert s.n_questions == 3
    assert s.accuracy == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert s.evidence_recall == pytest.approx(0.5)  # Q1 ok, Q2 missing; Q3 has no evidence
    assert s.accuracy_by_type["abstention"] == pytest.approx(0.5)
    assert s.fabrications_per_run == pytest.approx(0.5)
    assert s.failures == {"fabrication": 0.5, "reasoning_error": 0.5}


def test_summarize_l0_has_no_accuracy() -> None:
    s = summarize([record("Q1", 0)], runs=1)
    assert s.accuracy is None
    assert s.evidence_recall == 1.0
