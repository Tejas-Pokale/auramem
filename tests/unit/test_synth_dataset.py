import json
from pathlib import Path
from typing import Any

from evals.synth.dataset import build_synth
from evals.synth.generate import _key_tokens, check_render
from evals.synth.models import LedgerFact, RenderedSession, RenderedTurn


def fact(fid: str, statement: str, speaker: str = "user", value: str | None = None) -> LedgerFact:
    return LedgerFact(
        fact_id=fid, month="2025-03", session=1, told_day=4, kind="personal",
        speaker=speaker, statement=statement, value=value,
    )  # fmt: skip


def turn(role: str, text: str, *ids: str) -> RenderedTurn:
    return RenderedTurn(role=role, text=text, fact_ids=list(ids))


def test_key_tokens_are_numbers_and_names() -> None:
    f = fact("F1", "My rent went up to $2,025 at Harlan Property.", value="$2,025")
    assert _key_tokens(f) == ["2,025"]
    assert "Harlan" in _key_tokens(fact("F2", "I moved in with Harlan Property in May."))


def test_check_render_accepts_a_faithful_session() -> None:
    facts = [
        fact("F1", "My rent is $2,025.", value="$2,025"),
        fact("F2", "Try a budget.", "assistant"),
    ]
    session = RenderedSession(
        turns=[
            turn("user", "Rent is now $2,025 a month.", "F1"),
            turn("assistant", "Try a budget.", "F2"),
        ]
    )
    assert check_render(session, facts) == []


def test_check_render_reports_missing_values_roles_and_order() -> None:
    facts = [
        fact("F1", "My rent is $2,025.", value="$2,025"),
        fact("F2", "Try a budget.", "assistant"),
    ]
    session = RenderedSession(
        turns=[turn("assistant", "Hello", "F2"), turn("user", "Rent went up.", "F1")]
    )
    problems = check_render(session, facts)
    assert any("alternate" in p for p in problems)
    assert any("F1 turn lacks" in p for p in problems)
    unstated = check_render(
        RenderedSession(turns=[turn("user", "hi"), turn("assistant", "hey")]), facts
    )
    assert "F1 is not stated" in unstated and "F2 is not stated" in unstated


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_build_synth_interleaves_and_maps_evidence(tmp_path: Path) -> None:
    sessions = _write(tmp_path / "s.jsonl", [{
        "session_key": "2025-03-s01", "date": "2025-03-04", "unverified_fact_ids": [],
        "turns": [{"role": "user", "text": "Rent is $2,025.", "fact_ids": ["F1", "F2"]},
                  {"role": "assistant", "text": "Noted.", "fact_ids": []}],
    }])  # fmt: skip
    distractors = _write(tmp_path / "d.jsonl", [{
        "distractor_id": "D001", "date": "2025-01-02", "topic": "x",
        "turns": [{"role": "user", "text": "q"}, {"role": "assistant", "text": "a"}],
    }])  # fmt: skip
    data = build_synth(sessions, distractors)
    assert [s.key for s in data.sessions] == ["synth-D001", "synth-2025-03-s01"]
    assert data.evidence_locations() == {
        "F1": ("synth-2025-03-s01", 0),
        "F2": ("synth-2025-03-s01", 0),
    }
    assert data.fingerprint == build_synth(sessions, distractors).fingerprint
