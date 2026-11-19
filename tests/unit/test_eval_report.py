import csv
import json
from pathlib import Path

from evals.custom.report import write_markdown


def test_report_shows_context_answers_and_failures_first(tmp_path: Path) -> None:
    golden = tmp_path / "golden.csv"
    with golden.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_id", "question", "expected_answer"])
        w.writeheader()
        w.writerow({"question_id": "Q1", "question": "Where?", "expected_answer": "Pune"})
        w.writerow({"question_id": "Q2", "question": "When?", "expected_answer": "June"})
    base = {
        "test_type": "single_fact", "evidence_fact_ids": ["1"], "memory_tokens": 10,
        "facts_in_context": 1, "context_turn_ids": [], "guidance": ["note"],
    }  # fmt: skip
    records = [
        {**base, "question_id": "Q1", "run": 0, "missing_evidence": [], "verdict": "CORRECT",
         "answer": "Pune.", "reason": "ok", "memory_context": "## Facts\n- lives in Pune"},
        {**base, "question_id": "Q2", "run": 0, "missing_evidence": ["1"], "verdict": "WRONG",
         "failure": "retrieval_miss", "answer": "Unknown.", "reason": "missed",
         "memory_context": "```code``` inside"},
    ]  # fmt: skip
    run_file = tmp_path / "run.jsonl"
    run_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    text = write_markdown(run_file, golden).read_text(encoding="utf-8")

    assert text.index("## Q2") < text.index("## Q1")  # failing first
    assert "**Run 0: WRONG, retrieval_miss**" in text
    assert "- lives in Pune" in text
    assert "````text\n```code``` inside\n````" in text  # fence longer than inner backticks
    assert "**Missing from context:** 1." in text
