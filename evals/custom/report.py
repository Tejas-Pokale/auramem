"""Readable per-question report for an eval run, including the retrieved memory context.

    uv run python -m evals.custom.report evals/custom/runs/<run>.jsonl

Writes ``<run>.md`` next to the JSONL file (gitignored with it: it contains personal data).
``run.py`` calls this automatically after every run. Each question shows the expected answer,
every run's answer and verdict, which evidence was missing, the planner's notes, and the
memory block exactly as the answerer saw it (from the first run; retrieval is deterministic).
Failing questions come first.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

GOLDEN = Path(__file__).parent / "data" / "private" / "golden_v2.csv"


def _fence(text: str) -> str:
    """A fenced block that cannot be closed early by backticks inside the text."""
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}text\n{text}\n{ticks}"


def _question_block(qid: str, records: list[dict[str, Any]], golden: dict[str, str]) -> str:
    first = records[0]
    verdicts = [r.get("verdict") for r in records]
    correct = sum(v == "CORRECT" for v in verdicts)
    score = f"{correct}/{len(records)}" if any(verdicts) else "retrieval only"
    lines = [
        f"## {qid} [{first['test_type']}] {score}",
        "",
        f"**Question:** {golden.get('question', '?')}",
        "",
        f"**Expected:** {golden.get('expected_answer', '?')}",
        "",
    ]
    evidence = ", ".join(first["evidence_fact_ids"]) or "none (abstention)"
    missing = ", ".join(first["missing_evidence"]) or "none"
    lines.append(
        f"**Evidence:** {evidence}. **Missing from context:** {missing}. "
        f"**Memory:** {first['memory_tokens']} tokens, {first.get('facts_in_context', 0)} facts, "
        f"{len(first.get('context_turn_ids', []))} turns; top rerank score "
        f"{first.get('top_rerank_score')}; plan depth {first.get('plan_depth')}."
    )
    for r in records:
        if r.get("answer") is None:
            continue
        failure = f", {r['failure']}" if r.get("failure") else ""
        lines += [
            "",
            f"**Run {r['run']}: {r.get('verdict')}{failure}**",
            "",
            r["answer"],
            "",
            f"*Judge:* {r.get('reason')}",
        ]
    if first.get("guidance"):
        lines += ["", "**Retrieval notes:**", *(f"- {g}" for g in first["guidance"])]
    if first.get("mode") == "deep":
        searches = first.get("deep_tool_calls") or []
        lines += [
            "",
            f"**Deep mode:** {first.get('deep_steps')} steps, {len(searches)} searches, "
            f"stopped by {first.get('deep_stop_reason')}.",
            *(f"{n}. `{c}`" for n, c in enumerate(searches, start=1)),
        ]
    context = first.get("memory_context")
    lines += ["", "**Retrieved context:**", ""]
    if context is None:
        lines.append("_Not recorded (this run predates context logging)._")
    elif context == "":
        lines.append("_Nothing injected._")
    else:
        lines.append(_fence(context))
    return "\n".join(lines)


def write_markdown(run_file: Path, golden_path: Path = GOLDEN) -> Path:
    golden = {r["question_id"]: r for r in csv.DictReader(golden_path.open(encoding="utf-8"))}
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in run_file.open(encoding="utf-8"):
        record = json.loads(line)
        by_question[record["question_id"]].append(record)

    def sort_key(qid: str) -> tuple[int, str]:
        records = by_question[qid]
        wrong = sum(r.get("verdict") == "WRONG" for r in records)
        missing = bool(records[0]["missing_evidence"])
        return (-(wrong * 2 + missing), qid)

    blocks = [
        _question_block(qid, sorted(by_question[qid], key=lambda r: r["run"]), golden.get(qid, {}))
        for qid in sorted(by_question, key=sort_key)
    ]
    out = run_file.with_suffix(".md")
    header = f"# {run_file.stem}\n\n{len(by_question)} questions; failing questions first.\n"
    out.write_text(header + "\n\n".join(["", *blocks]) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_file", type=Path)
    print(f"wrote {write_markdown(parser.parse_args().run_file)}")


if __name__ == "__main__":
    main()
