"""Judge validation (PLAN.md section 10): hand-grade 30 judgments, require >= 90% agreement.

    uv run python -m evals.custom.handgrade sample runs/<run>.jsonl   # writes runs/handgrade_*.csv
    # fill the human_verdict column with CORRECT or WRONG, then:
    uv run python -m evals.custom.handgrade score runs/handgrade_<...>.csv

The sample is stratified by test type and judge verdict so both verdicts get checked.
Hide the judge's verdict while grading if you can (the column is last for that reason).
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from evals.custom.judge import JUDGE_VERSION
from evals.custom.run import GOLDEN, RUNS_DIR

SAMPLE_SIZE = 30
AGREEMENT_TARGET = 0.9
SEED = 30
COLUMNS = [
    "question_id",
    "run",
    "test_type",
    "question",
    "expected_answer",
    "answer",
    "human_verdict",
    "judge_verdict",
    "judge_reason",
]


def sample(run_file: Path) -> Path:
    golden = {r["question_id"]: r for r in csv.DictReader(GOLDEN.open(encoding="utf-8"))}
    records = [json.loads(line) for line in run_file.open(encoding="utf-8")]
    judged = [r for r in records if r.get("verdict")]
    buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for r in judged:
        buckets[(str(r["test_type"]), str(r["verdict"]))].append(r)
    rng = random.Random(SEED)
    for items in buckets.values():
        rng.shuffle(items)
    picked: list[dict[str, object]] = []
    while len(picked) < min(SAMPLE_SIZE, len(judged)):  # round-robin across buckets
        for key in sorted(buckets):
            if buckets[key] and len(picked) < SAMPLE_SIZE:
                picked.append(buckets[key].pop())
    out = RUNS_DIR / f"handgrade_{JUDGE_VERSION}_{run_file.stem}.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in sorted(picked, key=lambda r: (str(r["question_id"]), int(str(r["run"])))):
            g = golden[str(r["question_id"])]
            w.writerow(
                {
                    "question_id": r["question_id"],
                    "run": r["run"],
                    "test_type": r["test_type"],
                    "question": g["question"],
                    "expected_answer": g["expected_answer"],
                    "answer": r["answer"],
                    "human_verdict": "",
                    "judge_verdict": r["verdict"],
                    "judge_reason": r["reason"],
                }
            )
    return out


def score(graded: Path) -> float:
    rows = list(csv.DictReader(graded.open(encoding="utf-8")))
    labeled = [r for r in rows if r["human_verdict"].strip().upper() in {"CORRECT", "WRONG"}]
    if not labeled:
        raise SystemExit("no human_verdict values filled in")
    agree = [r for r in labeled if r["human_verdict"].strip().upper() == r["judge_verdict"]]
    rate = len(agree) / len(labeled)
    print(f"agreement {len(agree)}/{len(labeled)} = {rate:.0%} (target {AGREEMENT_TARGET:.0%})")
    for r in labeled:
        if r not in agree:
            print(
                f"  disagree {r['question_id']} run {r['run']} ({r['test_type']}): "
                f"human {r['human_verdict']}, judge {r['judge_verdict']}: {r['judge_reason']}"
            )
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample").add_argument("run_file", type=Path)
    sub.add_parser("score").add_argument("graded_file", type=Path)
    args = parser.parse_args()
    if args.cmd == "sample":
        print(f"wrote {sample(args.run_file)}")
    else:
        score(args.graded_file)


if __name__ == "__main__":
    main()
