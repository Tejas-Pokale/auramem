"""Load the generated synthetic benchmark as eval sessions (same shape as the custom tiers).

Persona sessions and distractors are interleaved chronologically. Timestamps are derived
deterministically from each session's date; every turn keeps the ledger fact ids it states,
which become the evidence map for L0 (fact -> session key, turn seq).
"""

import hashlib
import json
import random
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from evals.custom.build_sessions import EvalSession, EvalTurn, Tier, TierData

DATA = Path(__file__).parent / "data"
SESSIONS = DATA / "sessions.jsonl"
DISTRACTORS = DATA / "distractors.jsonl"
QUESTIONS = DATA / "questions.csv"
TIER: Tier = "synth500"
SEED = 500
TZ = timezone(timedelta(hours=-7), "MST")  # the persona lives in Denver


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _stamps(rng: random.Random, day: date, n: int) -> list[datetime]:
    at = datetime.combine(day, time(rng.randint(7, 22), rng.randint(0, 59)), TZ)
    out = []
    for i in range(n):
        out.append(at)
        at += timedelta(seconds=rng.randint(10, 50) if i % 2 == 0 else rng.randint(45, 300))
    return out


def build_synth(sessions_path: Path = SESSIONS, distractors_path: Path = DISTRACTORS) -> TierData:
    rng = random.Random(SEED)
    sessions: list[EvalSession] = []
    evidence: dict[str, tuple[str, int]] = {}
    for record in _read(sessions_path):
        key = f"synth-{record['session_key']}"
        turns_raw = record["turns"]
        stamps = _stamps(rng, date.fromisoformat(record["date"]), len(turns_raw))
        turns = []
        for seq, (t, at) in enumerate(zip(turns_raw, stamps, strict=True)):
            turns.append(EvalTurn(t["role"], t["text"], at))
            for fid in t.get("fact_ids", []):
                evidence.setdefault(fid, (key, seq))
        sessions.append(EvalSession(key=key, kind="fact", turns=turns))
    for record in _read(distractors_path):
        turns_raw = record["turns"]
        stamps = _stamps(rng, date.fromisoformat(record["date"]), len(turns_raw))
        turns = [
            EvalTurn(t["role"], t["text"], at) for t, at in zip(turns_raw, stamps, strict=True)
        ]
        sessions.append(
            EvalSession(key=f"synth-{record['distractor_id']}", kind="distractor", turns=turns)
        )
    sessions.sort(key=lambda s: (s.started_at, s.key))
    payload = json.dumps([asdict(s) for s in sessions], default=str, sort_keys=True)
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return TierData(tier=TIER, sessions=sessions, fingerprint=fingerprint, evidence=evidence)
