"""Turn corpus v2 + frozen distractors into dated conversations (PLAN.md section 10).

Deterministic (fixed seed): the same inputs always give the same sessions, turn texts and
timestamps. Each user fact becomes a user turn followed by an assistant reply (an authored
reply from the corpus when there is one, otherwise a short neutral acknowledgement).
Conflict and update pairs live in different sessions, older first (set in corpus_v2.csv).

Tiers: small = fact sessions only; medium/large add distractor sessions up to a token budget.
"""

import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal

DATA = Path(__file__).parent / "data"
CORPUS = DATA / "private" / "corpus_v2.csv"
DISTRACTORS = DATA / "distractors" / "distractors_v1.jsonl"

Tier = Literal["small", "medium", "large", "synth500"]
TIERS: tuple[Tier, ...] = ("small", "medium", "large")
# Distractor tokens added per tier; fact sessions add ~6K on top (section 10 tier table).
TIER_DISTRACTOR_TOKENS: dict[Tier, int] = {"small": 0, "medium": 34_000, "large": 114_000}
CHARS_PER_TOKEN = 4
SEED = 20261024
TZ = timezone(timedelta(hours=5, minutes=30), "IST")
DISTRACTOR_SPAN = (date(2025, 1, 3), date(2026, 10, 20))

ACKS = [
    "Got it, thanks for sharing.",
    "Noted.",
    "Thanks, that's helpful context.",
    "Understood.",
    "Okay, I'll keep that in mind.",
    "Makes sense.",
    "Thanks for letting me know.",
    "Good to know.",
    "Got it.",
    "That's useful context, thanks.",
]


@dataclass
class EvalTurn:
    role: Literal["user", "assistant"]
    text: str
    created_at: datetime
    fact_id: str | None = None


@dataclass
class EvalSession:
    key: str
    kind: Literal["fact", "distractor"]
    turns: list[EvalTurn] = field(default_factory=list)

    @property
    def started_at(self) -> datetime:
        return self.turns[0].created_at


@dataclass(frozen=True)
class TierData:
    tier: Tier
    sessions: list[EvalSession]
    fingerprint: str
    # Explicit fact -> (session key, seq) map, for datasets where a turn states several facts.
    evidence: dict[str, tuple[str, int]] | None = None

    def evidence_locations(self) -> dict[str, tuple[str, int]]:
        """fact_id -> (session key, seq) of the turn that states it."""
        if self.evidence is not None:
            return self.evidence
        return {
            t.fact_id: (s.key, seq)
            for s in self.sessions
            for seq, t in enumerate(s.turns)
            if t.fact_id is not None
        }

    def approx_tokens(self) -> int:
        return sum(len(t.text) for s in self.sessions for t in s.turns) // CHARS_PER_TOKEN


def _timestamps(rng: random.Random, day: date, n: int) -> list[datetime]:
    """A plausible time of day, then 1-4 minutes per user turn and 10-40 s per reply."""
    at = datetime.combine(day, time(rng.randint(9, 22), rng.randint(0, 59)), TZ)
    stamps = []
    for i in range(n):
        stamps.append(at)
        at += timedelta(seconds=rng.randint(10, 40) if i % 2 == 0 else rng.randint(60, 240))
    return stamps


def fact_sessions(path: Path = CORPUS, seed: int = SEED) -> list[EvalSession]:
    rng = random.Random(seed)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    sessions: dict[str, tuple[date, list[tuple[str, str, str]]]] = {}
    for row in rows:
        day = date.fromisoformat(row["session_date"])
        sessions.setdefault(row["session_key"], (day, []))[1].append(
            (row["role"], row["text"], row["fact_id"])
        )

    out: list[EvalSession] = []
    for key, (day, items) in sessions.items():
        pairs: list[tuple[str, str, str, str | None]] = []  # user text, fact, reply, reply fact
        i = 0
        while i < len(items):
            role, text, fact_id = items[i]
            if role != "user":
                raise ValueError(f"{key}: assistant row {fact_id} does not follow a user fact")
            if i + 1 < len(items) and items[i + 1][0] == "assistant":
                pairs.append((text, fact_id, items[i + 1][1], items[i + 1][2]))
                i += 2
            else:
                pairs.append((text, fact_id, rng.choice(ACKS), None))
                i += 1
        stamps = _timestamps(rng, day, 2 * len(pairs))
        turns: list[EvalTurn] = []
        for n, (user_text, fact_id, reply, reply_fact) in enumerate(pairs):
            turns.append(EvalTurn("user", user_text, stamps[2 * n], fact_id))
            turns.append(EvalTurn("assistant", reply, stamps[2 * n + 1], reply_fact))
        out.append(EvalSession(key=f"fact-{key}", kind="fact", turns=turns))
    return out


def distractor_sessions(path: Path = DISTRACTORS, seed: int = SEED) -> list[EvalSession]:
    rng = random.Random(seed + 1)
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    start, end = DISTRACTOR_SPAN
    span = (end - start).days
    out: list[EvalSession] = []
    for n, record in enumerate(records):
        raw = record["turns"]
        if len(raw) % 2:  # keep whole exchanges only
            raw = raw[:-1]
        day = start + timedelta(days=round(n * span / max(1, len(records) - 1)))
        stamps = _timestamps(rng, day, len(raw))
        turns = [EvalTurn(t["role"], t["text"], at) for t, at in zip(raw, stamps, strict=True)]
        out.append(
            EvalSession(key=f"distractor-{record['distractor_id']}", kind="distractor", turns=turns)
        )
    return out


def build_tier(tier: Tier) -> TierData:
    sessions = fact_sessions()
    budget = TIER_DISTRACTOR_TOKENS[tier]
    used = 0
    for session in distractor_sessions():
        cost = sum(len(t.text) for t in session.turns) // CHARS_PER_TOKEN
        if used + cost > budget:
            break
        sessions.append(session)
        used += cost
    sessions.sort(key=lambda s: (s.started_at, s.key))
    payload = json.dumps([asdict(s) for s in sessions], default=str, sort_keys=True)
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return TierData(tier=tier, sessions=sessions, fingerprint=fingerprint)


if __name__ == "__main__":
    for t in TIERS:
        data = build_tier(t)
        kinds = [s.kind for s in data.sessions]
        turns = sum(len(s.turns) for s in data.sessions)
        print(
            f"{t:6s} sessions={len(kinds)} (fact={kinds.count('fact')}, "
            f"distractor={kinds.count('distractor')}) turns={turns} "
            f"~tokens={data.approx_tokens()} fingerprint={data.fingerprint[:10]}"
        )
