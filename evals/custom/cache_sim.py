"""Multi-turn prompt-cache simulation (Phase 6 gate: cache hit rate above 70%).

A throwaway user gets a copy of an existing user's profile, then chats through several
conversations with real stored history (``remember`` after each reply, no extraction queue).
Midway, one observation is appended to the profile, as an extraction job would, which
invalidates the cached prefix once. The throwaway user is deleted at the end, and the source
user is only read.

    uv run python -m evals.custom.cache_sim --source-user eval-synth500-a7b461696138
"""

import argparse
import asyncio
import csv
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from memory.client import MemoryClient
from memory.config import get_settings
from memory.observe import estimate_tokens

QUESTIONS = Path(__file__).resolve().parents[1] / "synth" / "data" / "questions.csv"
SESSIONS = 3
MESSAGES_PER_SESSION = 7
PROFILE_UPDATE_BEFORE_SESSION = 2  # an extraction job runs between conversations 2 and 3
NEW_OBSERVATION = "- [2026-10-26] (medium) Started a weekly Sunday meal-prep routine."


async def main(source_user: str) -> None:
    settings = get_settings()
    client = MemoryClient.from_settings(settings)
    store = client.store
    sim = f"cache-sim-{uuid.uuid4().hex[:8]}"
    rows = list(csv.DictReader(QUESTIONS.open(encoding="utf-8")))
    messages = [
        r["question"] for r in rows[:: max(1, len(rows) // (SESSIONS * MESSAGES_PER_SESSION))]
    ]
    try:
        profile = await store.get_observation_log(source_user)
        assert profile is not None, f"{source_user} has no profile"
        now = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)
        calls: list[dict[str, int | str]] = []
        for s in range(SESSIONS):
            session = f"{sim}/s{s}"
            for m in range(MESSAGES_PER_SESSION):
                if s == 0 and m == 0:
                    # Seed the copied profile through the extraction commit (needs a session).
                    await client.remember(sim, session, "hello", "hi", asked_at=now,
                                          replied_at=now + timedelta(seconds=1))  # fmt: skip
                    current = await store.get_session(sim, session)
                    assert current is not None
                    await store.commit_extraction(
                        sim, session, entities=[], facts=[],
                        observations=profile.text.splitlines(),
                        expected_watermark=current.extracted_through_seq, up_to_seq=1,
                    )  # fmt: skip
                if s == PROFILE_UPDATE_BEFORE_SESSION and m == 0:
                    # An extraction job between conversations updates the profile.
                    prev = f"{sim}/s{s - 1}"
                    current = await store.get_session(sim, prev)
                    assert current is not None
                    await store.commit_extraction(
                        sim, prev, entities=[], facts=[],
                        observations=[NEW_OBSERVATION],
                        expected_watermark=current.extracted_through_seq,
                        up_to_seq=2 * MESSAGES_PER_SESSION - 1,  # two turns per message
                    )  # fmt: skip
                text = messages[(s * MESSAGES_PER_SESSION + m) % len(messages)]
                now += timedelta(minutes=2)
                reply = await client.respond(sim, session, text, now=now)
                await client.remember(sim, session, text, reply.text, asked_at=now,
                                      replied_at=now + timedelta(seconds=5))  # fmt: skip
                u = reply.usage or {}
                calls.append({
                    "session": s, "message": m,
                    "input": int(u.get("input_tokens", 0)),
                    "cache_read": int(u.get("cache_read_tokens", 0)),
                    "cache_write": int(u.get("cache_write_tokens", 0)),
                })  # fmt: skip
        hits = sum(1 for c in calls if int(c["cache_read"]) > 0)
        read = sum(int(c["cache_read"]) for c in calls)
        total_in = sum(int(c["input"]) for c in calls)
        print(f"profile copied: {estimate_tokens(profile.text, settings)} tokens (estimate)")
        for c in calls:
            # cache_write is as reported by langchain usage_metadata (may read 0 on writes).
            print(
                f"  s{c['session']} m{c['message']}: input {c['input']:>6}, "
                f"cache read {c['cache_read']:>5}, cache write {c['cache_write']:>5}"
            )
        print(f"calls {len(calls)}, calls with a cache hit {hits} ({hits / len(calls):.0%})")
        print(f"share of all input tokens served from cache: {read / total_in:.0%}")
    finally:
        counts = await store.delete_user_data(sim)
        print("deleted simulation user:", {k: v for k, v in counts.items() if v})
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-user", required=True)
    asyncio.run(main(parser.parse_args().source_user))
