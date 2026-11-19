"""Load test (PLAN.md section 9, Phase 8): N concurrent users on the read path, at no LLM cost.

Each simulated user sends its messages one after another, and all users run at once. Real
Neo4j, embeddings, and reranker; the answering model is a fake and the planner uses rules only,
so the run costs nothing and measures what the memory layer adds (success criterion: quick-mode
read path p95 under 600 ms, answering call excluded). Nothing is written: replies are not
persisted, so the eval data is untouched. Deep mode is not exercised (it needs the LLM).

    uv run python -m evals.load_test --user-id eval-synth500-a7b461696138 --users 20
"""

import argparse
import asyncio
import csv
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from memory.client import MemoryClient
from memory.config import get_settings
from memory.embed import Embedder
from memory.models import AnswerOutput
from memory.plan import Planner
from memory.rerank import Reranker
from memory.store.neo4j import Neo4jStore

QUESTIONS = Path(__file__).parent / "synth" / "data" / "questions.csv"
NOW = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)


class NoLLMAnswerer:
    model = "none (load test)"

    async def __call__(self, messages: object) -> tuple[AnswerOutput, dict[str, int]]:
        return AnswerOutput(answer="ok"), {}


def _p(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


async def run(user_id: str, users: int, per_user: int) -> dict[str, object]:
    settings = get_settings()
    store = Neo4jStore.connect(settings)
    embedder = Embedder(settings)
    reranker = Reranker(settings)
    await asyncio.to_thread(reranker.warmup)
    client = MemoryClient(
        store, embedder, NoLLMAnswerer(), settings,
        planner=Planner(settings, None), reranker=reranker,
    )  # fmt: skip
    if await store.count_turns(user_id) == 0:
        raise SystemExit(f"{user_id} has no stored turns")
    rows = list(csv.DictReader(QUESTIONS.open(encoding="utf-8")))
    retrieval: list[float] = []
    end_to_end: list[float] = []
    stages: dict[str, list[float]] = {}
    errors = 0

    async def simulated_user(n: int) -> None:
        nonlocal errors
        for i in range(per_user):
            message = rows[(n * per_user + i) % len(rows)]["question"]
            started = time.perf_counter()
            try:
                reply = await client.respond(
                    user_id, f"load-test/{n}", message, now=NOW, mode="quick"
                )
            except Exception:
                errors += 1
                continue
            end_to_end.append((time.perf_counter() - started) * 1000)
            timings = reply.memory.timings_ms
            retrieval.append(timings.get("total", 0.0))
            for stage, ms in timings.items():
                stages.setdefault(stage, []).append(ms)

    started = time.perf_counter()
    try:
        await asyncio.gather(*(simulated_user(n) for n in range(users)))
    finally:
        await embedder.aclose()
        await store.close()
    wall = time.perf_counter() - started
    done = len(end_to_end)
    return {
        "users": users,
        "requests": done,
        "errors": errors,
        "throughput_rps": round(done / wall, 2),
        "read_path_ms": {"p50": round(_p(retrieval, 0.5)), "p95": round(_p(retrieval, 0.95))},
        "end_to_end_ms_no_llm": {
            "p50": round(_p(end_to_end, 0.5)),
            "p95": round(_p(end_to_end, 0.95)),
        },
        "stage_p95_ms": {k: round(_p(v, 0.95)) for k, v in sorted(stages.items())},
        "stage_mean_ms": {k: round(statistics.mean(v)) for k, v in sorted(stages.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-id", required=True, help="an existing user with stored memory")
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--per-user", type=int, default=10)
    args = parser.parse_args()
    result = asyncio.run(run(args.user_id, args.users, args.per_user))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
