# Picking up where we left off (2026-10-25)

## Where things stand

- **All 9 phases of [plan.md](plan.md) are built and evaluated.**
  - Final score on the held-out personal questions: 96.1% (3 runs).
  - The full story is in [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md).
- **`auramem`**, the importable library ([src/auramem](../src/auramem)), and the **chatbot** ([examples/chatbot.py](../examples/chatbot.py)) are done.
  - The chatbot's default user is `tejas-pokale`.
  - It opens every session by continuing the last one.
- **Tests:** 371 pass, and ruff and mypy are clean.
- **Git:** nothing is committed (this folder is not a git repository yet). Before committing, review [evals/RESULTS.md](../evals/RESULTS.md), which contains names from the personal corpus.

## Starting again

```bash
# 1. Start Docker Desktop, then the services (Neo4j data is kept in a Docker volume)
docker compose up -d

# 2. Chat
uv run python examples/chatbot.py            # as tejas-pokale
uv run python examples/chatbot.py --debug    # with retrieval details after each answer

# 3. Optional: check everything still works
uv run pytest
```

The `tejas-pokale` memory starts empty. It fills from your own chats: facts are extracted at
`/quit`, on `/flush`, or automatically as a conversation grows.

## Stored data

| User id | What it is |
| --- | --- |
| `tejas-pokale` | Your chatbot memory (empty until you chat) |
| `eval-large-e3823ecad731` | Personal eval dataset, full corpus (extract_v5) |
| `eval-large-fdb0943d8ec8` | The same corpus without dates in fact embeddings (ablation) |
| `eval-synth500-a7b461696138` | Synthetic benchmark (extract_v4). Rerun evals on it with `EXTRACTOR_PROMPT_VERSION=v4`, or it will re-ingest |

## Known gaps (accepted)

- Evidence recall is 90%, against a 95% target. This comes from the smaller context, chosen for cost.
- Quick-mode search time is 852 ms at the 95th percentile, against a 600 ms target. The cause is the reranker on the laptop GPU.
- About 1.65 questions per second is the limit with many simultaneous users.
- Judge agreement and write-path lag weren't measured.
