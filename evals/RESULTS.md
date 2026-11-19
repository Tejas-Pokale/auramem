# Eval results

Newest last. Produced by `evals/custom/run.py`.

## 2026-10-25 01:30 IST: B0: Phase 2 baseline, turns only (L2, large tier, test split, 3 runs)

- git: `no-commit`; write path `turns_v1`; run file `20261025-013006_L2_large_test.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v1`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000; corpus 1474 turns

| Metric | Value |
| --- | --- |
| Questions | 43 |
| Answer accuracy | 95.3% |
| Evidence recall (L0) | 90.0% |
| Fabrications on abstention items (per run) | 0.0 of 3 |
| Memory tokens per query (mean / p95) | 1913 / 3442 |
| Retrieval latency ms (p50 / p95) | 95 / 384 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 50.0% |
| knowledge_update | 100.0% | 50.0% |
| multi_fact | 81.8% | 90.9% |
| preference_application | 100.0% | 50.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 1.0, retrieval_miss 1.0

## 2026-10-25 01:31 IST: B0-dev: Phase 2 baseline, turns only (L1, medium tier, dev split, 1 run)

- git: `no-commit`; write path `turns_v1`; run file `20261025-013144_L1_medium_dev.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v1`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000; corpus 678 turns

| Metric | Value |
| --- | --- |
| Questions | 65 |
| Answer accuracy | 95.4% |
| Evidence recall (L0) | 91.7% |
| Fabrications on abstention items (per run) | 0.0 of 5 |
| Memory tokens per query (mean / p95) | 1219 / 2995 |
| Retrieval latency ms (p50 / p95) | 90 / 358 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| inference | 50.0% | 0.0% |
| knowledge_update | 100.0% | 100.0% |
| multi_fact | 88.2% | 88.2% |
| preference_application | 100.0% | 66.7% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: retrieval_miss 3.0
## 2026-10-25: judge_v1 validation waived

The user accepted `judge_v1` verdicts without hand-grading (PLAN.md section 10 asks for
30 hand-graded judgments at >= 90% agreement). No agreement rate has been measured. The
unfilled sample is `evals/custom/runs/handgrade_judge_v1_20261025-013006_L2_large_test.csv`.
Scores in this file are judge-only until that check is done.

## 2026-10-25 02:20 IST: P3-dev: planner + rerank (L1, medium tier, dev split, 1 run)

- git: `no-commit`; write path `turns_v1`; run file `20261025-022043_L1_medium_dev.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v2`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.15, none_depth_min_score=0.5; corpus 678 turns

| Metric | Value |
| --- | --- |
| Questions | 65 |
| Answer accuracy | 72.3% |
| Evidence recall (L0) | 66.7% |
| Fabrications on abstention items (per run) | 0.0 of 5 |
| Memory tokens per query (mean / p95) | 68 / 143 |
| Retrieval latency ms (p50 / p95) | 815 / 3697 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 65) |
| Rerank latency ms (p95) | 1734 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| inference | 0.0% | 0.0% |
| knowledge_update | 25.0% | 25.0% |
| multi_fact | 64.7% | 64.7% |
| preference_application | 66.7% | 0.0% |
| sensitive_history | 33.3% | 33.3% |
| single_fact | 87.0% | 87.0% |
| temporal | 75.0% | 75.0% |

Failures per run: retrieval_miss 18.0

## 2026-10-25 02:25 IST: P3-dev tuning: rerank_min_score=0.001 (L1, medium tier, dev split, 1 run)

- git: `no-commit`; write path `turns_v1`; run file `20261025-022533_L1_medium_dev.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v2`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.001, none_depth_min_score=0.5; corpus 678 turns

| Metric | Value |
| --- | --- |
| Questions | 65 |
| Answer accuracy | 93.8% |
| Evidence recall (L0) | 90.0% |
| Fabrications on abstention items (per run) | 0.0 of 5 |
| Memory tokens per query (mean / p95) | 326 / 804 |
| Retrieval latency ms (p50 / p95) | 896 / 2233 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 65) |
| Rerank latency ms (p95) | 1983 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| inference | 50.0% | 0.0% |
| knowledge_update | 75.0% | 50.0% |
| multi_fact | 94.1% | 94.1% |
| preference_application | 100.0% | 100.0% |
| sensitive_history | 66.7% | 66.7% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: retrieval_miss 4.0

## 2026-10-25 02:26 IST: P3-dev tuning: rerank_min_score=0.0 (L1, medium tier, dev split, 1 run)

- git: `no-commit`; write path `turns_v1`; run file `20261025-022656_L1_medium_dev.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v2`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5; corpus 678 turns

| Metric | Value |
| --- | --- |
| Questions | 65 |
| Answer accuracy | 98.5% |
| Evidence recall (L0) | 93.3% |
| Fabrications on abstention items (per run) | 0.0 of 5 |
| Memory tokens per query (mean / p95) | 973 / 2306 |
| Retrieval latency ms (p50 / p95) | 813 / 2000 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 65) |
| Rerank latency ms (p95) | 1602 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| inference | 100.0% | 0.0% |
| knowledge_update | 75.0% | 50.0% |
| multi_fact | 100.0% | 100.0% |
| preference_application | 100.0% | 100.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: retrieval_miss 1.0

## 2026-10-25 02:30 IST: P3: Phase 3 gate, planner + rerank (rerank_min_score tuned on dev) (L2, large tier, test split, 3 runs)

- git: `no-commit`; write path `turns_v1`; run file `20261025-023036_L2_large_test.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v2`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5; corpus 1474 turns

| Metric | Value |
| --- | --- |
| Questions | 43 |
| Answer accuracy | 91.5% |
| Evidence recall (L0) | 85.0% |
| Fabrications on abstention items (per run) | 0.0 of 3 |
| Memory tokens per query (mean / p95) | 1397 / 2731 |
| Retrieval latency ms (p50 / p95) | 733 / 1856 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 129) |
| Rerank latency ms (p95) | 1763 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 50.0% |
| knowledge_update | 66.7% | 50.0% |
| multi_fact | 81.8% | 81.8% |
| preference_application | 100.0% | 50.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 50.0% | 50.0% |

Failures per run: reasoning_error 0.7, retrieval_miss 3.0

## 2026-10-25: Phase 3 gate verdict: NOT PASSED

P3 (planner + rerank, `rerank_min_score=0.0` tuned on dev) scored 91.5% on the Large/test
split vs B0 95.3% (evidence recall 85.0% vs 90.0%). Dev went the other way (98.5% vs 95.4%).

Newly failing test questions (classification per PLAN.md section 10):
- Q095 temporal, retrieval miss. The answer turn (joining JHEX) does not mention the
  question's anchor (leaving Meditab); the cross-encoder moved it from RRF rank 11 to rank 17,
  outside MAX_TURNS=15. Multi-hop weakness of reranking alone.
- Q090 knowledge_update, reasoning error in 2 of 3 runs with the evidence in context.

No fix was tuned on the test split. Both splits are near ceiling, so the B0/P3 difference is
two questions out of 43.
## 2026-10-25: Phase 3 gate waived by the user

The user chose to accept Phase 3 as built (P3: 91.5% test, 98.5% dev) and move to Phase 4.
Later phases compare against both B0 and P3.
## 2026-10-25 03:17 IST: P4-dev: facts (extract_v2) + planner + rerank (L1, medium tier, dev split, 1 run)

- git: `no-commit`; write path `turns_facts_v1`; run file `20261025-031736_L1_medium_dev.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v3`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=40; corpus 678 turns, 174 facts (174 mentions; by attribution {'user': 169, 'assistant': 5}; hypothetical 0)

| Metric | Value |
| --- | --- |
| Questions | 65 |
| Answer accuracy | 96.9% |
| Evidence recall (L0) | 96.7% |
| Fabrications on abstention items (per run) | 0.0 of 5 |
| Facts in context per query (mean) | 39.6 |
| Memory tokens per query (mean / p95) | 3593 / 5212 |
| Retrieval latency ms (p50 / p95) | 695 / 1786 |
| Planner latency ms (p50 / p95) | 0.1 / 0.3 (rules 65) |
| Rerank latency ms (p95) | 1671 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| inference | 100.0% | 0.0% |
| knowledge_update | 100.0% | 100.0% |
| multi_fact | 100.0% | 100.0% |
| preference_application | 66.7% | 100.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 75.0% | 100.0% |

Failures per run: reasoning_error 2.0

## 2026-10-25 03:35 IST: P4-dev (after fact time-range rendering fix) (L1, medium tier, dev split, 1 run)

- git: `no-commit`; write path `turns_facts_v1`; run file `20261025-033554_L1_medium_dev.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v3`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=40; corpus 678 turns, 174 facts (174 mentions; by attribution {'user': 169, 'assistant': 5}; hypothetical 0)

| Metric | Value |
| --- | --- |
| Questions | 65 |
| Answer accuracy | 100.0% |
| Evidence recall (L0) | 96.7% |
| Fabrications on abstention items (per run) | 0.0 of 5 |
| Facts in context per query (mean) | 39.6 |
| Memory tokens per query (mean / p95) | 3564 / 5171 |
| Retrieval latency ms (p50 / p95) | 906 / 2473 |
| Planner latency ms (p50 / p95) | 0.1 / 0.3 (rules 65) |
| Rerank latency ms (p95) | 1777 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| inference | 100.0% | 0.0% |
| knowledge_update | 100.0% | 100.0% |
| multi_fact | 100.0% | 100.0% |
| preference_application | 100.0% | 100.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

## 2026-10-25 03:46 IST: P4: Phase 4 gate, facts (extract_v2) + planner + rerank (L2, large tier, test split, 3 runs)

- git: `no-commit`; write path `turns_facts_v1`; run file `20261025-034650_L2_large_test.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v3`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=40; corpus 1474 turns, 369 facts (369 mentions; by attribution {'user': 336, 'assistant': 33}; hypothetical 0)

| Metric | Value |
| --- | --- |
| Questions | 43 |
| Answer accuracy | 93.0% |
| Evidence recall (L0) | 95.0% |
| Fabrications on abstention items (per run) | 0.7 of 3 |
| Facts in context per query (mean) | 40.0 |
| Memory tokens per query (mean / p95) | 4322 / 5767 |
| Retrieval latency ms (p50 / p95) | 742 / 1790 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 129) |
| Rerank latency ms (p95) | 1553 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 77.8% | n/a |
| assistant_recall | 100.0% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 50.0% |
| knowledge_update | 50.0% | 100.0% |
| multi_fact | 87.9% | 100.0% |
| preference_application | 100.0% | 50.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: fabrication 0.7, reasoning_error 2.3

## 2026-10-25: Phase 4 gate verdict: PARTIAL (temporal passed, knowledge_update did not)

P4 (facts via extract_v2 + planner + rerank, answer_v3) vs P3 on Large/test (3 runs):
overall 93.0% vs 91.5% (B0 95.3%); evidence recall 95.0% vs 85.0%; temporal 100% vs 50% (gate
item passed); knowledge_update 50.0% vs 66.7% (gate item not passed). Extraction golden tests:
10/10 pass.

Changed test questions (classification per PLAN.md section 10):
- Q095 temporal 0/3 -> 3/3 and Q066 multi_fact 0/3 -> 2/3: fixed by the fact index.
- Q090 knowledge_update 1/3 -> 0/3, reasoning error with evidence in context. Both the
  Meditab fact (three-year bond) and the JHEX fact (no bond) are in context with the same
  slot `employment_bond`, but nothing marks which is current. That is Phase 5 supersession
  (`is_latest`, SUPERSEDES, `[latest]` / `[superseded]` labels).
- Q079 abstention 3/3 -> 1/3, fabrication. The clean fact "JHEX package is 3.4 LPA" leads the
  answerer to state it as the current salary instead of flagging it as unconfirmed.

Fixes made during Phase 4 were found on dev or in unit/golden tests, never on test:
extract_v2 (considering != hypothetical), transient API retries, fact time-range rendering.

## 2026-10-25: disclosure, corpus term in extract_v2

`extract_v2` (used for P4-dev and P4) illustrated its "considering is not hypothetical" rule
with "The user is considering RunPod", a phrase from the eval corpus. The rule is generic,
but the example mirrors one evaluated fact (RunPod -> Vast.ai), so P4 may be slightly
optimistic on that item. `extract_v3` replaces the example with a neutral one, and later
prompts were scanned for corpus terms.
## 2026-10-25 04:26 IST: P5-dev: entities + supersession + S5/S7 (extract_v3, answer_v4) (L1, medium tier, dev split, 1 run)

- git: `no-commit`; write path `turns_facts_v1`; run file `20261025-042644_L1_medium_dev.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v4`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=40; corpus 678 turns, 179 facts (179 mentions; by attribution {'user': 173, 'assistant': 6}; hypothetical 0; superseded 4)

| Metric | Value |
| --- | --- |
| Questions | 65 |
| Answer accuracy | 100.0% |
| Evidence recall (L0) | 96.7% |
| Fabrications on abstention items (per run) | 0.0 of 5 |
| Facts in context per query (mean) | 40.7 |
| Memory tokens per query (mean / p95) | 3628 / 5244 |
| Retrieval latency ms (p50 / p95) | 906 / 2502 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 65) |
| Rerank latency ms (p95) | 1779 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| inference | 100.0% | 0.0% |
| knowledge_update | 100.0% | 100.0% |
| multi_fact | 100.0% | 100.0% |
| preference_application | 100.0% | 100.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

## 2026-10-25 04:53 IST: P5: Phase 5 gate, entities + supersession + S5/S7 (extract_v3, answer_v4) (L2, large tier, test split, 3 runs)

- git: `no-commit`; write path `turns_facts_v1`; run file `20261025-045323_L2_large_test.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v4`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=40; corpus 1474 turns, 337 facts (337 mentions; by attribution {'user': 316, 'assistant': 21}; hypothetical 0; superseded 5)

| Metric | Value |
| --- | --- |
| Questions | 43 |
| Answer accuracy | 96.9% |
| Evidence recall (L0) | 95.0% |
| Fabrications on abstention items (per run) | 0.0 of 3 |
| Facts in context per query (mean) | 40.5 |
| Memory tokens per query (mean / p95) | 4336 / 5719 |
| Retrieval latency ms (p50 / p95) | 1563 / 3530 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 129) |
| Rerank latency ms (p95) | 2936 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 50.0% |
| knowledge_update | 100.0% | 100.0% |
| multi_fact | 90.9% | 100.0% |
| preference_application | 100.0% | 50.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 97.9% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 1.3

## 2026-10-25: Phase 5 gate verdict: PASSED

P5 (scored entity resolution + tiebreak, ADD-only supersession with [latest]/[superseded]
labels, S5/S7, extract_v3, answer_v4) on Large/test (3 runs): 96.9% overall (P4 93.0%,
P3 91.5%, B0 95.3%). conflict 100% and knowledge_update 100% (gate items), abstention 100%,
0 fabrications. Dev (Medium): 100%. Golden extraction tests: 11/11.

Changed test questions vs P4:
- Q090 knowledge_update 0/3 -> 3/3. "employment_bond: Meditab three-year bond" is now
  [superseded] by "JHEX has no bond" [latest]; answers give the latest value with its date.
- Q079 abstention 1/3 -> 3/3. answer_v4 presents [latest] salary as "most recently reported,
  may have changed".
- Q066 multi_fact 2/3 -> 3/3.
- Q042 single_fact 3/3 -> 2/3, a false supersession (open issue). The Optum offer's stipend
  and the later Meditab internship's stipend share the slot `internship_stipend` and subject
  `user`, so the Optum figure was marked superseded, and one answer said the Optum stipend was
  "no longer confirmed". Same-slot facts about different opportunities (an offer vs the job
  taken) are not one attribute changing value. To revisit (scoping slots by organization or
  status, and/or Phase 8 consolidation).
- Q078 multi_fact 0/3 in every phase: the answer omits the "supporting skills" half.

Notes: one extraction batch on Large hit an output-parsing error and succeeded on the schema
retry. Retrieval latency in this run (p50 1563 ms, p95 3530 ms) is measured with 6 questions
in flight on one GPU; single-user latency is still to be measured (Phase 8 target: quick mode
p95 < 600 ms).
## 2026-10-25 06:40 IST: SYNTH-500: P5 system on 500 synthetic conversations, 200 questions (L1, synth500 tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_v1`; run file `20261025-064012_L1_synth500_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v4`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=40; corpus 2454 turns, 1212 facts (1245 mentions; by attribution {'user': 875, 'assistant': 334, 'third_party': 3}; hypothetical 4; superseded 106)

| Metric | Value |
| --- | --- |
| Questions | 200 |
| Answer accuracy | 86.5% |
| Evidence recall (L0) | 95.0% |
| Fabrications on abstention items (per run) | 4.0 of 19 |
| Facts in context per query (mean) | 41.1 |
| Memory tokens per query (mean / p95) | 4252 / 4924 |
| Retrieval latency ms (p50 / p95) | 850 / 3572 |
| Planner latency ms (p50 / p95) | 0.1 / 1999.6 (llm 18, rules 182) |
| Rerank latency ms (p95) | 1544 |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 78.9% | n/a |
| assistant_recall | 94.1% | 100.0% |
| conflict | 66.7% | 100.0% |
| inference | 80.0% | 100.0% |
| knowledge_update | 85.7% | 100.0% |
| multi_fact | 93.9% | 97.0% |
| preference_application | 54.5% | 36.4% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 74.2% | 96.8% |

Failures per run: extraction_miss 3.0, fabrication 4.0, reasoning_error 16.0, retrieval_miss 4.0

## 2026-10-25: SYNTH-500 analysis (new synthetic benchmark, fresh Neo4j)

Dataset: evals/synth (persona "Mari" Quintanilla-Reyes, Nov 2024 to Oct 2026). 500 conversations
(350 persona, 150 distractor), 2,454 turns, 772 ledger facts, 200 Sonnet-verified questions.
Ingested through the full P5 write path into a wiped Neo4j: 1,212 facts, 106 superseded.
Result: 86.5% (173/200), evidence recall 95.0%, 4 fabrications on 19 abstention items. The
200 questions were answered once (1 run).

Failure classification (27):
- preference_application 5/11 wrong (recall 36%): the formatting preference ("concise bullets,
  recommendation first") is never retrieved for generic tasks that don't mention it. Retrieval
  can't surface it; an always-on profile can. That is the strongest case for the Phase 6
  ObservationLog.
- conflict 3/9 wrong: two different stated values share a slot, so supersession labels the
  older one [superseded] and the answerer reports only the latest. Supersession hides real
  contradictions; needs a "conflicting" state when values differ without a stated change.
- temporal 8/31 wrong. At least 2 are dataset noise: the ground-truth event date (Apr 12,
  Apr 26) was never stated in the conversation, which was told a day later. The rest are
  date-reasoning slips (sign-up month vs race month, hedging between two days).
- abstention 4/19 judged fabrication: all four decline the asked item but add extra details,
  some contradicting the expected answer (e.g. a half-marathon PB that later improved).
  Partly judge strictness and stale expected answers.
- knowledge_update 3/21, multi_fact 2/33, assistant_recall 1/17, inference 1/5: partial or
  stale answers.

Dataset quality issues found (generator v1): Haiku renders occasionally changed relations
("my sister Rosa" / "my mom's neighbor Rosa" while the ledger says mother). The render check
verifies names and numbers but not relations. Some temporal ground truths use event dates not
stated in the text. Both make the 86.5% a slight underestimate.
## 2026-10-25 08:33 IST: SYNTH-500 v2: P6 (profile + conflicts, extract_v4, answer_v5) (L1, synth500 tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-083314_L1_synth500_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v5`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=40; corpus 2452 turns, 1009 facts (1010 mentions; by attribution {'user': 724, 'assistant': 284, 'third_party': 1}; hypothetical 8; superseded 94)

| Metric | Value |
| --- | --- |
| Questions | 200 |
| Answer accuracy | 91.5% |
| Evidence recall (L0) | 96.7% |
| Fabrications on abstention items (per run) | 3.0 of 20 |
| Facts in context per query (mean) | 41.6 |
| Memory tokens per query (mean / p95) | 4390 / 5129 |
| Retrieval latency ms (p50 / p95) | 924 / 3537 |
| Planner latency ms (p50 / p95) | 0.1 / 2129.1 (llm 19, rules 181) |
| Rerank latency ms (p95) | 1690 |
| Profile tokens (mean) / prompt-cache hit rate | 2192 / 99.0% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 85.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| conflict | 60.0% | 100.0% |
| knowledge_update | 95.2% | 95.2% |
| multi_fact | 91.2% | 100.0% |
| preference_application | 90.9% | 54.5% |
| sensitive_history | 80.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 88.2% | 100.0% |

Failures per run: fabrication 3.0, reasoning_error 14.0

## 2026-10-25 08:37 IST: SYNTH-500 v2: P6 ablation, no profile (L1, synth500 tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-083705_L1_synth500_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v5`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, no profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=40; corpus 2452 turns, 1009 facts (1010 mentions; by attribution {'user': 724, 'assistant': 284, 'third_party': 1}; hypothetical 8; superseded 94)

| Metric | Value |
| --- | --- |
| Questions | 200 |
| Answer accuracy | 89.5% |
| Evidence recall (L0) | 96.7% |
| Fabrications on abstention items (per run) | 3.0 of 20 |
| Facts in context per query (mean) | 41.6 |
| Memory tokens per query (mean / p95) | 4391 / 5129 |
| Retrieval latency ms (p50 / p95) | 1015 / 3258 |
| Planner latency ms (p50 / p95) | 0.1 / 2004.9 (llm 19, rules 181) |
| Rerank latency ms (p95) | 1808 |
| Profile tokens (mean) / prompt-cache hit rate | 0 / 99.0% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 85.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| conflict | 70.0% | 100.0% |
| knowledge_update | 85.7% | 95.2% |
| multi_fact | 91.2% | 100.0% |
| preference_application | 63.6% | 54.5% |
| sensitive_history | 80.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 88.2% | 100.0% |

Failures per run: extraction_miss 2.0, fabrication 3.0, reasoning_error 15.0, retrieval_miss 1.0

## 2026-10-25: SYNTH-500 v2 analysis, Phase 6 (profile) vs ablation

Dataset: generator v2 (relation check, stated dates only). The question set was regenerated,
so v2 scores are not directly comparable with the v1 run above (86.5%). The profile on/off pair
below is the clean comparison: same ingestion, same questions, only the profile differs.

Run history and disclosure: the first P6 run was stopped at extraction 188/500 because the
Reflector never applied. Its safety check required every high-priority date to survive,
while reflect_v1 tells the model to merge bullets under the newest date, so every rewrite was
rejected and the log grew to 3,362 tokens. Fix: the check is now coverage of names and numbers
in high-priority bullets (`reflector_min_high_coverage` = 0.8; the dry run on the real log kept
84%). The run then resumed from the extraction watermarks. Facts from sessions 1-188 were
extracted while the log was unreflected; this changes only the profile, not facts. The user
cancelled the Teju dev and L2 runs, so the Phase 6 gate was judged on synthetic data only.

Ingestion: 1,009 facts (94 superseded), 3 reflections over the whole ingestion (final log
2,192 tokens), 48 entity summaries. One extraction call returned a degenerate output
(`source_turn_seqs` repeated about 2,000 zeros, `observations` missing); it was retried and
succeeded.

| Test type | n | P6 (profile) | No profile | Delta |
| --- | --- | --- | --- | --- |
| overall | 200 | 91.5% | 89.5% | +2.0 |
| preference_application | 11 | 90.9% | 63.6% | +27.3 |
| knowledge_update | 21 | 95.2% | 85.7% | +9.5 |
| multi_fact | 34 | 91.2% | 91.2% | 0 |
| temporal | 34 | 88.2% | 88.2% | 0 |
| conflict | 10 | 60.0% | 70.0% | -10.0 |
| abstention | 20 | 85.0% | 85.0% | 0 |
| sensitive_history | 5 | 80.0% | 80.0% | 0 |
| single_fact, assistant_recall | | 100% | 100% | 0 |

9 questions flipped to correct with the profile (S057, S078, S082, S104, S175, S178, S179,
S181, S192) and 5 flipped to wrong (S049, S053, S108, S132, S191). S108 is a profile side
effect: the profile says the user "prefers the assistant change the subject" on the breakup
with Theo (and holds the date itself), and the answerer refused the user's own direct date
question. The answer prompt should say that a sensitivity note shapes tone and unprompted
mentions, and never blocks a direct question from the user.

Remaining failures (17):
- conflict 4/10: the extractor labels a changed value as an update (supersession), not a
  contradiction, so the answer reports only the latest value (S127, S134, S135). S132 is
  partly dataset ambiguity (a second May half marathon, in 2026, exists).
- abstention 3/20 judged fabrication: S149 and S156 do abstain but add true details from
  memory that the expected answer lacks (judge strictness). S139 gives the assistant's own
  illustrative rates while flagging them as not a quote.
- temporal 4: S108 (above), an off-by-one date (S103, July 14 vs 15), hedging (S123), and S120, where the
  answer lists two real Thanksgivings at the same house (the expected answer knows only one).
- Others: S053 (the expected answer is stale; the project is finished), S069, S087 and S184
  (retrieval/reasoning misses), S049 and S191 (partial answers).

Phase 6 gate on synthetic data: PARTIAL.
- preference_application improves: PASS (+27.3 vs no profile; 54.5% on v1 with P5).
- multi_fact improves: NOT MET (91.2% with and without the profile).
- Prompt-cache hit rate > 70%: 99% of answer calls read the cache, but this is a single-turn
  eval (the instructions block is cached too, which is why the ablation also shows 99%), not
  the multi-turn simulation the plan asks for.
## 2026-10-25 09:56 IST: TEJU-ALL before: baseline for context reduction (extract_v5, answer_v6, old context sizes) (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-095608_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v6`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=6000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=40; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 96.3% |
| Evidence recall (L0) | 96.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 40.5 |
| Memory tokens per query (mean / p95) | 4329 / 5598 |
| Retrieval latency ms (p50 / p95) | 1007 / 2729 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 108) |
| Rerank latency ms (p95) | 2430 |
| Profile tokens (mean) / prompt-cache hit rate | 1287 / 99.1% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 83.3% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 25.0% |
| knowledge_update | 100.0% | 100.0% |
| multi_fact | 96.4% | 100.0% |
| preference_application | 80.0% | 80.0% |
| sensitive_history | 80.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 4.0

## 2026-10-25 09:58 IST: TEJU-ALL after: context reduction (10 facts + 5 partners, 10 excerpt turns, 200-char assistant cut, compact fact labels, profile 2000->1000, answer_v7) (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-095811_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 96.3% |
| Evidence recall (L0) | 92.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 10.4 |
| Memory tokens per query (mean / p95) | 939 / 1333 |
| Retrieval latency ms (p50 / p95) | 786 / 2169 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 108) |
| Rerank latency ms (p95) | 1804 |
| Profile tokens (mean) / prompt-cache hit rate | 1906 / 98.1% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 83.3% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 25.0% |
| knowledge_update | 100.0% | 83.3% |
| multi_fact | 92.9% | 92.9% |
| preference_application | 80.0% | 60.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 2.0, retrieval_miss 2.0


## 2026-10-25: context reduction, before vs after (Teju large tier, all 108 questions, 1 run each)

Requested by the user: ingest the whole corpus once, run all golden questions on the current
system, apply the approved context changes, and rerun on the same ingestion (no re-ingestion).
Both runs use extract_v5. Before = answer_v6 with the old context sizes; after = answer_v7 with
10 facts (+ up to 5 supersession/conflict partners), 10 excerpt turns, assistant excerpt lines
cut at 200 chars, compact ISO fact labels, memory budget 4000, chars_per_token 2.7, and
profile limits 2000 -> 1000. The profile (about 1,900 tokens) was already under the new
2,000 limit, so it was not condensed. The after run therefore tests the read-side changes.

| | Before | After |
| --- | --- | --- |
| Answer accuracy | 96.3% (104/108) | 96.3% (104/108) |
| Evidence recall (L0) | 96.0% | 92.0% |
| Real input tokens per answer call (mean) | 8,857 | 4,093 (-54%) |
| of which uncached (memory + question) | 5,918 | 1,058 (-82%) |
| of which cached (instructions + profile) | 2,940 | 3,035 |
| Facts in context (mean) | 40.5 | 10.4 |
| Retrieval latency p50 / p95 ms | 1007 / 2729 | 786 / 2169 |
| Answer latency p50 ms | 1688 | 1692 |

Per type, unchanged except multi_fact 96.4% -> 92.9% and sensitive_history 80% -> 100%.
- Lost: Q066 (multi_fact, retrieval_miss). Two required tools sit in a fact outside the top
  10 and in no excerpt within the 10-turn cap.
- Gained: Q073 (sensitive_history). The before answer omitted one symptom; the after answer
  listed them all.
- Still failing in both: Q078 (multi_fact, a second required point missing), Q098
  (assistant_recall, said no record of earlier advice), Q104 (preference_application,
  headline emphasis).

Evidence recall fell 4 points (knowledge_update 100 -> 83.3, multi_fact 100 -> 92.9,
preference 80 -> 60) while accuracy held. Possible follow-up: a larger fact cap only when the
planner flags a several-items question.
## 2026-10-25 10:27 IST: SYNTH-500 v2: context reduction (10 facts + 5 partners, 10 excerpt turns, profile condensed 3247->1984, answer_v7; facts from extract_v4 ingestion) (L1, synth500 tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-102748_L1_synth500_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 2452 turns, 1009 facts (1010 mentions; by attribution {'user': 724, 'assistant': 284, 'third_party': 1}; hypothetical 8; superseded 94)

| Metric | Value |
| --- | --- |
| Questions | 200 |
| Answer accuracy | 88.0% |
| Evidence recall (L0) | 87.2% |
| Fabrications on abstention items (per run) | 4.0 of 20 |
| Facts in context per query (mean) | 10.9 |
| Memory tokens per query (mean / p95) | 1506 / 1804 |
| Retrieval latency ms (p50 / p95) | 854 / 3818 |
| Planner latency ms (p50 / p95) | 0.1 / 2112.1 (llm 19, rules 181) |
| Rerank latency ms (p95) | 1339 |
| Profile tokens (mean) / prompt-cache hit rate | 1984 / 98.5% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 80.0% | n/a |
| assistant_recall | 82.4% | 100.0% |
| conflict | 50.0% | 100.0% |
| knowledge_update | 90.5% | 52.4% |
| multi_fact | 91.2% | 94.1% |
| preference_application | 72.7% | 18.2% |
| sensitive_history | 80.0% | 100.0% |
| single_fact | 97.9% | 95.8% |
| temporal | 94.1% | 100.0% |

Failures per run: extraction_miss 1.0, fabrication 4.0, reasoning_error 16.0, retrieval_miss 3.0


## 2026-10-25: context reduction on SYNTH-500 v2 (same ingestion as the P6 run)

Reused the extract_v4 ingestion: run with EXTRACTOR_PROMPT_VERSION=v4 so the eval user id
matches; nothing was re-extracted. The profile was condensed to the new limits first, in two
Reflector passes (3,247 -> 2,146 -> 1,984 tokens; versions 169 and 170 kept for rollback). It
ends under the 2,000 limit but about double the 1,000 target, because reflect_v1 keeps every
high-priority bullet. Compared with the P6 run (old context sizes, answer_v5, 2,192 old-counter
profile tokens):

| | P6 (before) | Context reduction (after) |
| --- | --- | --- |
| Answer accuracy | 91.5% | 88.0% (-7 questions: 12 lost, 5 gained) |
| Evidence recall | 96.7% | 87.2% |
| Real input tokens per answer call | 10,924 | 5,025 (-54%) |
| of which uncached | 6,470 | 1,665 (-74%) |
| Fabrications on abstention items | 3 of 20 | 4 of 20 |

| Test type | Before | After |
| --- | --- | --- |
| assistant_recall | 100.0% | 82.4% |
| preference_application | 90.9% | 72.7% |
| conflict | 60.0% | 50.0% |
| abstention | 85.0% | 80.0% |
| knowledge_update | 95.2% | 90.5% |
| single_fact | 100.0% | 97.9% |
| temporal | 88.2% | 94.1% |
| multi_fact | 91.2% | 91.2% |
| sensitive_history | 80.0% | 80.0% |

Lost questions by cause:
- assistant_recall 3 (S165, S170, S171): the 200-character cut of assistant excerpt lines
  removed the advice itself. S170's answer says the advice "got cut off". This is the clearest
  regression, and it did not show on the Teju set, which has few assistant_recall items.
- preference_application 2 (S178, S181): recommendation-first is kept, but the judge wanted
  cleaner bullets. Evidence recall for the type fell to 18%: the preference now reaches the
  answer only through the profile, not through excerpts.
- Fewer facts (S091 knowledge_update, retrieval_miss on the old "before" value; S070 and S061
  multi_fact missing details).
- Judge strictness or noise (S025, S137, S144 add true or general details; S130 picks one
  conflicting value).
Gained: S108 (answer_v6/v7 rule, sensitive topics no longer refused), S103, S049, S053, S149.

Confounded run: profile condensing, context sizes, and answer_v5 -> v7 changed together.
## 2026-10-25 11:00 IST: SYNTH-500 v2: reduced context + fix A (matched assistant turns uncut) (L1, synth500 tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-110038_L1_synth500_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 2452 turns, 1009 facts (1010 mentions; by attribution {'user': 724, 'assistant': 284, 'third_party': 1}; hypothetical 8; superseded 94)

| Metric | Value |
| --- | --- |
| Questions | 200 |
| Answer accuracy | 90.5% |
| Evidence recall (L0) | 87.2% |
| Fabrications on abstention items (per run) | 3.0 of 20 |
| Facts in context per query (mean) | 10.9 |
| Memory tokens per query (mean / p95) | 1561 / 1884 |
| Retrieval latency ms (p50 / p95) | 1006 / 3184 |
| Planner latency ms (p50 / p95) | 0.1 / 2101.3 (llm 19, rules 181) |
| Rerank latency ms (p95) | 1436 |
| Profile tokens (mean) / prompt-cache hit rate | 1984 / 100.0% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 85.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| conflict | 50.0% | 100.0% |
| knowledge_update | 90.5% | 52.4% |
| multi_fact | 94.1% | 94.1% |
| preference_application | 54.5% | 18.2% |
| sensitive_history | 80.0% | 100.0% |
| single_fact | 100.0% | 95.8% |
| temporal | 97.1% | 100.0% |

Failures per run: extraction_miss 2.0, fabrication 3.0, reasoning_error 11.0, retrieval_miss 3.0

## 2026-10-25 11:02 IST: TEJU-ALL: reduced context + fix A (matched assistant turns uncut) (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-110245_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 97.2% |
| Evidence recall (L0) | 92.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 10.4 |
| Memory tokens per query (mean / p95) | 1297 / 2205 |
| Retrieval latency ms (p50 / p95) | 904 / 2479 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 108) |
| Rerank latency ms (p95) | 2205 |
| Profile tokens (mean) / prompt-cache hit rate | 1906 / 98.1% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 83.3% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 25.0% |
| knowledge_update | 100.0% | 83.3% |
| multi_fact | 92.9% | 92.9% |
| preference_application | 100.0% | 60.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 1.0, retrieval_miss 2.0


## 2026-10-25: fix A (matched assistant turns kept whole) and the multi-turn cache simulation

Fix A: assistant turns that search matched are no longer cut; only neighbor turns added for
context are cut at 200 characters. Both datasets were re-answered on their existing ingestion.

| Dataset | Reduced context | + fix A | Input tokens per call (uncached) |
| --- | --- | --- | --- |
| Teju large, all 108 | 104 (96.3%) | 105 (97.2%), best so far | 4,093 (1,058) -> 4,390 (1,356) |
| SYNTH-500 v2, 200 | 176 (88.0%) | 181 (90.5%) | 5,025 (1,665) -> 5,071 (1,659) |

Synthetic per type (reduced -> fix A): assistant_recall 14 -> 17/17 (recovered), multi_fact
31 -> 32, temporal 32 -> 33, abstention 16 -> 17, single_fact 47 -> 48,
preference_application 8 -> 6, others unchanged. 19 questions flipped between two runs that
differ only by fix A, so single-run differences of about 3 questions are within run-to-run
noise (answerer and judge variance).

Multi-turn cache simulation (`evals/custom/cache_sim.py`): a throwaway user with the synthetic
profile (1,984 tokens), 3 conversations of 7 messages with stored history, and one profile
update between conversations 2 and 3. 19 of 21 calls (90%) read the cached prefix. The misses
were the first call and the first call after the profile change. 73% of all input tokens were
served from cache. The cache_write count from langchain usage_metadata reads 0 even on misses
(reporting gap).

Phase 6 gate (synthetic + Teju): preference_application improves (PASS: no profile 7/11,
profile 10/11); prompt-cache hit rate > 70% in a multi-turn simulation (PASS: 90%);
multi_fact improves (NOT MET: 31-32/34 throughout). Deep mode (Phase 7) targets multi_fact.
## 2026-10-25 11:40 IST: P7 Teju: deep mode (Haiku 5.5 medium, uncapped, auto routing) (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-114054_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`, mode auto
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 97.2% |
| Evidence recall (L0) | 93.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 22.9 |
| Memory tokens per query (mean / p95) | 4367 / 14635 |
| Retrieval latency ms (p50 / p95) | 920 / 2518 |
| Planner latency ms (p50 / p95) | 0.1 / 0.3 (rules 108) |
| Rerank latency ms (p95) | 2077 |
| Profile tokens (mean) / prompt-cache hit rate | 1906 / 97.2% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 83.3% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 25.0% |
| knowledge_update | 100.0% | 83.3% |
| multi_fact | 96.4% | 96.4% |
| preference_application | 80.0% | 60.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 1.0, retrieval_miss 2.0

| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |
| --- | --- | --- | --- | --- |
| deep | 34 | 100.0% | 3783 / 6107 | 13962 |
| none | 6 | 100.0% | 2486 / 6375 | 4809 |
| quick | 68 | 95.6% | 3133 / 6145 | 4383 |

Deep mode (`claude-haiku-5-5`, effort medium, `deep_v1`): 1.0 steps and 0.0 searches per question; stopped by: answered 34

## 2026-10-25 11:45 IST: P7 synth: deep mode (Haiku 5.5 medium, uncapped, auto routing) (L1, synth500 tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-114556_L1_synth500_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`, mode auto
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 2452 turns, 1009 facts (1010 mentions; by attribution {'user': 724, 'assistant': 284, 'third_party': 1}; hypothetical 8; superseded 94)

| Metric | Value |
| --- | --- |
| Questions | 200 |
| Answer accuracy | 91.0% |
| Evidence recall (L0) | 91.7% |
| Fabrications on abstention items (per run) | 4.0 of 20 |
| Facts in context per query (mean) | 37.5 |
| Memory tokens per query (mean / p95) | 5228 / 10864 |
| Retrieval latency ms (p50 / p95) | 993 / 3266 |
| Planner latency ms (p50 / p95) | 0.1 / 2029.2 (llm 18, rules 181, rules_fallback 1) |
| Rerank latency ms (p95) | 1769 |
| Profile tokens (mean) / prompt-cache hit rate | 1984 / 100.0% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 80.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| conflict | 70.0% | 100.0% |
| knowledge_update | 95.2% | 66.7% |
| multi_fact | 97.1% | 100.0% |
| preference_application | 54.5% | 36.4% |
| sensitive_history | 80.0% | 100.0% |
| single_fact | 97.9% | 97.9% |
| temporal | 94.1% | 100.0% |

Failures per run: extraction_miss 2.0, fabrication 4.0, reasoning_error 11.0, retrieval_miss 1.0

| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |
| --- | --- | --- | --- | --- |
| deep | 102 | 94.1% | 4423 / 9380 | 14496 |
| none | 12 | 75.0% | 2980 / 5365 | 4676 |
| quick | 86 | 89.5% | 3695 / 7689 | 5075 |

Deep mode (`claude-haiku-5-5`, effort medium, `deep_v1`): 1.0 steps and 0.0 searches per question; stopped by: answered 102


## 2026-10-25: Phase 7 (deep mode) results and gate

Deep mode: claude-haiku-5-5 at medium effort (user decision), prompt deep_v1. No caps on context
or searches; it stops when the model answers, or after DEEP_PATIENCE=2 steps in a row with no
new item scoring >= DEEP_MIN_RELEVANCE (0.01). Backstops: a 600 s outage timeout and the
model's context window. Routing uses rules plus plan_v2. Both datasets were evaluated on their
existing ingestion (verified: no turns or sessions pending; the logs show reuse).
The synthetic set has a fixed dev/test split (59/141); the routing rules were written from dev
questions only.

The first synthetic attempt crashed on an Anthropic 500 error after the SDK's 2 retries.
`llm_max_retries` (4) now applies to every LLM call, and the run was repeated. One planner
LLM fallback timed out and fell back to the rules plan (by design).

| | Quick only (fix A) | Phase 7 (auto routing) |
| --- | --- | --- |
| Teju, all 108 | 105 | 105 |
| Teju, held-out test 43 | 40 | 40 |
| Teju multi_fact / temporal | 26/28, 6/6 | 27/28, 6/6 |
| Synthetic, all 200 | 181 | 182 |
| Synthetic, held-out test 141 | 126 | 127 |
| Synthetic multi_fact / temporal | 32/34, 33/34 | 33/34, 32/34 |

Routing and cost:
- Teju: 34 deep (all correct), 68 quick, 6 none. 1 of 74 non-deep answers changed (noise).
  Q066, lost when the context was reduced, is recovered in deep mode.
- Synthetic: 102 deep, 86 quick, 12 none. Deep-routed accuracy 96 -> 96 of 102.
  Non-deep 85 -> 86 of 98, with 7 flips (run-to-run noise is about 3-5 questions).
- Deep mode almost never searched: 3 of 102 synthetic and 0 of 34 Teju deep questions ran a
  tool. The uncapped first context (about 14,000-14,500 input tokens, vs about 5,000 in quick
  mode) usually holds the answer, so deep mode currently acts as "more context" more than as
  "more searching".
- End-to-end latency, deep (p50 / p95): Teju 3.8 / 6.1 s, synthetic 4.4 / 9.4 s.
  Quick: 3.1 / 6.1 s and 3.7 / 7.7 s.

Phase 7 gate: multi_fact improves (PASS on both, +1 each; within noise); temporal improves
(NOT MET: Teju unchanged at 6/6, synthetic -1, S120 routed quick); deep-mode p95 latency
recorded (PASS); quick mode unchanged (PASS, within noise).

## 2026-10-25: Phase 8 load test (read path, 20 concurrent users, no LLM cost)

`uv run python -m evals.load_test --user-id eval-synth500-a7b461696138 --users 20 --per-user 10`
uses real Neo4j, embeddings, and the GPU reranker, a fake answerer, and the rules-only planner.
It writes nothing and only exercises quick mode. Hardware: RTX 3050 Laptop GPU (4 GB).

| | 1 user (20 requests) | 20 concurrent users (200 requests) |
| --- | --- | --- |
| Errors | 0 | 0 |
| Throughput | 1.0 req/s | 1.0 req/s |
| Read path p50 / p95 | 898 / 1,569 ms | 9,361 / 16,622 ms |
| Rerank mean / p95 | 808 / 1,103 ms | 12,801 / 14,863 ms |
| Everything but rerank (mean) | about 170 ms | about 700 ms |

The cross-encoder (bge-reranker-v2-m3, 100 candidates per query, one GPU, serialized) caps
throughput at about 1 query per second. Under load, queueing for it is the whole latency.
Without it the read path is well under the 600 ms target. Options (not applied; for the
user's decision): fewer rerank candidates, a shorter reranker max_length, cross-request
batching, or a bigger GPU.
## 2026-10-25 20:07 IST: P8 ablation: no fact index (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-200705_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`, mode auto, no fact index
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 98.1% |
| Evidence recall (L0) | 89.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 0.0 |
| Memory tokens per query (mean / p95) | 5743 / 22619 |
| Retrieval latency ms (p50 / p95) | 1047 / 2433 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 108) |
| Rerank latency ms (p95) | 2228 |
| Profile tokens (mean) / prompt-cache hit rate | 1906 / 97.2% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 75.0% | 25.0% |
| knowledge_update | 100.0% | 33.3% |
| multi_fact | 96.4% | 96.4% |
| preference_application | 100.0% | 40.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: retrieval_miss 2.0

| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |
| --- | --- | --- | --- | --- |
| deep | 34 | 100.0% | 3692 / 6161 | 19482 |
| none | 6 | 100.0% | 2752 / 3180 | 3848 |
| quick | 68 | 97.1% | 3458 / 6136 | 3571 |

Deep mode (`claude-haiku-5-5`, effort medium, `deep_v1`): 1.0 steps and 0.0 searches per question; stopped by: answered 34

## 2026-10-25 20:09 IST: P8 ablation: no BM25 (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-200929_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`, mode auto, no BM25
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 97.2% |
| Evidence recall (L0) | 94.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 23.7 |
| Memory tokens per query (mean / p95) | 3016 / 9156 |
| Retrieval latency ms (p50 / p95) | 780 / 2082 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 108) |
| Rerank latency ms (p95) | 1864 |
| Profile tokens (mean) / prompt-cache hit rate | 1906 / 100.0% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 83.3% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 25.0% |
| knowledge_update | 100.0% | 83.3% |
| multi_fact | 92.9% | 96.4% |
| preference_application | 100.0% | 80.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 2.0, retrieval_miss 1.0

| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |
| --- | --- | --- | --- | --- |
| deep | 34 | 97.1% | 3832 / 5586 | 10495 |
| none | 6 | 100.0% | 3039 / 5889 | 4814 |
| quick | 68 | 97.1% | 3044 / 5443 | 4340 |

Deep mode (`claude-haiku-5-5`, effort medium, `deep_v1`): 1.0 steps and 0.0 searches per question; stopped by: answered 34

## 2026-10-25 20:11 IST: P8 ablation: no reranker (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-201106_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), no rerank, mode auto
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 97.2% |
| Evidence recall (L0) | 94.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 30.5 |
| Memory tokens per query (mean / p95) | 6683 / 24264 |
| Retrieval latency ms (p50 / p95) | 135 / 523 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 108) |
| Profile tokens (mean) / prompt-cache hit rate | 1906 / 100.0% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 83.3% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 25.0% |
| knowledge_update | 100.0% | 100.0% |
| multi_fact | 96.4% | 96.4% |
| preference_application | 80.0% | 60.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 2.0, retrieval_miss 1.0

| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |
| --- | --- | --- | --- | --- |
| deep | 34 | 100.0% | 2704 / 4630 | 20801 |
| none | 6 | 100.0% | 1500 / 2519 | 4704 |
| quick | 68 | 95.6% | 2076 / 5043 | 4237 |

Deep mode (`claude-haiku-5-5`, effort medium, `deep_v1`): 1.0 steps and 0.0 searches per question; stopped by: answered 34

## 2026-10-25 20:13 IST: P8 ablation: no planner (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-201312_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, no planner, rerank `BAAI/bge-reranker-v2-m3`, mode auto
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 95.4% |
| Evidence recall (L0) | 92.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 10.4 |
| Memory tokens per query (mean / p95) | 1296 / 2205 |
| Retrieval latency ms (p50 / p95) | 914 / 1875 |
| Rerank latency ms (p95) | 1688 |
| Profile tokens (mean) / prompt-cache hit rate | 1906 / 100.0% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 83.3% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 25.0% |
| knowledge_update | 100.0% | 83.3% |
| multi_fact | 89.3% | 92.9% |
| preference_application | 80.0% | 60.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 3.0, retrieval_miss 2.0

| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |
| --- | --- | --- | --- | --- |
| quick | 108 | 95.4% | 3045 / 5968 | 4384 |

## 2026-10-25 20:15 IST: P8 ablation: no ObservationLog profile (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-201522_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, no profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`, mode auto
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 96.3% |
| Evidence recall (L0) | 93.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 22.9 |
| Memory tokens per query (mean / p95) | 4367 / 14635 |
| Retrieval latency ms (p50 / p95) | 776 / 2138 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 108) |
| Rerank latency ms (p95) | 1869 |
| Profile tokens (mean) / prompt-cache hit rate | 0 / 97.2% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 83.3% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 25.0% |
| knowledge_update | 66.7% | 83.3% |
| multi_fact | 100.0% | 96.4% |
| preference_application | 80.0% | 60.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 2.0, retrieval_miss 2.0

| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |
| --- | --- | --- | --- | --- |
| deep | 34 | 100.0% | 4138 / 6140 | 12069 |
| none | 6 | 100.0% | 2341 / 3264 | 2916 |
| quick | 68 | 94.1% | 2874 / 5306 | 2490 |

Deep mode (`claude-haiku-5-5`, effort medium, `deep_v1`): 1.0 steps and 0.0 searches per question; stopped by: answered 34

## 2026-10-25 20:17 IST: P8 ablation: no graph expansion (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-201736_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`, mode auto, no graph expansion
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 99.1% |
| Evidence recall (L0) | 93.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 22.4 |
| Memory tokens per query (mean / p95) | 4447 / 15351 |
| Retrieval latency ms (p50 / p95) | 964 / 2675 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 108) |
| Rerank latency ms (p95) | 2288 |
| Profile tokens (mean) / prompt-cache hit rate | 1906 / 100.0% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 83.3% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 25.0% |
| knowledge_update | 100.0% | 83.3% |
| multi_fact | 100.0% | 96.4% |
| preference_application | 100.0% | 60.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 1.0

| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |
| --- | --- | --- | --- | --- |
| deep | 34 | 100.0% | 3452 / 5873 | 14161 |
| none | 6 | 100.0% | 2452 / 5191 | 4817 |
| quick | 68 | 98.5% | 2944 / 5646 | 4388 |

Deep mode (`claude-haiku-5-5`, effort medium, `deep_v1`): 1.0 steps and 0.0 searches per question; stopped by: answered 34

## 2026-10-25 20:19 IST: P8 ablation: no deep mode (quick only) (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-201942_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`, mode quick
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 98.1% |
| Evidence recall (L0) | 92.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 10.4 |
| Memory tokens per query (mean / p95) | 1297 / 2205 |
| Retrieval latency ms (p50 / p95) | 940 / 2366 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 108) |
| Rerank latency ms (p95) | 2198 |
| Profile tokens (mean) / prompt-cache hit rate | 1906 / 100.0% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 83.3% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 25.0% |
| knowledge_update | 100.0% | 83.3% |
| multi_fact | 96.4% | 92.9% |
| preference_application | 100.0% | 60.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 1.0, retrieval_miss 1.0

| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |
| --- | --- | --- | --- | --- |
| none | 6 | 100.0% | 3170 / 5590 | 4809 |
| quick | 102 | 98.0% | 2948 / 6058 | 4377 |

## 2026-10-25 20:39 IST: P8 ablation: facts embedded without dates (separate ingestion) (L1, large tier, all split, 1 run)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-203927_L1_large_all.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`, mode auto, facts embedded without dates
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 239 facts (239 mentions; by attribution {'user': 225, 'assistant': 14}; hypothetical 0; superseded 4)

| Metric | Value |
| --- | --- |
| Questions | 108 |
| Answer accuracy | 96.3% |
| Evidence recall (L0) | 92.0% |
| Fabrications on abstention items (per run) | 0.0 of 8 |
| Facts in context per query (mean) | 23.5 |
| Memory tokens per query (mean / p95) | 4255 / 14734 |
| Retrieval latency ms (p50 / p95) | 818 / 3447 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 108) |
| Rerank latency ms (p95) | 2554 |
| Profile tokens (mean) / prompt-cache hit rate | 1456 / 93.5% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 100.0% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 75.0% | 25.0% |
| knowledge_update | 83.3% | 66.7% |
| multi_fact | 96.4% | 96.4% |
| preference_application | 80.0% | 60.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 1.0, retrieval_miss 3.0

| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |
| --- | --- | --- | --- | --- |
| deep | 34 | 100.0% | 3481 / 5340 | 13461 |
| none | 6 | 100.0% | 2743 / 3836 | 4480 |
| quick | 68 | 94.1% | 3211 / 5840 | 4093 |

Deep mode (`claude-haiku-5-5`, effort medium, `deep_v1`): 1.0 steps and 0.0 searches per question; stopped by: answered 34

## 2026-10-25 20:42 IST: P8: final gate (L2, large tier, test split, 3 runs) (L2, large tier, test split, 3 runs)

- git: `no-commit`; write path `turns_facts_obs_v1`; run file `20261025-204213_L2_large_test.jsonl`
- models: answer `claude-haiku-5-5` (effort medium, `answer_v7`); judge `claude-sonnet-5-5` (effort medium, `judge_v1`); embeddings `embeddinggemma:300m` (85462619ee72)
- read path: turn signals S2+S4, ObservationLog profile, planner (rules + LLM fallback), rerank `BAAI/bge-reranker-v2-m3`, mode auto
- config: signal_top_k=50, rrf_k=60, max_turns=15, turn_neighbors=1, memory_token_budget=4000, rerank_candidates=100, rerank_min_score=0.0, none_depth_min_score=0.5, max_facts=10, max_fact_partners=5, max_excerpt_turns=10, excerpt_assistant_max_chars=200, chars_per_token=2.7, profile 2000->1000; corpus 1474 turns, 196 facts (196 mentions; by attribution {'assistant': 8, 'user': 188}; hypothetical 0; superseded 3)

| Metric | Value |
| --- | --- |
| Questions | 43 |
| Answer accuracy | 96.1% |
| Evidence recall (L0) | 90.0% |
| Fabrications on abstention items (per run) | 0.0 of 3 |
| Facts in context per query (mean) | 21.0 |
| Memory tokens per query (mean / p95) | 3560 / 13047 |
| Retrieval latency ms (p50 / p95) | 802 / 1750 |
| Planner latency ms (p50 / p95) | 0.1 / 0.2 (rules 129) |
| Rerank latency ms (p95) | 1594 |
| Profile tokens (mean) / prompt-cache hit rate | 1906 / 97.7% |

| Test type | Accuracy | Evidence recall |
| --- | --- | --- |
| abstention | 100.0% | n/a |
| assistant_recall | 50.0% | 100.0% |
| conflict | 100.0% | 100.0% |
| inference | 100.0% | 50.0% |
| knowledge_update | 100.0% | 100.0% |
| multi_fact | 93.9% | 90.9% |
| preference_application | 100.0% | 0.0% |
| sensitive_history | 100.0% | 100.0% |
| single_fact | 100.0% | 100.0% |
| temporal | 100.0% | 100.0% |

Failures per run: reasoning_error 1.0, retrieval_miss 0.7

| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |
| --- | --- | --- | --- | --- |
| deep | 33 | 100.0% | 3397 / 5631 | 13368 |
| none | 3 | 100.0% | 2768 / 4296 | 4513 |
| quick | 93 | 94.6% | 2998 / 6802 | 4459 |

Deep mode (`claude-haiku-5-5`, effort medium, `deep_v1`): 1.0 steps and 0.0 searches per question; stopped by: answered 33


## 2026-10-25: Phase 8 reranker throughput fix

Benchmark (20 queries x 100 real candidates, RTX 3050 Laptop): sorting pairs by length before
batching took 0.8 -> 2.0 queries/s with identical scores. Scoring requests that queue up
while the model is busy in one call took it to 2.3 queries/s (fp16 differences <= 0.0016).
A 256-token max_length gave nothing extra once sorted, so it was not applied (it would cut
long texts). Load test after the fix: single-user read path p95 1,569 -> 852 ms (rerank mean
808 -> 555 ms); 20 users 1.04 -> 1.65 req/s. At 20 back-to-back users, p95 is still
15.9 s. That load exceeds the laptop GPU's capacity, and with the reranker faster, embeddings
and Neo4j now queue too (baseline stage mean 0.1 -> 2.0 s under load).

## 2026-10-25: Phase 8 ablations (Teju large tier, all 108 questions, 1 run each)

Reference: the full system, P7 Teju run, 105/108 (97.2%). One component is removed at a time.
All runs except the date ablation reuse the stored ingestion (verified in logs).

| Removed | Correct (of 108) | Delta |
| --- | --- | --- |
| Fact index (S1/S3/S5/S6/S7) | 106 | +1 |
| BM25 (S3/S4) | 105 | 0 |
| Reranker | 105 | 0 |
| Planner | 103 | -2 |
| ObservationLog profile | 104 | -1 |
| Graph expansion (S7) | 107 | +2 |
| Deep mode (quick only) | 106 | +1 |
| Dates in fact embeddings (separate ingestion, extract_v5) | 104 | -1 |

Every delta is within run-to-run noise (about +-3 questions). On this corpus (170 facts, ~97%
baseline) no single component is decisive: retrieval over raw turns alone already finds
nearly all evidence. Component value shows on harder data: on SYNTH-500 the profile was worth
+27 points on preference_application (7/11 -> 10/11). The user chose Teju-only ablations to
limit cost; a synthetic ablation set would separate the components better.

## 2026-10-25: Phase 8 gate (L2, Teju large tier, test split, 3 runs)

Result: 96.1% (124/129 answers). Every test type is at 100% except assistant_recall
(Q098 wrong 3/3: "no record" of earlier advice) and multi_fact 93.9% (Q078 wrong 2/3,
retrieval miss). Deep-routed answers: 33/33 correct. Quick-routed: 88/93.

| Success criterion (PLAN.md section 1) | Target | Result | Verdict |
| --- | --- | --- | --- |
| Answer accuracy, test split, Large tier, 3 runs | >= 85% (stretch 90%) | 96.1% | MET (stretch met) |
| Evidence recall | >= 95% | 90.0% | GAP: the user cut context to 10 facts / 10 turns (tokens -54%, accuracy unchanged). Next fix: a bigger fact cap only for multi-item questions (proposed "fix B"), or more deep routing. |
| conflict and knowledge_update | 100% | 100% | MET |
| abstention | >= 90%, <= 1 fabrication | 100%, 0 | MET (3 items in the test split) |
| Judge agreement with hand labels | >= 90% | not measured | GAP: the user waived hand-grading. Next fix: grade 30 judgments by hand (evals/custom/handgrade.py exists). |
| Read-path added latency, quick mode | p95 < 600 ms | 852 ms single user (load test); 1,750 ms at eval concurrency | GAP: the reranker on a 4 GB laptop GPU. Without it, about 170 ms. Next fix: a bigger GPU, or rerank 50 candidates (tune on dev). |
| Memory context per query | 4,000-8,000 tokens | quick ~1,300-1,500, deep ~13,000 | DEVIATION by the user's decision (quick capped, deep uncapped) |
| Write-path lag | p95 < 30 s | not measured end to end | GAP: extraction runs after 10 minutes idle or 2,200 tokens by design (raw turns are searchable at once); job time not measured under load. Next fix: time queue -> commit in the worker with the new tracing. |

Phase 8 gate verdict: the criteria are met or documented with their gap and next fix, as the
plan requires. Additional Phase 8 deliverables: forget with preview and confirmation, export,
nightly consolidation, the re-embedding job, JSON tracing of every LLM call, the load test,
and ablation switches. 362 tests pass.
