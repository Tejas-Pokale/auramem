"""Custom eval runner (PLAN.md section 10).

    uv run python -m evals.custom.run --level L0 --tier medium --split dev
    uv run python -m evals.custom.run --level L1 --tier medium --split dev
    uv run python -m evals.custom.run --level L2 --label B0    # large tier, test split, 3 runs

L0: retrieval only - is every evidence turn in the assembled memory context? No LLM cost.
L1/L2: answer with the real read path, then grade with the judge.
Per-question records go to evals/custom/runs/ (gitignored: they contain personal data);
the aggregate is appended to evals/RESULTS.md.
"""

import argparse
import asyncio
import csv
import json
import logging
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any, Literal

import httpx

from evals.custom.build_sessions import TZ, build_tier
from evals.custom.ingest import WRITE_PATH_VERSION, ingest_tier
from evals.custom.judge import JUDGE_EFFORT, JUDGE_MODEL, JUDGE_VERSION, Judge
from evals.custom.report import write_markdown
from evals.synth.dataset import QUESTIONS as SYNTH_QUESTIONS
from evals.synth.dataset import TIER as SYNTH_TIER
from evals.synth.dataset import build_synth
from memory.client import MemoryClient, claude_deep_llms
from memory.config import Settings, get_settings
from memory.entities import ClaudeTiebreaker
from memory.extract import ExtractionDeps, Extractor, ExtractorLLM
from memory.graphs.read import ClaudeAnswerer
from memory.observe import claude_reflector, claude_summarizer
from memory.retrieve import Mode

log = logging.getLogger("evals.custom.run")

HERE = Path(__file__).parent
GOLDEN = HERE / "data" / "private" / "golden_v2.csv"
RUNS_DIR = HERE / "runs"
RESULTS = HERE.parent / "RESULTS.md"
QUESTION_TIME = dtime(12, 0)  # questions are asked at noon local time on asked_at
CONCURRENCY = 6

Level = Literal["L0", "L1", "L2"]
LEVEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "L0": {"tier": "medium", "split": "dev", "runs": 1},
    "L1": {"tier": "medium", "split": "dev", "runs": 1},
    "L2": {"tier": "large", "split": "test", "runs": 3},
}


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    expected_answer: str
    test_type: str
    evidence_fact_ids: list[str]
    asked_at: datetime
    must_abstain: bool
    split: str


@dataclass
class Record:
    question_id: str
    run: int
    test_type: str
    must_abstain: bool
    evidence_fact_ids: list[str]
    missing_evidence: list[str]
    memory_tokens: int
    retrieval_ms: float
    facts_in_context: int = 0
    memory_context: str = ""  # the exact memory block the answerer saw
    profile_tokens: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    context_fact_ids: list[str] = field(default_factory=list)
    context_turn_ids: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)
    evidence_without_facts: list[str] = field(default_factory=list)
    planner_ms: float | None = None
    rerank_ms: float | None = None
    plan_source: str | None = None
    plan_depth: str | None = None
    top_rerank_score: float | None = None
    answer: str | None = None
    verdict: str | None = None
    reason: str | None = None
    answer_ms: float | None = None
    failure: str | None = None
    mode: str | None = None  # quick | deep | none
    output_tokens: int = 0
    deep_steps: int = 0
    deep_tool_calls: list[str] = field(default_factory=list)
    deep_stop_reason: str | None = None

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + (self.answer_ms or 0.0)

    @property
    def evidence_ok(self) -> bool:
        return not self.missing_evidence

    @property
    def correct(self) -> bool:
        return self.verdict == "CORRECT"


@dataclass
class Summary:
    n_questions: int
    runs: int
    accuracy: float | None
    accuracy_by_type: dict[str, float]
    evidence_recall: float
    evidence_recall_by_type: dict[str, float]
    fabrications_per_run: float | None
    abstention_items: int
    memory_tokens_mean: float
    facts_per_query_mean: float
    memory_tokens_p95: float
    retrieval_ms_p50: float
    retrieval_ms_p95: float
    planner_ms_p50: float | None = None
    planner_ms_p95: float | None = None
    rerank_ms_p95: float | None = None
    plan_sources: dict[str, int] = field(default_factory=dict)
    cache_hit_rate: float | None = None
    profile_tokens_mean: float | None = None
    failures: dict[str, float] = field(default_factory=dict)
    modes: dict[str, int] = field(default_factory=dict)
    accuracy_by_mode: dict[str, float] = field(default_factory=dict)
    latency_by_mode: dict[str, tuple[float, float]] = field(default_factory=dict)  # p50, p95
    input_tokens_by_mode: dict[str, float] = field(default_factory=dict)
    deep_steps_mean: float | None = None
    deep_tool_calls_mean: float | None = None
    deep_stop_reasons: dict[str, int] = field(default_factory=dict)


def load_questions(split: str, path: Path = GOLDEN) -> list[Question]:
    out = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if split != "all" and row["split"] != split:
            continue
        out.append(
            Question(
                question_id=row["question_id"],
                question=row["question"],
                expected_answer=row["expected_answer"],
                test_type=row["test_type"],
                evidence_fact_ids=[f for f in row["evidence_fact_ids"].split(";") if f],
                asked_at=datetime.combine(date.fromisoformat(row["asked_at"]), QUESTION_TIME, TZ),
                must_abstain=row["must_abstain"] == "true",
                split=row["split"],
            )
        )
    return out


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def classify_failure(r: Record) -> str | None:
    """Section 10 failure classes.

    extraction_miss: evidence is missing from context and at least one missing evidence turn
    never produced a fact (raw turns are still searchable, so this is also a retrieval miss,
    but the fact index could not help). retrieval_miss: evidence missing although facts exist.
    """
    if r.verdict != "WRONG":
        return None
    if r.must_abstain:
        return "fabrication"
    if not r.evidence_ok:
        if set(r.missing_evidence) & set(r.evidence_without_facts):
            return "extraction_miss"
        return "retrieval_miss"
    return "reasoning_error"


def summarize(records: list[Record], runs: int) -> Summary:
    first_run = [r for r in records if r.run == 0]
    with_evidence = [r for r in first_run if r.evidence_fact_ids]
    by_type: dict[str, list[Record]] = defaultdict(list)
    for r in with_evidence:
        by_type[r.test_type].append(r)
    judged = [r for r in records if r.verdict is not None]

    accuracy = accuracy_by_type = fabrications = None
    if judged:
        per_run = [
            statistics.mean(r.correct for r in judged if r.run == run) for run in range(runs)
        ]
        accuracy = statistics.mean(per_run)
        types = sorted({r.test_type for r in judged})
        accuracy_by_type = {
            t: statistics.mean(r.correct for r in judged if r.test_type == t) for t in types
        }
        fabrications = sum(r.failure == "fabrication" for r in judged) / runs

    failures = Counter(r.failure for r in judged if r.failure)
    tokens = [float(r.memory_tokens) for r in first_run]
    latency = [r.retrieval_ms for r in records]
    planner = [r.planner_ms for r in records if r.planner_ms is not None]
    rerank = [r.rerank_ms for r in records if r.rerank_ms is not None]
    return Summary(
        n_questions=len(first_run),
        runs=runs,
        accuracy=accuracy,
        accuracy_by_type=accuracy_by_type or {},
        evidence_recall=statistics.mean(r.evidence_ok for r in with_evidence)
        if with_evidence
        else 1.0,
        evidence_recall_by_type={
            t: statistics.mean(r.evidence_ok for r in rs) for t, rs in sorted(by_type.items())
        },
        fabrications_per_run=fabrications,
        abstention_items=sum(r.must_abstain for r in first_run),
        memory_tokens_mean=statistics.mean(tokens) if tokens else 0.0,
        facts_per_query_mean=statistics.mean(r.facts_in_context for r in first_run)
        if first_run
        else 0.0,
        memory_tokens_p95=_p(tokens, 0.95),
        retrieval_ms_p50=_p(latency, 0.5),
        retrieval_ms_p95=_p(latency, 0.95),
        planner_ms_p50=_p(planner, 0.5) if planner else None,
        planner_ms_p95=_p(planner, 0.95) if planner else None,
        rerank_ms_p95=_p(rerank, 0.95) if rerank else None,
        plan_sources=dict(Counter(r.plan_source for r in records if r.plan_source)),
        cache_hit_rate=statistics.mean(r.cache_read_tokens > 0 for r in judged) if judged else None,
        profile_tokens_mean=statistics.mean(r.profile_tokens for r in judged) if judged else None,
        failures={k: v / runs for k, v in sorted(failures.items())},
        **_mode_stats(judged),
    )


def _mode_stats(judged: list[Record]) -> dict[str, Any]:
    """Quick vs deep: how many, accuracy, end-to-end latency, input tokens, deep-loop shape."""
    by_mode: dict[str, list[Record]] = defaultdict(list)
    for r in judged:
        by_mode[r.mode or "quick"].append(r)
    deep = by_mode.get("deep", [])
    return {
        "modes": {m: len(rs) for m, rs in sorted(by_mode.items())},
        "accuracy_by_mode": {
            m: statistics.mean(r.correct for r in rs) for m, rs in sorted(by_mode.items())
        },
        "latency_by_mode": {
            m: (_p([r.total_ms for r in rs], 0.5), _p([r.total_ms for r in rs], 0.95))
            for m, rs in sorted(by_mode.items())
        },
        "input_tokens_by_mode": {
            m: statistics.mean(r.input_tokens for r in rs) for m, rs in sorted(by_mode.items())
        },
        "deep_steps_mean": statistics.mean(r.deep_steps for r in deep) if deep else None,
        "deep_tool_calls_mean": statistics.mean(len(r.deep_tool_calls) for r in deep)
        if deep
        else None,
        "deep_stop_reasons": dict(Counter(r.deep_stop_reason or "" for r in deep)),
    }


async def evaluate(
    client: MemoryClient,
    judge: Judge | None,
    questions: list[Question],
    *,
    user_id: str,
    evidence_turn_ids: dict[str, str],
    turns_with_facts: set[str],
    runs: int,
    concurrency: int = CONCURRENCY,
    mode: Mode = "auto",
) -> list[Record]:
    sem = asyncio.Semaphore(concurrency)

    async def one(q: Question, run: int) -> Record:
        async with sem:
            session_id = f"{user_id}/question/{q.question_id}/run{run}"
            reply = None
            started = time.perf_counter()
            if judge is None:
                memory = await client.retriever.retrieve(
                    user_id, q.question, now=q.asked_at, current_session_id=session_id
                )
            else:
                # The reply carries the exact memory context the answerer saw.
                reply = await client.respond(
                    user_id, session_id, q.question, now=q.asked_at, mode=mode
                )
                memory = reply.memory
            elapsed = (time.perf_counter() - started) * 1000
            # Evidence is present if its source turn or a fact derived from it is in context.
            in_context = set(memory.turn_ids) | set(memory.covered_turn_ids)
            timings = memory.timings_ms
            record = Record(
                question_id=q.question_id,
                run=run,
                test_type=q.test_type,
                must_abstain=q.must_abstain,
                evidence_fact_ids=q.evidence_fact_ids,
                missing_evidence=[
                    f for f in q.evidence_fact_ids if evidence_turn_ids[f] not in in_context
                ],
                memory_tokens=memory.token_estimate,
                facts_in_context=len(memory.fact_ids),
                memory_context=memory.text,
                context_fact_ids=memory.fact_ids,
                context_turn_ids=memory.turn_ids,
                guidance=memory.guidance,
                evidence_without_facts=[
                    f for f in q.evidence_fact_ids if evidence_turn_ids[f] not in turns_with_facts
                ],
                retrieval_ms=timings.get("total", elapsed),
                planner_ms=timings.get("planner"),
                rerank_ms=timings.get("rerank"),
                plan_source=memory.plan_source,
                plan_depth=memory.plan.depth if memory.plan else None,
                top_rerank_score=memory.top_rerank_score,
            )
            if reply is None:
                return record
            record.answer_ms = elapsed - record.retrieval_ms
            record.answer = reply.text
            record.profile_tokens = reply.profile_tokens
            record.input_tokens = reply.usage.get("input_tokens", 0)
            record.output_tokens = reply.usage.get("output_tokens", 0)
            record.cache_read_tokens = reply.usage.get("cache_read_tokens", 0)
            record.mode = reply.mode
            if reply.deep is not None:
                record.deep_steps = reply.deep.steps
                record.deep_tool_calls = reply.deep.tool_calls
                record.deep_stop_reason = reply.deep.stop_reason
            verdict = await judge(  # type: ignore[misc]
                question=q.question,
                expected=q.expected_answer,
                test_type=q.test_type,
                must_abstain=q.must_abstain,
                answer=reply.text,
            )
            record.verdict, record.reason = verdict.verdict, verdict.reason
            record.failure = classify_failure(record)
            return record

    tasks = [one(q, run) for run in range(runs) for q in questions]
    return list(await asyncio.gather(*tasks))


def _git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except subprocess.CalledProcessError:
        return "no-commit"


async def _embedding_digest(settings: Settings) -> str:
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=5) as http:
            tags = (await http.get("/api/tags")).json()["models"]
        match = next(m for m in tags if m["name"] == settings.embedding_model)
        return str(match["digest"])[:12]
    except (httpx.HTTPError, StopIteration, KeyError):
        return "unknown"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_results(
    summary: Summary,
    *,
    label: str,
    level: str,
    tier: str,
    split: str,
    settings: Settings,
    digest: str,
    corpus_turns: int,
    fact_summary: dict[str, Any],
    run_file: str,
    components: str,
) -> str:
    s = settings
    types = sorted(set(summary.accuracy_by_type) | set(summary.evidence_recall_by_type))
    m = summary
    fabrications = (
        "n/a"
        if m.fabrications_per_run is None
        else f"{m.fabrications_per_run:.1f} of {m.abstention_items}"
    )
    answer = f"`{s.answer_model}` (effort {s.answer_effort}, `answer_{s.answer_prompt_version}`)"
    judge = f"`{JUDGE_MODEL}` (effort {JUDGE_EFFORT}, `{JUDGE_VERSION}`)"
    lines = [
        f"## {datetime.now(TZ):%Y-%m-%d %H:%M} IST: {label} ({level}, {tier} tier, {split} split"
        f", {summary.runs} run{'s' if summary.runs > 1 else ''})",
        "",
        f"- git: `{_git_sha()}`; write path `{WRITE_PATH_VERSION}`; run file `{run_file}`",
        f"- models: answer {answer}; judge {judge}; embeddings `{s.embedding_model}` ({digest})",
        f"- read path: {components}",
        f"- config: signal_top_k={s.signal_top_k}, rrf_k={s.rrf_k}, max_turns={s.max_turns}, "
        f"turn_neighbors={s.turn_neighbors}, memory_token_budget={s.memory_token_budget}, "
        f"rerank_candidates={s.rerank_candidates}, rerank_min_score={s.rerank_min_score}, "
        f"none_depth_min_score={s.none_depth_min_score}, max_facts={s.max_facts}, "
        f"max_fact_partners={s.max_fact_partners}, max_excerpt_turns={s.max_excerpt_turns}, "
        f"excerpt_assistant_max_chars={s.excerpt_assistant_max_chars}, "
        f"chars_per_token={s.chars_per_token}, profile {s.obslog_reflect_at}->{s.obslog_target}; "
        f"corpus {corpus_turns} turns, {fact_summary.get('facts', 0)} facts "
        f"({fact_summary.get('mentions', 0)} mentions; by attribution "
        f"{fact_summary.get('by_attribution', {})}; hypothetical "
        f"{fact_summary.get('hypothetical', 0)}; superseded {fact_summary.get('superseded', 0)})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Questions | {summary.n_questions} |",
        f"| Answer accuracy | {_pct(summary.accuracy)} |",
        f"| Evidence recall (L0) | {_pct(summary.evidence_recall)} |",
        f"| Fabrications on abstention items (per run) | {fabrications} |",
        f"| Facts in context per query (mean) | {m.facts_per_query_mean:.1f} |",
        f"| Memory tokens per query (mean / p95) | "
        f"{m.memory_tokens_mean:.0f} / {m.memory_tokens_p95:.0f} |",
        f"| Retrieval latency ms (p50 / p95) | "
        f"{m.retrieval_ms_p50:.0f} / {m.retrieval_ms_p95:.0f} |",
    ]
    if m.planner_ms_p95 is not None:
        sources = ", ".join(f"{k} {v}" for k, v in sorted(m.plan_sources.items()))
        lines.append(
            f"| Planner latency ms (p50 / p95) | {m.planner_ms_p50:.1f} / {m.planner_ms_p95:.1f}"
            f" ({sources}) |"
        )
    if m.rerank_ms_p95 is not None:
        lines.append(f"| Rerank latency ms (p95) | {m.rerank_ms_p95:.0f} |")
    if m.cache_hit_rate is not None:
        lines.append(
            f"| Profile tokens (mean) / prompt-cache hit rate | "
            f"{m.profile_tokens_mean:.0f} / {_pct(m.cache_hit_rate)} |"
        )
    lines += [
        "",
        "| Test type | Accuracy | Evidence recall |",
        "| --- | --- | --- |",
    ]
    for t in types:
        acc = summary.accuracy_by_type.get(t)
        rec = summary.evidence_recall_by_type.get(t)
        lines.append(f"| {t} | {_pct(acc)} | {'n/a' if rec is None else _pct(rec)} |")
    if summary.failures:
        failures = ", ".join(f"{k} {v:.1f}" for k, v in summary.failures.items())
        lines += ["", f"Failures per run: {failures}"]
    if m.modes:
        lines += [
            "",
            "| Mode | Questions | Accuracy | Latency ms end-to-end (p50 / p95) | Input tokens |",
            "| --- | --- | --- | --- | --- |",
        ]
        for mode, n in m.modes.items():
            p50, p95 = m.latency_by_mode[mode]
            lines.append(
                f"| {mode} | {n} | {_pct(m.accuracy_by_mode[mode])} | {p50:.0f} / {p95:.0f} | "
                f"{m.input_tokens_by_mode[mode]:.0f} |"
            )
    if m.deep_steps_mean is not None:
        stops = ", ".join(f"{k} {v}" for k, v in sorted(m.deep_stop_reasons.items()))
        lines += [
            "",
            f"Deep mode (`{s.deep_answer_model}`, effort {s.deep_answer_effort}, "
            f"`deep_{s.deep_prompt_version}`): {m.deep_steps_mean:.1f} steps and "
            f"{m.deep_tool_calls_mean:.1f} searches per question; stopped by: {stops}",
        ]
    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings().model_copy(
        update={
            "use_fact_index": not args.no_facts,
            "use_bm25": not args.no_bm25,
            "use_graph": not args.no_graph,
            "embed_fact_dates": not args.no_fact_dates,
        }
    )
    level: Level = args.level
    defaults = LEVEL_DEFAULTS[level]
    runs = args.runs or defaults["runs"]
    if args.dataset == "synth":
        # Fully synthetic benchmark (evals/synth): one tier, all questions.
        golden, tier, split = SYNTH_QUESTIONS, SYNTH_TIER, args.split or "all"
        data = build_synth()
    else:
        golden, tier, split = GOLDEN, args.tier or defaults["tier"], args.split or defaults["split"]
        data = build_tier(tier)

    questions = load_questions(split, golden)
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q.question_id in wanted]
    if args.limit:
        questions = questions[: args.limit]

    full = MemoryClient.from_settings(settings)
    store, embedder = full.store, full.embedder
    planner = None if args.no_planner else full.retriever.planner
    reranker = None if args.no_rerank else full.reranker
    components = ", ".join(
        [
            "turn signals S2+S4",
            "no profile" if args.no_profile else "ObservationLog profile",
            "planner (rules + LLM fallback)" if planner else "no planner",
            f"rerank `{settings.reranker_model}`" if reranker else "no rerank",
            f"mode {args.mode}",
            *(["no fact index"] if args.no_facts else []),
            *(["no BM25"] if args.no_bm25 else []),
            *(["no graph expansion"] if args.no_graph else []),
            *(["facts embedded without dates"] if args.no_fact_dates else []),
        ]
    )
    if reranker is not None:
        await asyncio.to_thread(reranker.warmup)  # keep model load out of latency numbers
    client = MemoryClient(
        store,
        embedder,
        ClaudeAnswerer(settings, model=settings.answer_model, effort=settings.answer_effort),
        settings,
        planner=planner,
        reranker=reranker,
        use_profile=not args.no_profile,
        deep_llms=None if args.mode == "quick" else claude_deep_llms(settings),
    )
    try:
        extraction = ExtractionDeps(
            store=store,
            embedder=embedder,
            extractor=Extractor(ExtractorLLM(settings), settings),
            settings=settings,
            tiebreaker=ClaudeTiebreaker(settings),
        )
        ingested = await ingest_tier(
            store,
            embedder,
            data,
            settings,
            extraction,
            reflector=claude_reflector(settings),
            summarizer=claude_summarizer(settings),
        )
        turns_with_facts = await store.turns_with_facts(ingested.user_id)
        judge = None if level == "L0" else Judge(settings)
        log.info(
            "evaluating %d questions x %d run(s) on %s", len(questions), runs, ingested.user_id
        )
        records = await evaluate(
            client,
            judge,
            questions,
            user_id=ingested.user_id,
            evidence_turn_ids=ingested.evidence_turn_ids,
            turns_with_facts=turns_with_facts,
            runs=runs,
            concurrency=args.concurrency,
            mode=args.mode,
        )
        digest = await _embedding_digest(settings)
    finally:
        await client.close()

    summary = summarize(records, runs)
    RUNS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    run_file = RUNS_DIR / f"{stamp}_{level}_{tier}_{split}.jsonl"
    with run_file.open("w", encoding="utf-8") as f:
        for r in sorted(records, key=lambda r: (r.question_id, r.run)):
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    report_file = write_markdown(run_file, golden)

    report = render_results(
        summary,
        label=args.label,
        level=level,
        tier=tier,
        split=split,
        settings=settings,
        digest=digest,
        corpus_turns=ingested.turns,
        fact_summary=ingested.fact_summary,
        run_file=run_file.name,
        components=components,
    )
    print(report)
    print(f"Per-question report with retrieved context: {report_file}")
    if args.log:
        if not RESULTS.exists():
            RESULTS.write_text(
                "# Eval results\n\nNewest last. Produced by `evals/custom/run.py`.\n\n",
                encoding="utf-8",
            )
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(report + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--level", choices=list(LEVEL_DEFAULTS), default="L1")
    parser.add_argument("--dataset", choices=["teju", "synth"], default="teju")
    parser.add_argument("--tier", choices=["small", "medium", "large"])
    parser.add_argument("--split", choices=["dev", "test", "all"])
    parser.add_argument("--runs", type=int)
    parser.add_argument("--limit", type=int, help="first N questions only (smoke runs)")
    parser.add_argument("--ids", help="comma-separated question ids only (smoke runs)")
    parser.add_argument("--label", default="run", help="name in RESULTS.md, e.g. B0")
    parser.add_argument("--no-log", dest="log", action="store_false", help="skip RESULTS.md")
    parser.add_argument("--no-planner", action="store_true", help="ablation: no query planner")
    parser.add_argument("--no-rerank", action="store_true", help="ablation: no cross-encoder")
    parser.add_argument("--no-facts", action="store_true", help="ablation: no fact index")
    parser.add_argument("--no-bm25", action="store_true", help="ablation: no keyword search")
    parser.add_argument("--no-graph", action="store_true", help="ablation: no graph expansion")
    parser.add_argument(
        "--no-fact-dates",
        action="store_true",
        help="ablation: facts embedded without their date (separate ingestion)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "quick", "deep"],
        default="auto",
        help="auto: planner routes quick/deep; quick: never deep (Phase 6 behavior); deep: always",
    )
    parser.add_argument(
        "--no-profile", action="store_true", help="ablation: no ObservationLog in the prompt"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help="questions in flight; use 1 to measure single-user latency (one GPU is shared)",
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "neo4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
