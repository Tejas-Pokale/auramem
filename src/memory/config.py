"""All settings and thresholds for the memory layer (PLAN.md section 8).

Every tunable number used anywhere in the codebase lives here. Each field can be
overridden by an environment variable of the same name in upper case
(for example ``RERANK_MIN_SCORE=0.2``) or by a ``.env`` file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Connections -------------------------------------------------------------
    anthropic_api_key: SecretStr | None = None
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("memory-dev-password")
    neo4j_database: str = "neo4j"
    redis_url: str = "redis://localhost:6379/0"
    ollama_url: str = "http://localhost:11434"

    # --- Models (CLAUDE.md "Models") ---------------------------------------------
    planner_model: str = "claude-haiku-5-5"
    extractor_model: str = "claude-haiku-5-5"
    observer_model: str = "claude-haiku-5-5"
    entity_summary_model: str = "claude-haiku-5-5"
    answer_model: str = "claude-haiku-5-5"
    reflector_model: str = "claude-sonnet-5-5"
    entity_tiebreak_model: str = "claude-sonnet-5-5"
    deep_answer_model: str = "claude-haiku-5-5"  # user decision 2026-10-25
    # SDK retries (with backoff) on connection errors, 429 and 5xx, for every LLM call.
    llm_max_retries: int = 4
    # Haiku 5.5 / Sonnet 5.5 reject non-default temperature; effort is the only knob.
    answer_effort: str = "medium"
    deep_answer_effort: str = "medium"
    answer_max_tokens: int = 4096
    answer_prompt_version: str = "v7"
    default_timezone: str = "UTC"
    embedding_model: str = "embeddinggemma:300m"
    embedding_dims: int = 768
    embedding_batch_size: int = 32
    ollama_timeout_seconds: float = 60.0
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 32
    reranker_max_length: int = 512
    reranker_device: str = "auto"  # "auto" picks cuda when available, else cpu

    # --- Write path (PLAN.md section 5) -----------------------------------------
    # Same trigger point as before the chars_per_token fix (1500 at 4.0 chars/token).
    extract_token_threshold: int = 2200
    extract_idle_seconds: int = 600
    extract_batch_max_turns: int = 25
    extract_batch_overlap: int = 5
    extract_related_facts: int = 30
    extract_schema_retries: int = 1
    extract_transient_retries: int = 4  # connection / 429 / 5xx, on top of the SDK's own
    extract_retry_backoff_seconds: float = 2.0  # doubled after each transient failure
    extractor_effort: str = "medium"
    extractor_max_tokens: int = 16000
    extractor_prompt_version: str = "v5"
    extract_lock_ttl_seconds: int = 900  # per-user extraction lock (safety net for the queue)
    worker_job_timeout_seconds: int = 900
    worker_max_jobs: int = 10
    gate_max_tokens: int = 4  # turns at or under this that match greetings/acks skip extraction
    min_fact_confidence: float = Field(0.6, ge=0, le=1)
    entity_merge: float = Field(0.85, ge=0, le=1)
    entity_llm_band_low: float = Field(0.65, ge=0, le=1)
    entity_weight_string: float = 0.5
    entity_weight_embedding: float = 0.3
    entity_weight_cooccurrence: float = 0.2
    entity_summary_debounce_seconds: int = 600
    entity_candidates: int = 5  # vector + full-text candidates scored per unmatched mention
    entity_tiebreak_effort: str = "low"
    entity_tiebreak_max_tokens: int = 1024
    entity_tiebreak_facts: int = 3  # facts about the candidate shown to the tiebreak
    entity_tiebreak_prompt_version: str = "v1"

    # --- Read path (PLAN.md section 6) ------------------------------------------
    signal_top_k: int = 50
    # Unique query terms sent to a full-text index; keeps long pasted messages under
    # Lucene's boolean clause limit (1024) even with two fields per index.
    fulltext_max_terms: int = 128
    temporal_fact_scan_limit: int = 500  # S6 candidates scored per query
    rrf_k: int = 60
    rerank_candidates: int = 100
    # PLAN.md starts at 0.15. Tuned on the dev split (2026-10-25): bge-reranker-v2-m3 ranks
    # evidence well (median rank 0) but its absolute scores on short turns are bimodal
    # (p25 of evidence 0.012), so any absolute cut dropped evidence: 0.15 -> 72.3% dev
    # accuracy, 0.001 -> 93.8%, 0.0 -> 98.5%. Selection is therefore rank-based (MAX_TURNS);
    # "inject nothing" is handled by NONE_DEPTH_MIN_SCORE for depth="none" messages.
    rerank_min_score: float = 0.0
    # depth="none" skips memory injection unless the best rerank score reaches this.
    none_depth_min_score: float = 0.5
    # Safety cap only; MAX_FACTS and MAX_EXCERPT_TURNS normally decide the size.
    memory_token_budget: int = 4000
    # Token estimate for budgeting (chars / chars_per_token). Not a tokenizer. Measured on
    # the answer model's real input_tokens (2026-10-25): about 2.7 chars per token.
    chars_per_token: float = 2.7
    # Context sizes chosen with the user (2026-10-25 audit: the first evidence fact was in
    # the top 10 for 95% of questions; excerpts held the only evidence for 11%).
    max_facts: int = 10
    max_fact_partners: int = 5  # superseded / replacing / conflicting partners of chosen facts
    max_turns: int = 15  # turn hits considered for excerpts
    max_excerpt_turns: int = 10  # turns rendered in excerpts, neighbors included
    turn_neighbors: int = 1
    excerpt_assistant_max_chars: int = 200  # longer assistant turns are cut in the excerpt
    graph_seed_facts: int = 10
    graph_hops_quick: int = 1
    graph_hops_deep: int = 2
    planner_prompt_version: str = "v2"
    planner_effort: str = "low"
    planner_max_tokens: int = 2048
    planner_llm_timeout_seconds: float = 6.0
    planner_recent_days: int = 30  # "recently" / "lately"
    planner_recent_turns: int = 4  # current-session context given to the LLM fallback
    planner_max_rewrites: int = 3
    planner_max_guidance: int = 5

    # --- Deep mode (PLAN.md section 6 step 8, Phase 7) ----------------------------
    # No caps on context size or number of searches (user decision 2026-10-25). The loop
    # stops when the model answers, or after DEEP_PATIENCE searches in a row find nothing
    # new with a rerank score >= DEEP_MIN_RELEVANCE (memory is finite, so this always ends).
    deep_prompt_version: str = "v1"
    deep_patience: int = 2
    deep_min_relevance: float = 0.01  # tune on dev only
    deep_signal_top_k: int = 200  # candidates per signal for each deep search
    deep_timeout_seconds: float = 600.0  # outage backstop only (API hangs or repeated errors)
    # The model's own context window, not a design cap: when the conversation would no longer
    # fit, the loop must answer with what it has.
    deep_context_window_tokens: int = 180_000

    # --- ObservationLog (PLAN.md sections 5 and 6) -------------------------------
    obslog_reflect_at: int = 2000
    obslog_target: int = 1000
    reflector_effort: str = "medium"
    reflector_max_tokens: int = 8000
    reflector_prompt_version: str = "v1"
    # --- Ablation switches (PLAN.md section 10; evals only, all on in production) ---
    use_fact_index: bool = True  # S1, S3, S5, S6, S7 (facts)
    use_bm25: bool = True  # S3, S4 (keyword search)
    use_graph: bool = True  # S7 (graph expansion)
    embed_fact_dates: bool = True  # write path: "(around March 2026)" in fact embeddings

    # --- Observability (PLAN.md section 8, Phase 8) -------------------------------
    log_format: Literal["text", "json"] = "text"
    log_level: str = "INFO"
    trace_file: str = ""  # also append every llm_call record to this JSONL file
    # USD per million tokens, per model: {"input", "output", "cache_read", "cache_write"}.
    # Empty by default (prices change; copy them from Anthropic's pricing page). Without
    # them the trace still logs every token count, so costs can be computed later.
    llm_prices: dict[str, dict[str, float]] = Field(default_factory=dict)

    # --- Nightly consolidation (PLAN.md section 5, Phase 8) -------------------------
    consolidation_hour: int = Field(3, ge=0, le=23)  # worker's local time
    consolidation_timeout_seconds: int = 3600

    # --- Opening message (a new conversation continues the previous one) ---------
    greeting_prompt_version: str = "v1"
    greeting_recent_turns: int = 8  # last turns of the previous conversation shown to it

    # --- Forget (PLAN.md section 5 step 11, Phase 8) ------------------------------
    # A forget request first shows what matched and waits for the user's yes (user decision
    # 2026-10-25); matches need this rerank score against the request.
    forget_min_relevance: float = 0.5
    forget_signal_top_k: int = 50
    forget_preview_ttl_minutes: int = 60
    forget_prompt_version: str = "v1"
    # Share of names/numbers from high-priority bullets a rewrite must keep to be accepted.
    reflector_min_high_coverage: float = 0.8
    entity_summary_effort: str = "low"
    entity_summary_max_tokens: int = 1024
    entity_summary_prompt_version: str = "v1"
    entity_summary_min_facts: int = 3  # entities with fewer facts don't get a summary
    entity_summary_facts: int = 10  # latest facts given to the summarizer

    # --- Migrations ----------------------------------------------------------------
    migrations_dir: Path = Path(__file__).parent / "store" / "migrations"
    migration_index_timeout_seconds: int = 300

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if not self.entity_llm_band_low <= self.entity_merge:
            raise ValueError("entity_llm_band_low must be <= entity_merge")
        if self.extract_batch_overlap >= self.extract_batch_max_turns:
            raise ValueError("extract_batch_overlap must be < extract_batch_max_turns")
        if self.obslog_target >= self.obslog_reflect_at:
            raise ValueError("obslog_target must be < obslog_reflect_at")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
