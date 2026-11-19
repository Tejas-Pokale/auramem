from typing import Any

import pytest
from pydantic import ValidationError

from memory.config import Settings


def make(**overrides: Any) -> Settings:
    # _env_file=None keeps a developer's local .env from leaking into tests.
    return Settings(_env_file=None, **overrides)


def test_defaults_match_plan_section_8() -> None:
    s = make()
    assert s.extract_token_threshold == 2200
    assert s.extract_idle_seconds == 600
    assert (s.extract_batch_max_turns, s.extract_batch_overlap) == (25, 5)
    assert s.min_fact_confidence == 0.6
    assert (s.entity_merge, s.entity_llm_band_low) == (0.85, 0.65)
    assert s.signal_top_k == 50
    assert s.rrf_k == 60
    assert s.rerank_candidates == 100
    assert s.rerank_min_score == 0.0  # plan starts at 0.15; tuned on dev (see config.py)
    assert s.memory_token_budget == 4000
    assert s.chars_per_token == 2.7
    assert (s.max_facts, s.max_fact_partners) == (10, 5)
    assert (s.max_turns, s.max_excerpt_turns, s.turn_neighbors) == (15, 10, 1)
    assert s.excerpt_assistant_max_chars == 200
    assert (s.deep_answer_model, s.deep_answer_effort) == ("claude-haiku-5-5", "medium")
    assert (s.deep_patience, s.deep_timeout_seconds) == (2, 600.0)
    assert (s.obslog_reflect_at, s.obslog_target) == (2000, 1000)


def test_entity_weights_sum_to_one() -> None:
    s = make()
    total = s.entity_weight_string + s.entity_weight_embedding + s.entity_weight_cooccurrence
    assert total == pytest.approx(1.0)


def test_models_match_claude_md() -> None:
    s = make()
    assert s.extractor_model == s.planner_model == "claude-haiku-5-5"
    assert s.reflector_model == s.entity_tiebreak_model == "claude-sonnet-5-5"
    assert s.embedding_model == "embeddinggemma:300m"
    assert s.embedding_dims == 768
    assert s.reranker_model == "BAAI/bge-reranker-v2-m3"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANK_MIN_SCORE", "0.2")
    monkeypatch.setenv("NEO4J_URI", "bolt://example:7687")
    s = make()
    assert s.rerank_min_score == 0.2
    assert s.neo4j_uri == "bolt://example:7687"


def test_rejects_inverted_entity_band() -> None:
    with pytest.raises(ValidationError):
        make(entity_merge=0.6, entity_llm_band_low=0.7)


def test_rejects_overlap_not_below_batch() -> None:
    with pytest.raises(ValidationError):
        make(extract_batch_max_turns=5, extract_batch_overlap=5)


def test_secrets_are_not_printed() -> None:
    s = make(neo4j_password="hunter22-secret")
    assert "hunter22-secret" not in repr(s)
