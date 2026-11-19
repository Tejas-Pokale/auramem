from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from memory.models import (
    ExtractedFact,
    ExtractionResult,
    FactType,
    NewFact,
    NewTurn,
    QueryPlan,
)

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


def fact_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "fact_id": "f1",
        "text": "Teju works at JHEX.",
        "text_for_embedding": "Teju works at JHEX. (around October 2026)",
        "fact_type": FactType.state,
        "attributed_to": "user",
        "mentioned_at": NOW,
        "confidence": 0.9,
        "extractor_version": "extract_v1",
        "derived_from_turn_ids": ["t1"],
    }
    return base | overrides


def test_extraction_result_parses_llm_json() -> None:
    payload = {
        "facts": [
            {
                "text": "The user started working at JHEX in October 2026.",
                "fact_type": "event",
                "attributed_to": "user",
                "subject": "user",
                "attribute_key": "works_at",
                "entities": [
                    {"surface": "JHEX", "canonical_guess": "JHEX", "entity_type": "organization"}
                ],
                "start": "2026-10-01T00:00:00Z",
                "time_precision": "month",
                "aliases": ["joined JHEX", "new job at JHEX"],
                "confidence": 0.95,
                "source_turn_seqs": [3],
            }
        ],
        "observations": [{"date": "2026-10-24T00:00:00Z", "priority": "high", "text": "x"}],
    }
    result = ExtractionResult.model_validate(payload)
    assert result.facts[0].fact_type is FactType.event
    assert result.forget_intents == []


def test_extracted_fact_rejects_too_many_aliases() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact(
            text="x",
            fact_type=FactType.event,
            attributed_to="user",
            subject="user",
            aliases=["a", "b", "c", "d", "e"],
            confidence=0.9,
            source_turn_seqs=[1],
        )


def test_query_plan_caps_rewrites() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(depth="quick", is_temporal=False, rewrites=["a", "b", "c", "d"])


@pytest.mark.parametrize("attributed_to", ["user", "assistant", "third_party:e42"])
def test_new_fact_accepts_attributions(attributed_to: str) -> None:
    assert NewFact(**fact_kwargs(attributed_to=attributed_to)).attributed_to == attributed_to


@pytest.mark.parametrize("attributed_to", ["third_party", "third_party:", "sister"])
def test_new_fact_rejects_bad_attribution(attributed_to: str) -> None:
    with pytest.raises(ValidationError):
        NewFact(**fact_kwargs(attributed_to=attributed_to))


def test_new_fact_needs_provenance() -> None:
    with pytest.raises(ValidationError):
        NewFact(**fact_kwargs(derived_from_turn_ids=[]))


def test_new_fact_defaults_flags_to_booleans() -> None:
    fact = NewFact(**fact_kwargs())
    assert fact.is_latest is True
    assert fact.is_hypothetical is False


def test_naive_datetimes_rejected() -> None:
    with pytest.raises(ValidationError):
        NewTurn(turn_id="t", role="user", text="hi", created_at=datetime(2026, 1, 1), seq=0)


def test_embedding_requires_model_name() -> None:
    with pytest.raises(ValidationError, match="together"):
        NewTurn(turn_id="t", role="user", text="hi", created_at=NOW, seq=0, embedding=[0.1])
