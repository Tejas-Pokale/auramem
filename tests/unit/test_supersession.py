from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from memory.config import Settings
from memory.entities import entity_score, route, string_similarity
from memory.extract import normalize_slot, supersession
from memory.models import Entity, EntityType, Fact, FactType
from memory.retrieve.assemble import fact_line

IST = timezone(timedelta(hours=5, minutes=30))
FEB = datetime(2026, 2, 8, tzinfo=IST)
AUG = datetime(2026, 8, 10, tzinfo=IST)


def fact(**overrides: Any) -> Fact:
    base: dict[str, Any] = {
        "fact_id": "f",
        "user_id": "u",
        "text": "x",
        "text_for_embedding": "x",
        "fact_type": FactType.state,
        "attributed_to": "user",
        "subject_entity_id": "u:self",
        "attribute_key": "works_at",
        "mentioned_at": FEB,
        "confidence": 0.9,
        "extractor_version": "test",
    }
    return Fact(**(base | overrides))


class TestSupersession:
    def test_later_mention_wins_when_starts_unknown(self) -> None:
        assert supersession(fact(mentioned_at=AUG), fact(mentioned_at=FEB)) == "new_wins"

    def test_older_backfill_loses(self) -> None:
        assert supersession(fact(mentioned_at=FEB), fact(mentioned_at=AUG)) == "old_wins"

    def test_start_beats_mention_order(self) -> None:
        new = fact(start=datetime(2025, 1, 1, tzinfo=IST), mentioned_at=AUG)
        old = fact(start=datetime(2026, 1, 1, tzinfo=IST), mentioned_at=FEB)
        assert supersession(new, old) == "old_wins"

    def test_one_start_missing_falls_back_to_mention(self) -> None:
        new = fact(mentioned_at=AUG)
        old = fact(start=datetime(2026, 2, 8, tzinfo=IST), mentioned_at=FEB)
        assert supersession(new, old) == "new_wins"

    @pytest.mark.parametrize(
        ("new", "old"),
        [
            ({"fact_type": FactType.event}, {}),  # events never supersede states
            ({}, {"fact_type": FactType.event}),
            ({"fact_type": FactType.plan}, {}),
            ({"attribute_key": "lives_in"}, {}),  # different slot
            ({"attribute_key": None}, {"attribute_key": None}),
            ({"subject_entity_id": "ent-asha"}, {}),  # different subject
            ({"subject_entity_id": None}, {"subject_entity_id": None}),
            ({"is_hypothetical": True}, {}),
            ({}, {"is_hypothetical": True}),
        ],
    )
    def test_no_supersession(self, new: dict[str, Any], old: dict[str, Any]) -> None:
        assert supersession(fact(mentioned_at=AUG, **new), fact(mentioned_at=FEB, **old)) is None

    def test_exact_tie_is_left_alone(self) -> None:
        assert supersession(fact(), fact()) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("works_at", "works_at"),
        ("Employer", "works_at"),
        ("current company", "works_at"),
        ("job-title", "job_title"),
        ("role", "job_title"),
        ("CTC", "salary"),
        ("bond", "employment_bond"),
        ("city", "lives_in"),
        ("gpu rental provider", "gpu_rental_provider"),
        ("", None),
        (None, None),
    ],
)
def test_normalize_slot(raw: str | None, expected: str | None) -> None:
    assert normalize_slot(raw) == expected


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def entity(name: str, aliases: list[str] | None = None) -> Entity:
    return Entity(
        entity_id=f"e-{name}",
        user_id="u",
        name=name,
        entity_type=EntityType.organization,
        aliases=aliases or [],
        mention_count=1,
    )


class TestEntityScoring:
    def test_string_similarity_normalizes_case_and_punctuation(self) -> None:
        assert string_similarity("JHEX Consulting", "jhex consulting.") == 1.0
        assert string_similarity("Meditab", "Optum") < 0.5

    def test_best_alias_counts(self) -> None:
        s = settings()
        cand = entity("Government College of Engineering, Yavatmal", ["GCOE Yavatmal"])
        score = entity_score("GCOE Yavatmal", cand, cosine=0.5, cooccurrence=0.0, settings=s)
        assert score == pytest.approx(0.5 * 1.0 + 0.3 * 0.5)

    def test_weights_combine(self) -> None:
        s = settings()
        score = entity_score("Vast", entity("Vast.ai"), cosine=1.0, cooccurrence=1.0, settings=s)
        sim = string_similarity("Vast", "Vast.ai")
        assert score == pytest.approx(0.5 * sim + 0.3 + 0.2)

    @pytest.mark.parametrize(
        ("score", "decision"),
        [(0.9, "merge"), (0.85, "merge"), (0.7, "tiebreak"), (0.65, "tiebreak"), (0.64, "new")],
    )
    def test_routing_bands(self, score: float, decision: str) -> None:
        assert route(score, settings()) == decision


class TestLabels:
    def test_latest_label_only_for_slotted_facts(self) -> None:
        assert fact_line(fact(text="Works at JHEX.", is_latest=True)).startswith("- [latest] ")
        plain = fact(text="Went hiking.", attribute_key=None)
        assert not fact_line(plain).startswith("- [")

    def test_superseded_label_has_date(self) -> None:
        old = fact(text="Works at Meditab.", is_latest=False, superseded_on=AUG)
        assert fact_line(old).startswith("- [superseded on 2026-08-10] Works at Meditab.")


class TestConflicts:
    def test_contradiction_links_instead_of_superseding(self) -> None:
        new = fact(mentioned_at=AUG, relation_to_existing="contradiction")
        assert supersession(new, fact(mentioned_at=FEB)) == "conflict"

    def test_contradiction_applies_to_past_events_too(self) -> None:
        new = fact(fact_type=FactType.event, relation_to_existing="contradiction", mentioned_at=AUG)
        old = fact(fact_type=FactType.event, mentioned_at=FEB)
        assert supersession(new, old) == "conflict"

    def test_contradiction_needs_the_same_slot(self) -> None:
        new = fact(attribute_key="lives_in", relation_to_existing="contradiction", mentioned_at=AUG)
        assert supersession(new, fact(mentioned_at=FEB)) is None

    def test_conflicting_label_wins_over_latest(self) -> None:
        line = fact_line(fact(text="Commute is 20 minutes.", conflicts_with_ids=["f2"]))
        assert line.startswith("- [conflicting] Commute is 20 minutes.")
