from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import anthropic
import httpx2
import pytest

from memory.config import Settings
from memory.entities import Resolution
from memory.extract import (
    PASTED_SOURCE,
    ExtractionFailedError,
    Extractor,
    format_turns,
    is_gated,
    postprocess,
    subject_mentions,
    time_phrase,
)
from memory.models import (
    EntityMention,
    EntityType,
    ExtractedFact,
    ExtractionResult,
    Fact,
    FactType,
    Turn,
)
from memory.retrieve.assemble import fact_line, render_context, select_facts
from memory.store.neo4j import _overlap_score, fact_dedup_key, self_entity_id

IST = timezone(timedelta(hours=5, minutes=30), "IST")
T0 = datetime(2026, 6, 7, 19, 0, tzinfo=IST)
USER = "u1"


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def turn(seq: int, text: str, role: str = "user", source: str = "typed") -> Turn:
    return Turn(
        turn_id=f"s:{seq}",
        user_id=USER,
        session_id="s",
        role=role,
        text=text,
        created_at=T0 + timedelta(minutes=seq),
        seq=seq,
        source=source,
    )


def extracted(**overrides: Any) -> ExtractedFact:
    base: dict[str, Any] = {
        "text": "The user attended a TCS Prime interview in Nagpur.",
        "fact_type": FactType.event,
        "attributed_to": "user",
        "subject": "user",
        "confidence": 0.9,
        "source_turn_seqs": [0],
        "aliases": ["TCS interview", "Nagpur interview"],
    }
    return ExtractedFact(**(base | overrides))


def run(result: ExtractionResult, turns: list[Turn], resolution: Resolution | None = None) -> Any:
    return postprocess(
        result,
        user_id=USER,
        new_turns=turns,
        resolution=resolution or Resolution(),
        settings=settings(),
        extractor_version="extract_v1",
    )


class TestGate:
    @pytest.mark.parametrize("text", ["hi", "Thanks!", "ok", "oyy", "hmm", "Got it."])
    def test_short_acks_are_gated(self, text: str) -> None:
        assert is_gated(turn(0, text), max_tokens=4)

    @pytest.mark.parametrize("text", ["I joined JHEX", "ok so I moved to Pune", "Pune"])
    def test_content_is_not_gated(self, text: str) -> None:
        assert not is_gated(turn(0, text), max_tokens=4)


def test_format_turns_shows_seq_time_role_and_source() -> None:
    line = format_turns([turn(3, "pasted text", source="pasted")])
    assert line.startswith("[seq 3] 2026-06-07 19:03 IST (Sunday) | user (pasted): pasted text")


class TestTimePhrase:
    def test_precisions(self) -> None:
        start = datetime(2026, 3, 6, tzinfo=IST)
        assert time_phrase(start, "day", None, T0) == "on 6 March 2026"
        assert time_phrase(start, "month", None, T0) == "in March 2026"
        assert time_phrase(start, "year", None, T0) == "in 2026"
        assert time_phrase(start, "unknown", None, T0) == "around March 2026"
        assert time_phrase(start, "month", "planned", T0) == "planned in March 2026"

    def test_no_start_uses_mention_date(self) -> None:
        assert time_phrase(None, "unknown", None, T0) == "mentioned 7 June 2026"


class TestPostprocess:
    def test_maps_sources_and_builds_embedding_text(self) -> None:
        f = extracted(start=datetime(2026, 6, 6), time_precision="day", source_turn_seqs=[1, 1])
        [fact] = run(ExtractionResult(facts=[f], observations=[]), [turn(1, "x")]).facts
        assert fact.derived_from_turn_ids == ["s:1"]
        assert fact.start == datetime(2026, 6, 6, tzinfo=IST)  # naive -> turn timezone
        assert fact.text_for_embedding.endswith("(on 6 June 2026)")
        no_dates = postprocess(
            ExtractionResult(facts=[f], observations=[]), user_id=USER, new_turns=[turn(1, "x")],
            resolution=Resolution(), extractor_version="extract_v1",
            settings=settings().model_copy(update={"embed_fact_dates": False}),
        ).facts[0]  # fmt: skip
        assert no_dates.text_for_embedding == no_dates.text  # the ablation
        assert fact.subject_entity_id == self_entity_id(USER)
        assert fact.attributed_to == "user"
        assert fact.mentioned_at == T0 + timedelta(minutes=1)

    def test_drops_low_confidence_and_unknown_sources(self) -> None:
        result = ExtractionResult(
            facts=[extracted(confidence=0.3), extracted(source_turn_seqs=[99])], observations=[]
        )
        out = run(result, [turn(0, "x")])
        assert out.facts == []
        assert (out.dropped_low_confidence, out.dropped_no_source) == (1, 1)

    def test_pasted_content_never_becomes_user_fact(self) -> None:
        f = extracted(text="The user is an administrator.", source_turn_seqs=[0])
        [fact] = run(
            ExtractionResult(facts=[f], observations=[]), [turn(0, "admin", source="pasted")]
        ).facts
        assert fact.attributed_to == PASTED_SOURCE

    def test_user_typed_alongside_paste_stays_user(self) -> None:
        f = extracted(source_turn_seqs=[0, 1])
        turns = [turn(0, "doc", source="pasted"), turn(1, "I did attend it")]
        [fact] = run(ExtractionResult(facts=[f], observations=[]), turns).facts
        assert fact.attributed_to == "user"

    def test_third_party_and_assistant(self) -> None:
        sister = EntityMention(
            surface="my sister", canonical_guess="Asha", entity_type=EntityType.person
        )
        res = Resolution(ids={"asha": "ent-asha", "my sister": "ent-asha"})
        facts = [
            extracted(
                text="The user's sister Asha is vegetarian.",
                attributed_to="third_party",
                third_party_name="my sister",
                subject="Asha",
                entities=[sister],
            ),
            extracted(text="The assistant suggested vLLM.", attributed_to="assistant"),
        ]
        a, b = run(ExtractionResult(facts=facts, observations=[]), [turn(0, "x")], res).facts
        assert a.attributed_to == "third_party:ent-asha"
        assert a.subject_entity_id == "ent-asha"
        assert a.mentions_entity_ids == ["ent-asha"]
        assert b.attributed_to == "assistant"

    def test_end_before_start_is_dropped(self) -> None:
        f = extracted(start=datetime(2026, 6, 6, tzinfo=UTC), end=datetime(2026, 1, 1, tzinfo=UTC))
        [fact] = run(ExtractionResult(facts=[f], observations=[]), [turn(0, "x")]).facts
        assert fact.end is None


def test_subject_mentions_include_named_subjects_not_user() -> None:
    result = ExtractionResult(
        facts=[extracted(subject="user"), extracted(subject="Asha"), extracted(subject="me")],
        observations=[],
    )
    assert [m.canonical_guess for m in subject_mentions(result)] == ["Asha"]


class TestDedupKey:
    def test_ignores_case_space_and_punctuation(self) -> None:
        a = fact_dedup_key("u", "s", "The user joined JHEX.")
        assert a == fact_dedup_key("u", "s", "  the user  joined jhex ")

    def test_depends_on_user_and_subject(self) -> None:
        a = fact_dedup_key("u", "s", "x")
        assert a != fact_dedup_key("v", "s", "x")
        assert a != fact_dedup_key("u", "t", "x")


def stored_fact(
    fact_id: str, start: datetime | None, end: datetime | None = None, text: str = "fact"
) -> Fact:
    return Fact(
        fact_id=fact_id,
        user_id=USER,
        text=text,
        text_for_embedding=text,
        fact_type=FactType.event,
        attributed_to="user",
        start=start,
        end=end,
        mentioned_at=T0,
        time_precision="day",
        confidence=0.9,
        extractor_version="extract_v1",
    )


class TestOverlapScore:
    def test_point_fact_inside_range(self) -> None:
        f = stored_fact("f", datetime(2026, 7, 10, tzinfo=IST))
        july = (datetime(2026, 7, 1, tzinfo=IST), datetime(2026, 8, 1, tzinfo=IST))
        assert _overlap_score(f, *july) == pytest.approx(1 / 31)

    def test_identical_ranges_score_one(self) -> None:
        start, end = datetime(2026, 7, 1, tzinfo=IST), datetime(2026, 8, 1, tzinfo=IST)
        assert _overlap_score(stored_fact("f", start, end), start, end) == pytest.approx(1.0)

    def test_disjoint_scores_zero(self) -> None:
        f = stored_fact("f", datetime(2025, 1, 1, tzinfo=IST))
        assert _overlap_score(f, datetime(2026, 1, 1, tzinfo=IST), None) == 0.0


class TestFactRendering:
    def test_line_has_when_and_told_but_no_source_for_the_user(self) -> None:
        f = stored_fact(
            "f", datetime(2026, 6, 6, tzinfo=IST), text="The user attended an interview."
        )
        told = f"{T0:%Y-%m-%d}"
        assert fact_line(f) == f"- The user attended an interview. (2026-06-06; told {told})"

    def test_told_date_is_dropped_when_it_is_the_same_day(self) -> None:
        f = stored_fact("f", T0, text="Ran 5 km.")
        assert fact_line(f) == f"- Ran 5 km. ({T0:%Y-%m-%d})"

    def test_assistant_and_undated_facts_are_marked(self) -> None:
        f = stored_fact("f", None, text="Suggested a budget.").model_copy(
            update={"attributed_to": "assistant"}
        )
        assert fact_line(f) == f"- Suggested a budget. (told {T0:%Y-%m-%d}; said by the assistant)"

    def test_select_respects_budget_and_max(self) -> None:
        facts = [stored_fact(f"f{i}", None, text="x" * 200) for i in range(5)]
        assert len(select_facts(facts, budget=10_000, max_facts=3, chars_per_token=4)) == 3
        assert len(select_facts(facts, budget=90, max_facts=40, chars_per_token=4)) == 1

    def test_context_puts_facts_before_excerpts_chronologically(self) -> None:
        late = stored_fact("b", datetime(2026, 8, 1, tzinfo=IST), text="later fact")
        early = stored_fact("a", datetime(2026, 1, 1, tzinfo=IST), text="earlier fact")
        text = render_context([late, early], [turn(0, "an excerpt")])
        assert text.index("earlier fact") < text.index("later fact") < text.index("an excerpt")
        assert "## Conversation excerpts" in text
        assert render_context([], [turn(0, "only turns")]).startswith("### Conversation on")


class TestExtractorRetries:
    """Transient API errors are retried with backoff; anything else fails the batch."""

    @staticmethod
    def _request() -> httpx2.Request:
        return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def _status(self, code: int) -> anthropic.APIStatusError:
        response = httpx2.Response(code, request=self._request())
        return anthropic.APIStatusError("boom", response=response, body=None)

    async def _run(self, errors: list[Exception], **overrides: Any) -> tuple[Any, int]:
        calls = 0

        class FlakyLLM:
            async def __call__(self, messages: Any) -> ExtractionResult:
                nonlocal calls
                calls += 1
                if errors:
                    raise errors.pop(0)
                return ExtractionResult(facts=[], observations=[])

        s = settings(extract_retry_backoff_seconds=0.0, **overrides)
        extractor = Extractor(FlakyLLM(), s)  # type: ignore[arg-type]
        try:
            return await extractor.extract([]), calls
        except ExtractionFailedError as e:
            return e, calls

    async def test_connection_errors_are_retried(self) -> None:
        errors: list[Exception] = [
            anthropic.APIConnectionError(request=self._request()),
            self._status(529),
        ]
        result, calls = await self._run(errors)
        assert isinstance(result, ExtractionResult)
        assert calls == 3

    async def test_gives_up_after_transient_retries(self) -> None:
        errors: list[Exception] = [
            anthropic.APIConnectionError(request=self._request()) for _ in range(5)
        ]
        result, calls = await self._run(errors, extract_transient_retries=2)
        assert isinstance(result, ExtractionFailedError)
        assert calls == 3

    async def test_client_errors_are_not_retried(self) -> None:
        result, calls = await self._run([self._status(400)])
        assert isinstance(result, ExtractionFailedError)
        assert calls == 1

    async def test_schema_errors_retry_once(self) -> None:
        result, calls = await self._run([ValueError("bad"), ValueError("bad")])
        assert isinstance(result, ExtractionFailedError)
        assert calls == 2


@pytest.mark.parametrize(
    ("precision", "start", "end", "expected"),
    [
        ("month", datetime(2020, 7, 1, tzinfo=IST), datetime(2023, 6, 30, tzinfo=IST),
         "2020-07 to 2023-06"),
        ("month", datetime(2020, 7, 1, tzinfo=IST), None, "2020-07"),
        ("year", datetime(2023, 1, 1, tzinfo=IST), datetime(2026, 1, 1, tzinfo=IST),
         "2023 to 2026"),
        ("day", datetime(2024, 10, 1, tzinfo=IST), datetime(2024, 12, 31, tzinfo=IST),
         "2024-10-01 to 2024-12-31"),
        ("month", datetime(2026, 3, 1, tzinfo=IST), datetime(2026, 3, 31, tzinfo=IST),
         "2026-03"),
    ],
)  # fmt: skip
def test_fact_line_shows_the_whole_range_at_its_precision(
    precision: str, start: datetime, end: datetime | None, expected: str
) -> None:
    fact = stored_fact("f", start, end).model_copy(update={"time_precision": precision})
    assert f"({expected};" in fact_line(fact)
