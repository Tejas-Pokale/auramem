"""Schemas for the synthetic dataset generator (structured LLM outputs and stored records)."""

from typing import Literal

from pydantic import BaseModel, Field

# --- Stage A: life plan ---------------------------------------------------------------------


class Person(BaseModel):
    name: str
    relation: str = Field(description="Relation to the user, e.g. sister, manager, friend")
    notes: str = Field(description="One or two facts about this person")


class Persona(BaseModel):
    full_name: str
    preferred_name: str
    birth_year: int
    hometown: str
    background: str = Field(description="Education and early career in 2-3 sentences")
    personality: str
    people: list[Person] = Field(description="8-14 recurring people in the user's life")


class AttributeValue(BaseModel):
    value: str
    from_date: str = Field(description="YYYY-MM-DD when this value became true")


class Attribute(BaseModel):
    key: str = Field(description="snake_case slot, e.g. employer, city, monthly_rent")
    description: str
    history: list[AttributeValue] = Field(description="Chronological values; first may predate")


class LifeEvent(BaseModel):
    date: str = Field(description="YYYY-MM-DD the event happened")
    title: str
    details: str = Field(description="Concrete details: names, places, numbers")


class PlannedConflict(BaseModel):
    topic: str
    first_value: str
    first_month: str = Field(description="YYYY-MM the first version is said")
    second_value: str
    second_month: str = Field(description="YYYY-MM the contradicting version is said")
    note: str = Field(description="Why they differ (misremembering, different contexts)")


class LifePlan(BaseModel):
    persona: Persona
    attributes: list[Attribute]
    events: list[LifeEvent]
    conflicts: list[PlannedConflict]
    preferences: list[str] = Field(description="Stable preferences and communication style")
    sensitive_topics: list[str] = Field(description="Health, family, money worries, with dates")


class PersonaPart(BaseModel):
    persona: Persona
    preferences: list[str]
    sensitive_topics: list[str]


class AttributesPart(BaseModel):
    attributes: list[Attribute]


class EventsPart(BaseModel):
    events: list[LifeEvent]


class ConflictsPart(BaseModel):
    conflicts: list[PlannedConflict]


# --- Stage B: monthly ledger ---------------------------------------------------------------

FactKind = Literal[
    "personal", "update", "event", "preference", "third_party", "assistant_advice",
    "hypothetical", "conflict", "sensitive",
]  # fmt: skip


class DraftFact(BaseModel):
    session: int = Field(description="Session number within the month, starting at 1")
    told_day: int = Field(ge=1, le=31, description="Day of the month the session happens")
    kind: FactKind
    speaker: Literal["user", "assistant"]
    statement: str = Field(
        description="What is said, as a natural sentence: first person for the user, "
        "second person advice for the assistant"
    )
    attribute: str | None = Field(None, description="Plan attribute key if single-valued")
    value: str | None = Field(None, description="The attribute value stated, if any")
    event_date: str | None = Field(None, description="YYYY-MM-DD the event happened, if any")


class MonthDraft(BaseModel):
    facts: list[DraftFact]


class LedgerFact(DraftFact):
    fact_id: str
    month: str  # YYYY-MM


# --- Stage C: conversations ----------------------------------------------------------------


class RenderedTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    fact_ids: list[str] = Field(default_factory=list, description="Ledger facts stated here")


class RenderedSession(BaseModel):
    turns: list[RenderedTurn]


class DistractorTopics(BaseModel):
    topics: list[str]


# --- Stage D: questions --------------------------------------------------------------------

TestType = Literal[
    "single_fact", "multi_fact", "knowledge_update", "temporal", "conflict", "abstention",
    "assistant_recall", "preference_application", "sensitive_history", "inference",
]  # fmt: skip


class DraftQuestion(BaseModel):
    question: str = Field(description="Asked by the user on 2026-11-01, in first person")
    expected_answer: str
    evidence_fact_ids: list[str] = Field(description="Ledger fact ids needed; empty if abstain")
    must_abstain: bool = False


class QuestionBatch(BaseModel):
    questions: list[DraftQuestion]


class QuestionVerdict(BaseModel):
    index: int
    keep: bool
    reason: str
    corrected_answer: str | None = Field(
        None, description="A corrected expected answer when the draft is slightly off"
    )


class VerificationBatch(BaseModel):
    verdicts: list[QuestionVerdict]
