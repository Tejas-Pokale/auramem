"""Structured outputs for every LLM call (PLAN.md section 7).

Each model is passed to ``with_structured_output``; nothing downstream consumes raw LLM text.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

TimePrecision = Literal["exact", "day", "week", "month", "year", "unknown"]
FactStatus = Literal["ongoing", "completed", "planned"]
RelationToExisting = Literal["new", "update", "restatement", "contradiction"]
Priority = Literal["high", "medium", "low"]
Depth = Literal["none", "quick", "deep"]


class FactType(StrEnum):
    event = "event"
    state = "state"
    plan = "plan"
    preference = "preference"
    relationship = "relationship"
    opinion = "opinion"
    trait = "trait"


class EntityType(StrEnum):
    person = "person"
    organization = "organization"
    location = "location"
    product = "product"
    concept = "concept"
    other = "other"


class EntityMention(BaseModel):
    surface: str = Field(description="As written in the conversation")
    canonical_guess: str = Field(description="Extractor's best normalized name")
    entity_type: EntityType


class ExtractedFact(BaseModel):
    text: str = Field(description="Self-contained narrative fact, third person")
    fact_type: FactType
    attributed_to: Literal["user", "assistant", "third_party"]
    third_party_name: str | None = None
    subject: str = Field(description="Who/what the fact is about; 'user' for the user")
    attribute_key: str | None = Field(
        None, description="Single-valued slot e.g. works_at, lives_in"
    )
    entities: list[EntityMention] = Field(default_factory=list)
    start: datetime | None = None
    end: datetime | None = None
    time_precision: TimePrecision = "unknown"
    status: FactStatus | None = None
    aliases: list[str] = Field(default_factory=list, max_length=4)
    confidence: float = Field(ge=0, le=1)
    is_hypothetical: bool = False
    importance: Literal[1, 2, 3] = 2
    relation_to_existing: RelationToExisting = Field(
        "new",
        description="Against stored facts on the same attribute: update = the value really "
        "changed; restatement = same value said again; contradiction = a different value for "
        "something that did not change (e.g. two dates for one past event); new otherwise",
    )
    source_turn_seqs: list[int]


class Observation(BaseModel):
    date: datetime
    priority: Priority
    text: str


class ForgetIntent(BaseModel):
    description: str = Field(description="What the user asked to forget")
    source_turn_seq: int


class ExtractionResult(BaseModel):
    facts: list[ExtractedFact]
    observations: list[Observation]
    forget_intents: list[ForgetIntent] = Field(default_factory=list)


class TimeRange(BaseModel):
    start: datetime | None
    end: datetime | None


class QueryPlan(BaseModel):
    depth: Depth
    is_temporal: bool
    time_range: TimeRange | None = None
    entities: list[str] = Field(default_factory=list)
    rewrites: list[str] = Field(default_factory=list, max_length=3)
    guidance: list[str] = Field(default_factory=list, max_length=5)


class EntityMergeDecision(BaseModel):
    same_entity: bool
    reason: str


class ReflectionOutput(BaseModel):
    text: str = Field(description="The rewritten observation log, one dated bullet per line")


class EntitySummaryOutput(BaseModel):
    summary: str = Field(description="Neutral 2-4 sentence profile of the entity")


class AnswerOutput(BaseModel):
    answer: str = Field(description="The reply to send to the user")


DeepTool = Literal["search_facts", "search_turns", "grep", "timeline"]


class DeepToolCall(BaseModel):
    """One memory search in deep mode (PLAN.md section 6 step 8)."""

    tool: DeepTool = Field(
        description=(
            "search_facts: meaning + keyword search over extracted facts; search_turns: the same "
            "over raw conversation turns; grep: exact text in turns and facts; timeline: every "
            "fact about one person, place, organization, or thing in date order"
        )
    )
    query: str = Field(
        description="Search text (search_*), the exact text to find (grep), or the name (timeline)"
    )
    start: datetime | None = Field(None, description="Optional start of a time range (ISO 8601)")
    end: datetime | None = Field(None, description="Optional end of a time range (ISO 8601)")
    entity: str | None = Field(
        None, description="search_facts only: limit to facts about this name"
    )


class DeepStep(BaseModel):
    """Each deep-mode step either searches (tool_calls) or answers (answer), never both."""

    reasoning: str = Field(description="Briefly: what is known so far and what is still missing")
    tool_calls: list[DeepToolCall] = Field(
        default_factory=list, description="Searches to run now (empty when answering)"
    )
    answer: str | None = Field(
        None, description="The final reply to the user, once the evidence is enough"
    )
