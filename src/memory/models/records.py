"""Storage records (PLAN.md section 4): what the store layer reads and writes.

``New*`` models are inputs to insert functions; the plain names are what reads return.
Read results never carry embeddings (they are large and only the indexes need them).
"""

from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, Field, model_validator

from memory.models.llm import EntityType, FactStatus, FactType, RelationToExisting, TimePrecision

Role = Literal["user", "assistant"]
TurnSource = Literal["typed", "pasted", "tool"]
# "user", "assistant", or "third_party:<entity_id>"
ATTRIBUTION_PATTERN = r"^(user|assistant|third_party:.+)$"


class _Embedded(BaseModel):
    embedding: list[float] | None = None
    embedding_model: str | None = None

    @model_validator(mode="after")
    def _model_with_embedding(self) -> Self:
        if (self.embedding is None) != (self.embedding_model is None):
            raise ValueError("embedding and embedding_model must be set together")
        return self


class User(BaseModel):
    user_id: str
    created_at: AwareDatetime
    timezone: str | None = None


class Session(BaseModel):
    session_id: str
    user_id: str
    started_at: AwareDatetime
    ended_at: AwareDatetime | None = None
    title: str | None = None
    extracted_through_seq: int = -1  # extraction watermark; -1 = nothing extracted yet


class NewTurn(_Embedded):
    turn_id: str
    role: Role
    text: str
    created_at: AwareDatetime
    seq: int = Field(ge=0)
    source: TurnSource = "typed"


class Turn(BaseModel):
    turn_id: str
    user_id: str
    session_id: str
    role: Role
    text: str
    created_at: AwareDatetime
    seq: int
    source: TurnSource
    embedding_model: str | None = None


class _FactFields(BaseModel):
    fact_id: str
    text: str
    text_for_embedding: str
    fact_type: FactType
    attributed_to: str = Field(pattern=ATTRIBUTION_PATTERN)
    subject_entity_id: str | None = None
    attribute_key: str | None = None
    start: AwareDatetime | None = None
    end: AwareDatetime | None = None
    mentioned_at: AwareDatetime
    time_precision: TimePrecision = "unknown"
    status: FactStatus | None = None
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    is_latest: bool = True
    is_hypothetical: bool = False
    importance: Literal[1, 2, 3] = 2
    relation_to_existing: RelationToExisting = "new"
    extractor_version: str


class NewFact(_FactFields, _Embedded):
    derived_from_turn_ids: list[str] = Field(min_length=1)
    mentions_entity_ids: list[str] = Field(default_factory=list)


class Fact(_FactFields):
    user_id: str
    embedding_model: str | None = None
    mention_dates: list[AwareDatetime] = Field(default_factory=list)
    source_turn_ids: list[str] = Field(default_factory=list, description="DERIVED_FROM turns")
    superseded_on: AwareDatetime | None = Field(
        None, description="When the replacing fact happened or was said (if superseded)"
    )
    superseded_by_ids: list[str] = Field(default_factory=list)
    supersedes_ids: list[str] = Field(default_factory=list)
    conflicts_with_ids: list[str] = Field(default_factory=list)
    mentioned_entity_ids: list[str] = Field(default_factory=list)


class NewEntity(_Embedded):
    entity_id: str
    name: str
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    summary: str | None = None
    mention_count: int = Field(1, ge=0)


class Entity(BaseModel):
    entity_id: str
    user_id: str
    name: str
    entity_type: EntityType
    aliases: list[str]
    summary: str | None = None
    summary_updated_at: AwareDatetime | None = None
    mention_count: int
    embedding_model: str | None = None


class ObservationLog(BaseModel):
    user_id: str
    text: str
    token_count: int
    version: int
    updated_at: AwareDatetime


class Hit[T: BaseModel](BaseModel):
    """One search result with its raw index score (comparable only within one query)."""

    item: T
    score: float
