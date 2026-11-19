"""Read-path outputs: the assembled memory block and the reply; forget previews and reports."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from memory.models.llm import QueryPlan


class MemoryContext(BaseModel):
    """The memory block injected into the answering prompt.

    ``text`` is empty when nothing relevant was found (then no memory is injected).
    """

    text: str
    token_estimate: int
    hit_turn_ids: list[str] = Field(description="Retrieved turns selected, best first")
    turn_ids: list[str] = Field(description="Every turn rendered, including neighbors")
    fact_ids: list[str] = Field(default_factory=list, description="Facts rendered")
    covered_turn_ids: list[str] = Field(
        default_factory=list, description="Source turns of the rendered facts"
    )
    candidates: int = Field(description="Fused candidates before selection")
    plan: QueryPlan | None = None
    plan_source: str | None = None
    top_rerank_score: float | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    deep: bool = False  # selected without caps for the deep-mode loop

    @property
    def guidance(self) -> list[str]:
        return self.plan.guidance if self.plan else []

    @classmethod
    def empty(cls, candidates: int = 0) -> "MemoryContext":
        return cls(text="", token_estimate=0, hit_turn_ids=[], turn_ids=[], candidates=candidates)


class DeepTrace(BaseModel):
    """What deep mode did: searches run, how it stopped, and its token use across steps."""

    steps: int = 0
    tool_calls: list[str] = Field(default_factory=list, description="tool: query, in order")
    new_items: list[int] = Field(default_factory=list, description="New relevant items per step")
    stop_reason: str = ""  # answered | no_progress | timeout | error
    elapsed_ms: float = 0.0


class Reply(BaseModel):
    text: str
    model: str
    prompt_version: str
    memory: MemoryContext
    profile_tokens: int = 0  # ObservationLog size included in the prompt
    usage: dict[str, int] = Field(default_factory=dict)  # tokens, incl. prompt-cache reads
    mode: str = "quick"  # quick | deep | none | forget
    deep: DeepTrace | None = None


class ForgetItem(BaseModel):
    kind: Literal["fact", "turn"]
    item_id: str
    text: str
    date: datetime
    role: str | None = None  # turns: who said it


class ForgetPreview(BaseModel):
    """What a forget request matched. Nothing is deleted until the user confirms."""

    preview_id: str | None  # None when nothing matched (nothing to confirm)
    description: str
    items: list[ForgetItem] = Field(default_factory=list)

    @property
    def fact_ids(self) -> list[str]:
        return [i.item_id for i in self.items if i.kind == "fact"]

    @property
    def turn_ids(self) -> list[str]:
        return [i.item_id for i in self.items if i.kind == "turn"]

    def render(self) -> str:
        """The message shown to the user, asking for permission."""
        if not self.items:
            return f'I couldn\'t find anything in my memory about "{self.description}".'
        lines = [f'I found {len(self.items)} things in my memory about "{self.description}":']
        for n, item in enumerate(self.items, start=1):
            who = "" if item.kind == "fact" else f"{(item.role or 'user').capitalize()} said: "
            lines.append(f"{n}. [{item.date:%Y-%m-%d}] {who}{item.text}")
        lines.append(
            "Should I permanently delete all of these? This can't be undone. "
            'Reply "yes" to delete them or "no" to keep them.'
        )
        return "\n".join(lines)


class ForgetReport(BaseModel):
    facts: int = 0
    turns: int = 0
    entities: int = 0
    profile_rewritten: bool = False
    summaries_refreshed: int = 0

    def render(self) -> str:
        return f"Done. I permanently deleted {self.facts} memories and {self.turns} messages" + (
            ", and removed it from your profile." if self.profile_rewritten else "."
        )
