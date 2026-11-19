"""LLM judge (PLAN.md section 10).

Sonnet 5.5 with the fixed prompt judge_v1.md. Sonnet 5.5 rejects non-default temperature,
so instead of temperature 0 the judge runs at a fixed effort; consistency comes from the
fixed prompt and rubric, and scores are averaged over runs at phase gates.
Never edit judge_v1.md in place: copy it to judge_v2.md and bump JUDGE_VERSION.
"""

from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from memory.config import Settings
from memory.graphs.llm import chat_model, structured

JUDGE_VERSION = "judge_v1"
JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE_EFFORT = "medium"
JUDGE_MAX_TOKENS = 2048
_PROMPT = (Path(__file__).parent / f"{JUDGE_VERSION}.md").read_text(encoding="utf-8")


class JudgeVerdict(BaseModel):
    reason: str = Field(description="One or two sentences explaining the verdict")
    verdict: Literal["CORRECT", "WRONG"]


class Judge:
    def __init__(self, settings: Settings) -> None:
        self._call = structured(
            chat_model(JUDGE_MODEL, settings, effort=JUDGE_EFFORT, max_tokens=JUDGE_MAX_TOKENS),
            JudgeVerdict,
        )

    async def __call__(
        self, *, question: str, expected: str, test_type: str, must_abstain: bool, answer: str
    ) -> JudgeVerdict:
        body = (
            f"Test type: {test_type}\n"
            f"Correct behavior is to abstain: {'yes' if must_abstain else 'no'}\n\n"
            f"<question>\n{question}\n</question>\n\n"
            f"<expected_answer>\n{expected}\n</expected_answer>\n\n"
            f"<actual_answer>\n{answer}\n</actual_answer>"
        )
        return await self._call([SystemMessage(_PROMPT), HumanMessage(body)])
