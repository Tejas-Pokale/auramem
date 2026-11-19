"""Prompt pieces shared by the quick answer and deep mode: the cached system prefix and dates."""

from datetime import datetime
from typing import Any

from langchain_core.messages import SystemMessage

from memory.prompts import load_prompt


def format_today(now: datetime) -> str:
    return f"{now:%A, %d %B %Y, %H:%M} ({now.tzname() or 'UTC'})"


def system_message(version: str, profile: str, *, name: str = "answer") -> SystemMessage:
    """Frozen instructions, then the user's profile; the cache breakpoint covers both."""
    blocks: list[str | dict[str, Any]] = [
        {"type": "text", "text": load_prompt(name, version, "system").text}
    ]
    if profile.strip():
        blocks.append({"type": "text", "text": f"<user_profile>\n{profile}\n</user_profile>"})
    last = blocks[-1]
    assert isinstance(last, dict)
    last["cache_control"] = {"type": "ephemeral"}
    return SystemMessage(content=blocks)
