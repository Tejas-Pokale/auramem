from datetime import UTC, datetime
from typing import Any

from memory.graphs.read import system_message
from memory.models import Observation
from memory.observe import _kept_high_priority, format_observations, high_priority_lines

LOG = """- [2026-01-05] (high) Prefers concise bullet answers with the recommendation first.
- [2026-02-10] (medium) Training for a half marathon.
- [2026-03-01] (high) Mother has type 2 diabetes; user tracks her readings.
- [2026-03-04] (low) Tried a new ramen place."""


def test_format_observations_is_one_dated_bullet_each() -> None:
    obs = [
        Observation(
            date=datetime(2026, 4, 2, tzinfo=UTC), priority="high", text="Moved   to Leeds."
        ),
        Observation(date=datetime(2026, 4, 3, tzinfo=UTC), priority="low", text="   "),
    ]
    assert format_observations(obs) == ["- [2026-04-02] (high) Moved to Leeds."]


def test_high_priority_lines() -> None:
    assert len(high_priority_lines(LOG)) == 2


HIGH_LOG = """- [2025-02-11] (high) Father Daniel had bypass surgery at Presbyterian Hospital.
- [2025-04-08] (high) Dr. Abramowitz found benign PVCs; annual checkups only.
- [2025-06-01] (high) Daniel finished cardiac rehab and is fishing again.
- [2025-03-01] (medium) Salary rose to $111,500."""


def test_merging_under_the_newest_date_is_allowed() -> None:
    merged = (
        "- [2025-06-01] (high) Father Daniel had bypass surgery at Presbyterian Hospital "
        "(2025-02), finished cardiac rehab and is fishing again.\n"
        "- [2025-04-08] (high) Dr. Abramowitz: benign PVCs, annual checkups only."
    )
    assert _kept_high_priority(HIGH_LOG, merged, min_coverage=0.8)


def test_dropping_high_priority_substance_is_rejected() -> None:
    dropped = "- [2025-06-01] (high) Daniel is fishing again.\n- [2025-03-01] (medium) $111,500."
    assert not _kept_high_priority(HIGH_LOG, dropped, min_coverage=0.8)


def test_low_priority_content_may_go() -> None:
    kept = (
        "- [2025-06-01] (high) Father Daniel: bypass at Presbyterian Hospital, cardiac rehab "
        "done, fishing again.\n- [2025-04-08] (high) Dr. Abramowitz: benign PVCs, annual "
        "checkups only."
    )
    assert _kept_high_priority(HIGH_LOG, kept, min_coverage=0.8)


def _blocks(message: Any) -> list[dict[str, Any]]:
    blocks = message.content
    assert isinstance(blocks, list)
    return blocks


def test_system_message_caches_instructions_plus_profile() -> None:
    blocks = _blocks(system_message("v5", "- [2026-01-05] (high) Prefers bullets."))
    assert len(blocks) == 2
    assert "<user_profile>" in blocks[1]["text"]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[0]


def test_system_message_without_profile_still_caches_instructions() -> None:
    [only] = _blocks(system_message("v5", ""))
    assert only["cache_control"] == {"type": "ephemeral"}
