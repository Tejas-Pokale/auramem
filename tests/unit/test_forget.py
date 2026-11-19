from datetime import UTC, datetime

import pytest

from memory.forget import CANCEL, CONFIRM, forget_request
from memory.models import ForgetItem, ForgetPreview, ForgetReport


@pytest.mark.parametrize(
    ("message", "what"),
    [
        ("Please forget that my sister is pregnant.", "my sister is pregnant"),
        ("Can you forget about my old flat?", "my old flat"),
        ("Delete what I told you about my salary from memory", "my salary"),
        ("erase everything about my old job", "my old job"),
        ("I want you to forget my old address", "my old address"),
        ("I forgot my keys at home", None),
        ("What did I forget to pack for the trip?", None),
        ("Don't forget my mom's birthday", None),
        ("Remove the background from this photo", None),
    ],
)
def test_forget_request(message: str, what: str | None) -> None:
    assert forget_request(message) == what


@pytest.mark.parametrize("message", ["yes", "Yes please", "ok", "go ahead", "delete them"])
def test_confirm(message: str) -> None:
    assert CONFIRM.match(message) and not CANCEL.match(message)


@pytest.mark.parametrize("message", ["no", "No, keep it", "cancel", "don't"])
def test_cancel(message: str) -> None:
    assert CANCEL.match(message) and not CONFIRM.match(message)


def test_preview_lists_items_and_asks_permission() -> None:
    at = datetime(2026, 3, 1, tzinfo=UTC)
    preview = ForgetPreview(
        preview_id="p1",
        description="my old flat",
        items=[
            ForgetItem(kind="fact", item_id="f1", text="The user rented a flat.", date=at),
            ForgetItem(kind="turn", item_id="t1", text="My flat is small.", date=at, role="user"),
        ],
    )
    text = preview.render()
    assert "1. [2026-03-01] The user rented a flat." in text
    assert "2. [2026-03-01] User said: My flat is small." in text
    assert "permanently delete" in text and "can't be undone" in text
    assert (preview.fact_ids, preview.turn_ids) == (["f1"], ["t1"])


def test_report_message() -> None:
    assert ForgetReport(facts=2, turns=3, profile_rewritten=True).render() == (
        "Done. I permanently deleted 2 memories and 3 messages, and removed it from your profile."
    )
