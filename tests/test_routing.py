"""Artjeck's slash-command router maps console input to an (action, argument)."""
import pytest

from secondbrain.agent import _route_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("/exit", ("exit", "")),
        ("/quit", ("exit", "")),
        ("/help", ("help", "")),
        ("/status", ("status", "")),
        ("/tasks", ("task_list", "open")),
        ("/tasks all", ("task_list", "all")),
        ("/learn I prefer PDF invoices", ("learn", "I prefer PDF invoices")),
        ("remember the gateway port is 4000", ("learn", "the gateway port is 4000")),
        ("/ingest ~/notes", ("ingest", "~/notes")),
        ("/task buy milk", ("task_add", "buy milk")),
        ("task: buy milk", ("task_add", "buy milk")),
        ("/done 7", ("task_done", "7")),
        ("what is my career goal?", ("ask", "what is my career goal?")),
    ],
)
def test_route_text(text, expected):
    assert _route_text(text) == expected


def test_plain_question_routes_to_ask_and_is_trimmed():
    assert _route_text("   how does routing work?  ") == ("ask", "how does routing work?")
