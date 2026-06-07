"""Artjeck's slash-command router maps console input to an (action, argument)."""
import pytest

from secondbrain.agent import _route_text, run_turn


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
        ("/task", ("task_add", "")),
        ("/task buy milk", ("task_add", "buy milk")),
        ("task: buy milk", ("task_add", "buy milk")),
        ("/done", ("task_done", "")),
        ("/done 7", ("task_done", "7")),
        ("check system memory", ("system_status", "memory")),
        ("whats the storage space?", ("system_status", "storage")),
        ("/system", ("system_status", "all")),
        ("whats the weather?", ("weather", "")),
        ("/weather Seattle", ("weather", "Seattle")),
        ("weather in Portland, OR", ("weather", "Portland, OR")),
        ("whats the weather in Athens?", ("weather", "Athens")),
        ("how are you?", ("chat", "how are you")),
        ("hello", ("chat", "hello")),
        ("what is my career goal?", ("ask", "what is my career goal?")),
    ],
)
def test_route_text(text, expected):
    assert _route_text(text) == expected


def test_plain_question_routes_to_ask_and_is_trimmed():
    assert _route_text("   how does routing work?  ") == ("ask", "how does routing work?")


def test_small_talk_gets_direct_agent_reply():
    result = run_turn("how are you?")

    assert result["action"] == "chat"
    assert "ready" in result["answer"]
    assert "sources" not in result


def test_weather_gets_direct_agent_reply(monkeypatch):
    import secondbrain.agent as agent

    monkeypatch.setattr(agent, "format_weather", lambda location: f"weather for {location or 'default'}")

    result = run_turn("whats the weather in Athens?")

    assert result["action"] == "weather"
    assert result["answer"] == "weather for Athens"
    assert "sources" not in result
