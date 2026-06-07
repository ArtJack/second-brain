"""Console display helpers for Artjeck/second-brain CLI output."""

from io import StringIO

from rich.console import Console

import secondbrain.cli as cli


def test_print_answer_can_hide_sources(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream, force_terminal=False, width=120))

    cli._print_answer(
        {
            "answer": "The context does not contain that.",
            "sources": [{"n": 1, "source": "/Users/artjack/private.md", "distance": 0.1}],
        },
        show_sources=False,
    )

    output = stream.getvalue()
    assert "The context does not contain that." in output
    assert "/Users/artjack/private.md" not in output
