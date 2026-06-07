from __future__ import annotations

from types import SimpleNamespace


def _chunk(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def test_answer_stream_yields_text_deltas(monkeypatch):
    import secondbrain.llm as llm

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            assert kwargs["temperature"] == 0.1
            return [_chunk("hello "), _chunk("world"), _chunk("")]

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(llm, "_client", fake_client)

    assert "".join(llm.answer_stream("q", "[1] context")) == "hello world"
