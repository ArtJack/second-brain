"""Learned-memory file writing (no embedding/network — write_memory only)."""
from secondbrain import memory


def test_slug_normalizes_and_falls_back():
    assert memory._slug("My Preferred Invoice Format!") == "my-preferred-invoice-format"
    assert memory._slug("") == "memory"
    assert memory._slug("!!!") == "memory"


def test_write_memory_creates_inspectable_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.cfg, "memory_dir", tmp_path)

    path = memory.write_memory("the gateway listens on port 4000", source="user")

    assert path.exists()
    assert path.parent == tmp_path
    body = path.read_text(encoding="utf-8")
    assert "the gateway listens on port 4000" in body
    assert "type: learned_memory" in body
    assert "source: user" in body
