from datetime import UTC, datetime

import pytest

from secondbrain.browser_check import BrowserPage, capture_url, render_browser_note, save_browser_note


def test_capture_url_rejects_non_http_url(tmp_path):
    with pytest.raises(ValueError):
        capture_url("file:///etc/passwd", output_dir=tmp_path)


def test_save_browser_note_writes_metadata(tmp_path):
    page = BrowserPage(
        url="https://example.com",
        final_url="https://example.com/",
        title="Example Domain",
        text="Rendered text",
        fetched_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        screenshot=tmp_path / "example.png",
    )

    note = save_browser_note(page, output_dir=tmp_path)
    text = note.read_text()

    assert note.name.startswith("example-domain-")
    assert "# Browser Check: Example Domain" in text
    assert "- Source URL: https://example.com" in text
    assert "- Screenshot:" in text
    assert "Rendered text" in text


def test_render_browser_note_handles_empty_text():
    page = BrowserPage(
        url="https://example.com",
        final_url="https://example.com/",
        title="Example Domain",
        text="",
        fetched_at="2026-06-04T00:00:00+00:00",
        screenshot=None,
    )

    assert "(no visible text extracted)" in render_browser_note(page)
