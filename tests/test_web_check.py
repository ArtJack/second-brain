from datetime import UTC, datetime

import pytest

from secondbrain.web_check import WebPage, fetch_page, parse_content, render_web_note, save_web_note


def test_parse_content_extracts_title_and_skips_script():
    html = """
    <html>
      <head><title>Example Domain</title><script>secret()</script></head>
      <body><h1>Example Domain</h1><p>This domain is for examples.</p></body>
    </html>
    """

    title, text = parse_content(html, content_type="text/html")

    assert title == "Example Domain"
    assert "Example Domain" in text
    assert "This domain is for examples." in text
    assert "secret" not in text


def test_parse_content_plain_text():
    title, text = parse_content("hello\nworld", content_type="text/plain")

    assert title == ""
    assert text == "hello\nworld"


def test_fetch_page_rejects_non_http_url():
    with pytest.raises(ValueError):
        fetch_page("file:///etc/passwd")


def test_save_web_note_writes_source_metadata(tmp_path):
    page = WebPage(
        url="https://example.com",
        final_url="https://example.com/",
        status=200,
        content_type="text/html",
        title="Example Domain",
        text="Example text",
        fetched_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )

    path = save_web_note(page, output_dir=tmp_path)
    text = path.read_text()

    assert path.name.startswith("example-domain-")
    assert "# Web Check: Example Domain" in text
    assert "- Source URL: https://example.com" in text
    assert "Example text" in text


def test_render_web_note_handles_empty_text():
    page = WebPage(
        url="https://example.com",
        final_url="https://example.com/",
        status=200,
        content_type="text/html",
        title="Example Domain",
        text="",
        fetched_at="2026-06-04T00:00:00+00:00",
    )

    assert "(no readable text extracted)" in render_web_note(page)
