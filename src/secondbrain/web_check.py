"""Read-only web page capture for source-backed second-brain notes."""
from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_WEB_DIR = Path("/Volumes/DISK/AI/artjeck/inbox/web")
MAX_BYTES = 2_000_000
USER_AGENT = "ArtjeckSecondBrain/0.1 (+read-only web check)"


@dataclass(frozen=True)
class WebPage:
    url: str
    final_url: str
    status: int
    content_type: str
    title: str
    text: str
    fetched_at: str


class ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.text_parts.append(text)

    @property
    def title(self) -> str:
        return _clean_text(" ".join(self.title_parts), limit=180)

    @property
    def text(self) -> str:
        return _clean_text(" ".join(self.text_parts), limit=12000)


def fetch_page(url: str, *, timeout_s: int = 20, max_bytes: int = MAX_BYTES) -> WebPage:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("web-check only supports absolute http(s) URLs")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raw = raw[:max_bytes]
            content_type = response.headers.get("content-type", "")
            final_url = response.geturl()
            status = int(getattr(response, "status", 200))
    except urllib.error.HTTPError as exc:
        raw = exc.read(max_bytes)
        content_type = exc.headers.get("content-type", "")
        final_url = exc.geturl()
        status = int(exc.code)
    text = _decode(raw, content_type)
    title, readable = parse_content(text, content_type=content_type)
    return WebPage(
        url=url,
        final_url=final_url,
        status=status,
        content_type=content_type,
        title=title or final_url,
        text=readable,
        fetched_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )


def parse_content(content: str, *, content_type: str = "") -> tuple[str, str]:
    if "html" not in content_type.lower() and "<html" not in content[:500].lower():
        return "", _clean_text(content, limit=12000)
    parser = ReadableHTMLParser()
    parser.feed(content)
    return parser.title, parser.text


def save_web_note(page: WebPage, *, output_dir: Path = DEFAULT_WEB_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_slug(page.title)}-{_url_hash(page.final_url)}.md"
    path.write_text(render_web_note(page))
    return path


def render_web_note(page: WebPage) -> str:
    return "\n".join(
        [
            f"# Web Check: {page.title}",
            "",
            f"- Source URL: {page.url}",
            f"- Final URL: {page.final_url}",
            f"- HTTP status: {page.status}",
            f"- Content type: {page.content_type or 'unknown'}",
            f"- Fetched: {page.fetched_at}",
            "",
            "## Page Text",
            "",
            page.text or "(no readable text extracted)",
            "",
        ]
    )


def check_url(url: str, *, output_dir: Path = DEFAULT_WEB_DIR, timeout_s: int = 20) -> dict:
    page = fetch_page(url, timeout_s=timeout_s)
    note = save_web_note(page, output_dir=output_dir)
    return {"page": page, "path": note}


def _decode(raw: bytes, content_type: str) -> str:
    match = re.search(r"charset=([^;\s]+)", content_type, re.I)
    encodings = [match.group(1)] if match else []
    encodings.extend(["utf-8", "latin-1"])
    for encoding in encodings:
        try:
            return raw.decode(encoding, errors="replace")
        except LookupError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean_text(text: str, *, limit: int) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug[:80] or "web-page"


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
