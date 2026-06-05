"""Real-browser page capture for JavaScript-rendered sites."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_BROWSER_DIR = Path("/Volumes/DISK/AI/artjeck/inbox/browser")


@dataclass(frozen=True)
class BrowserPage:
    url: str
    final_url: str
    title: str
    text: str
    fetched_at: str
    screenshot: Path | None = None


def capture_url(
    url: str,
    *,
    output_dir: Path = DEFAULT_BROWSER_DIR,
    timeout_ms: int = 30000,
    wait_until: str = "networkidle",
    screenshot: bool = True,
) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("browser-check only supports absolute http(s) URLs")

    output_dir.mkdir(parents=True, exist_ok=True)
    page = _capture_page(url, output_dir=output_dir, timeout_ms=timeout_ms, wait_until=wait_until, screenshot=screenshot)
    note = save_browser_note(page, output_dir=output_dir)
    return {"page": page, "path": note}


def _capture_page(
    url: str,
    *,
    output_dir: Path,
    timeout_ms: int,
    wait_until: str,
    screenshot: bool,
) -> BrowserPage:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is not installed. Run `uv sync` and try again.") from exc

    with sync_playwright() as p:
        browser = _launch_browser(p)
        try:
            context = browser.new_context(
                viewport={"width": 1365, "height": 900},
                user_agent="ArtjeckSecondBrain/0.1 (+read-only browser check)",
            )
            page = context.new_page()
            page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            page.wait_for_timeout(1000)
            title = page.title()
            text = page.locator("body").inner_text(timeout=timeout_ms)
            final_url = page.url
            screenshot_path = None
            if screenshot:
                screenshot_path = output_dir / f"{_slug(title or final_url)}-{_url_hash(final_url)}.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
        finally:
            browser.close()
    return BrowserPage(
        url=url,
        final_url=final_url,
        title=_clean_text(title or final_url, limit=180),
        text=_clean_text(text, limit=16000),
        fetched_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        screenshot=screenshot_path,
    )


def _launch_browser(playwright):
    try:
        return playwright.chromium.launch(channel="chrome", headless=True)
    except Exception:
        return playwright.chromium.launch(headless=True)


def save_browser_note(page: BrowserPage, *, output_dir: Path = DEFAULT_BROWSER_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    note = output_dir / f"{_slug(page.title)}-{_url_hash(page.final_url)}.md"
    note.write_text(render_browser_note(page))
    return note


def render_browser_note(page: BrowserPage) -> str:
    screenshot_line = f"- Screenshot: {page.screenshot}" if page.screenshot else "- Screenshot: not captured"
    return "\n".join(
        [
            f"# Browser Check: {page.title}",
            "",
            f"- Source URL: {page.url}",
            f"- Final URL: {page.final_url}",
            f"- Fetched: {page.fetched_at}",
            screenshot_line,
            "",
            "## Visible Page Text",
            "",
            page.text or "(no visible text extracted)",
            "",
        ]
    )


def _clean_text(text: str, *, limit: int) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug[:80] or "browser-page"


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
