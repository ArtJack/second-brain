"""Read-only health checks for Artjeck's local lab and projects."""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from .config import cfg
from .project_context import discover_project_roots


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: str
    detail: str
    command: str = ""
    duration_ms: int = 0


def health_dir(root: Path | None = None) -> Path:
    return root or cfg.state_db.parent / "health"


def latest_health_report(root: Path | None = None) -> Path | None:
    base = health_dir(root)
    if not base.exists():
        return None
    reports = sorted(base.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def run_health(*, output_dir: Path | None = None, timeout_s: int = 45) -> dict:
    checks: list[HealthCheck] = []
    checks.extend(_service_checks(timeout_s=min(timeout_s, 10)))
    checks.extend(_project_checks(timeout_s=timeout_s))

    out_dir = health_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report = out_dir / f"{stamp}-health.md"
    markdown = render_health_report(stamp=stamp, checks=checks)
    report.write_text(markdown)
    return {
        "path": str(report),
        "checks": [check.__dict__ for check in checks],
        "passed": sum(1 for check in checks if check.status == "pass"),
        "failed": sum(1 for check in checks if check.status == "fail"),
        "skipped": sum(1 for check in checks if check.status == "skip"),
        "markdown": markdown,
    }


def render_health_report(*, stamp: str, checks: list[HealthCheck]) -> str:
    passed = sum(1 for check in checks if check.status == "pass")
    failed = sum(1 for check in checks if check.status == "fail")
    skipped = sum(1 for check in checks if check.status == "skip")
    lines = [
        f"# Health Report - {stamp}",
        "",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        f"- Skipped: {skipped}",
        "",
        "## Checks",
        "",
    ]
    for check in checks:
        lines.append(f"- {check.status.upper()} {check.name}: {check.detail}")
        if check.command:
            lines.append(f"  - Command: `{check.command}`")
        if check.duration_ms:
            lines.append(f"  - Duration: {check.duration_ms} ms")
    return "\n".join(lines).rstrip() + "\n"


def _service_checks(timeout_s: int) -> list[HealthCheck]:
    return [
        _http_check("ollama", _ollama_url(), timeout_s=timeout_s),
        _http_check("qdrant", cfg.qdrant_url.rstrip("/") + "/collections", timeout_s=timeout_s),
        _launchd_check("com.secondbrain.overnight"),
    ]


def _ollama_url() -> str:
    parsed = urlparse(cfg.base_url)
    base = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else cfg.base_url.rstrip("/v1")
    return base.rstrip("/") + "/api/tags"


def _http_check(name: str, url: str, *, timeout_s: int) -> HealthCheck:
    start = datetime.now()
    try:
        headers = {}
        if name == "qdrant" and cfg.qdrant_api_key:
            headers["api-key"] = cfg.qdrant_api_key
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = getattr(response, "status", 200)
            ok = 200 <= int(status) < 300
            detail = f"HTTP {status} at {url}"
            return HealthCheck(name=name, status="pass" if ok else "fail", detail=detail, duration_ms=_elapsed_ms(start))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return HealthCheck(name=name, status="fail", detail=f"{url}: {exc}", duration_ms=_elapsed_ms(start))


def _launchd_check(label: str) -> HealthCheck:
    res = _run(["launchctl", "list"], cwd=Path.home(), timeout_s=10)
    if res.returncode != 0:
        return HealthCheck(name="launchd nightly", status="fail", detail=res.output, command=res.command, duration_ms=res.duration_ms)
    loaded = label in res.output
    return HealthCheck(
        name="launchd nightly",
        status="pass" if loaded else "fail",
        detail=f"{label} {'loaded' if loaded else 'not loaded'}",
        command=res.command,
        duration_ms=res.duration_ms,
    )


def _project_checks(timeout_s: int) -> list[HealthCheck]:
    checks = []
    for root in discover_project_roots(limit=12):
        checks.append(_project_check(root, timeout_s=timeout_s))
    return checks


def _project_check(root: Path, *, timeout_s: int) -> HealthCheck:
    command = _project_command(root)
    if not command:
        return HealthCheck(name=f"project {root.name}", status="skip", detail=f"no safe test/build command found ({root})")
    res = _run(command, cwd=root, timeout_s=timeout_s)
    status = "pass" if res.returncode == 0 else "fail"
    detail = _summarize_output(res.output)
    return HealthCheck(
        name=f"project {root.name}",
        status=status,
        detail=detail,
        command=res.command,
        duration_ms=res.duration_ms,
    )


def _project_command(root: Path) -> list[str]:
    pyproject = root / "pyproject.toml"
    if pyproject.exists() and (root / "tests").exists():
        if (root / "uv.lock").exists():
            return ["uv", "run", "pytest"]
        return ["python3", "-m", "pytest"]
    package = root / "package.json"
    if package.exists():
        try:
            scripts = json.loads(package.read_text()).get("scripts", {})
        except Exception:
            scripts = {}
        if "test" in scripts:
            return ["npm", "test"]
        if "build" in scripts:
            return ["npm", "run", "build"]
    return []


@dataclass(frozen=True)
class _RunResult:
    command: str
    returncode: int
    output: str
    duration_ms: int


def _run(command: list[str], *, cwd: Path, timeout_s: int) -> _RunResult:
    start = datetime.now()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=_clean_env(),
        )
        output = "\n".join(part for part in [proc.stdout.strip(), proc.stderr.strip()] if part)
        return _RunResult(" ".join(command), proc.returncode, output, _elapsed_ms(start))
    except subprocess.TimeoutExpired as exc:
        output = "\n".join(
            part.decode() if isinstance(part, bytes) else part
            for part in [exc.stdout, exc.stderr]
            if part
        ).strip()
        return _RunResult(" ".join(command), 124, f"timed out after {timeout_s}s\n{output}".strip(), _elapsed_ms(start))
    except OSError as exc:
        return _RunResult(" ".join(command), 127, str(exc), _elapsed_ms(start))


def _elapsed_ms(start: datetime) -> int:
    return int((datetime.now() - start).total_seconds() * 1000)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    return env


def _summarize_output(output: str, limit: int = 500) -> str:
    text = " ".join(output.split())
    if not text:
        return "no output"
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."
