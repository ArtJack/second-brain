"""Local machine status helpers for Artjeck's agent loop."""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
from pathlib import Path

from .config import cfg


def _run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()


def _bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024 or unit == "PiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"


def memory_status() -> dict:
    if platform.system() == "Darwin":
        total = int(_run(["sysctl", "-n", "hw.memsize"]))
        vm_stat = _run(["vm_stat"])
        page_size_match = re.search(r"page size of (\d+) bytes", vm_stat)
        page_size = int(page_size_match.group(1)) if page_size_match else 4096
        pages: dict[str, int] = {}
        for line in vm_stat.splitlines():
            match = re.match(r"(.+?):\s+(\d+)\.", line)
            if match:
                pages[match.group(1).strip()] = int(match.group(2))
        available_pages = (
            pages.get("Pages free", 0)
            + pages.get("Pages inactive", 0)
            + pages.get("Pages speculative", 0)
        )
        available = available_pages * page_size
        used = max(total - available, 0)
        return {"total": total, "available": available, "used": used, "percent_used": used / total}

    if platform.system() == "Linux":
        data: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, value = line.split(":", 1)
            data[name] = int(value.strip().split()[0]) * 1024
        total = data.get("MemTotal")
        available = data.get("MemAvailable")
        if total and available is not None:
            used = max(total - available, 0)
            return {"total": total, "available": available, "used": used, "percent_used": used / total}

    return {"total": None, "available": None, "used": None, "percent_used": None}


def _default_storage_paths() -> list[Path]:
    candidates = [
        Path("/"),
        cfg.memory_dir,
        cfg.state_db.parent,
        cfg.persist_dir,
        Path("/Volumes/DISK"),
    ]
    paths: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        target = path if path.exists() else path.parent
        try:
            resolved = str(target.resolve())
        except OSError:
            continue
        if resolved not in seen and target.exists():
            paths.append(Path(resolved))
            seen.add(resolved)
    return paths


def storage_status(paths: list[Path] | None = None) -> list[dict]:
    rows = []
    for path in paths or _default_storage_paths():
        usage = shutil.disk_usage(path)
        rows.append(
            {
                "path": str(path),
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent_used": usage.used / usage.total if usage.total else None,
            }
        )
    return rows


def format_system_status(kind: str = "all") -> str:
    kind = kind or "all"
    lines: list[str] = []
    if kind in {"all", "memory"}:
        mem = memory_status()
        percent = "unknown" if mem["percent_used"] is None else f"{mem['percent_used']:.1%}"
        lines.extend(
            [
                "Memory:",
                f"  total     : {_bytes(mem['total'])}",
                f"  used      : {_bytes(mem['used'])} ({percent})",
                f"  available : {_bytes(mem['available'])}",
            ]
        )
    if kind in {"all", "storage"}:
        if lines:
            lines.append("")
        lines.append("Storage:")
        for row in storage_status():
            percent = "unknown" if row["percent_used"] is None else f"{row['percent_used']:.1%}"
            lines.append(
                f"  {row['path']}: {_bytes(row['free'])} free / {_bytes(row['total'])} total ({percent} used)"
            )
    return "\n".join(lines)
