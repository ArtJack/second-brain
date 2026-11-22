"""Central configuration, loaded from .env.

Defaults point at a local Ollama directly (free, no key). Switch OPENAI_BASE_URL +
models to the LiteLLM gateway (.env.gateway.example) for the Claude-quality path —
the rest of the code doesn't change, because both speak the OpenAI API.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
import time
from pathlib import Path

from dotenv import dotenv_values, find_dotenv, load_dotenv


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load(path: str) -> tuple[dict[str, str], str | None]:
    """`load_dotenv()` at import, returning what it put into os.environ and a digest
    of the file, so that refresh() can later change exactly that and nothing the
    process inherited."""
    text = Path(path).read_text(encoding="utf-8") if path else None
    inherited = set(os.environ)
    load_dotenv(stream=io.StringIO(text or ""))
    from_file = {key: os.environ[key] for key in os.environ.keys() - inherited}
    return from_file, (_digest(text) if text is not None else None)


_ENV_FILE = find_dotenv()
_from_file, _stamp = _load(_ENV_FILE)

_ROOT = Path(__file__).resolve().parents[2]


class Config:
    def __init__(self) -> None:
        self.base_url: str = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        self.api_key: str = os.getenv("OPENAI_API_KEY", "ollama")
        self.embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.chat_model: str = os.getenv("CHAT_MODEL", "llama3.1:8b")
        self.vision_model: str = os.getenv("VISION_MODEL", "vision")
        self.store_backend: str = os.getenv("SB_STORE", "chroma").lower()
        self.persist_dir: Path = Path(os.getenv("SB_DATA") or (_ROOT / "data" / "chroma"))
        self.memory_dir: Path = Path(os.getenv("SB_MEMORY_DIR") or (_ROOT / "data" / "memory"))
        self.state_db: Path = Path(os.getenv("SB_STATE_DB") or (_ROOT / "data" / "artjeck.sqlite3"))
        self.collection: str = os.getenv("SB_COLLECTION", "second_brain")
        # Reference material — study texts, standards, other people's documentation —
        # kept apart from the owner's own notes. Mixed into one collection it wins on
        # volume: 2,935 of 4,483 chunks were ISTQB syllabus PDFs on 2026-09-12, and
        # they answered questions that were about this system.
        self.reference_collection: str = os.getenv("SB_REFERENCE_COLLECTION", "second_brain_reference")
        self.qdrant_url: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
        self.qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
        self.chunk_size: int = int(os.getenv("SB_CHUNK_SIZE", "1200"))
        self.chunk_overlap: int = int(os.getenv("SB_CHUNK_OVERLAP", "150"))
        self.top_k: int = int(os.getenv("SB_TOP_K", "5"))
        self.weather_location: str | None = os.getenv("SB_WEATHER_LOCATION") or None
        # The SDK default is 600 s with two retries. On a home lab reachable over a
        # tailnet, that is indistinguishable from a hang — and the MCP server that
        # waits on it is shared by every session on the network.
        self.llm_timeout_s: float = float(os.getenv("SB_LLM_TIMEOUT_S", "60"))
        # Roots the MCP ingest tool may read from. The tool is reachable over the
        # tailnet by anything holding SB_MCP_TOKEN, and paired with `recall` an
        # unrestricted ingest is an arbitrary-file-read primitive: ingest the
        # file, then ask for it back. Defaults to this repo and the memory
        # directory, which is everything the owner actually ingests by hand.
        roots = os.getenv("SB_INGEST_ROOTS", "")
        self.ingest_roots: list[str] = (
            [r for r in roots.split(os.pathsep) if r.strip()]
            if roots.strip()
            else [str(_ROOT), str(Path(os.getenv("SB_MEMORY_DIR") or (_ROOT / "data" / "memory")))]
        )
        self.hybrid_enabled: bool = os.getenv("SB_HYBRID", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }


cfg = Config()

# A file modified this recently may still be mid-write: `sops -d … > .env`
# truncates it first, and a half-written file would point a running server at
# the defaults. A later call reads it.
_SETTLE_S = 1.0
_lock = threading.Lock()


def refresh() -> bool:
    """Re-read .env if it changed since this process read it; True when cfg was updated.

    A process reads .env once, at import, and the MCP server lives as long as the
    session that started it. On 2026-10-16 fourteen open sessions were still
    sending a gateway key rotated three days before, or answering from a store
    that had moved. The rules are load_dotenv()'s: a variable the process
    inherited keeps winning over the file, and only what the file put into
    os.environ is updated, or removed when the file drops it. A missing or
    unreadable file keeps the current settings.
    """
    global _stamp
    if not _ENV_FILE:
        return False
    with _lock:
        try:
            # Read before stat: the mtime checked is then never older than the text.
            text = Path(_ENV_FILE).read_text(encoding="utf-8")
            modified = os.stat(_ENV_FILE).st_mtime
        except OSError:
            return False
        digest = _digest(text)
        if digest == _stamp or time.time() - modified < _SETTLE_S:
            return False
        values = {k: v for k, v in dotenv_values(stream=io.StringIO(text)).items() if v is not None}
        for key, value in list(_from_file.items()):
            if os.environ.get(key) != value:
                del _from_file[key]  # the process set it since: no longer the file's
            elif key not in values:
                del os.environ[key]
                del _from_file[key]
        for key, value in values.items():
            if key in _from_file or key not in os.environ:
                os.environ[key] = value
                _from_file[key] = value
        cfg.__dict__.update(Config().__dict__)
        _stamp = digest
        return True


def collections_for(corpus: str | None) -> list[str]:
    """Which collections a named corpus means, in the order they should be searched.

    Mirrors the web API's `corpus` field so the CLI, the MCP server and the HTTP
    layer all speak the same word. `None` is `personal`, so every existing caller
    keeps the exact behaviour it had before this existed.
    """
    name = (corpus or "personal").strip().lower()
    if name == "personal":
        return [cfg.collection]
    if name == "reference":
        return [cfg.reference_collection]
    if name == "all":
        return [cfg.collection, cfg.reference_collection]
    raise ValueError(f"unknown corpus {corpus!r}; expected one of: personal, reference, all")
