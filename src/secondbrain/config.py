"""Central configuration, loaded from .env.

Defaults point at a local Ollama directly (free, no key). Switch OPENAI_BASE_URL +
models to the LiteLLM gateway (.env.gateway.example) for the Claude-quality path —
the rest of the code doesn't change, because both speak the OpenAI API.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    """A path setting from the environment, with a relative value anchored to this checkout.

    `.env` writes storage paths as `./data/...`. Taken as given, they followed the
    working directory, so a process started in another folder opened an empty task
    database and keyword index there, and said nothing. `_ROOT` is the package's
    own checkout, so a worktree keeps its own `data/`. An absolute value is used as
    given: that is how worktree code is pointed at production state.
    """
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else _ROOT / path


class Config:
    def __init__(self) -> None:
        self.base_url: str = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        self.api_key: str = os.getenv("OPENAI_API_KEY", "ollama")
        self.embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.chat_model: str = os.getenv("CHAT_MODEL", "llama3.1:8b")
        self.vision_model: str = os.getenv("VISION_MODEL", "vision")
        self.store_backend: str = os.getenv("SB_STORE", "chroma").lower()
        self.persist_dir: Path = _env_path("SB_DATA", _ROOT / "data" / "chroma")
        self.memory_dir: Path = _env_path("SB_MEMORY_DIR", _ROOT / "data" / "memory")
        self.state_db: Path = _env_path("SB_STATE_DB", _ROOT / "data" / "artjeck.sqlite3")
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
            else [str(_ROOT), str(self.memory_dir)]
        )
        self.hybrid_enabled: bool = os.getenv("SB_HYBRID", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }


cfg = Config()


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
