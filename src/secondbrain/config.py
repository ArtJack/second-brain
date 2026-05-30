"""Central configuration, loaded from .env.

Defaults point at the GTX Ollama directly (free, no key). Switch OPENAI_BASE_URL +
models to the LiteLLM gateway (.env.gateway.example) for the Claude-quality path —
the rest of the code doesn't change, because both speak the OpenAI API.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]


class Config:
    def __init__(self) -> None:
        self.base_url: str = os.getenv("OPENAI_BASE_URL", "http://192.168.1.159:11434/v1")
        self.api_key: str = os.getenv("OPENAI_API_KEY", "ollama")
        self.embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.chat_model: str = os.getenv("CHAT_MODEL", "llama3.1:8b")
        self.persist_dir: Path = Path(os.getenv("SB_DATA") or (_ROOT / "data" / "chroma"))
        self.collection: str = os.getenv("SB_COLLECTION", "second_brain")
        self.chunk_size: int = int(os.getenv("SB_CHUNK_SIZE", "1200"))
        self.chunk_overlap: int = int(os.getenv("SB_CHUNK_OVERLAP", "150"))
        self.top_k: int = int(os.getenv("SB_TOP_K", "5"))


cfg = Config()
