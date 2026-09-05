"""Central configuration for the NSE Intelligence system.

Every module reads its settings from here instead of calling os.environ
directly. That gives us one place to document each knob, one place to apply
defaults, and makes it trivial to override settings in tests.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# The project root is the folder containing this file. Resolving paths
# relative to it means every entry point (Streamlit, MCP server, scheduler)
# sees the same chroma_db/ and logs/ no matter which directory launched it.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "gemma4:e4b"))
    ollama_embed_model: str = field(default_factory=lambda: _env("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    ollama_num_ctx: int = field(default_factory=lambda: int(_env("OLLAMA_NUM_CTX", "8192")))

    chroma_persist_dir: Path = field(default_factory=lambda: _resolve(_env("CHROMA_PERSIST_DIR", "./chroma_db")))
    chroma_collection: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "nse_documents"))

    rag_refresh_interval_minutes: int = field(default_factory=lambda: int(_env("RAG_REFRESH_INTERVAL_MINUTES", "5")))
    nse_session_refresh_minutes: int = field(default_factory=lambda: int(_env("NSE_SESSION_REFRESH_MINUTES", "15")))
    max_items_per_feed: int = field(default_factory=lambda: int(_env("MAX_ITEMS_PER_FEED", "300")))

    # When a question names a known symbol, pre-fetch quote/filings/RAG chunks
    # before the agent runs so small models still ground every answer in data.
    agent_auto_context: bool = field(default_factory=lambda: _env("AGENT_AUTO_CONTEXT", "true").lower() == "true")

    log_dir: Path = field(default_factory=lambda: _resolve(_env("LOG_DIR", "./logs")))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # Chunking targets from the spec: ~512 tokens with ~50 token overlap.
    # We measure in characters (roughly 4 chars per token) so the splitter
    # never needs to download a tokenizer vocabulary - the system stays offline.
    chunk_size_chars: int = 2000
    chunk_overlap_chars: int = 200


settings = Settings()


def configure_logging(component: str) -> None:
    """Route loguru output to stderr and to a per-component rotating log file."""
    from loguru import logger

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    # stderr, never stdout: the MCP server uses stdout as its protocol channel.
    logger.add(sys.stderr, level=settings.log_level)
    logger.add(
        settings.log_dir / f"{component}.log",
        level=settings.log_level,
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
    )
