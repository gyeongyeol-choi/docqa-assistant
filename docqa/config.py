"""Configuration management for DocQA Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in minimal test environments
    load_dotenv = None


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    llm_provider: str = "offline"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    use_embeddings: bool = False
    docs_dir: Path = Path("./sample_docs")
    index_path: Path = Path("./.docqa_index.json")
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        """Load settings from `.env` (if present) and process environment."""
        if load_dotenv is not None:
            load_dotenv(dotenv_path=env_file, override=False)

        return cls(
            llm_provider=os.getenv("DOCQA_LLM_PROVIDER", "offline").strip().lower(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("DOCQA_GEMINI_MODEL", "gemini-2.5-flash").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("DOCQA_OPENAI_MODEL", "gpt-4o-mini").strip(),
            use_embeddings=os.getenv("DOCQA_USE_EMBEDDINGS", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            docs_dir=Path(os.getenv("DOCQA_DOCS_DIR", "./sample_docs")),
            index_path=Path(os.getenv("DOCQA_INDEX_PATH", "./.docqa_index.json")),
            chunk_size=max(100, int(os.getenv("DOCQA_CHUNK_SIZE", "700"))),
            chunk_overlap=max(0, int(os.getenv("DOCQA_CHUNK_OVERLAP", "120"))),
            top_k=max(1, int(os.getenv("DOCQA_TOP_K", "4"))),
        )
