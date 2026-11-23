"""Configuration management for DocQA Assistant."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in minimal test environments
    load_dotenv = None


def resolve_resource_path(relative: str | Path) -> Path:
    """Resolve a bundled resource path for both source and PyInstaller frozen runs."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    llm_provider: str = "offline"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
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
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
            anthropic_model=os.getenv(
                "DOCQA_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"
            ).strip(),
            use_embeddings=os.getenv("DOCQA_USE_EMBEDDINGS", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            docs_dir=Path(os.getenv("DOCQA_DOCS_DIR", "./sample_docs")),
            index_path=Path(os.getenv("DOCQA_INDEX_PATH", "./.docqa_index.json")),
            chunk_size=max(100, int(os.getenv("DOCQA_CHUNK_SIZE", "700"))),
            chunk_overlap=max(0, int(os.getenv("DOCQA_CHUNK_OVERLAP", "120"))),
            top_k=max(1, int(os.getenv("DOCQA_TOP_K", "4"))),
        )

    @classmethod
    def for_desktop(cls, env_file: str | Path | None = None) -> "Settings":
        """Load settings with CWD-independent defaults suitable for desktop app launches."""
        base = cls.from_env(env_file=env_file)

        docs_dir = base.docs_dir
        if not os.getenv("DOCQA_DOCS_DIR") and not docs_dir.exists():
            bundled_docs = resolve_resource_path("sample_docs")
            if bundled_docs.exists():
                docs_dir = bundled_docs

        index_path = base.index_path
        if not os.getenv("DOCQA_INDEX_PATH") and not index_path.exists():
            app_dir = Path.home() / ".docqa"
            app_dir.mkdir(parents=True, exist_ok=True)
            index_path = app_dir / "index.json"

        return cls(
            llm_provider=base.llm_provider,
            gemini_api_key=base.gemini_api_key,
            gemini_model=base.gemini_model,
            openai_api_key=base.openai_api_key,
            openai_model=base.openai_model,
            anthropic_api_key=base.anthropic_api_key,
            anthropic_model=base.anthropic_model,
            use_embeddings=base.use_embeddings,
            docs_dir=docs_dir,
            index_path=index_path,
            chunk_size=base.chunk_size,
            chunk_overlap=base.chunk_overlap,
            top_k=base.top_k,
        )
