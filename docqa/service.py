"""Core orchestration service for indexing documents and answering questions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from docqa.config import Settings
from docqa.llm import LLMClient, create_llm
from docqa.loader import load_documents
from docqa.retriever import (
    HybridRetriever,
    RetrievalHit,
    embed_text_gemini,
    embed_text_openai,
)


@dataclass(frozen=True)
class Citation:
    """Source citation returned alongside an answer."""

    index: int
    chunk_id: str
    source: str
    section: str
    score: float
    excerpt: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnswerResult:
    """Structured response from a Q&A query."""

    question: str
    answer: str
    provider: str
    citations: list[Citation]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "provider": self.provider,
            "citations": [c.to_dict() for c in self.citations],
        }


def _resolve_embed_fn(settings: Settings) -> Callable[[str], list[float]] | None:
    if not settings.use_embeddings:
        return None
    if settings.gemini_api_key:
        return lambda text: embed_text_gemini(text, settings.gemini_api_key)
    if settings.openai_api_key:
        return lambda text: embed_text_openai(text, settings.openai_api_key)
    return None


class DocQAService:
    """High-level service managing document index lifecycle and Q&A."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.llm = llm or create_llm(self.settings)
        self.embed_fn = _resolve_embed_fn(self.settings)
        self.retriever = self._load_or_init_retriever()

    def _load_or_init_retriever(self) -> HybridRetriever:
        if self.settings.index_path.exists():
            return HybridRetriever.load(
                self.settings.index_path,
                embed_fn=self.embed_fn,
            )
        return HybridRetriever(embed_fn=self.embed_fn)

    def index_directory(self, docs_dir: Path | None = None) -> dict[str, int | str]:
        """Load, chunk, and index all documents in `docs_dir`."""
        target = docs_dir or self.settings.docs_dir
        chunks = load_documents(
            target,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.retriever.index_chunks(chunks)
        self.retriever.save(self.settings.index_path)
        sources = {c.source for c in chunks}
        return {
            "docs_dir": str(target),
            "documents": len(sources),
            "chunks": len(chunks),
            "index_path": str(self.settings.index_path),
        }

    def list_sources(self) -> list[dict[str, int | str]]:
        """Summarize indexed documents and chunk counts."""
        counts: dict[str, int] = {}
        for chunk in self.retriever.chunks:
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        return [
            {"source": source, "chunks": count}
            for source, count in sorted(counts.items())
        ]

    def ask(self, question: str, top_k: int | None = None) -> AnswerResult:
        """Retrieve relevant chunks and generate a cited answer."""
        if not self.retriever.chunks and self.settings.docs_dir.exists():
            self.index_directory(self.settings.docs_dir)

        k = top_k or self.settings.top_k
        hits: list[RetrievalHit] = self.retriever.search(question, top_k=k)
        answer = self.llm.generate(question, hits)
        citations = [
            Citation(
                index=idx,
                chunk_id=hit.chunk.chunk_id,
                source=hit.chunk.source,
                section=hit.chunk.section,
                score=round(hit.score, 4),
                excerpt=hit.chunk.text[:240],
            )
            for idx, hit in enumerate(hits, start=1)
        ]
        return AnswerResult(
            question=question,
            answer=answer,
            provider=self.llm.provider_name,
            citations=citations,
        )
