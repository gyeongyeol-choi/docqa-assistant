"""Hybrid document retriever combining BM25 lexical search and optional dense vectors."""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from docqa.loader import DocumentChunk

TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_]+")


def tokenize(text: str) -> list[str]:
    """Tokenize English/Korean text into lowercase terms and Hangul bigrams."""
    base_tokens = [tok.lower() for tok in TOKEN_RE.findall(text)]
    tokens = list(base_tokens)
    for tok in base_tokens:
        # Add character 2-grams for Korean tokens >= 3 chars to improve partial matching.
        if len(tok) >= 3 and any("\uac00" <= ch <= "\ud7a3" for ch in tok):
            tokens.extend(tok[i : i + 2] for i in range(len(tok) - 1))
    return tokens


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if not vec_a or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_text_gemini(text: str, api_key: str) -> list[float]:
    """Fetch a dense embedding vector from the Gemini Embeddings REST API."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/text-embedding-004:embedContent?key={api_key}"
    )
    payload = json.dumps(
        {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return [float(x) for x in body["embedding"]["values"]]


def embed_text_openai(text: str, api_key: str) -> list[float]:
    """Fetch a dense embedding vector from the OpenAI Embeddings REST API."""
    url = "https://api.openai.com/v1/embeddings"
    payload = json.dumps(
        {
            "model": "text-embedding-3-small",
            "input": text,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return [float(x) for x in body["data"][0]["embedding"]]


@dataclass(frozen=True)
class RetrievalHit:
    """A retrieved chunk along with its relevance score."""

    chunk: DocumentChunk
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0


class HybridRetriever:
    """In-memory hybrid retriever with JSON persistence."""

    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
        embeddings: list[list[float]] | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks: list[DocumentChunk] = chunks or []
        self.embeddings: list[list[float]] = embeddings or []
        self.embed_fn = embed_fn
        self.k1 = k1
        self.b = b

        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avg_len: float = 0.0
        self._build_bm25()

    def _build_bm25(self) -> None:
        self._doc_freqs = []
        self._doc_lens = []
        df: Counter[str] = Counter()

        for chunk in self.chunks:
            tokens = tokenize(f"{chunk.section} {chunk.text}")
            tf = Counter(tokens)
            self._doc_freqs.append(tf)
            self._doc_lens.append(len(tokens))
            df.update(tf.keys())

        n_docs = len(self.chunks)
        self._avg_len = (sum(self._doc_lens) / n_docs) if n_docs else 0.0
        self._idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Replace indexed chunks and optionally compute dense embeddings."""
        self.chunks = list(chunks)
        self.embeddings = []
        if self.embed_fn is not None and self.chunks:
            self.embeddings = [self.embed_fn(c.text) for c in self.chunks]
        self._build_bm25()

    def _bm25_scores(self, query: str) -> list[float]:
        if not self.chunks:
            return []
        q_tokens = tokenize(query)
        scores = [0.0] * len(self.chunks)
        avg_len = self._avg_len or 1.0

        for idx, tf in enumerate(self._doc_freqs):
            doc_len = self._doc_lens[idx]
            score = 0.0
            for term in q_tokens:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_len))
                score += idf * ((freq * (self.k1 + 1.0)) / denom)
            scores[idx] = score
        return scores

    def search(self, query: str, top_k: int = 4) -> list[RetrievalHit]:
        """Retrieve top-k chunks using BM25 or Reciprocal Rank Fusion (BM25 + Vector)."""
        if not self.chunks or not query.strip():
            return []

        bm25 = self._bm25_scores(query)
        has_vectors = (
            self.embed_fn is not None
            and len(self.embeddings) == len(self.chunks)
            and len(self.chunks) > 0
        )

        if not has_vectors:
            ranked = sorted(
                range(len(self.chunks)),
                key=lambda i: bm25[i],
                reverse=True,
            )
            return [
                RetrievalHit(
                    chunk=self.chunks[i],
                    score=bm25[i],
                    bm25_score=bm25[i],
                )
                for i in ranked[:top_k]
                if bm25[i] > 0.0
            ]

        q_vec = self.embed_fn(query)  # type: ignore[misc]
        vec_scores = [cosine_similarity(q_vec, emb) for emb in self.embeddings]

        bm25_order = sorted(range(len(self.chunks)), key=lambda i: bm25[i], reverse=True)
        vec_order = sorted(
            range(len(self.chunks)), key=lambda i: vec_scores[i], reverse=True
        )

        rrf_k = 60.0
        rrf_scores = [0.0] * len(self.chunks)
        for rank, idx in enumerate(bm25_order, start=1):
            if bm25[idx] > 0.0:
                rrf_scores[idx] += 1.0 / (rrf_k + rank)
        for rank, idx in enumerate(vec_order, start=1):
            if vec_scores[idx] > 0.0:
                rrf_scores[idx] += 1.0 / (rrf_k + rank)

        ranked = sorted(range(len(self.chunks)), key=lambda i: rrf_scores[i], reverse=True)
        return [
            RetrievalHit(
                chunk=self.chunks[i],
                score=rrf_scores[i],
                bm25_score=bm25[i],
                vector_score=vec_scores[i],
            )
            for i in ranked[:top_k]
            if rrf_scores[i] > 0.0
        ]

    def save(self, path: Path) -> None:
        """Persist indexed chunks and embeddings to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.to_dict() for c in self.chunks],
            "embeddings": self.embeddings,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: Path,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> "HybridRetriever":
        """Load a persisted index from disk."""
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk.from_dict(item) for item in data.get("chunks", [])]
        embeddings = data.get("embeddings", [])
        return cls(chunks=chunks, embeddings=embeddings, embed_fn=embed_fn)
