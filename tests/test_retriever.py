"""Tests for BM25 and hybrid RRF retrieval."""

from pathlib import Path
import tempfile
import unittest

from docqa.loader import DocumentChunk
from docqa.retriever import HybridRetriever


class TestRetriever(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            DocumentChunk("1", "arch.md", "Pipeline", 0, "BM25와 벡터 검색을 RRF로 결합합니다."),
            DocumentChunk("2", "deploy.md", "Docker", 0, "포트 8000으로 FastAPI 서버를 실행합니다."),
        ]

    def test_bm25_search_ranks_relevant_chunk_first(self) -> None:
        retriever = HybridRetriever()
        retriever.index_chunks(self.chunks)
        hits = retriever.search("FastAPI Docker 포트", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.source, "deploy.md")

    def test_hybrid_rrf_and_persistence(self) -> None:
        def fake_embed(text: str) -> list[float]:
            return [1.0, 0.0] if "RRF" in text or "하이브리드" in text else [0.0, 1.0]

        retriever = HybridRetriever(embed_fn=fake_embed)
        retriever.index_chunks(self.chunks)
        hits = retriever.search("하이브리드", top_k=2)
        self.assertEqual(hits[0].chunk.source, "arch.md")

        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "index.json"
            retriever.save(index_path)
            loaded = HybridRetriever.load(index_path, embed_fn=fake_embed)
            self.assertEqual(len(loaded.chunks), 2)
            self.assertEqual(loaded.search("Docker", top_k=1)[0].chunk.source, "deploy.md")


if __name__ == "__main__":
    unittest.main()
