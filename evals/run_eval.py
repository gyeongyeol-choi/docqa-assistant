"""Lightweight retrieval evaluation script (Hit@K and MRR)."""

from __future__ import annotations

import json
from pathlib import Path

from docqa.loader import load_documents
from docqa.retriever import HybridRetriever


def evaluate(top_k: int = 3) -> None:
    root = Path(__file__).resolve().parent.parent
    docs_dir = root / "sample_docs"
    eval_path = Path(__file__).resolve().parent / "qa_eval.json"

    chunks = load_documents(docs_dir)
    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    dataset = json.loads(eval_path.read_text(encoding="utf-8"))
    hits = 0
    mrr_total = 0.0

    for item in dataset:
        results = retriever.search(item["question"], top_k=top_k)
        sources = [r.chunk.source for r in results]
        expected = item["expected_source"]
        if expected in sources:
            hits += 1
            rank = sources.index(expected) + 1
            mrr_total += 1.0 / rank
            status = f"PASS (rank={rank})"
        else:
            status = "FAIL"
        print(f"[{status}] {item['question']} -> {sources}")

    n = len(dataset)
    print(f"\nSummary: Hit@{top_k} = {hits}/{n} ({hits / n:.1%}), MRR = {mrr_total / n:.3f}")


if __name__ == "__main__":
    evaluate()
