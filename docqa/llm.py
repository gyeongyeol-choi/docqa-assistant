"""LLM provider abstractions for grounded Q&A generation."""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol

from docqa.config import Settings
from docqa.retriever import RetrievalHit


class LLMClient(Protocol):
    """Protocol for answer generation backends."""

    provider_name: str

    def generate(self, question: str, hits: list[RetrievalHit]) -> str:
        """Generate a grounded answer from retrieved document hits."""


def build_grounded_prompt(question: str, hits: list[RetrievalHit]) -> str:
    """Build a strict RAG prompt instructing the model to cite sources."""
    context_blocks = []
    for idx, hit in enumerate(hits, start=1):
        c = hit.chunk
        context_blocks.append(
            f"[{idx}] Source: {c.source} | Section: {c.section}\n{c.text}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)
    return (
        "You are a precise technical documentation assistant.\n"
        "Answer the user's question using ONLY the provided context snippets.\n"
        "Cite your sources inline using bracketed numbers like [1], [2].\n"
        "If the context does not contain enough information, state clearly what is missing.\n"
        "Reply in the same language as the user's question.\n\n"
        f"## Context\n{context_text}\n\n"
        f"## Question\n{question}\n\n"
        "## Answer"
    )


class OfflineExtractiveLLM:
    """Zero-dependency fallback that summarizes retrieved passages with citations."""

    provider_name = "offline"

    def generate(self, question: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "관련 문서를 찾지 못했습니다. 문서를 다시 인덱싱하거나 다른 키워드로 질문해 주세요."

        lines = [
            f"질문(\"{question}\")과 가장 관련도가 높은 문서 근거를 정리했습니다 (오프라인 추출 모드):",
            "",
        ]
        for idx, hit in enumerate(hits, start=1):
            snippet = " ".join(hit.chunk.text.splitlines()).strip()
            if len(snippet) > 260:
                snippet = snippet[:257].rstrip() + "..."
            lines.append(
                f"- [{idx}] **{hit.chunk.source}** (`{hit.chunk.section}`): {snippet}"
            )
        lines.append("")
        lines.append(
            "💡 `.env`에서 `DOCQA_LLM_PROVIDER=gemini` 또는 `openai`와 API 키를 설정하면 자연어 종합 답변을 생성합니다."
        )
        return "\n".join(lines)


class GeminiLLM:
    """Gemini REST API client."""

    provider_name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when DOCQA_LLM_PROVIDER=gemini.")
        self.api_key = api_key
        self.model = model

    def generate(self, question: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "관련 문서를 찾지 못했습니다."
        prompt = build_grounded_prompt(question, hits)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent?key={self.api_key}"
        )
        payload = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        candidates = body.get("candidates", [])
        if not candidates:
            return "모델 응답이 비어 있습니다."
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip()


class OpenAILLM:
    """OpenAI Chat Completions REST API client."""

    provider_name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when DOCQA_LLM_PROVIDER=openai.")
        self.api_key = api_key
        self.model = model

    def generate(self, question: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "관련 문서를 찾지 못했습니다."
        prompt = build_grounded_prompt(question, hits)
        url = "https://api.openai.com/v1/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


def create_llm(settings: Settings) -> LLMClient:
    """Instantiate the configured LLM client."""
    if settings.llm_provider == "gemini":
        return GeminiLLM(api_key=settings.gemini_api_key, model=settings.gemini_model)
    if settings.llm_provider == "openai":
        return OpenAILLM(api_key=settings.openai_api_key, model=settings.openai_model)
    return OfflineExtractiveLLM()
