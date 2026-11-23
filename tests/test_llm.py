"""Unit tests for LLM providers and factory."""

import json
import unittest
from unittest.mock import MagicMock, patch

from docqa.config import Settings
from docqa.llm import ClaudeLLM, create_llm
from docqa.loader import DocumentChunk
from docqa.retriever import RetrievalHit


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class TestClaudeLLM(unittest.TestCase):
    def test_create_llm_claude_requires_api_key(self) -> None:
        settings = Settings(llm_provider="claude", anthropic_api_key="")
        with self.assertRaises(ValueError):
            create_llm(settings)

    def test_create_llm_claude_returns_client(self) -> None:
        settings = Settings(
            llm_provider="claude",
            anthropic_api_key="test-key",
            anthropic_model="claude-sonnet-4-20250514",
        )
        client = create_llm(settings)
        self.assertIsInstance(client, ClaudeLLM)
        self.assertEqual(client.provider_name, "claude")

    def test_claude_generate_empty_hits(self) -> None:
        client = ClaudeLLM(api_key="test-key")
        self.assertEqual(client.generate("질문?", []), "관련 문서를 찾지 못했습니다.")

    @patch("docqa.llm.urllib.request.urlopen")
    def test_claude_generate_calls_anthropic_messages_api(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _FakeHTTPResponse(
            {"content": [{"type": "text", "text": "포트 8000을 사용합니다 [1]."}]}
        )
        client = ClaudeLLM(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        hit = RetrievalHit(
            chunk=DocumentChunk(
                chunk_id="c1-0",
                source="guide.md",
                section="Setup",
                chunk_index=0,
                text="서버는 포트 8000을 사용합니다.",
            ),
            score=1.23,
        )

        answer = client.generate("포트 번호는?", [hit])
        self.assertEqual(answer, "포트 8000을 사용합니다 [1].")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "https://api.anthropic.com/v1/messages")
        headers = {k.lower(): v for k, v in req.header_items()}
        self.assertEqual(headers["x-api-key"], "sk-ant-test")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "claude-sonnet-4-20250514")
        self.assertEqual(body["messages"][0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
