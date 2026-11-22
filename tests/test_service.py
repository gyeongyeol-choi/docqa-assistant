"""Integration tests for DocQAService."""

from pathlib import Path
import tempfile
import unittest

from docqa.config import Settings
from docqa.service import DocQAService


class TestService(unittest.TestCase):
    def test_index_and_ask_offline_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "setup.md").write_text(
                "# 실행 방법\nDocker 컨테이너는 포트 8000을 사용합니다.\n",
                encoding="utf-8",
            )
            settings = Settings(
                llm_provider="offline",
                docs_dir=docs_dir,
                index_path=root / "index.json",
            )
            service = DocQAService(settings=settings)
            summary = service.index_directory()
            self.assertEqual(summary["documents"], 1)
            self.assertEqual(summary["chunks"], 1)

            result = service.ask("Docker 포트는 몇 번인가요?")
            self.assertEqual(result.provider, "offline")
            self.assertEqual(len(result.citations), 1)
            self.assertEqual(result.citations[0].source, "setup.md")
            self.assertIn("[1]", result.answer)


if __name__ == "__main__":
    unittest.main()
