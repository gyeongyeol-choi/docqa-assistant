"""Tests for document loading and section-aware chunking."""

from pathlib import Path
import tempfile
import unittest

from docqa.loader import chunk_document, load_documents


class TestLoader(unittest.TestCase):
    def test_markdown_section_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md = root / "guide.md"
            md.write_text(
                "# Intro\nHello world.\n\n## Setup\nInstall dependencies.\n",
                encoding="utf-8",
            )
            chunks = chunk_document(md, root_dir=root, chunk_size=200, chunk_overlap=20)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0].section, "Intro")
            self.assertEqual(chunks[1].section, "Setup")
            self.assertEqual(chunks[0].source, "guide.md")

    def test_load_documents_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "notes"
            sub.mkdir()
            (sub / "note.txt").write_text("Important architecture note.", encoding="utf-8")
            (sub / "ignore.bin").write_bytes(b"\x00\x01")
            chunks = load_documents(root)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].source, "notes/note.txt")


if __name__ == "__main__":
    unittest.main()
