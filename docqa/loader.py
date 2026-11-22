"""Document loading and section-aware chunking utilities."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class DocumentChunk:
    """A single indexed text chunk with source metadata."""

    chunk_id: str
    source: str
    section: str
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        """Serialize chunk metadata and text to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentChunk":
        """Deserialize a chunk from a dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            source=data["source"],
            section=data.get("section", "General"),
            chunk_index=int(data["chunk_index"]),
            text=data["text"],
        )


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required to load PDF files. Install it with `pip install pypdf`."
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def extract_text(path: Path) -> str:
    """Extract raw text from a supported document file."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return _read_text_file(path)
    if suffix == ".pdf":
        return _read_pdf_file(path)
    raise ValueError(f"Unsupported document format: {path.suffix}")


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown text into (section_title, section_body) pairs."""
    sections: list[tuple[str, list[str]]] = []
    current_title = "Overview"
    current_lines: list[str] = []

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_title, current_lines))
            current_title = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines and "\n".join(current_lines).strip():
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines).strip()) for title, lines in sections]


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks while respecting word boundaries."""
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    overlap = min(chunk_overlap, max(0, chunk_size // 2))
    chunks: list[str] = []
    start = 0
    length = len(cleaned)

    while start < length:
        end = min(length, start + chunk_size)
        if end < length:
            # Prefer splitting on paragraph or whitespace near the boundary.
            split_at = cleaned.rfind("\n\n", start + chunk_size // 2, end)
            if split_at == -1:
                split_at = cleaned.rfind(" ", start + chunk_size // 2, end)
            if split_at != -1:
                end = split_at

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_document(
    path: Path,
    root_dir: Path | None = None,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    """Load a single file and split it into section-aware DocumentChunks."""
    raw = extract_text(path)
    if not raw.strip():
        return []

    source = (
        str(path.relative_to(root_dir)).replace("\\", "/")
        if root_dir is not None
        else path.name
    )

    if path.suffix.lower() in {".md", ".markdown"}:
        sections = _split_markdown_sections(raw)
    else:
        sections = [("Document", raw)]

    result: list[DocumentChunk] = []
    chunk_idx = 0
    for section_title, section_body in sections:
        for piece in _chunk_text(section_body, chunk_size, chunk_overlap):
            digest = hashlib.sha1(
                f"{source}:{chunk_idx}:{piece[:64]}".encode("utf-8")
            ).hexdigest()[:12]
            result.append(
                DocumentChunk(
                    chunk_id=f"{digest}-{chunk_idx}",
                    source=source,
                    section=section_title,
                    chunk_index=chunk_idx,
                    text=piece,
                )
            )
            chunk_idx += 1

    return result


def load_documents(
    docs_dir: Path,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    """Recursively load and chunk all supported documents under `docs_dir`."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    files: Iterable[Path] = sorted(
        p
        for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    chunks: list[DocumentChunk] = []
    for file_path in files:
        chunks.extend(
            chunk_document(
                file_path,
                root_dir=docs_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks
