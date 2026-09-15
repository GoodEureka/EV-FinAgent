"""PDF parsing and page-aware chunking for annual reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import BinaryIO, Iterable

@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    page: int
    text: str


def _normalize(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_pdf(source: str | Path | bytes | BinaryIO) -> list[PageText]:
    """Extract text while preserving the PDF's one-based page numbers."""
    import pymupdf as fitz  # Loaded lazily so pure logic can be tested independently.

    if isinstance(source, (str, Path)):
        document = fitz.open(str(source))
    elif isinstance(source, bytes):
        document = fitz.open(stream=source, filetype="pdf")
    else:
        document = fitz.open(stream=source.read(), filetype="pdf")

    try:
        return [
            PageText(page=index + 1, text=_normalize(page.get_text("text")))
            for index, page in enumerate(document)
        ]
    finally:
        document.close()


def chunk_pages(
    pages: Iterable[PageText], chunk_size: int = 3500, overlap: int = 350
) -> list[TextChunk]:
    """Split each page independently so a chunk never loses its citation page."""
    if chunk_size <= overlap or overlap < 0:
        raise ValueError("chunk_size must be greater than a non-negative overlap")

    chunks: list[TextChunk] = []
    for page in pages:
        if not page.text:
            continue
        start = 0
        part = 1
        while start < len(page.text):
            end = min(start + chunk_size, len(page.text))
            chunks.append(
                TextChunk(
                    chunk_id=f"p{page.page}-c{part}",
                    page=page.page,
                    text=page.text[start:end],
                )
            )
            if end == len(page.text):
                break
            start = end - overlap
            part += 1
    return chunks


def pages_as_dicts(pages: Iterable[PageText]) -> list[dict]:
    return [asdict(page) for page in pages]
