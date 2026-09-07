"""
Loads the mixed-format rulebook and splits it into section-level chunks.

Chunking is by *section*, not by fixed token window, because every answer has to
carry a real section reference like "AR-3.2". A fixed-size splitter would cut
across clause boundaries and make citations meaningless.

Three input formats are handled:
  - markdown prose        (01, 02, 03)
  - markdown with tables  (04 — tables are kept whole, never split mid-row)
  - PDF                   (05 — text extracted, then re-sectioned by heading)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

# Human-readable titles, keyed by the document ID prefix used in section IDs.
DOCUMENT_TITLES: Dict[str, str] = {
    "AR": "Academic Regulations — B.Tech Programme",
    "HH": "Hostel Handbook",
    "SP": "Merit Scholarship Policy",
    "FS": "Fee Schedule and Payment Deadlines",
    "EC": "Examination Committee Charter",
}

# Matches "### AR-3.2 Minimum Attendance..." and "## Section AR-3. Attendance"
_SECTION_HEADING = re.compile(
    r"^(?P<hashes>#{2,4})\s*(?:Section\s+)?(?P<sid>[A-Z]{2}-\d+(?:\.\d+)?)\.?\s*(?P<title>.*)$"
)
# Matches PDF headings, which arrive without markdown hashes.
_PDF_HEADING = re.compile(r"^(?:Section\s+)?(?P<sid>EC-\d+(?:\.\d+)?)\.?\s+(?P<title>.+)$")


@dataclass
class Chunk:
    """One retrievable section of the rulebook."""

    section_id: str
    heading: str
    text: str
    source_file: str
    source_format: str
    document: str = field(init=False)

    def __post_init__(self) -> None:
        prefix = self.section_id.split("-")[0]
        self.document = DOCUMENT_TITLES.get(prefix, self.source_file)

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded — heading included, it carries real signal."""
        return f"{self.section_id} {self.heading}. {self.text}"


def _flush(
    sid: str | None,
    heading: str,
    buf: List[str],
    source_file: str,
    source_format: str,
    out: List[Chunk],
) -> None:
    if sid is None:
        return
    body = "\n".join(buf).strip()
    if not body:
        # A parent heading with no prose of its own (e.g. "Section AR-3. Attendance")
        # is skipped; its child clauses carry the content.
        return
    out.append(
        Chunk(
            section_id=sid,
            heading=heading.strip(),
            text=body,
            source_file=source_file,
            source_format=source_format,
        )
    )


def _load_markdown(path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    sid: str | None = None
    heading = ""
    buf: List[str] = []
    in_table = False

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()

        # Keep tables intact: a heading can never appear inside one.
        stripped = line.strip()
        if stripped.startswith("|"):
            in_table = True
        elif in_table and not stripped:
            in_table = False

        match = None if in_table else _SECTION_HEADING.match(line)
        if match:
            fmt = "table" if _has_table(buf) else "markdown"
            _flush(sid, heading, buf, path.name, fmt, chunks)
            sid = match.group("sid")
            heading = match.group("title") or sid
            buf = []
            continue

        if sid is not None:
            buf.append(line)

    _flush(sid, heading, buf, path.name, "table" if _has_table(buf) else "markdown", chunks)
    return chunks


def _has_table(buf: List[str]) -> bool:
    return sum(1 for line in buf if line.strip().startswith("|")) >= 2


def _load_pdf(path: Path) -> List[Chunk]:
    try:
        import pymupdf  # type: ignore
    except ImportError:  # pragma: no cover - older wheels expose the legacy name
        import fitz as pymupdf  # type: ignore

    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    chunks: List[Chunk] = []
    sid: str | None = None
    heading = ""
    buf: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _PDF_HEADING.match(line)
        if match:
            _flush(sid, heading, buf, path.name, "pdf", chunks)
            sid = match.group("sid")
            heading = match.group("title")
            buf = []
            continue
        if sid is not None:
            buf.append(line)

    _flush(sid, heading, buf, path.name, "pdf", chunks)
    return chunks


def iter_corpus_files(corpus_dir: Path = CORPUS_DIR) -> Iterator[Path]:
    for path in sorted(corpus_dir.iterdir()):
        if path.name.startswith("_"):
            continue  # build scripts are not corpus
        if path.suffix.lower() in {".md", ".pdf"}:
            yield path


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> List[Chunk]:
    """Load every corpus file and return section-level chunks."""
    chunks: List[Chunk] = []
    for path in iter_corpus_files(corpus_dir):
        if path.suffix.lower() == ".pdf":
            chunks.extend(_load_pdf(path))
        else:
            chunks.extend(_load_markdown(path))

    if not chunks:
        raise RuntimeError(f"No chunks loaded from {corpus_dir}. Is the corpus present?")

    seen: set[str] = set()
    for chunk in chunks:
        if chunk.section_id in seen:
            raise RuntimeError(f"Duplicate section id in corpus: {chunk.section_id}")
        seen.add(chunk.section_id)
    return chunks


def corpus_word_count(corpus_dir: Path = CORPUS_DIR) -> int:
    return sum(len(chunk.text.split()) for chunk in load_corpus(corpus_dir))


if __name__ == "__main__":
    loaded = load_corpus()
    print(f"{len(loaded)} sections loaded")
    by_format: Dict[str, int] = {}
    for c in loaded:
        by_format[c.source_format] = by_format.get(c.source_format, 0) + 1
    print("by format:", by_format)
    print("indexed words:", sum(len(c.text.split()) for c in loaded))
