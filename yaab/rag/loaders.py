"""Document loaders — turn files and URLs into :class:`Document` objects.

RAG ingestion starts from `Document`s; these loaders produce them from the
common formats so you point YAAB at your files instead of pre-extracting text.
Heavy parsers (PDF, rich HTML) are imported lazily so the core stays light; text,
Markdown, CSV, and JSON loaders are dependency-free, and HTML degrades to a
regex tag-stripper when BeautifulSoup isn't installed.

    from yaab.rag.loaders import load, load_directory
    docs = load("manual.pdf") + load_directory("./knowledge", glob="**/*.md")
    kb.add(docs)
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Optional

from .types import Document


def _doc(text: str, source: str, **meta: Any) -> Document:
    return Document(text=text, source=source, metadata=meta)


# --- per-format loaders -------------------------------------------------
def load_text(path: str, *, encoding: str = "utf-8") -> list[Document]:
    """Load a plain-text (or Markdown) file as a single Document."""
    with open(path, encoding=encoding) as fh:
        return [_doc(fh.read(), path, format="text")]


def load_markdown(path: str, *, encoding: str = "utf-8") -> list[Document]:
    docs = load_text(path, encoding=encoding)
    docs[0].metadata["format"] = "markdown"
    return docs


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n\s*\n\s*\n+")


def html_to_text(html: str) -> str:
    """Extract readable text from HTML (BeautifulSoup if present, else regex)."""
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
    except ImportError:
        # Dependency-free fallback: drop scripts/styles, strip tags, unescape.
        import html as _html

        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.I)
        text = _html.unescape(_TAG_RE.sub("", html))
    return _WS_RE.sub("\n\n", text).strip()


def load_html(path: str, *, encoding: str = "utf-8") -> list[Document]:
    with open(path, encoding=encoding) as fh:
        return [_doc(html_to_text(fh.read()), path, format="html")]


def load_pdf(path: str) -> list[Document]:
    """Load a PDF as one Document per page (requires ``pypdf``)."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("pypdf is required for PDF loading. `pip install pypdf`.") from exc
    reader = PdfReader(path)
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(_doc(text, path, format="pdf", page=i + 1))
    return docs


def load_csv(path: str, *, encoding: str = "utf-8", text_columns: Optional[list[str]] = None) -> list[Document]:
    """Load a CSV as one Document per row (``col: value`` lines)."""
    docs: list[Document] = []
    with open(path, encoding=encoding, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            cols = text_columns or list(row.keys())
            text = "\n".join(f"{c}: {row.get(c, '')}" for c in cols)
            docs.append(_doc(text, path, format="csv", row=i))
    return docs


def load_json(path: str, *, encoding: str = "utf-8") -> list[Document]:
    """Load JSON: one Document per element if it's a list, else one Document."""
    with open(path, encoding=encoding) as fh:
        data = json.load(fh)
    items = data if isinstance(data, list) else [data]
    return [
        _doc(json.dumps(item, ensure_ascii=False, indent=2), path, format="json", index=i)
        for i, item in enumerate(items)
    ]


# --- dispatch -----------------------------------------------------------
_LOADERS = {
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
    ".csv": load_csv,
    ".json": load_json,
}


def load(path: str) -> list[Document]:
    """Load a file into Documents, dispatching on its extension."""
    ext = Path(path).suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        # Unknown extension: best-effort as text.
        return load_text(path)
    return loader(path)


def load_directory(
    directory: str, *, glob: str = "**/*", recursive: bool = True
) -> list[Document]:
    """Load every supported file under ``directory`` matching ``glob``."""
    root = Path(directory)
    paths = root.glob(glob) if recursive else root.glob(glob.replace("**/", ""))
    docs: list[Document] = []
    for p in sorted(paths):
        if p.is_file() and p.suffix.lower() in _LOADERS:
            try:
                docs.extend(load(str(p)))
            except Exception:  # noqa: BLE001 - skip unreadable files, keep going
                continue
    return docs


def load_bytes(data: bytes, *, source: str, fmt: str = "text") -> list[Document]:
    """Load from in-memory bytes (e.g. an upload) using a named format."""
    if fmt == "html":
        return [_doc(html_to_text(data.decode("utf-8", "replace")), source, format="html")]
    if fmt == "csv":
        rows = csv.DictReader(io.StringIO(data.decode("utf-8", "replace")))
        return [
            _doc("\n".join(f"{k}: {v}" for k, v in row.items()), source, format="csv", row=i)
            for i, row in enumerate(rows)
        ]
    return [_doc(data.decode("utf-8", "replace"), source, format=fmt)]


__all__ = [
    "load",
    "load_directory",
    "load_bytes",
    "load_text",
    "load_markdown",
    "load_html",
    "load_pdf",
    "load_csv",
    "load_json",
    "html_to_text",
]
