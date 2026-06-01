"""Code snippets in onboarding docs must compile and reference real API.

For each documented file we extract the ```python fenced blocks and check:

1. **Parse** -- every block is valid Python (top-level ``await`` is allowed,
   matching how async snippets are written in docs).
2. **Imports resolve** -- every ``import``/``from ... import`` line works against
   the installed package, so docs can't reference API that doesn't exist.
3. **Names are defined** -- every name a block uses is defined or imported
   somewhere in that file's blocks (snippets build on each other top-to-bottom),
   is a builtin, or is an explicitly listed illustrative placeholder.
"""

from __future__ import annotations

import ast
import builtins
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# Names that docs use as placeholders defined "off-screen" (e.g. "your db here").
# Anything NOT in this set that a snippet uses without defining is a doc bug.
ILLUSTRATIVE: dict[str, set[str]] = {
    "quickstart.md": {"my_db"},
    "get-started.md": {"extractor", "transformer", "loader", "legal", "finance", "risk"},
    "state.md": {"agent", "pdf_bytes", "new_bytes"},
    "streaming-events.md": {"agent"},
}

DOC_FILES = sorted(ILLUSTRATIVE)

_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks(name: str) -> list[str]:
    return [m.group(1) for m in _FENCE.finditer((DOCS / name).read_text(encoding="utf-8"))]


def _parse(source: str) -> ast.Module:
    """Parse a snippet, allowing top-level await."""
    return compile(
        source,
        "<snippet>",
        "exec",
        flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
    )


@pytest.mark.parametrize("doc", DOC_FILES)
def test_snippets_parse(doc: str) -> None:
    found = _blocks(doc)
    assert found, f"{doc} has no ```python snippets"
    for i, src in enumerate(found, start=1):
        try:
            _parse(src)
        except SyntaxError as exc:
            pytest.fail(f"{doc} snippet {i} is not valid Python: {exc}\n---\n{src}")


@pytest.mark.parametrize("doc", DOC_FILES)
def test_snippet_imports_resolve(doc: str) -> None:
    for i, src in enumerate(_blocks(doc), start=1):
        for node in ast.walk(_parse(src)):
            if isinstance(node, ast.Import | ast.ImportFrom):
                stmt = ast.unparse(node)
                try:
                    exec(stmt, {})  # noqa: S102 - imports only, from our own docs
                except ImportError as exc:
                    pytest.fail(f"{doc} snippet {i}: `{stmt}` fails: {exc}")


@pytest.mark.parametrize("doc", DOC_FILES)
def test_snippet_names_are_defined(doc: str) -> None:
    defined: set[str] = set(dir(builtins)) | ILLUSTRATIVE.get(doc, set())
    undefined: set[str] = set()
    for src in _blocks(doc):
        tree = _parse(src)
        # First pass: everything this block defines (snippets are cumulative).
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                defined |= {a.asname or a.name for a in node.names}
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.alias):
                defined.add((node.asname or node.name).split(".")[0])
        # Second pass: every name this block reads.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in defined
            ):
                undefined.add(node.id)
    assert not undefined, (
        f"{doc} snippets use names that are never defined, imported, or declared "
        f"illustrative: {sorted(undefined)}"
    )
