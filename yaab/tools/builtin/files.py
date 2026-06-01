"""Sandboxed file tools — read/write/list under a fixed root directory.

The danger with file tools is path traversal letting an agent escape its
working area and read ``/etc/passwd`` or clobber arbitrary files. Every path
here is resolved (``Path.resolve()`` collapses ``..`` and symlinks) and verified
to stay *under* the configured root before any I/O happens — anything that
escapes returns an ``error: ...`` string instead of touching the filesystem.

The root is supplied at build time via :func:`make_file_tools` /
:func:`file_toolset` (or the ``root=`` kwarg when fetched from the component
registry), so different agents can be confined to different sandboxes::

    from yaab.tools.builtin.files import file_toolset
    agent = Agent("a", model=..., tools=file_toolset(root="./workspace"))

Errors are returned as model-readable strings (the agent loop feeds tool
results back to the model) rather than raised.
"""

from __future__ import annotations

from pathlib import Path

from ..base import FunctionTool, tool

#: Hard cap on bytes written/read regardless of caller-supplied limits, so a
#: runaway tool call can't exhaust memory or disk in one shot.
_MAX_BYTES = 1_000_000


def _safe_path(root: Path, path: str) -> Path | None:
    """Resolve ``path`` against ``root`` and return it only if it stays inside.

    Returns ``None`` when the resolved target escapes ``root`` (traversal via
    ``..``, absolute paths, or symlinks) — callers turn that into an error.
    """
    try:
        candidate = (root / path).resolve()
    except (OSError, ValueError):
        return None
    if candidate == root or root in candidate.parents:
        return candidate
    return None


def make_file_tools(*, root: str) -> tuple[FunctionTool, FunctionTool, FunctionTool]:
    """Build ``(read_file, write_file, list_directory)`` confined to ``root``.

    ``root`` is created if missing and resolved once; all subsequent paths are
    validated against it. The three tools are named ``file_read``,
    ``file_write``, and ``file_list`` for the model and the registry.
    """
    base = Path(root).resolve()
    base.mkdir(parents=True, exist_ok=True)

    @tool(name="file_read")
    async def read_file(path: str, max_chars: int = 10_000) -> str:
        """Read a text file under the sandbox root and return its contents.

        ``path`` is relative to the sandbox root; ``..`` traversal and absolute
        paths that escape the root are rejected. Returns up to ``max_chars``
        characters. Missing/unreadable files come back as ``error: ...``.
        """
        target = _safe_path(base, path)
        if target is None:
            return f"error: path {path!r} escapes the sandbox root"
        if not target.is_file():
            return f"error: no such file: {path}"
        try:
            data = target.read_bytes()[:_MAX_BYTES]
            return data.decode("utf-8", "replace")[:max_chars]
        except OSError as exc:
            return f"error: failed to read {path}: {exc}"

    @tool(name="file_write")
    async def write_file(path: str, content: str) -> str:
        """Write text to a file under the sandbox root (creating parent dirs).

        ``path`` is relative to the sandbox root; traversal outside it is
        rejected. Overwrites an existing file. Returns a confirmation string, or
        ``error: ...`` on failure.
        """
        target = _safe_path(base, path)
        if target is None:
            return f"error: path {path!r} escapes the sandbox root"
        if len(content.encode("utf-8")) > _MAX_BYTES:
            return f"error: content exceeds {_MAX_BYTES} bytes"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"error: failed to write {path}: {exc}"
        return f"wrote {len(content)} chars to {path}"

    @tool(name="file_list")
    async def list_directory(path: str = ".", glob: str = "*") -> str:
        """List entries in a directory under the sandbox root matching ``glob``.

        ``path`` is relative to the sandbox root; traversal outside it is
        rejected. Directory entries are suffixed with ``/``. Returns a
        newline-separated listing, or ``error: ...`` on failure.
        """
        target = _safe_path(base, path)
        if target is None:
            return f"error: path {path!r} escapes the sandbox root"
        if not target.is_dir():
            return f"error: not a directory: {path}"
        try:
            names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.glob(glob))
        except OSError as exc:
            return f"error: failed to list {path}: {exc}"
        return "\n".join(names) if names else "(empty)"

    return read_file, write_file, list_directory


def file_toolset(*, root: str) -> list[FunctionTool]:
    """Return the three sandboxed file tools as a list (handy for ``tools=``)."""
    return list(make_file_tools(root=root))


__all__ = ["make_file_tools", "file_toolset"]
