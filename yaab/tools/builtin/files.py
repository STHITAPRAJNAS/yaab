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

import difflib
import hashlib
import os
import tempfile
from pathlib import Path

from ...capabilities import Capability
from ...types import RunContext
from ..base import FunctionTool, tool

#: Hard cap on bytes written/read regardless of caller-supplied limits, so a
#: runaway tool call can't exhaust memory or disk in one shot.
_MAX_BYTES = 1_000_000

#: ``ctx.state`` key holding ``{resolved_path: sha256}`` of every file read this
#: run. ``file_edit`` consults it so an edit can only touch a file the agent has
#: actually read, and only while the on-disk content still matches what it read
#: (a stale read — file changed underneath — is rejected). ``temp:`` so it never
#: persists into checkpoints.
_FILES_READ_KEY = "temp:__files_read__"


#: In-root paths an agent must not poison — corrupting these can achieve code
#: execution outside the sandbox at the next dev/CI action (supply-chain escape).
_PROTECTED = (".git", ".github", ".hg", ".svn")


def _is_protected(root: Path, target: Path) -> bool:
    try:
        rel = target.relative_to(root).parts
    except ValueError:
        return True
    lowered = [p.lower() for p in rel]
    return bool(rel) and (any(p in _PROTECTED for p in lowered) or lowered[-1].endswith(".lock"))


def _has_symlink_component(root: Path, rel_path: str) -> bool:
    """True if any component of ``root/rel_path`` is a symlink/junction.

    Walks the **unresolved** join (``Path.resolve`` would collapse the symlinks we
    are trying to detect), so a symlink planted along the path is rejected and a
    write/read cannot follow it out of root. Absolute or ``..`` components are also
    rejected.
    """
    cur = root
    try:
        for part in Path(rel_path).parts:
            if part in ("..", "/", "\\") or (len(part) == 2 and part.endswith(":")):
                return True  # traversal or absolute/drive component
            cur = cur / part
            if cur.is_symlink():
                return True
    except (OSError, ValueError):
        return True
    return False


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


def _capped_text(target: Path) -> str:
    """Decode the file's first ``_MAX_BYTES`` as UTF-8 (lossy on bad bytes)."""
    return target.read_bytes()[:_MAX_BYTES].decode("utf-8", "replace")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_read(ctx: RunContext, key: str, digest: str) -> None:
    """Note that ``key`` was read with content hash ``digest`` (reassigns the
    whole dict so the write survives whatever storage ``ctx.state`` uses)."""
    reg = ctx.state.get(_FILES_READ_KEY)
    reg = {**reg, key: digest} if isinstance(reg, dict) else {key: digest}
    ctx.state[_FILES_READ_KEY] = reg


def _atomic_write(target: Path, content: str) -> None:
    """Write ``content`` to ``target`` atomically (temp file + ``os.replace``).

    A crash mid-write leaves the destination untouched rather than truncated.
    Caller is responsible for path-safety / protection checks.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".yaab-tmp-")
    try:
        try:
            # newline="" writes the string verbatim — no os-specific newline
            # translation. We read content back via read_bytes (which preserves
            # CRLF), so translating on write would double the \r and corrupt edits.
            fh = os.fdopen(fd, "w", encoding="utf-8", newline="")
        except Exception:
            os.close(fd)  # fdopen failed to take ownership of the fd
            raise
        with fh:
            fh.write(content)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _unified_diff(path: str, old: str, new: str) -> str:
    """A git-style unified diff of an edit, capped so it can't flood the model."""
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    return diff if len(diff) <= 4000 else diff[:4000] + "\n... (diff truncated)"


def make_file_tools(*, root: str) -> tuple[FunctionTool, FunctionTool, FunctionTool]:
    """Build ``(read_file, write_file, list_directory)`` confined to ``root``.

    ``root`` is created if missing and resolved once; all subsequent paths are
    validated against it. The three tools are named ``file_read``,
    ``file_write``, and ``file_list`` for the model and the registry.
    """
    base = Path(root).resolve()
    base.mkdir(parents=True, exist_ok=True)

    @tool(name="file_read", capabilities={Capability.FS_READ})
    async def read_file(ctx: RunContext, path: str, max_chars: int = 10_000) -> str:
        """Read a text file under the sandbox root and return its contents.

        ``path`` is relative to the sandbox root; ``..`` traversal and absolute
        paths that escape the root are rejected. Returns up to ``max_chars``
        characters. Missing/unreadable files come back as ``error: ...``.
        Reading a file also unlocks it for :func:`file_edit`.
        """
        target = _safe_path(base, path)
        if target is None:
            return f"error: path {path!r} escapes the sandbox root"
        if _has_symlink_component(base, path):
            return f"error: path {path!r} contains a symlink and is rejected"
        if not target.is_file():
            return f"error: no such file: {path}"
        try:
            text = _capped_text(target)
        except OSError as exc:
            return f"error: failed to read {path}: {exc}"
        # Record the full (capped) content hash so file_edit can detect a stale
        # read; the model only sees the max_chars slice.
        _record_read(ctx, str(target), _hash(text))
        return text[:max_chars]

    @tool(name="file_write", capabilities={Capability.FS_WRITE_IN_ROOT})
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
        if _is_protected(base, target):
            return f"error: writing to protected path {path!r} is not allowed"
        if _has_symlink_component(base, path):
            return f"error: path {path!r} contains a symlink and is rejected"
        try:
            _atomic_write(target, content)
        except OSError as exc:
            return f"error: failed to write {path}: {exc}"
        return f"wrote {len(content)} chars to {path}"

    @tool(name="file_list", capabilities={Capability.FS_READ})
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


def make_file_edit(*, root: str) -> FunctionTool:
    """Build the ``file_edit`` tool — surgical, read-gated, single-match edits.

    Unlike ``file_write`` (which clobbers a whole file), ``file_edit`` replaces an
    exact ``old_string`` with ``new_string``. It refuses to edit a file the agent
    has not read this run, and refuses if the file changed on disk since that read
    (both via the read-registry in ``ctx.state``); it refuses an ``old_string``
    that is absent or — without ``replace_all`` — ambiguous. On success it writes
    atomically and returns a unified diff.
    """
    base = Path(root).resolve()
    base.mkdir(parents=True, exist_ok=True)

    @tool(name="file_edit", capabilities={Capability.FS_WRITE_IN_ROOT})
    async def edit_file(
        ctx: RunContext,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Replace ``old_string`` with ``new_string`` in a file under the root.

        The file must have been read (``file_read``) earlier this run and be
        unchanged since. ``old_string`` must occur exactly once unless
        ``replace_all`` is set. Returns a unified diff, or ``error: ...``.
        """
        target = _safe_path(base, path)
        if target is None:
            return f"error: path {path!r} escapes the sandbox root"
        if _has_symlink_component(base, path):
            return f"error: path {path!r} contains a symlink and is rejected"
        if _is_protected(base, target):
            return f"error: editing protected path {path!r} is not allowed"
        if not target.is_file():
            return f"error: no such file: {path}"
        if not old_string:
            return "error: old_string must not be empty"

        key = str(target)
        reg = ctx.state.get(_FILES_READ_KEY)
        digest = reg.get(key) if isinstance(reg, dict) else None
        if digest is None:
            return f"error: file {path!r} must be read with file_read before editing"
        try:
            current = _capped_text(target)
        except OSError as exc:
            return f"error: failed to read {path}: {exc}"
        if _hash(current) != digest:
            return (
                f"error: file {path!r} changed on disk since it was read; re-read it before editing"
            )

        count = current.count(old_string)
        if count == 0:
            return f"error: old_string not found in {path}"
        if count > 1 and not replace_all:
            return (
                f"error: old_string matches {count} times in {path}; "
                "add surrounding context for a unique match or pass replace_all=true"
            )

        new_content = (
            current.replace(old_string, new_string)
            if replace_all
            else current.replace(old_string, new_string, 1)
        )
        if len(new_content.encode("utf-8")) > _MAX_BYTES:
            return f"error: result exceeds {_MAX_BYTES} bytes"
        try:
            _atomic_write(target, new_content)
        except OSError as exc:
            return f"error: failed to write {path}: {exc}"

        # Keep the read-registry current so a follow-up edit on this file still
        # passes the staleness check.
        _record_read(ctx, key, _hash(new_content))
        n = count if replace_all else 1
        diff = _unified_diff(path, current, new_content)
        return f"edited {path} ({n} replacement{'s' if n != 1 else ''})\n{diff}"

    return edit_file


def file_toolset(*, root: str) -> list[FunctionTool]:
    """Return the sandboxed file tools as a list (handy for ``tools=``).

    Includes ``file_read``, ``file_write``, ``file_list``, and ``file_edit``.
    """
    return [*make_file_tools(root=root), make_file_edit(root=root)]


__all__ = ["make_file_tools", "make_file_edit", "file_toolset"]
