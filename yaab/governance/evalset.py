"""Portable evaluation files — the ``.evalset.json`` format.

ADK ships eval suites as ``*.evalset.json`` files so they are portable across
machines, checked into version control, and runnable by a CLI/UI without any
Python. YAAB's in-memory :class:`~yaab.governance.eval.Case`/``Dataset`` are
great for code-first evals but are *not* a stable on-disk contract. This module
adds that contract:

- :class:`EvalCase` — one example: a (possibly multi-turn) conversation, an
  optional expected output, and an optional expected *tool trajectory* so the
  :class:`~yaab.governance.eval.ToolTrajectoryMatch` metric can score process,
  not just the final string.
- :class:`EvalSet` — a named, versioned collection of cases that round-trips to
  JSON (with a ``schema_version`` for forward compatibility) and converts both
  *to* a :class:`~yaab.governance.eval.Dataset` (so the existing ``Experiment``
  machinery runs it) and *from* existing :class:`~yaab.governance.eval.Case`
  objects (so code-first suites can be exported and shared).

The file extension convention is ``.evalset.json`` to match ADK and make these
files easy to discover.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .eval import Case, Dataset

#: Bumped only on a breaking change to the on-disk shape; readers should accept
#: any value they understand and ignore unknown trailing fields for forward
#: compatibility (pydantic already ignores extras on load).
EVALSET_SCHEMA_VERSION = 1

#: The conventional suffix for evalset files (matches ADK's ``.evalset.json``).
EVALSET_SUFFIX = ".evalset.json"


class EvalCase(BaseModel):
    """One portable evaluation example.

    A single-turn case is just a one-entry ``conversation``; multi-turn cases
    list the user turns in order (the agent/task is expected to drive the turns
    in between). ``expected_tool_trajectory`` is an ordered list of
    ``{"name": str, "arguments"?: dict}`` steps for trajectory scoring.
    """

    id: str
    #: User turns, in order. Single-turn evals have exactly one entry.
    conversation: list[str] = Field(default_factory=list)
    expected_output: str | None = None
    #: Ordered expected tool calls; each step is ``{"name", "arguments"?}``.
    expected_tool_trajectory: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_case(self) -> Case:
        """Convert to a yaab :class:`~yaab.governance.eval.Case`.

        The *last* user turn is the input the task receives; the full
        conversation is preserved under ``metadata["conversation"]`` so
        multi-turn-aware tasks can replay it, and the expected trajectory is
        stashed under ``metadata["expected_tool_trajectory"]`` where
        :class:`~yaab.governance.eval.ToolTrajectoryMatch` looks for it.
        """
        metadata: dict[str, Any] = dict(self.metadata)
        metadata["conversation"] = list(self.conversation)
        if self.expected_tool_trajectory is not None:
            metadata["expected_tool_trajectory"] = self.expected_tool_trajectory
        last_turn = self.conversation[-1] if self.conversation else ""
        return Case(
            name=self.id,
            inputs=last_turn,
            expected=self.expected_output,
            metadata=metadata,
        )

    @classmethod
    def from_case(cls, case: Case) -> EvalCase:
        """Build an :class:`EvalCase` from a yaab :class:`Case`.

        A ``conversation`` already in the case's metadata wins; otherwise the
        case's ``inputs`` becomes a single-turn conversation. The expected tool
        trajectory is read from metadata if present.
        """
        metadata = dict(case.metadata)
        conversation = metadata.pop("conversation", None)
        if not conversation:
            conversation = [str(case.inputs)] if case.inputs is not None else []
        trajectory = metadata.pop("expected_tool_trajectory", None)
        expected = case.expected if case.expected is None else str(case.expected)
        return cls(
            id=case.name or "",
            conversation=list(conversation),
            expected_output=expected,
            expected_tool_trajectory=trajectory,
            metadata=metadata,
        )


class EvalSet(BaseModel):
    """A named, versioned, portable collection of :class:`EvalCase`s."""

    name: str
    #: Suite version (author-controlled, distinct from the file ``schema_version``).
    version: str = "1"
    cases: list[EvalCase] = Field(default_factory=list)
    #: Unix epoch seconds when the set was created.
    creation_timestamp: float = Field(default_factory=time.time)

    # --- file I/O (the .evalset.json contract) ---------------------------
    def save(self, path: str | Path) -> Path:
        """Write the set to ``path`` as pretty JSON with a ``schema_version``.

        Returns the path written. The on-disk object is the model dump plus a
        leading ``schema_version`` so future readers can branch on format.
        """
        path = Path(path)
        body: dict[str, Any] = {"schema_version": EVALSET_SCHEMA_VERSION}
        body.update(self.model_dump(mode="json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> EvalSet:
        """Read an evalset back from ``path`` (ignores ``schema_version``).

        Unknown top-level fields (including ``schema_version``) are dropped by
        pydantic, which is what gives us forward compatibility across minor
        format additions.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw.pop("schema_version", None)
        return cls.model_validate(raw)

    # --- interop with the existing Experiment machinery ------------------
    def to_dataset(self) -> Dataset:
        """Convert to a yaab :class:`~yaab.governance.eval.Dataset`.

        The returned dataset can be handed straight to
        :class:`~yaab.governance.eval.Experiment`, so an evalset file becomes a
        runnable suite with no extra glue.
        """
        return Dataset(name=self.name, cases=[c.to_case() for c in self.cases])

    @classmethod
    def from_cases(
        cls,
        cases: list[Case],
        *,
        name: str = "evalset",
        version: str = "1",
    ) -> EvalSet:
        """Build an :class:`EvalSet` from existing yaab :class:`Case` objects."""
        return cls(name=name, version=version, cases=[EvalCase.from_case(c) for c in cases])
