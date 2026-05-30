"""Built-in prompt management & versioning.

Prompts are first-class, versioned artifacts — not strings buried in code. A
:class:`PromptTemplate` renders with ``{placeholder}`` substitution; a
:class:`PromptRegistry` stores immutable versions, tracks the active one, and
records a content hash so a prompt change is an auditable event (and can be
pinned by a compiled/optimized agent for deterministic production behavior).
"""

from __future__ import annotations

import hashlib
import time

from pydantic import BaseModel, Field

from .exceptions import YaabError


class PromptVersion(BaseModel):
    version: int
    template: str
    created_at: float = Field(default_factory=time.time)
    notes: str = ""
    hash: str = ""

    def render(self, **values: object) -> str:
        try:
            return self.template.format(**values)
        except KeyError as exc:
            raise YaabError(f"missing prompt variable: {exc}") from exc


class PromptTemplate(BaseModel):
    """A named prompt with an immutable, hash-stamped version history."""

    name: str
    versions: list[PromptVersion] = Field(default_factory=list)
    active: int = 1

    @staticmethod
    def _hash(template: str) -> str:
        return hashlib.sha256(template.encode()).hexdigest()[:16]

    @classmethod
    def create(cls, name: str, template: str, *, notes: str = "") -> PromptTemplate:
        pt = cls(name=name)
        pt.add_version(template, notes=notes)
        return pt

    def add_version(self, template: str, *, notes: str = "") -> PromptVersion:
        version = PromptVersion(
            version=len(self.versions) + 1,
            template=template,
            notes=notes,
            hash=self._hash(template),
        )
        self.versions.append(version)
        self.active = version.version
        return version

    def get(self, version: int | None = None) -> PromptVersion:
        target = version or self.active
        for v in self.versions:
            if v.version == target:
                return v
        raise YaabError(f"prompt '{self.name}' has no version {target}")

    def render(self, *, version: int | None = None, **values: object) -> str:
        return self.get(version).render(**values)


class PromptRegistry:
    """A store of versioned prompts, addressable by ``name`` (and optional version)."""

    def __init__(self) -> None:
        self._prompts: dict[str, PromptTemplate] = {}

    def register(self, name: str, template: str, *, notes: str = "") -> PromptTemplate:
        if name in self._prompts:
            self._prompts[name].add_version(template, notes=notes)
        else:
            self._prompts[name] = PromptTemplate.create(name, template, notes=notes)
        return self._prompts[name]

    def get(self, name: str) -> PromptTemplate:
        if name not in self._prompts:
            raise YaabError(f"unknown prompt '{name}'")
        return self._prompts[name]

    def render(self, name: str, /, *, version: int | None = None, **values: object) -> str:
        return self.get(name).render(version=version, **values)

    def list(self) -> list[str]:
        return list(self._prompts)


__all__ = ["PromptTemplate", "PromptVersion", "PromptRegistry"]
