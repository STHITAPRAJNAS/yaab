"""YAAB's first-class content type: ``Content`` made of typed ``Part``s.

Where Google ADK exposes ``google.genai.Content``/``Part``, YAAB ships its own
provider-neutral, multimodal content model so an agent message is never just a
string. A :class:`Content` is a role plus an ordered list of :class:`Part`s, and
each part is one of: text, inline binary data (a "blob"), a file reference, a
thought/reasoning trace, a tool call, or a tool result.

This is the canonical wire type: it renders down to the OpenAI/LiteLLM
multimodal ``content`` array, round-trips through sessions and checkpoints, and
streams over SSE. Plain strings still work everywhere — they are sugar for a
single text part.
"""

from __future__ import annotations

import base64
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from .types import Role, ToolCall


class PartKind(str, Enum):
    TEXT = "text"
    DATA = "data"  # inline binary blob (image/audio/...)
    FILE = "file"  # reference to a file by URI
    THOUGHT = "thought"  # reasoning / chain-of-thought trace
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class Part(BaseModel):
    """One unit of content. Use the classmethod constructors, not the raw fields."""

    kind: PartKind = PartKind.TEXT
    text: Optional[str] = None
    mime_type: Optional[str] = None
    data: Optional[str] = None  # base64-encoded bytes for DATA parts
    uri: Optional[str] = None  # for FILE parts
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[Any] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # --- constructors --------------------------------------------------
    @classmethod
    def text_part(cls, text: str) -> "Part":
        return cls(kind=PartKind.TEXT, text=text)

    @classmethod
    def data_part(cls, data: bytes, mime_type: str) -> "Part":
        return cls(
            kind=PartKind.DATA,
            mime_type=mime_type,
            data=base64.b64encode(data).decode("ascii"),
        )

    @classmethod
    def file_part(cls, uri: str, mime_type: Optional[str] = None) -> "Part":
        return cls(kind=PartKind.FILE, uri=uri, mime_type=mime_type)

    @classmethod
    def thought_part(cls, text: str) -> "Part":
        return cls(kind=PartKind.THOUGHT, text=text)

    @classmethod
    def tool_call_part(cls, call: ToolCall) -> "Part":
        return cls(kind=PartKind.TOOL_CALL, tool_call=call)

    @classmethod
    def tool_result_part(cls, result: Any) -> "Part":
        return cls(kind=PartKind.TOOL_RESULT, tool_result=result)

    # --- rendering -----------------------------------------------------
    def decoded(self) -> bytes:
        """Return the raw bytes for a DATA part."""
        if self.kind is not PartKind.DATA or self.data is None:
            raise ValueError("decoded() is only valid for DATA parts")
        return base64.b64decode(self.data)

    def to_provider(self) -> Optional[dict[str, Any]]:
        """Render to an OpenAI-style multimodal content item (or None)."""
        if self.kind in (PartKind.TEXT, PartKind.THOUGHT):
            return {"type": "text", "text": self.text or ""}
        if self.kind is PartKind.DATA:
            url = f"data:{self.mime_type};base64,{self.data}"
            return {"type": "image_url", "image_url": {"url": url}}
        if self.kind is PartKind.FILE:
            return {"type": "image_url", "image_url": {"url": self.uri}}
        return None  # tool parts are carried out-of-band on the message


class Content(BaseModel):
    """A role plus an ordered list of typed parts — YAAB's canonical message."""

    role: Role = Role.USER
    parts: list[Part] = Field(default_factory=list)

    @classmethod
    def from_text(cls, text: str, role: Role = Role.USER) -> "Content":
        return cls(role=role, parts=[Part.text_part(text)])

    @classmethod
    def coerce(cls, value: Union[str, "Content"], role: Role = Role.USER) -> "Content":
        if isinstance(value, Content):
            return value
        return cls.from_text(str(value), role=role)

    @property
    def text(self) -> str:
        """Concatenate all text/thought parts."""
        return "".join(p.text or "" for p in self.parts if p.kind in (PartKind.TEXT, PartKind.THOUGHT))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [p.tool_call for p in self.parts if p.tool_call is not None]

    def is_multimodal(self) -> bool:
        return any(p.kind in (PartKind.DATA, PartKind.FILE) for p in self.parts)

    def to_provider_content(self) -> Union[str, list[dict[str, Any]]]:
        """Render the parts into a provider ``content`` value.

        Returns a plain string for text-only content (the common, cheap case) or
        the multimodal array for mixed content.
        """
        if not self.is_multimodal():
            return self.text
        items = [item for p in self.parts if (item := p.to_provider()) is not None]
        return items

    def to_message(self) -> "Any":
        """Lower a Content into the flat :class:`~yaab.types.Message` wire type."""
        from .types import Message

        return Message(role=self.role, content=self.text, tool_calls=self.tool_calls)


__all__ = ["Part", "PartKind", "Content"]
