"""Signatures — typed input→output specs for optimizable modules (DSPy-style).

A signature declares what a module does, not how. It can be written inline
(``"question -> answer"``) or with explicit field descriptions. The module
renders the signature into a prompt and parses the model's response back into
the named output fields.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FieldSpec(BaseModel):
    name: str
    description: str = ""


class Signature(BaseModel):
    """A parsed input→output specification."""

    instructions: str = ""
    inputs: list[FieldSpec] = Field(default_factory=list)
    outputs: list[FieldSpec] = Field(default_factory=list)

    @classmethod
    def parse(cls, spec: str, *, instructions: str = "") -> Signature:
        """Parse a ``"a, b -> c, d"`` style signature string."""
        if "->" not in spec:
            raise ValueError("signature must contain '->' (e.g. 'question -> answer')")
        left, right = spec.split("->", 1)
        inputs = [FieldSpec(name=n.strip()) for n in left.split(",") if n.strip()]
        outputs = [FieldSpec(name=n.strip()) for n in right.split(",") if n.strip()]
        return cls(instructions=instructions, inputs=inputs, outputs=outputs)

    def render_prompt(self, values: dict[str, str], demos: list[dict] | None = None) -> str:
        """Render the instruction + few-shot demos + the current inputs."""
        lines: list[str] = []
        if self.instructions:
            lines.append(self.instructions)
        in_names = ", ".join(f.name for f in self.inputs)
        out_names = ", ".join(f.name for f in self.outputs)
        lines.append(f"Given the fields [{in_names}], produce the fields [{out_names}].")
        for demo in demos or []:
            lines.append("\n---")
            for f in self.inputs:
                lines.append(f"{f.name}: {demo.get(f.name, '')}")
            for f in self.outputs:
                lines.append(f"{f.name}: {demo.get(f.name, '')}")
        lines.append("\n---")
        for f in self.inputs:
            lines.append(f"{f.name}: {values.get(f.name, '')}")
        for f in self.outputs:
            lines.append(f"{f.name}:")
        return "\n".join(lines)

    def parse_output(self, text: str) -> dict[str, str]:
        """Extract output fields from a ``field: value`` formatted response."""
        result: dict[str, str] = {}
        names = {f.name.lower(): f.name for f in self.outputs}
        for line in text.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                canonical = names.get(key.strip().lower())
                if canonical:
                    result[canonical] = value.strip()
        # Single-output fallback: the whole text is the answer.
        if not result and len(self.outputs) == 1:
            result[self.outputs[0].name] = text.strip()
        return result
