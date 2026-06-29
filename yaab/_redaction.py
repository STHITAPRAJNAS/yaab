"""Secret redaction shared across cassettes, traces, and the audit log.

Any bytes a tool returns can carry a secret (a ``.env`` read, an API response
with a token). Those flow into persisted records — cassettes (committed to git),
the trace store (a shared DB), the audit chain — so all three scrub serialized
content before writing. This is **defense in depth**, not a guarantee: it catches
known token shapes, not every possible secret.
"""

from __future__ import annotations

import re
from typing import Any

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{8,}"),  # OpenAI-style key
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),  # GitHub PAT
    re.compile(r"gh[opsu]_[A-Za-z0-9]{20,}"),  # other GitHub token types
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack token
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),  # Google API key
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),  # PEM
]

_REDACTED = "[REDACTED]"


def scrub_secrets(value: Any) -> Any:
    """Recursively replace secret-shaped substrings with ``[REDACTED]``."""
    if isinstance(value, str):
        out = value
        for pat in _SECRET_PATTERNS:
            out = pat.sub(_REDACTED, out)
        return out
    if isinstance(value, dict):
        return {k: scrub_secrets(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_secrets(v) for v in value]
    return value
