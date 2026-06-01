"""Tool-level authentication: credentials and an OAuth2 consent surface.

This is the YAAB equivalent of ADK's tool ``auth_scheme`` + ``auth_credential``.
A tool can declare *what* auth it needs (a :class:`ToolAuth`) and *how* to get a
credential — a static one, or a ``credential_provider`` that the framework calls
at execution time with the live :class:`RunContext`, so per-user OAuth tokens can
be looked up via ``ctx.identity``. When no credential can be produced, resolution
raises :class:`ToolAuthRequired`, which carries the consent URL and scopes so a
UI (or the agent itself, via the error string the runtime emits) can drive the
user through consent and then retry the call.

Why a separate exception here instead of in :mod:`yaab.exceptions`? Tool auth is
a self-contained concern owned by this module, and keeping the exception local
avoids a cross-cutting edit to the shared exception hierarchy. It still derives
from :class:`~yaab.exceptions.YaabError` so a broad ``except`` catches it.
"""

from __future__ import annotations

import base64
import inspect
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from ..exceptions import YaabError

if TYPE_CHECKING:  # avoid importing RunContext at runtime (keeps the module light)
    from ..types import RunContext


#: The auth schemes a tool can require. ``api_key`` rides in a named header;
#: ``bearer``/``oauth2`` both render to an ``Authorization: Bearer`` header (the
#: distinction is provenance — oauth2 implies a token obtained via a consent
#: flow); ``basic`` is HTTP Basic (user:pass, base64).
AuthScheme = Literal["api_key", "bearer", "oauth2", "basic"]


class ToolCredential(BaseModel):
    """A resolved credential the framework injects into a tool call.

    The fields are a small superset so one model covers all four schemes:

    * ``api_key``: ``value`` is the key; ``header`` names where it rides.
    * ``bearer`` / ``oauth2``: ``token`` is the bearer token.
    * ``basic``: ``value`` is the username and ``token`` is the password.

    ``expires_at`` (epoch seconds) lets short-lived OAuth tokens be detected as
    stale so :meth:`ToolAuth.resolve` can re-fetch a fresh one instead of sending
    an expired token.
    """

    kind: AuthScheme
    #: The API key (api_key) or the username (basic).
    value: str | None = None
    #: The bearer/oauth2 token, or the password (basic).
    token: str | None = None
    #: Header name for an ``api_key`` credential (case preserved as given).
    header: str = "x-api-key"
    #: Optional expiry (epoch seconds). ``None`` means the credential never
    #: expires on its own.
    expires_at: float | None = None

    def is_expired(self, *, now: float | None = None) -> bool:
        """Return ``True`` if ``expires_at`` is set and in the past.

        A small clock skew isn't accounted for here — providers should issue a
        fresh token rather than one on the edge of expiry.
        """
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at


# A provider returns a credential given the run context. It may be sync or async
# so callers can do a blocking lookup or hit an async token store / IdP.
CredentialProvider = Callable[["RunContext"], "ToolCredential"]


class ToolAuthRequired(YaabError):
    """Raised when a tool needs authorization that couldn't be resolved.

    Carries everything a caller or UI needs to drive an OAuth2-style consent
    flow and retry: the ``consent_url`` to send the user to, the requested
    ``scopes``, and the ``tool`` name for context. The agent runtime renders this
    into a model-visible ``error: ... requires authorization`` string so the
    model can tell the user — see :meth:`as_model_error`.
    """

    def __init__(
        self,
        *,
        tool: str = "",
        consent_url: str | None = None,
        scopes: list[str] | None = None,
    ) -> None:
        self.tool = tool
        self.consent_url = consent_url
        self.scopes = scopes or []
        super().__init__(self._message())

    def _message(self) -> str:
        where = f" for tool '{self.tool}'" if self.tool else ""
        url = f" (visit {self.consent_url})" if self.consent_url else ""
        scopes = f" scopes: {', '.join(self.scopes)}" if self.scopes else ""
        return f"authorization required{where}{url}{scopes}".rstrip()

    def as_model_error(self) -> str:
        """Render the model-visible ``error:`` string the agent loop surfaces.

        Format mirrors the rest of the toolset's error convention so the model
        sees a consistent, actionable message it can relay to the user.
        """
        url = self.consent_url or "(no consent URL configured)"
        scopes = ", ".join(self.scopes) if self.scopes else "none"
        return f"error: tool {self.tool} requires authorization: visit {url} (scopes: {scopes})"


class ToolAuth(BaseModel):
    """Auth requirements for a tool, plus the logic to resolve a credential.

    Mirrors ADK's ``auth_scheme`` + ``auth_credential`` pairing: declare the
    scheme and either a static ``credential`` or a ``credential_provider`` that
    yields one per call. For OAuth2, set ``consent_url`` + ``scopes`` so an
    unresolved credential produces an actionable consent prompt.
    """

    model_config = {"arbitrary_types_allowed": True}

    scheme: AuthScheme
    #: A pre-resolved credential. Used directly unless it has expired, in which
    #: case the ``credential_provider`` (if any) is consulted for a fresh one.
    credential: ToolCredential | None = None
    #: A sync or async callable that returns a :class:`ToolCredential` given the
    #: :class:`RunContext`. Receives ``ctx`` so it can key off ``ctx.identity``
    #: for per-user tokens. Typed as a bare ``Callable`` (not the forward-ref'd
    #: :data:`CredentialProvider`) so pydantic needn't resolve ``RunContext``.
    credential_provider: Callable | None = Field(default=None, exclude=True)
    #: Where to send the user to grant consent (OAuth2). Surfaced in
    #: :class:`ToolAuthRequired` when no credential is available.
    consent_url: str | None = None
    #: OAuth2 scopes requested; informational, surfaced in the consent prompt.
    scopes: list[str] = Field(default_factory=list)

    async def resolve(self, ctx: RunContext, *, tool_name: str = "") -> ToolCredential:
        """Resolve a usable credential for this call.

        Resolution order:

        1. The static ``credential`` if present and not expired.
        2. The ``credential_provider`` (awaited if it returns a coroutine),
           which receives ``ctx`` so it can look up a per-identity token.
        3. Otherwise raise :class:`ToolAuthRequired` carrying the consent URL
           and scopes.

        An *expired* static credential is skipped (not returned) so a stale
        OAuth token triggers a refresh via the provider rather than being sent.
        """
        cred = self.credential
        if cred is not None and not cred.is_expired():
            return cred

        if self.credential_provider is not None:
            result = self.credential_provider(ctx)
            if inspect.isawaitable(result):
                result = await result
            return result

        raise ToolAuthRequired(
            tool=tool_name,
            consent_url=self.consent_url,
            scopes=list(self.scopes),
        )


def as_headers(credential: ToolCredential) -> dict[str, str]:
    """Render a :class:`ToolCredential` into HTTP request headers.

    * ``api_key`` -> ``{header: value}`` (header name from the credential).
    * ``bearer`` / ``oauth2`` -> ``{'Authorization': 'Bearer <token>'}``.
    * ``basic`` -> ``{'Authorization': 'Basic <base64(value:token)>'}``.

    Raises :class:`~yaab.exceptions.YaabError` if a required field is missing,
    so a misconfigured credential fails loudly rather than sending a header with
    a literal ``None`` in it.
    """
    kind = credential.kind
    if kind == "api_key":
        if credential.value is None:
            raise YaabError("api_key credential is missing 'value'")
        return {credential.header: credential.value}
    if kind in ("bearer", "oauth2"):
        if credential.token is None:
            raise YaabError(f"{kind} credential is missing 'token'")
        return {"Authorization": f"Bearer {credential.token}"}
    if kind == "basic":
        user = credential.value or ""
        password = credential.token or ""
        encoded = base64.b64encode(f"{user}:{password}".encode()).decode()
        return {"Authorization": f"Basic {encoded}"}
    raise YaabError(f"unsupported credential kind: {kind!r}")  # pragma: no cover


__all__ = [
    "AuthScheme",
    "ToolCredential",
    "ToolAuth",
    "ToolAuthRequired",
    "CredentialProvider",
    "as_headers",
]
