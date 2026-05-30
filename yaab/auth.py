"""Authentication schemes for serving agents.

Pluggable, protocol-based auth so the same agent can be exposed behind no auth
(dev), a bearer token, an API key, or OAuth 2.1 (the A2A standard) without
changing the agent. A scheme maps an incoming request's headers to an
``identity`` string that flows into the run context and the audit log.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .exceptions import YaabError


class AuthError(YaabError):
    """Raised when authentication fails."""


@runtime_checkable
class AuthScheme(Protocol):
    name: str

    def authenticate(self, headers: dict[str, str]) -> str | None:
        """Return the caller identity, or raise :class:`AuthError`."""
        ...

    def describe(self) -> dict:
        """Security-scheme metadata for the A2A agent card."""
        ...


class NoAuth:
    """Open access (development default). Identity is ``anonymous``."""

    name = "none"

    def authenticate(self, headers: dict[str, str]) -> str | None:
        return "anonymous"

    def describe(self) -> dict:
        return {"type": "none"}


class BearerTokenAuth:
    """Static bearer-token auth mapping tokens to identities."""

    name = "bearer"

    def __init__(self, tokens: dict[str, str]) -> None:
        # token -> identity
        self._tokens = tokens

    def authenticate(self, headers: dict[str, str]) -> str | None:
        header = headers.get("authorization") or headers.get("Authorization", "")
        if not header.lower().startswith("bearer "):
            raise AuthError("missing or malformed Authorization header")
        token = header.split(" ", 1)[1].strip()
        identity = self._tokens.get(token)
        if identity is None:
            raise AuthError("invalid bearer token")
        return identity

    def describe(self) -> dict:
        return {"type": "http", "scheme": "bearer"}


class APIKeyAuth:
    """API-key auth via a header (default ``x-api-key``)."""

    name = "api_key"

    def __init__(self, keys: dict[str, str], header: str = "x-api-key") -> None:
        self._keys = keys
        self.header = header.lower()

    def authenticate(self, headers: dict[str, str]) -> str | None:
        lowered = {k.lower(): v for k, v in headers.items()}
        key = lowered.get(self.header)
        identity = self._keys.get(key) if key else None
        if identity is None:
            raise AuthError("invalid or missing API key")
        return identity

    def describe(self) -> dict:
        return {"type": "apiKey", "in": "header", "name": self.header}


class OAuth2:
    """OAuth 2.1 scheme descriptor (token validation delegated to a callback).

    A2A mandates OAuth 2.1 for agent-to-agent auth. The ``validator`` takes a
    bearer token and returns an identity (or raises). Wire it to your IdP's
    token introspection / JWKS verification.
    """

    name = "oauth2"

    def __init__(self, validator, *, authorization_url: str = "", token_url: str = "") -> None:
        self._validator = validator
        self.authorization_url = authorization_url
        self.token_url = token_url

    def authenticate(self, headers: dict[str, str]) -> str | None:
        header = headers.get("authorization") or headers.get("Authorization", "")
        if not header.lower().startswith("bearer "):
            raise AuthError("missing OAuth bearer token")
        token = header.split(" ", 1)[1].strip()
        identity = self._validator(token)
        if not identity:
            raise AuthError("OAuth token rejected")
        return identity

    def describe(self) -> dict:
        return {
            "type": "oauth2",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": self.authorization_url,
                    "tokenUrl": self.token_url,
                    "scopes": {},
                }
            },
        }


__all__ = ["AuthScheme", "AuthError", "NoAuth", "BearerTokenAuth", "APIKeyAuth", "OAuth2"]
