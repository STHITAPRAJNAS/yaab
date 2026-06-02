"""Rate limiters as a swappable backend kind.

A rate limiter caps how fast an agent calls a model. In one process the
in-process token bucket (:class:`~yaab.models.resilient.RateLimiter`) is enough.
Behind a load balancer across many replicas, each replica's local limiter would
grant its own full budget, so a configured ``rate=10`` would silently become
``10 x replicas``. The shared :class:`~yaab.models.distributed_ratelimit.RedisRateLimiter`
keeps the budget in one store so the limit is *global*.

Both expose the same ``async def acquire()``, so swapping one for the other is a
one-line change. Each is registered under the ``ratelimiter`` component kind so
it can be selected by name: ``yaab.extensions.get("ratelimiter", "redis", ...)``.
"""

from __future__ import annotations

from typing import Any

from .models.resilient import RateLimiter

# ``RedisRateLimiter`` is re-exported lazily via ``__getattr__`` below so the
# redis driver stays an optional import; F822 does not see the lazy name in a
# non-package module, so it is suppressed here intentionally.
__all__ = ["RateLimiter", "RedisRateLimiter"]  # noqa: F822


def __getattr__(name: str) -> Any:
    # Lazy import so the redis driver is only needed when the shared limiter is used.
    if name == "RedisRateLimiter":
        from .models.distributed_ratelimit import RedisRateLimiter

        return RedisRateLimiter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _register_backends() -> None:
    """Register rate limiters as ``ratelimiter`` components (selectable by name)."""
    from .extensions import register

    register("ratelimiter", "memory", lambda **kw: RateLimiter(**kw))

    def _redis(**kw: Any) -> Any:
        from .models.distributed_ratelimit import RedisRateLimiter

        return RedisRateLimiter(**kw)

    register("ratelimiter", "redis", _redis)


_register_backends()
