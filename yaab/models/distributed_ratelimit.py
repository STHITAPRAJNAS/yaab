"""Shared rate limiter that caps requests across every replica.

The in-process :class:`~yaab.models.resilient.RateLimiter` protects one process.
When the same agent runs behind a load balancer across many replicas, each
replica's local limiter would grant its own full budget, so a configured
``rate=10`` would become ``10 x replicas``. :class:`RedisRateLimiter` fixes that
by keeping the token bucket in a shared store keyed by a bucket name, so the
budget is *global*: ``rate=10`` is ten permits per window no matter how many
replicas draw from it.

It exposes the same ``async def acquire()`` as the local limiter, so it is a
drop-in for anything that accepts an ``acquire()``-compatible limiter.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

# Atomic token-bucket refill-and-take. Returns 1 if a permit was granted, else 0.
# Kept server-side so concurrent replicas can never double-spend a permit:
# the read-modify-write of the bucket happens in a single round trip.
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local per = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local data = redis.call('GET', key)
local tokens, updated
if data then
  local sep = string.find(data, '|', 1, true)
  tokens = tonumber(string.sub(data, 1, sep - 1))
  updated = tonumber(string.sub(data, sep + 1))
else
  tokens = rate
  updated = now
end
local elapsed = now - updated
if elapsed < 0 then elapsed = 0 end
tokens = math.min(rate, tokens + elapsed * (rate / per))
local granted = 0
if tokens >= cost then
  tokens = tokens - cost
  granted = 1
end
redis.call('SET', key, tostring(tokens) .. '|' .. tostring(now))
return granted
"""


class RedisRateLimiter:
    """A global token bucket: at most ``rate`` permits per ``per`` seconds.

    All limiter objects that share the same ``bucket`` name and store enforce a
    single combined budget, so the limit holds across every replica.

    Args:
        rate: Maximum permits available per window.
        per: Window length in seconds (default 60).
        bucket: Shared name identifying this budget; limiters with the same name
            and store draw from the same pool.
        url: Connection URL used when no ``client`` is injected.
        prefix: Key namespace prefix.
        client: An optional pre-built client (used in tests to inject a fake);
            when provided, ``url`` is ignored and no driver import is attempted.
        poll: Seconds to wait between retries while the bucket is empty.
    """

    def __init__(
        self,
        rate: int,
        per: float = 60.0,
        *,
        bucket: str = "default",
        url: str = "redis://localhost:6379/0",
        prefix: str = "yaab:ratelimit",
        client: Any = None,
        poll: float = 0.05,
    ) -> None:
        self.rate = rate
        self.per = per
        self.bucket = bucket
        self._prefix = prefix
        self._poll = poll
        self._redis: Any
        if client is not None:
            self._redis = client
        else:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "redis is required for RedisRateLimiter. "
                    "Install with `pip install 'yaab-sdk[redis]'`."
                ) from exc
            self._redis = redis.Redis.from_url(url)

    def _key(self) -> str:
        return f"{self._prefix}:{self.bucket}"

    def _try_acquire(self, cost: float = 1.0) -> bool:
        """One atomic refill-and-take; True if a permit was granted."""
        now = self._now()
        granted = self._redis.eval(
            _TOKEN_BUCKET_LUA, 1, self._key(), self.rate, self.per, cost, now
        )
        return bool(int(granted))

    def _now(self) -> float:
        # Use the store's clock when it exposes one (test fakes freeze time);
        # otherwise fall back to wall-clock so refill tracks real elapsed time.
        clock = getattr(self._redis, "now", None)
        if isinstance(clock, (int, float)):
            return float(clock)
        return time.time()

    async def acquire(self) -> None:
        """Block until a permit is available, then consume one.

        Retries the atomic take with a short poll so multiple coroutines and
        replicas fairly contend for the shared budget without busy-spinning.
        """
        while True:
            if await asyncio.to_thread(self._try_acquire):
                return
            await asyncio.sleep(self._poll)


__all__ = ["RedisRateLimiter"]
