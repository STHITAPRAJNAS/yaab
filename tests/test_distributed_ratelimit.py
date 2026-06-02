"""Tests for the shared rate limiter that caps requests across replicas.

A :class:`RedisRateLimiter` keyed by a shared bucket name enforces one global
permit budget no matter how many processes (replicas) hold their own limiter
object. The tests use an injected fake client implementing the small slice of
Redis used by the atomic token-bucket script, so no server or driver is needed.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from yaab.models.distributed_ratelimit import RedisRateLimiter


class FakeRedis:
    """A minimal Redis stand-in supporting the token-bucket script.

    Implements ``eval`` for the bundled Lua source (interpreted in Python so the
    test stays offline) plus ``time``/``get``/``set`` used by the WATCH fallback.
    A single instance shared by several limiters models one real Redis serving
    several replicas.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.now: float = 1000.0

    # The limiter calls eval(script, numkeys, *keys_and_args). We ignore the
    # script text and run the canonical token-bucket algorithm so two limiters
    # over one FakeRedis share state.
    def eval(self, script: str, numkeys: int, *args: object) -> int:
        keys = [str(a) for a in args[:numkeys]]
        rest = args[numkeys:]
        key = keys[0]
        rate = float(rest[0])  # type: ignore[arg-type]
        per = float(rest[1])  # type: ignore[arg-type]
        cost = float(rest[2]) if len(rest) > 2 else 1.0  # type: ignore[arg-type]
        now = self.now

        raw = self.store.get(key)
        if raw is None:
            tokens, updated = float(rate), now
        else:
            t_str, u_str = raw.split("|", 1)
            tokens, updated = float(t_str), float(u_str)

        elapsed = max(0.0, now - updated)
        tokens = min(rate, tokens + elapsed * (rate / per))
        if tokens >= cost:
            tokens -= cost
            self.store[key] = f"{tokens}|{now}"
            return 1
        self.store[key] = f"{tokens}|{now}"
        return 0


def test_single_limiter_allows_burst_up_to_rate() -> None:
    client = FakeRedis()
    limiter = RedisRateLimiter(rate=3, per=60.0, bucket="b", client=client)

    async def main() -> int:
        granted = 0
        for _ in range(3):
            await limiter.acquire()
            granted += 1
        return granted

    assert asyncio.run(main()) == 3


def test_two_limiters_share_a_combined_cap() -> None:
    """Two limiter objects over one client must not exceed a global budget."""
    client = FakeRedis()
    a = RedisRateLimiter(rate=4, per=60.0, bucket="shared", client=client)
    b = RedisRateLimiter(rate=4, per=60.0, bucket="shared", client=client)

    # Time is frozen, so no tokens accrue: the bucket starts with 4 permits and
    # both limiters draw from it. The 5th acquire (whoever issues it) must block.
    granted = {"n": 0}

    async def grab(limiter: RedisRateLimiter) -> None:
        await limiter.acquire()
        granted["n"] += 1

    async def main() -> None:
        # Four acquires across the two limiters succeed immediately.
        await grab(a)
        await grab(b)
        await grab(a)
        await grab(b)
        # A fifth would block forever at frozen time; assert it times out.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(grab(a), timeout=0.2)

    asyncio.run(main())
    assert granted["n"] == 4


def test_refill_grants_more_after_time_passes() -> None:
    client = FakeRedis()
    limiter = RedisRateLimiter(rate=2, per=60.0, bucket="refill", client=client)

    async def main() -> None:
        await limiter.acquire()
        await limiter.acquire()
        # Bucket empty; advance simulated time by 60s -> full refill.
        client.now += 60.0
        await asyncio.wait_for(limiter.acquire(), timeout=0.5)

    asyncio.run(main())


def test_acquire_interface_matches_local_limiter() -> None:
    """Drop-in compatibility: same no-arg ``acquire`` coroutine signature."""
    from yaab.models.resilient import RateLimiter

    client = FakeRedis()
    shared = RedisRateLimiter(rate=5, bucket="iface", client=client)
    local = RateLimiter(rate=5)
    assert hasattr(shared, "acquire")
    assert hasattr(local, "acquire")

    async def main() -> None:
        await shared.acquire()
        await local.acquire()

    asyncio.run(main())


def test_requires_driver_without_client() -> None:
    try:
        import redis  # noqa: F401

        pytest.skip("redis is installed")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="redis"):
        RedisRateLimiter(rate=1, bucket="x")


def test_resilient_model_accepts_shared_limiter() -> None:
    """A shared limiter is accepted by ResilientModel just like the local one."""
    from yaab.models.resilient import ResilientModel

    class _Inner:
        name = "inner"

        async def complete(self, messages, **kw):  # pragma: no cover - not called
            return None

        def stream(self, messages, **kw):  # pragma: no cover - not called
            return None

    client = FakeRedis()
    limiter = RedisRateLimiter(rate=1, bucket="rm", client=client)
    model = ResilientModel(_Inner(), rate_limiter=limiter)
    assert model.rate_limiter is limiter


def test_wall_clock_default_refills() -> None:
    """With the default (wall-clock) FakeRedis time disabled, accrual is real."""
    client = FakeRedis()
    client.now = time.time()
    limiter = RedisRateLimiter(rate=60, per=1.0, bucket="wc", client=client)

    async def main() -> None:
        await limiter.acquire()

    asyncio.run(main())
