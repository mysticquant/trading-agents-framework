"""In-memory, time-limited memoization for evidence sources. A backtest asks
for the same ticker's raw data once per simulated day; this turns that into
one network fetch per ticker per run, while still refreshing in a
long-running live loop. Failures are remembered too, for a shorter time, so
a source that is down (or refusing us) isn't hit again for every ticker on
every simulated day — that would also risk rate limits or an SEC IP block."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class AsyncTTLCache(Generic[T]):
    def __init__(
        self,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        failure_ttl_seconds: float = 300.0,
    ) -> None:
        self._ttl = ttl_seconds
        self._failure_ttl = failure_ttl_seconds
        self._clock = clock
        self._values: dict[str, tuple[float, T]] = {}
        self._failures: dict[str, tuple[float, Exception]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, key: str, fetch: Callable[[], Awaitable[T]]) -> T:
        # One lock per key so concurrent callers for the same ticker share a
        # single fetch instead of each hitting the network.
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._values.get(key)
            if cached is not None and self._clock() - cached[0] < self._ttl:
                return cached[1]
            failed = self._failures.get(key)
            if failed is not None and self._clock() - failed[0] < self._failure_ttl:
                raise failed[1]
            try:
                value = await fetch()
            except Exception as exc:
                self._failures[key] = (self._clock(), exc)
                raise
            self._failures.pop(key, None)
            self._values[key] = (self._clock(), value)
            return value
