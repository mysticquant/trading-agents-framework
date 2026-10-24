"""In-memory, time-limited memoization for evidence sources. A backtest asks
for the same ticker's raw data once per simulated day; this turns that into
one network fetch per ticker per run, while still refreshing in a
long-running live loop."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class AsyncTTLCache(Generic[T]):
    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._values: dict[str, tuple[float, T]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, key: str, fetch: Callable[[], Awaitable[T]]) -> T:
        # One lock per key so concurrent callers for the same ticker share a
        # single fetch instead of each hitting the network.
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._values.get(key)
            if cached is not None and self._clock() - cached[0] < self._ttl:
                return cached[1]
            value = await fetch()
            self._values[key] = (self._clock(), value)
            return value
