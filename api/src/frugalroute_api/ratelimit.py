"""A simple in-memory per-IP token-bucket rate limiter (split-11 §2).

Each client IP gets a bucket of ``capacity`` tokens that refills at
``refill_per_s`` tokens/second. Each request costs one token; when a bucket is
empty the request is rejected with a ``Retry-After`` hint. This throttles a
single noisy caller; it is **distinct** from the Anthropic-side 429 (which the
SDK retries and split-09 surfaces as a ``retry`` event).

The clock is injectable (``time_fn``) so the reset-after-window behavior is
deterministically testable without sleeping.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class RateLimiter:
    """Per-key token bucket. ``allow(key)`` returns ``(ok, retry_after_s)``."""

    def __init__(
        self,
        *,
        capacity: int,
        refill_per_s: float,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}.")
        if refill_per_s < 0:
            raise ValueError(f"refill_per_s must be >= 0, got {refill_per_s}.")
        self._capacity = capacity
        self._refill = refill_per_s
        self._time = time_fn
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        """Consume one token for ``key``. Returns ``(allowed, retry_after_seconds)``."""
        now = self._time()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self._capacity), updated_at=now)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill)
                bucket.updated_at = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0
            # Empty: estimate when the next token arrives.
            if self._refill > 0:
                retry_after = max(1, int((1.0 - bucket.tokens) / self._refill + 0.999))
            else:
                retry_after = 1
            return False, retry_after
