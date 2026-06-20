"""Unit tests for the token-bucket rate limiter (split-11)."""

from __future__ import annotations

import pytest

from frugalroute_api.ratelimit import RateLimiter


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError):
        RateLimiter(capacity=0, refill_per_s=1.0)
    with pytest.raises(ValueError):
        RateLimiter(capacity=2, refill_per_s=-1.0)


def test_burst_then_throttle_then_refill() -> None:
    clock = {"t": 0.0}
    rl = RateLimiter(capacity=2, refill_per_s=1.0, time_fn=lambda: clock["t"])
    assert rl.allow("ip")[0] is True
    assert rl.allow("ip")[0] is True
    ok, retry_after = rl.allow("ip")
    assert ok is False and retry_after >= 1
    # One second later, one token has refilled.
    clock["t"] = 1.0
    assert rl.allow("ip")[0] is True


def test_zero_refill_never_recovers_within_window() -> None:
    clock = {"t": 0.0}
    rl = RateLimiter(capacity=1, refill_per_s=0.0, time_fn=lambda: clock["t"])
    assert rl.allow("ip")[0] is True
    ok, retry_after = rl.allow("ip")
    assert ok is False and retry_after == 1
    clock["t"] = 1000.0  # no refill rate → still throttled
    assert rl.allow("ip")[0] is False


def test_separate_ips_have_separate_buckets() -> None:
    rl = RateLimiter(capacity=1, refill_per_s=0.0, time_fn=lambda: 0.0)
    assert rl.allow("a")[0] is True
    assert rl.allow("b")[0] is True  # different IP, fresh bucket
    assert rl.allow("a")[0] is False
