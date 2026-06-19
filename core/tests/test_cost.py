"""No-key tests for the cache-aware cost engine (split-01 cases 1–7)."""

from __future__ import annotations

import pytest

from frugalroute.llm import cost_usd

ABS = 1e-9


def test_output_only_each_tier() -> None:
    # 1M output tokens at each tier's output price.
    assert cost_usd("claude-opus-4-8", 0, 1_000_000) == pytest.approx(25.0, abs=ABS)
    assert cost_usd("claude-haiku-4-5", 0, 1_000_000) == pytest.approx(5.0, abs=ABS)
    assert cost_usd("claude-sonnet-4-6", 0, 1_000_000) == pytest.approx(15.0, abs=ABS)


def test_fresh_input_only_each_tier() -> None:
    # 1M fresh (uncached) input tokens at each tier's input price.
    assert cost_usd("claude-opus-4-8", 1_000_000, 0) == pytest.approx(5.0, abs=ABS)
    assert cost_usd("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.0, abs=ABS)
    assert cost_usd("claude-sonnet-4-6", 1_000_000, 0) == pytest.approx(3.0, abs=ABS)


def test_cache_write_bucket() -> None:
    # 1M cache-write tokens on Haiku: 1.0 * 1.25 = 1.25.
    assert cost_usd("claude-haiku-4-5", 0, 0, cache_write_tokens=1_000_000) == pytest.approx(
        1.25, abs=ABS
    )


def test_cache_read_bucket() -> None:
    # 1M cache-read tokens on Haiku: 1.0 * 0.10 = 0.10.
    assert cost_usd("claude-haiku-4-5", 0, 0, cache_read_tokens=1_000_000) == pytest.approx(
        0.10, abs=ABS
    )


def test_mixed_realistic_calls() -> None:
    # Hand-computed fixtures reused by split 03's break-even math.
    assert cost_usd("claude-opus-4-8", 150, 250) == pytest.approx(0.0070, abs=ABS)
    assert cost_usd("claude-haiku-4-5", 150, 250) == pytest.approx(0.0014, abs=ABS)
    assert cost_usd("claude-haiku-4-5", 250, 30) == pytest.approx(0.0004, abs=ABS)


def test_all_buckets_combined() -> None:
    # Fresh + cache-write + cache-read + output on Haiku, hand-computed:
    #   100/1e6*1 + 200/1e6*1*1.25 + 300/1e6*1*0.10 + 50/1e6*5
    #   = 1e-4 + 2.5e-4 + 3e-5 + 2.5e-4 = 6.3e-4
    got = cost_usd(
        "claude-haiku-4-5",
        100,
        50,
        cache_write_tokens=200,
        cache_read_tokens=300,
    )
    assert got == pytest.approx(6.3e-4, abs=ABS)


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        cost_usd("gpt-5.5", 100, 100)


def test_zero_everything_is_zero() -> None:
    assert cost_usd("claude-opus-4-8", 0, 0) == pytest.approx(0.0, abs=ABS)
