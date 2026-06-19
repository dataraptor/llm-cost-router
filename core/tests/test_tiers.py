"""No-key tests for the config-driven tier list (split-01 cases 8–9)."""

from __future__ import annotations

from frugalroute.llm import cheap_tier, strong_tier


def test_default_two_tier_list() -> None:
    assert cheap_tier() == "claude-haiku-4-5"
    assert strong_tier() == "claude-opus-4-8"


def test_three_tier_override_no_code_change() -> None:
    tiers = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]
    # cheapest is still Haiku, strongest is still Opus — purely config-driven.
    assert cheap_tier(tiers) == "claude-haiku-4-5"
    assert strong_tier(tiers) == "claude-opus-4-8"
