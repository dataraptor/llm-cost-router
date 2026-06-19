"""Live cascade smoke (test 19) — full route against the gpt-5.5 backend.

This exercises the whole cascade end-to-end against the live Azure OpenAI gpt-5.5
backend through the Anthropic-shaped adapter, including the structured gate
(``messages.parse``) added in split 03. Marked ``@pytest.mark.azure`` (the live
backend for this build); auto-skipped without ``AZURE_OPENAI_API_KEY``. The
native-Anthropic ``@pytest.mark.api`` variant skips cleanly without a key.

Assertions are on structure/ranges, never exact accuracy (the API is
non-deterministic), per the build-spec §12 testing strategy.
"""

from __future__ import annotations

import pytest

from frugalroute.llm import DEFAULT_TIERS
from frugalroute.models import GateVerdict, RouteResult
from frugalroute.router import route

_GSM8K_QUERY = "If a box has 12 apples and 5 are removed, how many apples remain?"


def _assert_well_formed(result: RouteResult) -> None:
    assert isinstance(result, RouteResult)
    assert result.strategy == "cascade"
    assert result.tier_used in DEFAULT_TIERS
    assert isinstance(result.answer, str)
    assert isinstance(result.escalated, bool)
    assert isinstance(result.refused, bool)
    assert result.cost_usd > 0  # at least the cheap call + gate were billed
    assert result.latency_s >= 0
    # Cascade always either ran the gate (verdict present) or skipped it on a cheap
    # refusal (gate is None, refused True) — never an incoherent in-between.
    assert result.gate is None or isinstance(result.gate, GateVerdict)
    if result.gate is None:
        assert result.escalated and result.refused


@pytest.mark.azure
def test_cascade_route_live_smoke() -> None:
    from frugalroute.azure_client import get_azure_client

    result = route(
        _GSM8K_QUERY, strategy="cascade", benchmark="gsm8k", tau=0.8, client=get_azure_client()
    )
    _assert_well_formed(result)


@pytest.mark.api
def test_cascade_route_live_smoke_native() -> None:
    # Native-Anthropic variant: skips cleanly without ANTHROPIC_API_KEY.
    result = route(_GSM8K_QUERY, strategy="cascade", benchmark="gsm8k", tau=0.8)
    _assert_well_formed(result)
