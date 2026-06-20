"""Test 16 (R10): live POST /api/route over HTTP returns a well-formed RouteResult.

``@pytest.mark.azure`` exercises the gpt-5.5 adapter (this build's live backend) by
flipping the app to ``backend=azure``; ``@pytest.mark.api`` exercises the native
Anthropic path. Both auto-skip without their key. Asserts structure/ranges only
(model output is non-deterministic), never an exact answer or accuracy.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from frugalroute import llm


def _assert_well_formed_route(body: dict[str, object]) -> None:
    assert body["strategy"] == "cascade"
    assert body["tier_used"] in llm.DEFAULT_TIERS
    assert isinstance(body["cost_usd"], float) and body["cost_usd"] > 0
    assert body["correct"] is None  # live mode never grades
    assert "cost_breakdown" in body
    assert isinstance(body["cost_breakdown"]["always_strong_usd"], float)


@pytest.mark.azure
def test_route_live_azure(client: TestClient, override_settings: Callable[..., None]) -> None:
    override_settings(backend="azure")
    resp = client.post(
        "/api/route",
        json={"strategy": "cascade", "example_id": "gsm8k-1142", "tau": 0.8},
    )
    assert resp.status_code == 200, resp.text
    _assert_well_formed_route(resp.json())


@pytest.mark.api
def test_route_live_native(client: TestClient) -> None:
    resp = client.post(
        "/api/route",
        json={"strategy": "cascade", "example_id": "gsm8k-1142", "tau": 0.8},
    )
    assert resp.status_code == 200, resp.text
    _assert_well_formed_route(resp.json())
