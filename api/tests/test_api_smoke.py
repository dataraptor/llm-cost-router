"""Test 16 (R10): live POST /api/route over HTTP returns a well-formed RouteResult.

``@pytest.mark.azure`` exercises the gpt-5.5 adapter (this build's live backend) by
flipping the app to ``backend=azure``; ``@pytest.mark.api`` exercises the native
Anthropic path. Both auto-skip without their key. Asserts structure/ranges only
(model output is non-deterministic), never an exact answer or accuracy.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

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


def _parse_sse(text: str) -> list[tuple[str, Any]]:
    frames: list[tuple[str, Any]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event: str | None = None
        data: str | None = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        frames.append((event or "message", json.loads(data) if data is not None else None))
    return frames


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


@pytest.mark.azure
def test_route_stream_live_azure(
    client: TestClient, override_settings: Callable[..., None]
) -> None:
    # Split-09: the live SSE stream emits ordered events and a terminal `done`
    # whose body matches the synchronous contract (well-formed RouteResult).
    override_settings(backend="azure")
    resp = client.get(
        "/api/route/stream",
        params={"strategy": "cascade", "example_id": "gsm8k-1142", "tau": 0.8},
    )
    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    frames = _parse_sse(resp.text)
    types = [t for t, _ in frames]
    assert types[0] == "phase" and types[-1] == "done"
    _assert_well_formed_route(next(d for t, d in frames if t == "done"))
