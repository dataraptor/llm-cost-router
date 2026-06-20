"""Split-09 tests 5-7: GET /api/route/stream (SSE) over the TestClient.

No-key: ``frugalroute.route_events`` (and ``frugalroute.route``) are monkeypatched
with fakes so the streaming framing is exercised deterministically without a
network. The TestClient collects the full ``text/event-stream`` body, which we
parse back into ordered ``(event, data)`` frames.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import frugalroute
import pytest
from fastapi.testclient import TestClient
from frugalroute.models import RouteResult, route_result_to_dict
from frugalroute.router import RouteEvent

from conftest import patch_route

CHEAP = "claude-haiku-4-5"
STRONG = "claude-opus-4-8"


def _parse_sse(text: str) -> list[tuple[str, Any]]:
    """Parse an SSE body into ordered ``(event_type, data)`` frames."""
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


def _accepted_events(result: RouteResult) -> Iterator[RouteEvent]:
    yield RouteEvent("phase", {"phase": "gen", "tier": CHEAP})
    yield RouteEvent("candidate", {"answer": "cand", "tier": CHEAP, "cost_usd": 0.0014})
    yield RouteEvent("cost", {"cost_usd_cumulative": 0.0014})
    yield RouteEvent("phase", {"phase": "gate", "tier": CHEAP})
    yield RouteEvent(
        "gate", {"sufficient": True, "confidence": 0.9, "reason": "ok", "cost_usd": 0.0004}
    )
    yield RouteEvent("cost", {"cost_usd_cumulative": 0.0018})
    yield RouteEvent("done", route_result_to_dict(result))


def _escalated_events(result: RouteResult) -> Iterator[RouteEvent]:
    yield RouteEvent("phase", {"phase": "gen", "tier": CHEAP})
    yield RouteEvent("candidate", {"answer": "cand", "tier": CHEAP, "cost_usd": 0.0014})
    yield RouteEvent("cost", {"cost_usd_cumulative": 0.0014})
    yield RouteEvent("phase", {"phase": "gate", "tier": CHEAP})
    yield RouteEvent(
        "gate", {"sufficient": False, "confidence": 0.4, "reason": "doubt", "cost_usd": 0.0004}
    )
    yield RouteEvent("phase", {"phase": "escalate", "tier": STRONG})
    yield RouteEvent("cost", {"cost_usd_cumulative": 0.0088})
    yield RouteEvent("done", route_result_to_dict(result))


def _patch_events(
    monkeypatch: pytest.MonkeyPatch, gen_factory: Callable[[], Iterator[RouteEvent]]
) -> None:
    monkeypatch.setattr(frugalroute, "route_events", lambda *a, **k: gen_factory())


# --- 5. accepted: ordered frames; terminal done == POST /api/route body -----


def test_accepted_stream_order_and_done_equals_post(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    make_route_result: Callable[..., RouteResult],
) -> None:
    from frugalroute.models import GateVerdict

    result = make_route_result(
        strategy="cascade",
        escalated=False,
        tier_used=CHEAP,
        gate=GateVerdict(sufficient=True, confidence=0.9, reason="ok"),
        cost_usd=0.0018,
    )
    _patch_events(monkeypatch, lambda: _accepted_events(result))
    patch_route(monkeypatch, lambda *a, **k: result)

    resp = client.get("/api/route/stream", params={"strategy": "cascade", "query": "2+2?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert resp.headers.get("cache-control") == "no-cache"

    frames = _parse_sse(resp.text)
    assert [t for t, _ in frames] == ["phase", "candidate", "cost", "phase", "gate", "cost", "done"]

    done = next(d for t, d in frames if t == "done")
    post_body = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"}).json()
    assert done == post_body  # terminal done body deep-equals the sync contract


def test_escalated_stream_includes_escalate_phase(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    make_route_result: Callable[..., RouteResult],
) -> None:
    from frugalroute.models import GateVerdict

    result = make_route_result(
        strategy="cascade",
        escalated=True,
        tier_used=STRONG,
        gate=GateVerdict(sufficient=False, confidence=0.4, reason="doubt"),
        cost_usd=0.0088,
    )
    _patch_events(monkeypatch, lambda: _escalated_events(result))

    frames = _parse_sse(
        client.get("/api/route/stream", params={"strategy": "cascade", "query": "hard"}).text
    )
    phases = [d["phase"] for t, d in frames if t == "phase"]
    assert phases == ["gen", "gate", "escalate"]
    done = next(d for t, d in frames if t == "done")
    assert done["escalated"] is True
    assert done["cost_breakdown"]["exceeds_always_strong"] is True


# --- 6. bad params → pre-stream 422, never a 200 empty stream ---------------


@pytest.mark.parametrize(
    "params",
    [
        {"strategy": "cascade", "query": "x", "tau": 9},  # out of [0,1]
        {"strategy": "cascade", "query": "x", "theta": -0.1},  # out of [0,1]
        {"strategy": "cascade"},  # neither query nor example_id
        {"strategy": "cascade", "query": "x", "example_id": "y"},  # both
        {"strategy": "nonsense", "query": "x"},  # unknown strategy
    ],
)
def test_bad_params_pre_stream_422(client: TestClient, params: dict[str, Any]) -> None:
    resp = client.get("/api/route/stream", params=params)
    assert resp.status_code == 422
    assert "event-stream" not in resp.headers.get("content-type", "")
    assert resp.json()["error"]["type"] == "bad-request"


def test_unknown_example_pre_stream_404(client: TestClient) -> None:
    resp = client.get("/api/route/stream", params={"strategy": "cascade", "example_id": "nope"})
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not-found"


# --- 7. missing key → an `error` event, not a dropped connection ------------


def test_missing_key_is_error_event(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> Iterator[RouteEvent]:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it before live calls.")

    monkeypatch.setattr(frugalroute, "route_events", _boom)

    resp = client.get("/api/route/stream", params={"strategy": "cascade", "query": "2+2?"})
    assert resp.status_code == 200  # the stream opens; the failure is an event
    frames = _parse_sse(resp.text)
    assert frames[-1][0] == "error"
    assert frames[-1][1]["type"] == "missing-key"


def test_typed_apierror_mid_stream_preserves_its_type(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A typed APIError raised mid-stream is surfaced as an event with its OWN type
    # (not re-wrapped as a generic api-error).
    from frugalroute_api import errors

    def _events(*_a: Any, **_k: Any) -> Iterator[RouteEvent]:
        yield RouteEvent("phase", {"phase": "gen", "tier": CHEAP})
        raise errors.missing_key("no backend configured")

    monkeypatch.setattr(frugalroute, "route_events", _events)

    frames = _parse_sse(
        client.get("/api/route/stream", params={"strategy": "cascade", "query": "x"}).text
    )
    assert frames[-1][0] == "error"
    assert frames[-1][1] == {"type": "missing-key", "message": "no backend configured"}


def test_engine_error_is_error_event(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    import httpx

    def _events(*_a: Any, **_k: Any) -> Iterator[RouteEvent]:
        yield RouteEvent("phase", {"phase": "gen", "tier": CHEAP})
        raise anthropic.APIConnectionError(
            message="upstream boom",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

    monkeypatch.setattr(frugalroute, "route_events", _events)

    frames = _parse_sse(
        client.get("/api/route/stream", params={"strategy": "cascade", "query": "x"}).text
    )
    assert frames[0][0] == "phase"  # the stream had started before the failure
    assert frames[-1][0] == "error"
    assert frames[-1][1]["type"] == "api-error"
