"""Tests 4-10: POST /api/route — §7 passthrough, extras, validation, errors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from frugalroute.models import GateVerdict, RouteResult

from conftest import patch_route

# §7 RouteResult keys + the documented derived UI extras.
ROUTE_RESULT_KEYS = {
    "query",
    "strategy",
    "tier_used",
    "escalated",
    "answer",
    "correct",
    "gate",
    "p_strong",
    "refused",
    "cost_usd",
    "latency_s",
    "prompt_version",
}
EXTRA_KEYS = {"decision_margin", "cost_breakdown"}


def _returns(result: RouteResult) -> Callable[..., RouteResult]:
    def _fn(*_args: Any, **_kwargs: Any) -> RouteResult:
        return result

    return _fn


def _raises(exc: Exception) -> Callable[..., RouteResult]:
    def _fn(*_args: Any, **_kwargs: Any) -> RouteResult:
        raise exc

    return _fn


def test_cascade_accepted(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    make_route_result: Callable[..., RouteResult],
) -> None:
    result = make_route_result(
        strategy="cascade",
        escalated=False,
        tier_used="claude-haiku-4-5",
        gate=GateVerdict(sufficient=True, confidence=0.92, reason="clean"),
        cost_usd=0.0015,
    )
    patch_route(monkeypatch, _returns(result))

    resp = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"})
    assert resp.status_code == 200
    body = resp.json()
    assert ROUTE_RESULT_KEYS <= set(body)
    assert EXTRA_KEYS <= set(body)
    assert body["correct"] is None  # live mode never grades
    assert body["gate"] == {"sufficient": True, "confidence": 0.92, "reason": "clean"}
    assert body["cost_breakdown"]["label"] == "= Haiku + gate"
    assert body["cost_breakdown"]["exceeds_always_strong"] is False
    assert body["p_strong"] is None
    assert body["decision_margin"] is None


def test_cascade_escalated_exceeds_opus(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    make_route_result: Callable[..., RouteResult],
) -> None:
    result = make_route_result(
        strategy="cascade",
        escalated=True,
        tier_used="claude-opus-4-8",
        gate=GateVerdict(sufficient=False, confidence=0.40, reason="doubt"),
        cost_usd=0.012,  # cheap + gate + opus > always-opus
    )
    patch_route(monkeypatch, _returns(result))

    body = client.post("/api/route", json={"strategy": "cascade", "query": "hard"}).json()
    assert body["escalated"] is True
    assert body["cost_breakdown"]["label"] == "= Haiku + gate + Opus"
    assert body["cost_breakdown"]["exceeds_always_strong"] is True
    # No fabricated per-term USD: exactly these three keys.
    assert set(body["cost_breakdown"]) == {"label", "always_strong_usd", "exceeds_always_strong"}


def test_predictive_margin(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    make_route_result: Callable[..., RouteResult],
) -> None:
    result = make_route_result(
        strategy="predictive",
        escalated=True,
        tier_used="claude-opus-4-8",
        gate=None,
        p_strong=0.72,
        cost_usd=0.0065,
    )
    patch_route(monkeypatch, _returns(result))

    body = client.post("/api/route", json={"strategy": "predictive", "query": "x"}).json()
    assert body["gate"] is None
    assert body["p_strong"] == 0.72
    # decision_margin = p_strong - theta_used (default theta 0.6).
    assert body["decision_margin"] == pytest.approx(0.72 - 0.6, abs=1e-9)


@pytest.mark.parametrize(
    "payload",
    [
        {"strategy": "cascade"},  # neither query nor example_id
        {"strategy": "cascade", "query": "x", "example_id": "y"},  # both
        {"strategy": "cascade", "query": "x", "tau": 1.7},  # tau out of [0,1]
        {"strategy": "cascade", "query": "x", "theta": -0.1},  # theta out of [0,1]
        {"strategy": "nonsense", "query": "x"},  # unknown strategy
    ],
)
def test_validation_422(client: TestClient, payload: dict[str, Any]) -> None:
    resp = client.post("/api/route", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "bad-request"


def test_missing_key_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_route(
        monkeypatch,
        _raises(RuntimeError("ANTHROPIC_API_KEY is not set. Export it before live calls.")),
    )
    resp = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"})
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["type"] == "missing-key"
    assert "ANTHROPIC_API_KEY" in err["message"]


def test_refusal_passthrough(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    make_route_result: Callable[..., RouteResult],
) -> None:
    result = make_route_result(
        strategy="cascade",
        escalated=True,
        tier_used="claude-opus-4-8",
        gate=None,
        answer="",
        refused=True,
        cost_usd=0.0089,
    )
    patch_route(monkeypatch, _returns(result))

    resp = client.post("/api/route", json={"strategy": "cascade", "query": "bad"})
    assert resp.status_code == 200  # a refusal is data, not an error
    assert resp.json()["refused"] is True


def test_api_error_502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    import httpx

    exc = anthropic.APIConnectionError(
        message="upstream boom",
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    patch_route(monkeypatch, _raises(exc))

    resp = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"})
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "api-error"


def test_unknown_example_404(client: TestClient) -> None:
    resp = client.post("/api/route", json={"strategy": "cascade", "example_id": "does-not-exist"})
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not-found"


def test_oversized_query_422(client: TestClient) -> None:
    # A pathological multi-megabyte body is a clean 422, not an upstream call.
    resp = client.post("/api/route", json={"strategy": "cascade", "query": "x" * 2_000_000})
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "bad-request"
