"""Split-11 §2/§3: API hardening — request-id, back-pressure, rate-limit,
timeout, metrics, and access-log hygiene (no key, TestClient).

Each test builds a **fresh app** via ``create_app(settings=...)`` so the per-app
runtime state (rate limiter, concurrency semaphore, metrics) is isolated and the
hardening config is whatever the test sets — no shared module-level app.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import frugalroute
import pytest
from fastapi.testclient import TestClient
from frugalroute import obs
from frugalroute.models import GateVerdict, RouteResult

from frugalroute_api.app import create_app
from frugalroute_api.config import Settings
from frugalroute_api.ratelimit import RateLimiter

SENTINEL = "sk-ant-SENTINEL-never-logged-0123456789"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test from known engine config (no stray FRUGALROUTE_* / key)."""
    for var in (
        "FRUGALROUTE_MAX_CONCURRENCY",
        "FRUGALROUTE_REQUEST_TIMEOUT_S",
        "FRUGALROUTE_LOG_LEVEL",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    obs.reset_runtime()
    yield
    obs.reset_runtime()


def _make_result(
    *, cost_usd: float = 0.0015, escalated: bool = False, refused: bool = False
) -> RouteResult:
    return RouteResult(
        query="q",
        strategy="cascade",
        tier_used="claude-opus-4-8" if escalated else "claude-haiku-4-5",
        escalated=escalated,
        answer="" if refused else "The answer is 4.",
        correct=None,
        gate=GateVerdict(sufficient=not escalated, confidence=0.9, reason="ok"),
        p_strong=None,
        refused=refused,
        cost_usd=cost_usd,
        latency_s=0.01,
        prompt_version="v1",
    )


def _build(
    monkeypatch: pytest.MonkeyPatch, route_fn: Callable[..., RouteResult], **settings_kw: Any
) -> TestClient:
    """A TestClient over a fresh app with ``frugalroute.route`` patched."""
    monkeypatch.setattr(frugalroute, "route", route_fn)
    app = create_app(Settings(**settings_kw))
    return TestClient(app, raise_server_exceptions=False)


@contextmanager
def _capture() -> Iterator[list[logging.LogRecord]]:
    """Capture every record emitted on the ``frugalroute`` logger tree."""
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _H()
    logger = logging.getLogger("frugalroute")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _rendered(records: list[logging.LogRecord]) -> str:
    formatter = obs.JsonFormatter()
    return "\n".join(formatter.format(r) for r in records)


# --- Test 6: request id -----------------------------------------------------
def test_request_id_echoed_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build(monkeypatch, lambda *a, **k: _make_result())
    with _capture() as records:
        resp = client.post(
            "/api/route",
            json={"strategy": "cascade", "query": "2+2?"},
            headers={"X-Request-ID": "my-trace-id"},
        )
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == "my-trace-id"
    assert "my-trace-id" in _rendered(records)


def test_request_id_generated_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build(monkeypatch, lambda *a, **k: _make_result())
    with _capture() as records:
        resp = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"})
    assert resp.status_code == 200
    generated = resp.headers.get("X-Request-ID")
    assert generated and len(generated) >= 8
    # The generated id appears in an access log line.
    lines = [json.loads(obs.JsonFormatter().format(r)) for r in records]
    access = [line for line in lines if line.get("msg") == "request"]
    assert any(line.get("request_id") == generated for line in access)


# --- Test 7: concurrency back-pressure --------------------------------------
def test_over_capacity_returns_503_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build(monkeypatch, lambda *a, **k: _make_result())
    app = client.app
    # Saturate the back-pressure semaphore so the next engine request is shed.
    app.state.concurrency = threading.BoundedSemaphore(1)
    assert app.state.concurrency.acquire(blocking=False)
    try:
        resp = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"})
        assert resp.status_code == 503
        assert resp.json()["error"]["type"] == "busy"
        assert "Retry-After" in resp.headers
    finally:
        app.state.concurrency.release()
    # Recovers once capacity frees up.
    ok = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"})
    assert ok.status_code == 200


# --- Test 8: per-IP rate limit ----------------------------------------------
def test_rate_limit_429_then_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build(monkeypatch, lambda *a, **k: _make_result(), rate_limit_enabled=True)
    # Replace the limiter with one driven by a controllable clock — deterministic,
    # no sleeping. Capacity 2, refill 1 token/s.
    clock = {"t": 1000.0}
    client.app.state.limiter = RateLimiter(capacity=2, refill_per_s=1.0, time_fn=lambda: clock["t"])
    body = {"strategy": "cascade", "query": "q"}
    # Burst of 2 allowed, the 3rd is throttled (clock frozen → no refill).
    assert client.post("/api/route", json=body).status_code == 200
    assert client.post("/api/route", json=body).status_code == 200
    throttled = client.post("/api/route", json=body)
    assert throttled.status_code == 429
    assert throttled.json()["error"]["type"] == "rate-limited"
    assert "Retry-After" in throttled.headers
    # Advance the clock past the window → tokens refill → allowed again.
    clock["t"] += 5.0
    assert client.post("/api/route", json=body).status_code == 200


# --- Test 9: server timeout -------------------------------------------------
def test_server_timeout_returns_504(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRUGALROUTE_REQUEST_TIMEOUT_S", "0.2")

    def _slow(*_a: Any, **_k: Any) -> RouteResult:
        time.sleep(0.6)
        return _make_result()

    client = _build(monkeypatch, _slow)
    start = time.monotonic()
    resp = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"})
    elapsed = time.monotonic() - start
    assert resp.status_code == 504
    assert resp.json()["error"]["type"] == "api-error"
    assert elapsed < 0.5, f"timed out request hung for {elapsed:.2f}s"


# --- Test 10: metrics -------------------------------------------------------
def test_metrics_counters_and_cost_total(monkeypatch: pytest.MonkeyPatch) -> None:
    def _route(query: str, **_k: Any) -> RouteResult:
        if query == "hard":
            return _make_result(cost_usd=0.005, escalated=True, refused=True)
        return _make_result(cost_usd=0.002)

    client = _build(monkeypatch, _route)
    for _ in range(3):
        assert (
            client.post("/api/route", json={"strategy": "cascade", "query": "easy"}).status_code
            == 200
        )
    assert (
        client.post("/api/route", json={"strategy": "cascade", "query": "hard"}).status_code == 200
    )

    metrics = client.get("/api/metrics").json()
    assert metrics["requests_total"] == 4
    assert metrics["cost_usd_total"] == pytest.approx(3 * 0.002 + 0.005, abs=1e-9)
    assert metrics["escalation_rate"] == pytest.approx(0.25, abs=1e-9)
    assert metrics["refused_total"] == 1
    assert metrics["latency_p50_s"] is not None
    assert metrics["latency_p95_s"] is not None


# --- Test 11: access-log hygiene --------------------------------------------
def test_access_logs_omit_key_and_query_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    obs.reset_runtime()
    client = _build(monkeypatch, lambda *a, **k: _make_result())
    secret_query = "PLEASE-DO-NOT-LOG-THIS-QUERY-BODY"
    with _capture() as records:
        resp = client.post("/api/route", json={"strategy": "cascade", "query": secret_query})
    assert resp.status_code == 200
    rendered = _rendered(records)
    assert SENTINEL not in rendered
    assert secret_query not in rendered


# --- R11 adversarial: hammer concurrency + rate-limit with a sentinel set ----
def test_adversarial_load_shedding_and_no_secret_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    obs.reset_runtime()

    def _slow(*_a: Any, **_k: Any) -> RouteResult:
        time.sleep(0.05)
        return _make_result()

    client = _build(
        monkeypatch,
        _slow,
        rate_limit_enabled=True,
        rate_limit_burst=5,
        rate_limit_refill_per_s=5.0,
    )
    app = client.app
    app.state.concurrency = threading.BoundedSemaphore(2)

    statuses: list[int] = []
    lock = threading.Lock()

    def hammer() -> None:
        try:
            r = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"})
            with lock:
                statuses.append(r.status_code)
                if r.status_code in (429, 503):
                    assert "Retry-After" in r.headers
        except Exception as exc:  # noqa: BLE001 - record so a hang/crash fails loudly
            with lock:
                statuses.append(-1)
            raise exc

    with _capture() as records:
        threads = [threading.Thread(target=hammer) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not any(t.is_alive() for t in threads), "a request hung (never returned)"

    # It sheds load with typed responses, never crashes, and recovers.
    assert statuses, "no requests completed"
    assert -1 not in statuses
    assert all(s in (200, 429, 503) for s in statuses), statuses
    assert any(s in (429, 503) for s in statuses), "expected some load to be shed"
    time.sleep(0.3)
    assert client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"}).status_code in (
        200,
        429,
    )
    # The sentinel key appears nowhere across all captured logs.
    assert SENTINEL not in _rendered(records)
