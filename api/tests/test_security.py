"""Split-14 security hardening — input limits, CORS lock, error hygiene, and
no path-traversal (no key, TestClient).

Mirrors ``test_hardening``'s fresh-app pattern: each test builds its own
``create_app(Settings(...))`` so settings (CORS, etc.) are isolated.
"""

from __future__ import annotations

from typing import Any

import frugalroute
import pytest
from fastapi.testclient import TestClient
from frugalroute.models import GateVerdict, RouteResult

from frugalroute_api.app import create_app
from frugalroute_api.config import Settings
from frugalroute_api.middleware import MAX_BODY_BYTES
from frugalroute_api.schemas import MAX_QUERY_CHARS


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "FRUGALROUTE_MAX_CONCURRENCY",
        "FRUGALROUTE_REQUEST_TIMEOUT_S",
        "FRUGALROUTE_LOG_LEVEL",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _result() -> RouteResult:
    return RouteResult(
        query="q",
        strategy="cascade",
        tier_used="claude-haiku-4-5",
        escalated=False,
        answer="The answer is 4.",
        correct=None,
        gate=GateVerdict(sufficient=True, confidence=0.9, reason="ok"),
        p_strong=None,
        refused=False,
        cost_usd=0.0012,
        latency_s=0.01,
        prompt_version="v1",
    )


def _client(monkeypatch: pytest.MonkeyPatch, **settings_kw: Any) -> TestClient:
    monkeypatch.setattr(frugalroute, "route", lambda *a, **k: _result())
    app = create_app(Settings(**settings_kw))
    return TestClient(app, raise_server_exceptions=False)


def _is_error_envelope(body: dict[str, Any]) -> bool:
    """Every non-2xx body is exactly ``{"error": {type, message, detail?}}``."""
    if set(body.keys()) != {"error"}:
        return False
    err = body["error"]
    return isinstance(err, dict) and {"type", "message"} <= set(err.keys())


# --- Input limits -----------------------------------------------------------
def test_oversized_body_is_typed_413(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-100KB body is rejected up front with a typed 413, not a crash."""
    client = _client(monkeypatch)
    huge = "x" * (MAX_BODY_BYTES + 1000)
    resp = client.post("/api/route", json={"strategy": "cascade", "query": huge})
    assert resp.status_code == 413
    body = resp.json()
    assert _is_error_envelope(body)
    assert body["error"]["type"] == "bad-request"


def test_overlong_query_is_typed_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """A query over the char cap (but under the body cap) is a clean 422."""
    client = _client(monkeypatch)
    long_q = "a" * (MAX_QUERY_CHARS + 1)
    assert len(long_q.encode()) < MAX_BODY_BYTES  # passes the body gate, hits pydantic
    resp = client.post("/api/route", json={"strategy": "cascade", "query": long_q})
    assert resp.status_code == 422
    assert _is_error_envelope(resp.json())


def test_grid_length_is_capped_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pathologically long eval grid is rejected by the schema cap."""
    client = _client(monkeypatch)
    resp = client.post(
        "/api/eval",
        json={"strategy": "cascade", "benchmark": "gsm8k", "quick": True, "grid": [0.5] * 500},
    )
    assert resp.status_code == 422
    assert _is_error_envelope(resp.json())


# --- No path-traversal / SSRF via example_id --------------------------------
@pytest.mark.parametrize(
    "bad_id",
    [
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "/etc/shadow",
        "gsm8k-1142/../../secret",
        "http://169.254.169.254/latest/meta-data/",
    ],
)
def test_crafted_example_id_is_typed_404_no_traversal(
    monkeypatch: pytest.MonkeyPatch, bad_id: str
) -> None:
    """example_id is a dict key lookup, never a filesystem/URL path → typed 404."""
    client = _client(monkeypatch)
    resp = client.post("/api/route", json={"strategy": "cascade", "example_id": bad_id})
    assert resp.status_code == 404
    body = resp.json()
    assert _is_error_envelope(body)
    assert body["error"]["type"] == "not-found"


# --- CORS lock --------------------------------------------------------------
def test_cors_locked_rejects_disallowed_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a locked origin list, a disallowed Origin gets no ACAO header."""
    client = _client(monkeypatch, cors_origins=["https://app.example"])
    resp = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}


def test_cors_locked_allows_configured_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, cors_origins=["https://app.example"])
    resp = client.get("/api/health", headers={"Origin": "https://app.example"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "https://app.example"


def test_cors_default_is_wildcard_for_local_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default (local dev) is ``*``; prod locks it (compose passes empty)."""
    client = _client(monkeypatch)  # default cors_origins=["*"]
    resp = client.get("/api/health", headers={"Origin": "https://anything.example"})
    assert resp.headers.get("access-control-allow-origin") == "*"


# --- Error hygiene under fuzz: always structured, never a stack trace --------
_TRACE_MARKERS = ("Traceback (most recent", 'File "', "\n  File ", '.py", line')


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/route", {}),  # missing required fields
        ("post", "/api/route", {"strategy": "nonsense", "query": "x"}),  # bad enum
        ("post", "/api/route", {"strategy": "cascade"}),  # neither query nor example_id
        ("post", "/api/route", {"strategy": "cascade", "query": "x", "tau": 9.9}),  # OOB
        ("post", "/api/route", {"strategy": "cascade", "query": "x", "example_id": "y"}),  # both
        ("post", "/api/eval", {"strategy": "cascade", "benchmark": "nope"}),  # bad enum
        ("get", "/api/does-not-exist", None),  # unknown route
        ("post", "/api/route", "not-json-at-all"),  # malformed body
    ],
)
def test_fuzz_failures_are_structured_no_trace(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str, payload: Any
) -> None:
    client = _client(monkeypatch)
    if method == "get":
        resp = client.get(path)
    elif isinstance(payload, str):
        resp = client.post(path, content=payload, headers={"content-type": "application/json"})
    else:
        resp = client.post(path, json=payload)
    assert resp.status_code >= 400
    body = resp.json()
    assert _is_error_envelope(body), body
    blob = resp.text
    for marker in _TRACE_MARKERS:
        assert marker not in blob, f"stack-trace marker {marker!r} leaked: {blob[:200]}"


def test_unexpected_engine_error_is_structured_no_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected engine exception → typed 502, no traceback in the body."""

    def _boom(*a: Any, **k: Any) -> RouteResult:
        raise RuntimeError("internal boom at /opt/secret/path.py line 42")

    monkeypatch.setattr(frugalroute, "route", _boom)
    app = create_app(Settings())
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"})
    assert resp.status_code == 502
    body = resp.json()
    assert _is_error_envelope(body)
    assert "Traceback" not in resp.text
