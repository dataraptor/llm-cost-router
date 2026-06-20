"""Adversarial / safety-net tests: no failure path escapes the structured model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


def test_unexpected_exception_is_structured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception outside the routing try-block still yields the structured body."""
    import frugalroute

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("unexpected internal failure")

    monkeypatch.setattr(frugalroute, "load_examples", _boom)
    resp = client.get("/api/examples")
    assert resp.status_code == 502
    body = resp.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"type", "message", "detail"}
    assert body["error"]["type"] == "api-error"


def test_malformed_sample_is_structured(
    client: TestClient, override_settings: Callable[..., None], tmp_path: Path
) -> None:
    path = tmp_path / "sample.json"
    path.write_text("{not valid json", encoding="utf-8")
    override_settings(sample_run_path=path)
    resp = client.get("/api/eval/sample")
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "api-error"


def test_all_error_bodies_share_one_shape(client: TestClient) -> None:
    """Every typed error renders the same {error:{type,message,detail}} envelope."""
    cases = [
        client.get("/api/route/stream"),  # 501 not-implemented
        client.post("/api/route", json={"strategy": "cascade"}),  # 422 bad-request
        client.post("/api/route", json={"strategy": "cascade", "example_id": "nope"}),  # 404
    ]
    for resp in cases:
        body = resp.json()
        assert set(body) == {"error"}
        assert set(body["error"]) == {"type", "message", "detail"}
        assert isinstance(body["error"]["message"], str) and body["error"]["message"]
