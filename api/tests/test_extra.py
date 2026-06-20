"""Coverage of the real feature branches: backend selection, router load, example
resolution, root redirect, config fallback, and the remaining error mappings."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from conftest import patch_route


def _record_route(captured: dict[str, Any], result: Any) -> Callable[..., Any]:
    def _fn(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return result

    return _fn


def test_root_redirects_to_docs(client: TestClient) -> None:
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/docs"


def test_example_id_resolves_query(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    make_route_result: Callable[..., Any],
) -> None:
    captured: dict[str, Any] = {}
    patch_route(monkeypatch, _record_route(captured, make_route_result()))
    resp = client.post("/api/route", json={"strategy": "cascade", "example_id": "gsm8k-1142"})
    assert resp.status_code == 200
    # The bundled example's query was looked up and passed positionally.
    assert "Natalia" in captured["args"][0]


def test_backend_azure_injects_client(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    override_settings: Callable[..., None],
    make_route_result: Callable[..., Any],
) -> None:
    sentinel = object()
    import frugalroute.azure_client as az

    monkeypatch.setattr(az, "get_azure_client", lambda: sentinel)
    captured: dict[str, Any] = {}
    patch_route(monkeypatch, _record_route(captured, make_route_result()))
    override_settings(backend="azure")

    resp = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"})
    assert resp.status_code == 200
    assert captured["kwargs"]["client"] is sentinel


def test_router_path_loads_router(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    override_settings: Callable[..., None],
    make_route_result: Callable[..., Any],
    tmp_path: Path,
) -> None:
    router_file = tmp_path / "router.joblib"
    router_file.write_text("placeholder", encoding="utf-8")
    sentinel = object()
    import frugalroute

    monkeypatch.setattr(frugalroute, "load_router", lambda _p: sentinel)
    captured: dict[str, Any] = {}
    patch_route(
        monkeypatch,
        _record_route(captured, make_route_result(strategy="predictive", gate=None, p_strong=0.7)),
    )
    override_settings(router_path=router_file)

    resp = client.post("/api/route", json={"strategy": "predictive", "query": "x"})
    assert resp.status_code == 200
    assert captured["kwargs"]["router"] is sentinel


def test_health_azure_backend_reflects_azure_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, override_settings: Callable[..., None]
) -> None:
    override_settings(backend="azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-test")
    assert client.get("/api/health").json()["has_api_key"] is True
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert client.get("/api/health").json()["has_api_key"] is False


def test_route_value_error_is_bad_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*_a: Any, **_k: Any) -> Any:
        raise ValueError("Predictive routing requires a trained router.")

    patch_route(monkeypatch, _raise)
    resp = client.post("/api/route", json={"strategy": "predictive", "query": "x"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "bad-request"


def test_route_unexpected_engine_error_is_api_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*_a: Any, **_k: Any) -> Any:
        raise TypeError("engine blew up unexpectedly")

    patch_route(monkeypatch, _raise)
    resp = client.post("/api/route", json={"strategy": "cascade", "query": "x"})
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "api-error"


def test_route_apierror_passthrough(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from frugalroute_api import errors

    def _raise(*_a: Any, **_k: Any) -> Any:
        raise errors.not_found("engine said not found")

    patch_route(monkeypatch, _raise)
    resp = client.post("/api/route", json={"strategy": "cascade", "query": "x"})
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not-found"


def test_config_always_strong_fallback_when_no_sample(
    client: TestClient, override_settings: Callable[..., None], tmp_path: Path
) -> None:
    override_settings(sample_run_path=tmp_path / "missing.json")
    body = client.get("/api/config").json()
    # Falls back to the documented §5 constant when no sample run is present.
    assert body["always_strong_cost_ref_usd"] == pytest.approx(0.0070, abs=1e-9)


def test_eval_missing_key_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from conftest import patch_run_eval

    def _raise(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    patch_run_eval(monkeypatch, _raise)
    resp = client.post(
        "/api/eval", json={"strategy": "cascade", "benchmark": "gsm8k", "quick": True}
    )
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "missing-key"


def test_eval_apierror_passthrough(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from conftest import patch_run_eval
    from frugalroute_api import errors

    def _raise(*_a: Any, **_k: Any) -> Any:
        raise errors.not_found("engine said not found")

    patch_run_eval(monkeypatch, _raise)
    resp = client.post(
        "/api/eval", json={"strategy": "cascade", "benchmark": "gsm8k", "quick": True}
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not-found"


def test_cors_origins_csv_parsed() -> None:
    from frugalroute_api.config import Settings as ApiSettings

    settings = ApiSettings(cors_origins="https://a.com, https://b.com")
    assert settings.cors_origins == ["https://a.com", "https://b.com"]
