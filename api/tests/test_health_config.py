"""Tests 1-2: health + config sourced from core (no duplicated numbers)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from frugalroute import llm
from frugalroute.prompts import PROMPT_VERSION


def test_health_reflects_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Default backend ("") → has_api_key reflects ANTHROPIC_API_KEY presence.
    monkeypatch.setenv("FRUGALROUTE_BACKEND", "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["has_api_key"] is False
    assert isinstance(body["version"], str) and body["version"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    assert client.get("/api/health").json()["has_api_key"] is True


def test_config_sourced_from_core(client: TestClient) -> None:
    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()

    # prompt_version + tiers come from core, not retyped literals.
    assert body["prompt_version"] == PROMPT_VERSION
    assert body["model_tiers"] == list(llm.DEFAULT_TIERS)
    assert body["strategies"] == ["cascade", "predictive"]

    # Pricing numbers equal core's PRICING for each served tier (renamed keys only).
    for tier in llm.DEFAULT_TIERS:
        assert body["pricing"][tier]["input_per_mtok"] == llm.PRICING[tier]["input"]
        assert body["pricing"][tier]["output_per_mtok"] == llm.PRICING[tier]["output"]
    # Only the active tiers are exposed (not sonnet / gpt-5.5).
    assert set(body["pricing"]) == set(llm.DEFAULT_TIERS)

    assert body["defaults"] == {"tau": 0.8, "theta": 0.6}
    assert body["pricing_pinned_date"] == "2026-06-19"
    assert isinstance(body["always_strong_cost_ref_usd"], float)
    assert body["always_strong_cost_ref_usd"] > 0
