"""Tests 14-15: §7 round-trip fidelity + the honest streaming placeholder."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from frugalroute.harness import ItemRun, assemble_report, report_to_dict
from frugalroute.models import GateVerdict, RouteResult

from conftest import patch_route

TIERS = ["claude-haiku-4-5", "claude-opus-4-8"]


def test_route_result_roundtrip(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A core RouteResult → endpoint JSON: every §7 field present with the right type."""
    result = RouteResult(
        query="2+2?",
        strategy="cascade",
        tier_used="claude-haiku-4-5",
        escalated=False,
        answer="The answer is 4.",
        correct=None,
        gate=GateVerdict(sufficient=True, confidence=0.9, reason="ok"),
        p_strong=None,
        refused=False,
        cost_usd=0.0015,
        latency_s=0.4,
        prompt_version="v1",
    )

    def _fn(*_a: Any, **_k: Any) -> RouteResult:
        return result

    patch_route(monkeypatch, _fn)
    body = client.post("/api/route", json={"strategy": "cascade", "query": "2+2?"}).json()

    # Every §7 field carried through with the right Python/JSON type.
    assert body["query"] == result.query
    assert body["strategy"] == result.strategy
    assert body["tier_used"] == result.tier_used
    assert body["escalated"] is False
    assert body["answer"] == result.answer
    assert body["correct"] is None
    assert body["gate"] == {"sufficient": True, "confidence": 0.9, "reason": "ok"}
    assert body["p_strong"] is None
    assert body["refused"] is False
    assert isinstance(body["cost_usd"], float)
    assert isinstance(body["latency_s"], float)
    assert body["prompt_version"] == "v1"


def test_eval_report_roundtrip(
    client: TestClient, override_settings: Callable[..., None], tmp_path: Path
) -> None:
    """A core EvalReport → committed bundle → served JSON is byte-identical (per §7)."""
    runs = [
        [
            ItemRun(
                "a",
                {"claude-haiku-4-5": True, "claude-opus-4-8": True},
                {"claude-haiku-4-5": 0.001, "claude-opus-4-8": 0.007},
                {"claude-haiku-4-5": False, "claude-opus-4-8": False},
                gate_sufficient=True,
                gate_confidence=0.9,
                gate_cost=0.0005,
                p_strong=0.2,
            )
        ]
    ]
    report = assemble_report(runs, "cascade", TIERS, taus=[0.5, 0.8])
    expected = report_to_dict(report)
    bundle = {
        "reports": [expected],
        "benchmark": "gsm8k",
        "frozen_split": {"n_test": 1, "n_calibration": 4, "small_n": True},
        "generated_at": "2026-06-20T00:00:00+00:00",
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    override_settings(sample_run_path=path)

    served = client.get("/api/eval/sample").json()
    assert served["reports"][0] == expected  # field-for-field, no drift


def test_stream_placeholder_501(client: TestClient) -> None:
    resp = client.get("/api/route/stream")
    assert resp.status_code == 501
    assert resp.json()["error"]["type"] == "not-implemented"
