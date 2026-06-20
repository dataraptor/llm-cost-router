"""Tests 11-13: /api/eval/sample (200/404) + POST /api/eval (quick gate)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from frugalroute.harness import EvalRun, ItemRun, assemble_report

from conftest import patch_run_eval

TIERS = ["claude-haiku-4-5", "claude-opus-4-8"]

EVAL_REPORT_KEYS = {
    "strategy",
    "points",
    "baselines",
    "oracle",
    "retention_at_target",
    "retention_at_target_spread",
    "cost_reduction_at_target",
    "cost_reduction_at_target_spread",
    "n_refused",
    "prompt_version",
    "model_tiers",
    "n_runs",
}
POINT_KEYS = {
    "operating_param",
    "quality",
    "quality_spread",
    "cost_usd_per_query",
    "cost_spread",
    "escalation_rate",
    "n",
}


def _runs() -> list[list[ItemRun]]:
    return [
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
            ),
            ItemRun(
                "b",
                {"claude-haiku-4-5": False, "claude-opus-4-8": True},
                {"claude-haiku-4-5": 0.001, "claude-opus-4-8": 0.007},
                {"claude-haiku-4-5": False, "claude-opus-4-8": False},
                gate_sufficient=False,
                gate_confidence=0.3,
                gate_cost=0.0005,
                p_strong=0.8,
            ),
        ]
    ]


def _make_eval_run() -> EvalRun:
    cascade = assemble_report(_runs(), "cascade", TIERS, taus=[0.5, 0.8, 1.0])
    return EvalRun(
        reports={"cascade": cascade},
        repeats=[],
        meta={"benchmark": "gsm8k", "n": 8, "n_calibration": 32, "n_runs": 1},
    )


def test_eval_sample_present(client: TestClient) -> None:
    # The bundled dev sample exists → 200 with §7-complete reports.
    resp = client.get("/api/eval/sample")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"reports", "benchmark", "frozen_split", "generated_at"}
    assert len(body["reports"]) >= 1
    report = body["reports"][0]
    assert EVAL_REPORT_KEYS <= set(report)
    assert POINT_KEYS <= set(report["points"][0])
    # Leaderboard-relevant fields populated.
    assert {"always_cheap", "always_strong", "random"} <= set(report["baselines"])
    assert {"quality", "cost"} <= set(report["oracle"])
    assert isinstance(report["retention_at_target"], float)
    assert isinstance(report["cost_reduction_at_target"], float)


def test_eval_sample_missing_404(
    client: TestClient, override_settings: Callable[..., None], tmp_path: Path
) -> None:
    override_settings(sample_run_path=tmp_path / "nope.json")
    resp = client.get("/api/eval/sample")
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not-found"


def test_eval_non_quick_points_at_cli(client: TestClient) -> None:
    resp = client.post(
        "/api/eval", json={"strategy": "cascade", "benchmark": "gsm8k", "quick": False}
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["type"] == "bad-request"
    assert "eval" in body["error"]["message"].lower()


def test_eval_quick_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(*_args: Any, **_kwargs: Any) -> EvalRun:
        return _make_eval_run()

    patch_run_eval(monkeypatch, _fake)
    resp = client.post(
        "/api/eval", json={"strategy": "cascade", "benchmark": "gsm8k", "quick": True}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"reports", "benchmark", "frozen_split", "generated_at"}
    assert body["benchmark"] == "gsm8k"
    assert body["frozen_split"]["n_test"] == 8
    assert body["frozen_split"]["small_n"] is True
    assert EVAL_REPORT_KEYS <= set(body["reports"][0])
    assert body["generated_at"]  # stamped
