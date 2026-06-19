"""No-key tests for the data contracts (split-01 cases 14–16)."""

from __future__ import annotations

import json

from frugalroute.models import EvalReport, FrontierPoint, GateVerdict, RouteResult


def test_gate_verdict_round_trips() -> None:
    verdict = GateVerdict(sufficient=True, confidence=0.9, reason="ok")
    dumped = verdict.model_dump()
    assert dumped == {"sufficient": True, "confidence": 0.9, "reason": "ok"}
    assert GateVerdict.model_validate(dumped) == verdict


def test_gate_verdict_schema_is_api_legal() -> None:
    schema = GateVerdict.model_json_schema()
    blob = json.dumps(schema)
    # No numeric/length bounds (the Anthropic structured-output schema rejects them).
    assert "minLength" not in blob
    assert "maxLength" not in blob
    assert "minimum" not in blob
    assert "maximum" not in blob
    # additionalProperties:false (from extra="forbid"); no recursion ($ref).
    assert schema.get("additionalProperties") is False
    assert "$ref" not in blob
    assert "$defs" not in blob


def test_route_result_allows_none_fields() -> None:
    result = RouteResult(
        query="2+2?",
        strategy="cascade",
        tier_used="claude-haiku-4-5",
        escalated=False,
        answer="The answer is 4.",
        correct=None,  # None in the live demo
        gate=GateVerdict(sufficient=True, confidence=0.95, reason="commits to 4"),
        p_strong=None,  # None for cascade
        refused=False,
        cost_usd=0.0018,
        latency_s=0.42,
        prompt_version="v1",
    )
    assert result.correct is None
    assert result.p_strong is None
    assert result.gate is not None and result.gate.sufficient is True


def test_frontier_point_carries_spread() -> None:
    point = FrontierPoint(
        operating_param=0.7,
        quality=0.91,
        quality_spread=0.012,
        cost_usd_per_query=0.0031,
        cost_spread=0.0002,
        escalation_rate=0.4,
        n=84,
    )
    assert point.quality_spread == 0.012
    assert point.cost_spread == 0.0002


def test_eval_report_carries_distributional_fields() -> None:
    report = EvalReport(
        strategy="cascade",
        points=[],
        baselines={
            "always_cheap": {
                "quality": 0.6,
                "quality_spread": 0.01,
                "cost": 0.0014,
                "cost_spread": 0.0,
            },
            "always_strong": {
                "quality": 0.95,
                "quality_spread": 0.008,
                "cost": 0.0070,
                "cost_spread": 0.0,
            },
            "random": {
                "quality": 0.77,
                "quality_spread": 0.02,
                "cost": 0.0042,
                "cost_spread": 0.0003,
            },
        },
        oracle={"quality": 0.97, "quality_spread": 0.005, "cost": 0.0030},
        retention_at_target=0.93,
        retention_at_target_spread=0.012,
        cost_reduction_at_target=0.55,
        cost_reduction_at_target_spread=0.03,
        n_refused=2,
        prompt_version="v1",
        model_tiers=["claude-haiku-4-5", "claude-opus-4-8"],
        n_runs=3,
    )
    assert report.retention_at_target_spread == 0.012
    assert report.cost_reduction_at_target_spread == 0.03
    assert report.baselines["always_strong"]["quality_spread"] == 0.008
    assert report.oracle["quality_spread"] == 0.005
    assert report.n_runs == 3
