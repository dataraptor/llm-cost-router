"""Harness wiring (no-key): cheap-refusal re-threshold, collect, run_eval(both).

These exercise the harness's data-collection + orchestration paths with the fake
client (and a fake predictive router), so the cascade cheap-refusal cost rule, the
shared single-pass collection, and the both-strategy ``run_eval`` assemble correctly
without a key. The live generation/training itself is covered by the @api smoke.
"""

from __future__ import annotations

from typing import Any

import pytest

from frugalroute.benchmarks import load_benchmark
from frugalroute.harness import (
    ItemRun,
    cascade_point,
    collect_repeat,
    run_eval,
)
from frugalroute.models import GateVerdict

TIERS = ["claude-haiku-4-5", "claude-opus-4-8"]


def test_cascade_point_cheap_refusal_escalates_without_gate() -> None:
    # Cheap refused → escalate to strong, cost = c_cheap + c_strong (NO gate cost),
    # mirroring router._route_cascade's cheap-refusal path.
    run = ItemRun(
        item_id="r",
        tier_grades={"claude-haiku-4-5": False, "claude-opus-4-8": True},
        tier_costs={"claude-haiku-4-5": 0.001, "claude-opus-4-8": 0.007},
        tier_refused={"claude-haiku-4-5": True, "claude-opus-4-8": False},
        gate_sufficient=None,
        gate_cost=0.0,
    )
    quality, cost, escalation = cascade_point([run], TIERS, 0.8)
    assert quality == pytest.approx(1.0)  # strong was correct
    assert cost == pytest.approx(0.008)  # c_cheap + c_strong, no gate
    assert escalation == pytest.approx(1.0)


def _eval_client(fake_client, fake_usage) -> Any:
    return fake_client(
        text="The answer is 72.",
        parsed_output=GateVerdict(sufficient=True, confidence=0.9, reason="ok"),
        usage=fake_usage(input_tokens=100, output_tokens=40),
    )


def test_collect_repeat_cascade_builds_shared_cache(fake_client, fake_usage) -> None:
    client = _eval_client(fake_client, fake_usage)
    items = load_benchmark("gsm8k", n=4)
    runs = collect_repeat(client, items, TIERS, "gsm8k", need_gate=True)
    assert len(runs) == 4
    first = runs[0]
    assert set(first.tier_grades) == set(TIERS)
    assert set(first.tier_costs) == set(TIERS)
    assert first.gate_sufficient is True
    assert first.gate_confidence == pytest.approx(0.9)
    assert first.gate_cost > 0
    assert first.p_strong is None  # no router → no predictive margin


class _FakeRouter:
    """A trained predictive router stand-in (no embedder / sklearn needed)."""

    tiers = list(TIERS)
    label_run_ids = ["labels-gsm8k-abc123"]

    def predict_tier(
        self, query: str, theta: float | None = None, *, embedder: Any = None
    ) -> tuple[str, float]:
        return "claude-opus-4-8", 0.7


def test_run_eval_both_with_injected_router(fake_client, fake_usage) -> None:
    client = _eval_client(fake_client, fake_usage)
    run = run_eval(
        "gsm8k",
        strategy="both",
        repeats=1,
        n=12,
        client=client,
        router=_FakeRouter(),
        timestamp="T",
    )
    assert set(run.reports) == {"cascade", "predictive"}
    assert run.meta["label_run_ids"] == ["labels-gsm8k-abc123"]
    assert run.meta["timestamp"] == "T"
    assert run.reports["predictive"].points
    # p_strong=0.7 routes to strong for θ < 0.7 and to cheap at θ=0.7 → the sweep
    # actually moves, so escalation varies across the grid.
    escalations = {p.escalation_rate for p in run.reports["predictive"].points}
    assert len(escalations) >= 2
