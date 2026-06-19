"""Frontier selection + dominance + the losing region — tests 11-13 and R14.

``cost_reduction_at_target`` picks the lowest-cost point at/above the retention
target (and flags the no-point case, never fakes it). On a constructed sweep the
cascade dominates the random baseline (R5), and a hostile difficulty mix drives the
cascade into the losing region — a frontier point that costs MORE than always-strong
— which the report surfaces rather than hides (R14, build-spec §8).
"""

from __future__ import annotations

import pytest

from frugalroute.harness import ItemRun, assemble_report
from frugalroute.metrics import cost_reduction_at_target, frontier_points
from frugalroute.models import FrontierPoint

TIERS = ["cheap", "strong"]
C_CHEAP, C_GATE, C_STRONG = 0.001, 0.0004, 0.007


def _fp(param: float, quality: float, cost: float) -> FrontierPoint:
    return FrontierPoint(
        operating_param=param,
        quality=quality,
        quality_spread=0.0,
        cost_usd_per_query=cost,
        cost_spread=0.0,
        escalation_rate=0.0,
        n=10,
    )


def test_target_selects_lowest_cost_at_or_above_retention() -> None:
    # 11. retention >= 0.95 → quality >= 0.9405; the cheapest such point wins.
    points = [
        _fp(0.5, 0.90, 0.002),  # retention 0.909 — below target
        _fp(0.7, 0.96, 0.004),  # 0.970 — qualifies, cheapest qualifying
        _fp(0.9, 0.98, 0.006),  # 0.990 — qualifies, costlier
        _fp(1.0, 0.99, 0.008),
    ]
    result = cost_reduction_at_target(points, strong_quality=0.99, strong_cost=0.008)
    assert result["reached_target"] is True
    assert result["operating_param"] == pytest.approx(0.7)
    assert result["cost"] == pytest.approx(0.004)
    assert result["cost_reduction"] == pytest.approx(1 - 0.004 / 0.008)


def test_no_point_reaches_target_is_flagged_not_faked() -> None:
    # 12. No exception, no fake number — the closest (highest-retention) point + a flag.
    points = [_fp(0.5, 0.50, 0.002), _fp(0.7, 0.60, 0.004)]
    result = cost_reduction_at_target(points, strong_quality=1.0, strong_cost=0.008)
    assert result["reached_target"] is False
    assert result["operating_param"] == pytest.approx(0.7)  # highest retention
    assert result["retention"] == pytest.approx(0.60)  # real number, not 0.95


def test_empty_points_is_na() -> None:
    result = cost_reduction_at_target([], strong_quality=1.0, strong_cost=0.008)
    assert result["reached_target"] is False
    assert result["retention"] != result["retention"]  # nan


def test_all_nonfinite_retention_is_na() -> None:
    # Points exist but every quality is nan (e.g. an empty slice) → N/A, not a fake.
    points = [_fp(0.5, float("nan"), 0.002), _fp(0.9, float("nan"), 0.006)]
    result = cost_reduction_at_target(points, strong_quality=1.0, strong_cost=0.008)
    assert result["reached_target"] is False
    assert result["retention"] != result["retention"]  # nan


def _run(
    item_id: str,
    cheap_ok: bool,
    strong_ok: bool,
    gate_sufficient: bool,
    gate_confidence: float,
) -> ItemRun:
    return ItemRun(
        item_id=item_id,
        tier_grades={"cheap": cheap_ok, "strong": strong_ok},
        tier_costs={"cheap": C_CHEAP, "strong": C_STRONG},
        tier_refused={"cheap": False, "strong": False},
        gate_sufficient=gate_sufficient,
        gate_confidence=gate_confidence,
        gate_cost=C_GATE,
    )


def test_frontier_dominates_random_baseline() -> None:
    # 13. Cheap is right on most items (high acceptance) → a cascade point sits
    #     up-and-left of the random baseline (higher quality AND lower cost).
    repeat = [_run(f"easy{i}", True, True, True, 0.95) for i in range(8)]
    repeat += [_run(f"hard{i}", False, True, False, 0.2) for i in range(2)]
    report = assemble_report([repeat], "cascade", TIERS)
    random_point = report.baselines["random"]
    dominating = [
        point
        for point in report.points
        if point.quality >= random_point["quality"]
        and point.cost_usd_per_query <= random_point["cost"]
    ]
    assert dominating, "expected a cascade point up-and-left of the random baseline"


def test_frontier_surfaces_the_losing_region() -> None:
    # R14 (adversarial, mandatory). A mix the gate always doubts → the cascade
    # escalates on every item at every τ, so its per-query cost (c_cheap+c_gate+
    # c_strong) EXCEEDS always-strong (c_strong). The report must SHOW this cliff
    # (a point with cost > strong cost; a negative cost-reduction), not hide it.
    repeat = [_run(f"x{i}", False, True, False, 0.1) for i in range(10)]
    report = assemble_report([repeat], "cascade", TIERS)
    strong_cost = report.baselines["always_strong"]["cost"]
    losing = [p for p in report.points if p.cost_usd_per_query > strong_cost]
    assert losing, "the losing region (cost > always-strong) must be visible on the frontier"
    # And the honest headline reflects it: full retention but a NEGATIVE cost cut.
    assert report.retention_at_target == pytest.approx(1.0)
    assert report.cost_reduction_at_target < 0.0


def test_frontier_points_sorted_by_cost() -> None:
    points = [_fp(0.9, 0.98, 0.006), _fp(0.5, 0.90, 0.002), _fp(0.7, 0.96, 0.004)]
    ordered = frontier_points(points)
    assert [p.cost_usd_per_query for p in ordered] == [0.002, 0.004, 0.006]


def _predictive_run(item_id: str, cheap_ok: bool, strong_ok: bool, p_strong: float) -> ItemRun:
    return ItemRun(
        item_id=item_id,
        tier_grades={"cheap": cheap_ok, "strong": strong_ok},
        tier_costs={"cheap": C_CHEAP, "strong": C_STRONG},
        tier_refused={"cheap": False, "strong": False},
        gate_sufficient=True,
        gate_confidence=0.9,
        gate_cost=C_GATE,
        p_strong=p_strong,
    )


def test_predictive_report_renders_with_theta_headline() -> None:
    # The predictive path: high p_strong routes the hard items to strong, easy ones
    # stay cheap → a clean θ frontier; the rendered report names θ, not τ.
    from frugalroute.harness import EvalRun, assemble_report, format_report

    repeat = [_predictive_run(f"easy{i}", True, True, 0.1) for i in range(6)]
    repeat += [_predictive_run(f"hard{i}", False, True, 0.9) for i in range(4)]
    cascade = assemble_report([repeat], "cascade", TIERS)
    predictive = assemble_report([repeat], "predictive", TIERS)
    run = EvalRun(
        reports={"cascade": cascade, "predictive": predictive},
        repeats=[repeat],
        meta={"benchmark": "gsm8k", "n": 10, "model_tiers": TIERS, "n_refused": 2},
    )
    text = format_report(run)
    assert "Frontier (predictive):" in text
    assert "theta" in text
    assert "FrugalRoute-predictive" in text
    assert "predictive @ theta=" in text
    assert "n_refused: 2" in text  # the refusal chip is surfaced when > 0


def test_assemble_report_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        assemble_report([[]], "bogus", TIERS)
