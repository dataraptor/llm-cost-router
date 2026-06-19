"""Distributional aggregation, 0-item N/A, persistence round-trip — tests 14-16.

The spread is the population standard deviation over the R repeats: identical runs
→ 0 spread; differing runs → mean is the average and spread the labelled stat. A
0-item benchmark renders N/A end-to-end (never 0). A persisted run round-trips with
all §7 fields incl. the distributional spreads, the τ/θ grid, and label-run IDs.
"""

from __future__ import annotations

import math
import statistics

import pytest

from frugalroute.harness import (
    EvalRun,
    ItemRun,
    assemble_report,
    format_headline,
    format_report,
    read_run,
    report_to_dict,
    run_eval,
    write_run,
)

TIERS = ["cheap", "strong"]
C_CHEAP, C_GATE, C_STRONG = 0.001, 0.0004, 0.007


def _repeat(strong_grades: list[bool]) -> list[ItemRun]:
    # gate sufficient but confidence 0.5 < 1.0 → τ=1.0 always escalates, so the
    # cascade@1.0 quality equals the strong-tier accuracy for that repeat.
    return [
        ItemRun(
            item_id=f"i{idx}",
            tier_grades={"cheap": False, "strong": ok},
            tier_costs={"cheap": C_CHEAP, "strong": C_STRONG},
            tier_refused={"cheap": False, "strong": False},
            gate_sufficient=True,
            gate_confidence=0.5,
            gate_cost=C_GATE,
        )
        for idx, ok in enumerate(strong_grades)
    ]


def _point_at(report, param: float):
    return next(p for p in report.points if p.operating_param == param)


def test_identical_repeats_have_zero_spread() -> None:
    # 14. R=3 identical frontiers → every spread is 0.
    repeat = _repeat([True, False])
    report = assemble_report([repeat, repeat, repeat], "cascade", TIERS)
    for point in report.points:
        assert point.quality_spread == 0.0
        assert point.cost_spread == 0.0
    assert report.retention_at_target_spread == 0.0
    assert report.cost_reduction_at_target_spread == 0.0


def test_differing_repeats_report_mean_and_population_stdev() -> None:
    # 14. Three repeats with strong accuracy 1.0 / 0.5 / 0.0 at τ=1.0.
    report = assemble_report(
        [_repeat([True, True]), _repeat([True, False]), _repeat([False, False])],
        "cascade",
        TIERS,
    )
    point = _point_at(report, 1.0)
    assert point.quality == pytest.approx(0.5)  # mean of 1.0, 0.5, 0.0
    assert point.quality_spread == pytest.approx(statistics.pstdev([1.0, 0.5, 0.0]), abs=1e-9)


def test_zero_item_benchmark_renders_na_end_to_end() -> None:
    # 15. n=0 needs no key (no generations) and must render N/A, never 0.
    run = run_eval("gsm8k", strategy="cascade", repeats=1, n=0, client=None)
    report = run.reports["cascade"]
    assert math.isnan(report.retention_at_target)
    assert math.isnan(report.cost_reduction_at_target)
    headline = format_headline(report, benchmark="gsm8k", n_test=0)
    assert "N/A" in headline
    assert "N/A" in format_report(run)


def test_persisted_run_round_trips_with_all_fields(tmp_path) -> None:
    # 16. Write → read; the report (incl. spread fields) and provenance survive.
    repeats = [_repeat([True, True]), _repeat([True, False]), _repeat([False, False])]
    report = assemble_report(repeats, "cascade", TIERS, taus=[0.5, 0.8, 1.0])
    meta = {
        "benchmark": "gsm8k",
        "timestamp": "20260620T120000",
        "n": 2,
        "n_runs": 3,
        "prompt_version": report.prompt_version,
        "model_tiers": TIERS,
        "taus": [0.5, 0.8, 1.0],
        "thetas": [0.4, 0.6],
        "label_run_ids": ["labels-gsm8k-deadbeef"],
        "n_refused": report.n_refused,
    }
    run = EvalRun(reports={"cascade": report}, repeats=repeats, meta=meta)

    path = write_run(run, tmp_path / "run.jsonl")
    back = read_run(path)

    # Report round-trips field-for-field (incl. distributional spreads).
    assert report_to_dict(back["reports"]["cascade"]) == report_to_dict(report)
    restored = back["reports"]["cascade"]
    assert restored.points[0].quality_spread == report.points[0].quality_spread
    assert "quality_spread" in restored.baselines["always_strong"]

    # Provenance: meta carries the grid + label-run IDs + tiers; items carry grades.
    assert back["meta"]["taus"] == [0.5, 0.8, 1.0]
    assert back["meta"]["thetas"] == [0.4, 0.6]
    assert back["meta"]["label_run_ids"] == ["labels-gsm8k-deadbeef"]
    assert back["meta"]["model_tiers"] == TIERS
    assert back["meta"]["prompt_version"] == report.prompt_version
    assert back["meta"]["n_runs"] == 3
    item_rows = back["items"]
    assert item_rows and {"item_id", "tier", "grade", "cost", "refused"} <= set(item_rows[0])
    # 3 repeats × 2 items × 2 tiers = 12 per-item provenance rows.
    assert len(item_rows) == 12
