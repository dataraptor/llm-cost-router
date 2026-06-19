"""Atomic metric functions (build-spec §11) — tests 1-4.

Pure, no-key. Empty input is the documented ``nan`` sentinel (rendered N/A),
never ``0``; zero/undefined denominators never raise.
"""

from __future__ import annotations

import math

import pytest

from frugalroute.metrics import accuracy, cost_reduction, mean_cost, mean_spread, retention


def test_accuracy_basic() -> None:
    # 1.
    assert accuracy([True, True, False, True]) == pytest.approx(0.75)


def test_accuracy_empty_is_nan_not_zero() -> None:
    # 1. Empty → nan sentinel (N/A), NOT a misleading 0.
    assert math.isnan(accuracy([]))


def test_mean_cost_basic() -> None:
    # 2.
    assert mean_cost([0.001, 0.003]) == pytest.approx(0.002)


def test_mean_cost_empty_is_nan() -> None:
    assert math.isnan(mean_cost([]))


def test_retention_and_cost_reduction_definitions() -> None:
    # 3. retention = quality / strong_quality; cost_reduction = 1 - cost/strong_cost.
    assert retention(0.95, 0.97) == pytest.approx(0.979, abs=1e-3)
    assert cost_reduction(0.003, 0.007) == pytest.approx(0.571, abs=1e-3)


def test_zero_and_nonfinite_denominator_safe() -> None:
    # 4. No ZeroDivision crash; undefined reference → nan.
    assert math.isnan(retention(0.5, 0.0))
    assert math.isnan(cost_reduction(0.5, 0.0))
    assert math.isnan(retention(0.5, float("nan")))
    assert math.isnan(cost_reduction(0.5, float("nan")))


def test_mean_spread_single_run_is_zero() -> None:
    mean, spread = mean_spread([0.4])
    assert mean == pytest.approx(0.4)
    assert spread == 0.0


def test_mean_spread_empty_is_nan() -> None:
    mean, spread = mean_spread([])
    assert math.isnan(mean) and math.isnan(spread)
