"""The oracle ceiling (build-spec §11) — tests 5-8.

The oracle routes each item to the cheapest tier that is actually correct (held-out
grades), else the cheapest tier. It is an upper envelope: ``quality >= strong`` and
``cost <= strong``. Fixtures are hand-computed and exact (``abs=1e-9``).
"""

from __future__ import annotations

import pytest

from frugalroute.metrics import baselines, oracle

TIERS = ["cheap", "strong"]
_UNIT_COST = {"cheap": 0.001, "strong": 0.007}


def _costs(n: int) -> list[dict[str, float]]:
    return [dict(_UNIT_COST) for _ in range(n)]


# A 4-item mix exercising every routing case:
#   A both correct · B strong-only · C cheap-only · D neither.
_GRADES = [
    {"cheap": True, "strong": True},
    {"cheap": False, "strong": True},
    {"cheap": True, "strong": False},
    {"cheap": False, "strong": False},
]


def test_cheapest_correct_ceiling() -> None:
    # 5. quality = mean(any correct) = 3/4; cost uses cheap wherever cheap suffices:
    #    A→cheap, B→strong, C→cheap, D→cheap = (0.001+0.007+0.001+0.001)/4 = 0.0025.
    result = oracle(_GRADES, _costs(4), TIERS)
    assert result["quality"] == pytest.approx(0.75, abs=1e-9)
    assert result["cost"] == pytest.approx(0.0025, abs=1e-9)


def test_no_tier_correct_takes_cheapest_and_counts_wrong() -> None:
    # 6. An item no tier solves → oracle takes the cheapest, graded wrong, cheap cost.
    result = oracle([{"cheap": False, "strong": False}], _costs(1), TIERS)
    assert result["quality"] == pytest.approx(0.0, abs=1e-9)
    assert result["cost"] == pytest.approx(0.001, abs=1e-9)


def test_oracle_dominates_strong_quality() -> None:
    # 7. Cheap is right on C where strong is wrong → oracle quality >= strong quality.
    result = oracle(_GRADES, _costs(4), TIERS)
    base = baselines(_GRADES, _costs(4), TIERS)
    assert result["quality"] >= base["always_strong"]["quality"]
    assert result["quality"] == pytest.approx(0.75) and base["always_strong"][
        "quality"
    ] == pytest.approx(0.5)


def test_oracle_cost_le_strong_cost() -> None:
    # 8.
    result = oracle(_GRADES, _costs(4), TIERS)
    base = baselines(_GRADES, _costs(4), TIERS)
    assert result["cost"] <= base["always_strong"]["cost"]


def test_oracle_empty_input_is_nan() -> None:
    result = oracle([], [], TIERS)
    assert result["quality"] != result["quality"]  # nan
    assert result["cost"] != result["cost"]


def test_oracle_empty_tiers_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        oracle([{"cheap": True}], [{"cheap": 0.001}], [])
