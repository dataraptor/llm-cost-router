"""The plotted baselines (build-spec §11) — tests 9-10.

``always_cheap``/``always_strong`` are exact single-tier points; ``random`` routes a
fraction to strong uniformly at random, deterministic given the seed and bounded
between the cheap and strong baselines.
"""

from __future__ import annotations

import pytest

from frugalroute.metrics import baselines

TIERS = ["cheap", "strong"]


def test_always_cheap_and_strong_are_exact() -> None:
    # 9. cheap == cheap-tier quality/cost; strong == strong-tier quality/cost.
    grades = [{"cheap": True, "strong": False}, {"cheap": False, "strong": True}]
    costs = [{"cheap": 0.001, "strong": 0.007}, {"cheap": 0.001, "strong": 0.007}]
    base = baselines(grades, costs, TIERS)
    assert base["always_cheap"]["quality"] == pytest.approx(0.5)
    assert base["always_cheap"]["cost"] == pytest.approx(0.001)
    assert base["always_strong"]["quality"] == pytest.approx(0.5)
    assert base["always_strong"]["cost"] == pytest.approx(0.007)


def _dominant_fixture(n: int = 20) -> tuple[list[dict[str, bool]], list[dict[str, float]]]:
    # Strong is right on every item, cheap on none → a random mix sits strictly
    # between the two baselines.
    grades = [{"cheap": False, "strong": True} for _ in range(n)]
    costs = [{"cheap": 0.001, "strong": 0.007} for _ in range(n)]
    return grades, costs


def test_random_is_deterministic_and_reproducible() -> None:
    # 10. Same seed → identical point, re-run reproduces it.
    grades, costs = _dominant_fixture()
    first = baselines(grades, costs, TIERS, rng_seed=42)["random"]
    second = baselines(grades, costs, TIERS, rng_seed=42)["random"]
    assert first == second


def test_random_sits_between_cheap_and_strong() -> None:
    # 10. quality/cost bounded by the cheap and strong baselines.
    grades, costs = _dominant_fixture()
    base = baselines(grades, costs, TIERS, rng_seed=42)
    random_point = base["random"]
    assert (
        base["always_cheap"]["quality"]
        <= random_point["quality"]
        <= base["always_strong"]["quality"]
    )
    assert base["always_cheap"]["cost"] <= random_point["cost"] <= base["always_strong"]["cost"]
    # A different seed gives a different (still bounded) draw.
    other = baselines(grades, costs, TIERS, rng_seed=7)["random"]
    assert base["always_cheap"]["cost"] <= other["cost"] <= base["always_strong"]["cost"]


def test_baselines_empty_input_is_nan() -> None:
    base = baselines([], [], TIERS)
    for name in ("always_cheap", "always_strong", "random"):
        assert base[name]["quality"] != base[name]["quality"]  # nan
        assert base[name]["cost"] != base[name]["cost"]


def test_baselines_empty_tiers_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        baselines([{"cheap": True}], [{"cheap": 0.001}], [])
