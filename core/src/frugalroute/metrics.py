"""Eval metrics: accuracy, cost, the oracle ceiling, baselines, and the frontier.

These are the **rigor** of FrugalRoute — pure functions, no I/O and no key, so the
oracle/baseline/retention math is exact and fully unit-tested on synthetic per-item
grades (build-spec §11). The harness (:mod:`frugalroute.harness`) collects the
per-item per-tier grades + costs once (@api) and feeds them here; every frontier
point and headline number is then deterministic arithmetic over that cache.

Conventions:
- **Empty input → ``nan``** (a documented sentinel the report layer renders as
  ``"N/A"``), never ``0`` (which would be a misleading number, §17).
- **Zero denominator → ``nan``** for ``retention`` / ``cost_reduction`` (no
  ``ZeroDivisionError``; the always-strong reference is undefined).
- **Spread is the population standard deviation over the R eval repeats** (so R=1
  and identical repeats both yield ``0.0``); labelled wherever it is reported.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

from frugalroute.models import FrontierPoint

NAN = float("nan")


# ----------------------------------------------------------------------------
# Atomic metrics (pure)
# ----------------------------------------------------------------------------
def accuracy(correct_flags: Sequence[bool]) -> float:
    """Mean of the boolean correctness flags. Empty → ``nan`` (rendered N/A)."""
    if not correct_flags:
        return NAN
    return statistics.fmean(1.0 if flag else 0.0 for flag in correct_flags)


def mean_cost(costs: Sequence[float]) -> float:
    """Mean per-query cost in USD. Empty → ``nan`` (rendered N/A)."""
    if not costs:
        return NAN
    return statistics.fmean(costs)


def retention(quality: float, strong_quality: float) -> float:
    """``quality / strong_quality`` — the headline retention vs always-strong.

    Returns ``nan`` when ``strong_quality`` is ``0`` or non-finite (the reference
    is undefined), never raising ``ZeroDivisionError``.
    """
    if not math.isfinite(strong_quality) or strong_quality == 0.0:
        return NAN
    return quality / strong_quality


def cost_reduction(cost: float, strong_cost: float) -> float:
    """``1 - cost / strong_cost`` — the headline cost cut vs always-strong.

    Returns ``nan`` when ``strong_cost`` is ``0`` or non-finite (undefined),
    never raising.
    """
    if not math.isfinite(strong_cost) or strong_cost == 0.0:
        return NAN
    return 1.0 - cost / strong_cost


# ----------------------------------------------------------------------------
# The oracle ceiling + the baselines (computed from held-out per-item grades)
# ----------------------------------------------------------------------------
def oracle(
    per_item_tier_grades: Sequence[dict[str, bool]],
    per_item_tier_costs: Sequence[dict[str, float]],
    tiers: Sequence[str],
) -> dict[str, float]:
    """The unachievable cost-quality **ceiling** (build-spec §11).

    For each item, route to the **cheapest tier that is actually correct** (using
    the held-out grades); if no tier is correct, take the cheapest tier
    (escalating would not have helped). Then:

    - ``quality = mean(any tier correct)`` — an upper envelope ≥ always-strong
      quality, since the oracle picks a correct tier whenever *any* exists.
    - ``cost = mean(cost of the chosen tier per item)`` — uses the cheap-tier
      cost wherever cheap suffices, so it is ≤ always-strong cost.

    No online router can reach this (it uses ground truth). Empty input → both
    ``nan``. Raises ``ValueError`` on an empty ``tiers`` list.
    """
    if not tiers:
        raise ValueError("tiers must be non-empty to compute the oracle.")
    if not per_item_tier_grades:
        return {"quality": NAN, "cost": NAN}

    correct_flags: list[bool] = []
    chosen_costs: list[float] = []
    for grades, costs in zip(per_item_tier_grades, per_item_tier_costs, strict=True):
        any_correct = any(grades.get(tier, False) for tier in tiers)
        chosen = _cheapest_correct_tier(grades, tiers)
        correct_flags.append(any_correct)
        chosen_costs.append(costs[chosen])
    return {"quality": accuracy(correct_flags), "cost": mean_cost(chosen_costs)}


def _cheapest_correct_tier(grades: dict[str, bool], tiers: Sequence[str]) -> str:
    """The earliest (cheapest) tier graded correct, else the cheapest tier."""
    for tier in tiers:
        if grades.get(tier, False):
            return tier
    return tiers[0]


def baselines(
    per_item_tier_grades: Sequence[dict[str, bool]],
    per_item_tier_costs: Sequence[dict[str, float]],
    tiers: Sequence[str],
    *,
    rng_seed: int = 0,
    strong_fraction: float = 0.5,
) -> dict[str, dict[str, float]]:
    """The three plotted baselines (build-spec §11).

    - ``always_cheap`` — the cheapest tier (``tiers[0]``) on every item: the cost
      and quality floor.
    - ``always_strong`` — the strongest tier (``tiers[-1]``) on every item: the
      100% reference retention/cost-reduction are measured against.
    - ``random`` — route each item to the strong tier with probability
      ``strong_fraction`` **uniformly at random**, seeded by ``rng_seed`` so the
      point is fully deterministic and reproducible; it sits between the cheap and
      strong baselines and shows the router beats chance.

    Each entry is ``{"quality", "cost"}``. Empty input → all ``nan``.
    """
    if not tiers:
        raise ValueError("tiers must be non-empty to compute baselines.")
    cheap, strong = tiers[0], tiers[-1]

    cheap_point = _fixed_tier_point(per_item_tier_grades, per_item_tier_costs, cheap)
    strong_point = _fixed_tier_point(per_item_tier_grades, per_item_tier_costs, strong)

    rng = random.Random(rng_seed)
    rand_grades: list[bool] = []
    rand_costs: list[float] = []
    for grades, costs in zip(per_item_tier_grades, per_item_tier_costs, strict=True):
        tier = strong if rng.random() < strong_fraction else cheap
        rand_grades.append(grades.get(tier, False))
        rand_costs.append(costs[tier])
    random_point = {"quality": accuracy(rand_grades), "cost": mean_cost(rand_costs)}

    return {"always_cheap": cheap_point, "always_strong": strong_point, "random": random_point}


def _fixed_tier_point(
    per_item_tier_grades: Sequence[dict[str, bool]],
    per_item_tier_costs: Sequence[dict[str, float]],
    tier: str,
) -> dict[str, float]:
    """Quality + cost of always routing to a single ``tier``."""
    grades = [item.get(tier, False) for item in per_item_tier_grades]
    costs = [item[tier] for item in per_item_tier_costs]
    return {"quality": accuracy(grades), "cost": mean_cost(costs)}


# ----------------------------------------------------------------------------
# Frontier helpers
# ----------------------------------------------------------------------------
def frontier_points(sweep: Sequence[FrontierPoint]) -> list[FrontierPoint]:
    """Return the sweep sorted by ascending cost (left-to-right plot order)."""
    return sorted(sweep, key=lambda point: point.cost_usd_per_query)


def cost_reduction_at_target(
    points: Sequence[FrontierPoint],
    strong_quality: float,
    strong_cost: float,
    target_retention: float = 0.95,
) -> dict[str, float | bool]:
    """The single comparable headline: cheapest point at/above the retention target.

    Among ``points`` whose retention (``quality / strong_quality``) is ≥
    ``target_retention``, pick the one with the **lowest cost** and report its
    ``operating_param`` / ``retention`` / ``cost_reduction`` / ``quality`` /
    ``cost`` with ``reached_target=True``.

    If **no** point reaches the target, fall back to the highest-retention point,
    report its numbers, and set ``reached_target=False`` — the case is *flagged*,
    never faked. Empty ``points`` (or an undefined strong reference) →
    ``nan``/``False``.
    """
    if not points or not math.isfinite(strong_quality) or strong_quality == 0.0:
        return _no_target_result()

    scored = [(point, retention(point.quality, strong_quality)) for point in points]
    scored = [(point, ret) for point, ret in scored if math.isfinite(ret)]
    if not scored:
        return _no_target_result()

    qualifying = [(point, ret) for point, ret in scored if ret >= target_retention]
    if qualifying:
        point, ret = min(qualifying, key=lambda pair: pair[0].cost_usd_per_query)
        reached = True
    else:
        point, ret = max(scored, key=lambda pair: pair[1])
        reached = False

    return {
        "operating_param": point.operating_param,
        "retention": ret,
        "cost_reduction": cost_reduction(point.cost_usd_per_query, strong_cost),
        "quality": point.quality,
        "cost": point.cost_usd_per_query,
        "reached_target": reached,
    }


def _no_target_result() -> dict[str, float | bool]:
    """The N/A headline result (no points / undefined reference)."""
    return {
        "operating_param": NAN,
        "retention": NAN,
        "cost_reduction": NAN,
        "quality": NAN,
        "cost": NAN,
        "reached_target": False,
    }


# ----------------------------------------------------------------------------
# Distributional aggregation over the R eval repeats
# ----------------------------------------------------------------------------
def mean_spread(values: Sequence[float]) -> tuple[float, float]:
    """``(mean, population-stdev)`` over the R repeats. Empty → ``(nan, nan)``.

    The spread is the **population** standard deviation, so a single repeat (R=1)
    or identical repeats both yield ``0.0`` spread.
    """
    if not values:
        return NAN, NAN
    mean = statistics.fmean(values)
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, spread
