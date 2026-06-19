"""Break-even economics (build-spec §8) — tests 13–16.

These pin the cascade's central honesty: it only saves once acceptance clears
the break-even threshold, and below it there is a real losing region. The fixture
costs are the §5/§8 numbers (c_cheap=$0.0014, c_gate=$0.0004, c_strong=$0.0070).
"""

from __future__ import annotations

import pytest

from frugalroute.economics import break_even_acceptance, cascade_saves, mean_cascade_cost

C_CHEAP = 0.0014
C_GATE = 0.0004
C_STRONG = 0.0070


def test_break_even_acceptance_matches_spec_fixture() -> None:
    # 13. (c_cheap + c_gate) / c_strong == 0.0018 / 0.0070 ≈ 0.2571.
    assert break_even_acceptance(C_CHEAP, C_GATE, C_STRONG) == pytest.approx(0.2571, abs=1e-3)


def test_mean_cost_and_saves_above_break_even() -> None:
    # 14. At acceptance 0.60: 0.0018 + 0.4*0.0070 = 0.0046 < 0.0070 → saves.
    mean = mean_cascade_cost(C_CHEAP, C_GATE, C_STRONG, 0.60)
    assert mean == pytest.approx(0.0046, abs=1e-9)
    assert mean < C_STRONG
    assert cascade_saves(C_CHEAP, C_GATE, C_STRONG, 0.60) is True


def test_mean_cost_and_loses_below_break_even() -> None:
    # 15. At acceptance 0.20 (< break-even): 0.0018 + 0.8*0.0070 = 0.0074 > 0.0070 → loses.
    mean = mean_cascade_cost(C_CHEAP, C_GATE, C_STRONG, 0.20)
    assert mean == pytest.approx(0.0074, abs=1e-9)
    assert mean > C_STRONG
    assert cascade_saves(C_CHEAP, C_GATE, C_STRONG, 0.20) is False


def test_break_even_at_exact_threshold_is_not_saving() -> None:
    # The boundary itself does not save (strict '>'): at acceptance == break-even,
    # mean cost == c_strong exactly.
    be = break_even_acceptance(C_CHEAP, C_GATE, C_STRONG)
    assert mean_cascade_cost(C_CHEAP, C_GATE, C_STRONG, be) == pytest.approx(C_STRONG, abs=1e-12)
    assert cascade_saves(C_CHEAP, C_GATE, C_STRONG, be) is False


def test_break_even_zero_strong_raises() -> None:
    # 16. Undefined when c_strong <= 0.
    with pytest.raises(ValueError, match="c_strong must be > 0"):
        break_even_acceptance(C_CHEAP, C_GATE, 0.0)
    with pytest.raises(ValueError):
        break_even_acceptance(C_CHEAP, C_GATE, -1.0)
