"""Cascade break-even economics (build-spec §8).

Pure arithmetic, no I/O, no key. These functions make the cascade's central
subtlety explicit and testable: a cascade pays the cheap call **plus** the gate
on *every* query, so it only beats always-strong once the **acceptance rate**
clears a break-even threshold. Below that threshold the cascade *loses money* —
the "losing region" the frontier must be honest about.

Given the per-call costs ``c_cheap`` (cheap generation), ``c_gate`` (the judge),
and ``c_strong`` (strong generation), the per-query expected cascade cost is::

    mean = c_cheap + c_gate + (1 - acceptance) * c_strong

which is below the always-strong cost ``c_strong`` exactly when::

    acceptance > (c_cheap + c_gate) / c_strong   # == break_even_acceptance
"""

from __future__ import annotations


def break_even_acceptance(c_cheap: float, c_gate: float, c_strong: float) -> float:
    """Acceptance rate above which the cascade saves vs always-strong (§8).

    ``(c_cheap + c_gate) / c_strong``. Raises ``ValueError`` when
    ``c_strong <= 0`` (no strong cost ⇒ the comparison is undefined).
    """
    if c_strong <= 0:
        raise ValueError(f"c_strong must be > 0 (got {c_strong!r}); break-even is undefined.")
    return (c_cheap + c_gate) / c_strong


def mean_cascade_cost(c_cheap: float, c_gate: float, c_strong: float, acceptance: float) -> float:
    """Per-query expected cascade cost: ``c_cheap + c_gate + (1-acceptance)*c_strong``.

    The cheap call and the gate are paid on every query; the strong call is paid
    only on the ``(1 - acceptance)`` fraction that escalates.
    """
    return c_cheap + c_gate + (1.0 - acceptance) * c_strong


def cascade_saves(c_cheap: float, c_gate: float, c_strong: float, acceptance: float) -> bool:
    """True iff the cascade's expected cost is below always-strong at ``acceptance``.

    Equivalent to ``acceptance > break_even_acceptance(...)`` and to
    ``mean_cascade_cost(...) < c_strong``.
    """
    return acceptance > break_even_acceptance(c_cheap, c_gate, c_strong)
