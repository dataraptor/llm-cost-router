"""Cascade quality gate — the cheap structured judge (build-spec §5/§6).

The gate runs the *cheapest* tier as a strict judge over a QUESTION and a
candidate cheap ANSWER, parsing a :class:`~frugalroute.models.GateVerdict` via the
structured-output path of :func:`frugalroute.llm.call`. It is quality-first and
refusal-safe:

- ``stop_reason`` is checked **before** any content is read (handled inside
  ``llm.call``); a gate refusal yields a conservative *escalate* verdict.
- An empty / malformed structured output (no parseable verdict) also yields a
  conservative *escalate* verdict — the router must never crash on junk.
- ``confidence`` is **clamped to [0, 1]** here (the JSON schema cannot bound it,
  per the API constraints), so downstream threshold logic is always well-defined.

The gate sends no sampling / effort / thinking params (enforced by ``llm.call``)
and never runs on the strong tier — it is one short, cheap call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frugalroute.llm import call, cheap_tier
from frugalroute.models import GateVerdict
from frugalroute.prompts import GATE_SYSTEM, gate_user


@dataclass
class GateOutcome:
    """Result of one gate (judge) call.

    ``refused`` is True only on a genuine gate **refusal** (``stop_reason ==
    "refusal"``); a merely malformed/empty structured output is *not* a refusal
    but still produces a conservative escalate ``verdict``.
    """

    verdict: GateVerdict
    refused: bool
    cost_usd: float
    latency_s: float


def _conservative_verdict(reason: str) -> GateVerdict:
    """A doubt-biased verdict that forces escalation (sufficient=False, conf=0)."""
    return GateVerdict(sufficient=False, confidence=0.0, reason=reason)


def _clamp_unit(value: float) -> float:
    """Clamp a confidence to the unit interval ``[0.0, 1.0]``."""
    return min(max(value, 0.0), 1.0)


def gate(
    client: Any,
    question: str,
    cheap_answer: str,
    *,
    gate_model: str | None = None,
) -> GateOutcome:
    """Judge the cheap ANSWER to QUESTION with the cheap structured judge.

    Returns a :class:`GateOutcome`. On a gate refusal or an unparseable verdict,
    the verdict is conservative (``sufficient=False, confidence=0.0``) so the
    router escalates — quality-first (build-spec §17). ``confidence`` is clamped
    to ``[0, 1]`` before it is exposed. Cost/latency come from the call's usage.
    """
    model = gate_model or cheap_tier()
    result = call(
        client,
        model,
        GATE_SYSTEM,
        gate_user(question, cheap_answer),
        parse_model=GateVerdict,
    )

    if result.refused:
        return GateOutcome(
            verdict=_conservative_verdict("gate refused → escalate"),
            refused=True,
            cost_usd=result.cost_usd,
            latency_s=result.latency_s,
        )

    parsed = result.parsed
    if not isinstance(parsed, GateVerdict):
        # Empty / malformed structured output — degrade to a conservative escalate
        # (not counted as a refusal; the gate responded, just unparseably).
        return GateOutcome(
            verdict=_conservative_verdict("gate returned no parseable verdict → escalate"),
            refused=False,
            cost_usd=result.cost_usd,
            latency_s=result.latency_s,
        )

    verdict = GateVerdict(
        sufficient=parsed.sufficient,
        confidence=_clamp_unit(float(parsed.confidence)),
        reason=parsed.reason,
    )
    return GateOutcome(
        verdict=verdict,
        refused=False,
        cost_usd=result.cost_usd,
        latency_s=result.latency_s,
    )
