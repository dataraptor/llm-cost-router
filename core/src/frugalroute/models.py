"""Data contracts for FrugalRoute (build-spec §7).

`GateVerdict` is the **only** type sent to the Anthropic API (as a structured
output). The rest are assembled in code. `GateVerdict` is pydantic; the result
and report types are plain dataclasses — they are built locally and never
serialized to the API, so they carry no schema constraints.

The §7 field list is authoritative for names and meaning. Three fields are
additive completions required by §9/§11 (distributional reporting) and the
predictive UI, and are annotated below:
  * the ``*_spread`` fields — mean ± spread must travel with every reported
    quality/cost/retention number (§9, §11);
  * ``RouteResult.p_strong`` — P(needs strong) for the predictive decision
    margin (set on the predictive path in split 04; ``None`` for cascade).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


class GateVerdict(BaseModel):
    """Cascade quality-gate verdict — the only structured output sent to the API.

    ``confidence`` is the gate's probability (0.0–1.0) that the candidate answer
    is correct. It is intentionally **not** range-constrained in the JSON schema
    (the Anthropic structured-output schema rejects numeric bounds); clamp or
    validate it in the gate layer (split 03), not here. ``extra="forbid"``
    yields ``additionalProperties: false`` in the emitted schema.
    """

    model_config = ConfigDict(extra="forbid")

    sufficient: bool
    confidence: float
    reason: str


@dataclass
class RouteResult:
    """Outcome of routing a single query (assembled in code, never sent)."""

    query: str
    strategy: str  # "cascade" | "predictive"
    tier_used: str  # model ID actually used for the returned answer
    escalated: bool  # cascade: did it go to the strong tier?
    answer: str
    correct: bool | None  # set only in eval (grader); None in the live demo
    gate: GateVerdict | None  # cascade only
    p_strong: float | None  # predictive only: P(needs strong); None for cascade
    refused: bool  # any tier returned stop_reason="refusal"
    cost_usd: float
    latency_s: float
    prompt_version: str


def route_result_to_dict(result: RouteResult) -> dict[str, Any]:
    """Serialize a ``RouteResult`` to a plain JSON-ready dict (§7 fields only).

    The ``gate`` (a :class:`GateVerdict`) is flattened via ``model_dump``; every
    other field is already JSON-native. This is the canonical ``done``-event
    payload of the streaming router (split 09) and round-trips losslessly with
    :func:`route_result_from_dict`.
    """
    return {
        "query": result.query,
        "strategy": result.strategy,
        "tier_used": result.tier_used,
        "escalated": result.escalated,
        "answer": result.answer,
        "correct": result.correct,
        "gate": result.gate.model_dump() if result.gate is not None else None,
        "p_strong": result.p_strong,
        "refused": result.refused,
        "cost_usd": result.cost_usd,
        "latency_s": result.latency_s,
        "prompt_version": result.prompt_version,
    }


def route_result_from_dict(data: dict[str, Any]) -> RouteResult:
    """Rebuild a ``RouteResult`` from :func:`route_result_to_dict` output.

    The inverse of :func:`route_result_to_dict`; reconstructs the ``gate``
    :class:`GateVerdict` when present. Used by the API to re-derive the full
    response (with the §7 derived extras) from a streamed ``done`` event.
    """
    gate = data.get("gate")
    return RouteResult(
        query=data["query"],
        strategy=data["strategy"],
        tier_used=data["tier_used"],
        escalated=data["escalated"],
        answer=data["answer"],
        correct=data.get("correct"),
        gate=GateVerdict(**gate) if gate is not None else None,
        p_strong=data.get("p_strong"),
        refused=data["refused"],
        cost_usd=data["cost_usd"],
        latency_s=data["latency_s"],
        prompt_version=data["prompt_version"],
    )


@dataclass
class FrontierPoint:
    """One operating point on a cost-quality Pareto frontier.

    ``quality_spread`` / ``cost_spread`` are the ± spread (standard deviation)
    of the metric over the R eval repeats.
    """

    operating_param: float  # τ (cascade) or θ (predictive)
    quality: float  # accuracy on the slice (mean over the R runs)
    quality_spread: float  # ± spread of quality over R runs (std)
    cost_usd_per_query: float  # mean over the R runs
    cost_spread: float  # ± spread of cost over R runs (std)
    escalation_rate: float
    n: int


@dataclass
class EvalReport:
    """A full evaluation of one strategy against the baselines and the oracle.

    Each baseline entry is ``{"quality", "quality_spread", "cost", "cost_spread"}``;
    ``oracle`` is ``{"quality", "quality_spread", "cost"}`` (the unachievable
    ceiling — a computed bound, not a strategy). The ``*_spread`` headline fields
    carry the distributional ± that the UI renders (§9/§11).
    """

    strategy: str
    points: list[FrontierPoint]
    baselines: dict[str, dict[str, float]]
    oracle: dict[str, float]
    retention_at_target: float
    retention_at_target_spread: float
    cost_reduction_at_target: float
    cost_reduction_at_target_spread: float
    n_refused: int
    prompt_version: str
    model_tiers: list[str]
    n_runs: int
