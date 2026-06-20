"""Request/response models + the §7 serialization adapter (split-06 §3/§4).

The engine's ``RouteResult`` / ``EvalReport`` are plain dataclasses (build-spec §7).
This module is the **single** place that maps them onto JSON, field-for-field, plus
the derived UI extras the contract defines (``decision_margin``, ``cost_breakdown``).
``EvalReport`` reuses the engine's own ``harness.report_to_dict`` so the HTTP shape
can never drift from the engine's persisted shape.
"""

from __future__ import annotations

from typing import Any, Literal

from frugalroute import llm
from frugalroute.harness import report_to_dict
from frugalroute.models import RouteResult
from pydantic import BaseModel, Field, model_validator

# Generous cap so a pathological multi-megabyte body is a clean 422, not an
# accidental (expensive) upstream call. Benchmark prompts are a few hundred chars.
MAX_QUERY_CHARS = 16384

# Short display names for the cost-breakdown label (presentation only; the numbers
# all come from the engine). Falls back to the raw model id for anything unmapped.
_SHORT_NAMES: dict[str, str] = {
    "claude-haiku-4-5": "Haiku",
    "claude-sonnet-4-6": "Sonnet",
    "claude-opus-4-8": "Opus",
    "gpt-5.5": "gpt-5.5",
}


def _short(model_id: str) -> str:
    return _SHORT_NAMES.get(model_id, model_id)


# ----------------------------------------------------------------------------
# Health / config / examples
# ----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str
    has_api_key: bool


class TierPrice(BaseModel):
    input_per_mtok: float
    output_per_mtok: float


class ConfigDefaults(BaseModel):
    tau: float
    theta: float


class ConfigResponse(BaseModel):
    prompt_version: str
    model_tiers: list[str]
    strategies: list[str]
    pricing: dict[str, TierPrice]
    always_strong_cost_ref_usd: float
    defaults: ConfigDefaults
    pricing_pinned_date: str


class ExampleOut(BaseModel):
    """An example picker entry — answers/labels are NOT served (live via /route)."""

    id: str
    benchmark: str
    label: str
    query: str


# ----------------------------------------------------------------------------
# Route
# ----------------------------------------------------------------------------
class RouteRequest(BaseModel):
    strategy: Literal["cascade", "predictive"]
    query: str | None = Field(default=None, max_length=MAX_QUERY_CHARS)
    example_id: str | None = None
    benchmark: Literal["gsm8k", "mmlu"] | None = None
    tau: float | None = Field(default=None, ge=0.0, le=1.0)
    theta: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _exactly_one_input(self) -> RouteRequest:
        has_query = bool(self.query and self.query.strip())
        has_example = bool(self.example_id and self.example_id.strip())
        if has_query == has_example:
            raise ValueError("Provide exactly one of 'query' or 'example_id'.")
        return self


class GateVerdictOut(BaseModel):
    sufficient: bool
    confidence: float
    reason: str


class CostBreakdown(BaseModel):
    """Faithful cost summary: a label + the reference + the exceeds flag.

    The engine exposes only the *total* ``cost_usd`` (no per-call USD), so this
    carries no fabricated per-term dollars — only which calls ran (``label``), the
    always-Opus reference, and whether this route cost more than always-Opus.
    """

    label: str
    always_strong_usd: float
    exceeds_always_strong: bool


class RouteResponse(BaseModel):
    # --- RouteResult, exactly per §7 ---
    query: str
    strategy: str
    tier_used: str
    escalated: bool
    answer: str
    correct: bool | None
    gate: GateVerdictOut | None
    p_strong: float | None
    refused: bool
    cost_usd: float
    latency_s: float
    prompt_version: str
    # --- derived UI extras (computed from RouteResult + config; never fabricated) ---
    decision_margin: float | None
    cost_breakdown: CostBreakdown


def _cost_label(result: RouteResult) -> str:
    """Which calls actually ran, as a human-readable label (faithful to the run)."""
    cheap = _short(llm.DEFAULT_TIERS[0])
    if result.strategy == "predictive":
        return f"= {_short(result.tier_used)}"
    # cascade
    if not result.escalated:
        return f"= {cheap} + gate"
    # escalated: the gate ran unless the cheap tier refused (gate is None then)
    if result.gate is not None:
        return f"= {cheap} + gate + {_short(result.tier_used)}"
    return f"= {cheap} + {_short(result.tier_used)}"


def route_response(
    result: RouteResult, *, theta_used: float | None, always_strong_usd: float
) -> RouteResponse:
    """Adapt a ``RouteResult`` to the HTTP response (§7 + derived extras).

    ``decision_margin`` is ``p_strong - theta_used`` on the predictive path (the
    same ``theta`` the engine decided with) and ``None`` for cascade. The
    cost-breakdown flags honestly when an escalated cascade exceeded always-Opus.
    """
    gate = GateVerdictOut(**result.gate.model_dump()) if result.gate is not None else None
    decision_margin: float | None = None
    if result.strategy == "predictive" and result.p_strong is not None and theta_used is not None:
        decision_margin = result.p_strong - theta_used
    return RouteResponse(
        query=result.query,
        strategy=result.strategy,
        tier_used=result.tier_used,
        escalated=result.escalated,
        answer=result.answer,
        correct=result.correct,
        gate=gate,
        p_strong=result.p_strong,
        refused=result.refused,
        cost_usd=result.cost_usd,
        latency_s=result.latency_s,
        prompt_version=result.prompt_version,
        decision_margin=decision_margin,
        cost_breakdown=CostBreakdown(
            label=_cost_label(result),
            always_strong_usd=always_strong_usd,
            exceeds_always_strong=result.cost_usd > always_strong_usd,
        ),
    )


# ----------------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------------
class EvalRequest(BaseModel):
    strategy: Literal["cascade", "predictive", "both"]
    benchmark: Literal["gsm8k", "mmlu"]
    quick: bool = True
    # Cap the grid length so a pathological many-thousand-point sweep can't be
    # requested over HTTP (each point is cheap arithmetic, but the list is unbounded
    # user input); a real sweep is a handful of taus/thetas.
    grid: list[float] | None = Field(default=None, max_length=64)
    repeats: int | None = Field(default=None, ge=1, le=10)


def bundle_to_json(
    reports: list[Any],
    *,
    benchmark: str,
    n_test: int,
    n_calibration: int,
    small_n: bool,
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the `/api/eval/sample`-shaped bundle from ``EvalReport`` objects.

    Each report is serialized via the engine's ``report_to_dict`` so the HTTP shape
    is identical to the engine's persisted shape (round-trip guaranteed).
    """
    return {
        "reports": [report_to_dict(report) for report in reports],
        "benchmark": benchmark,
        "frozen_split": {
            "n_test": n_test,
            "n_calibration": n_calibration,
            "small_n": small_n,
        },
        "generated_at": generated_at,
    }
