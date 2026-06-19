"""The routing orchestrator (build-spec §5).

``route()`` is the public entry point with the build-spec §13 surface
``route(query, strategy="cascade", tau=0.8)``. ``query`` is first-positional and
``client`` is keyword-only (defaulting to :func:`frugalroute.llm.get_client`),
so live callers stay ergonomic while no-key tests inject a fake client.

**Strategy A — Cascade** (implemented here):

1. generate on the cheap tier;
2. gate the cheap answer with the cheap structured judge;
3. accept iff ``verdict.sufficient AND verdict.confidence >= tau`` — otherwise
   escalate to the strong tier.

The reported ``cost_usd`` is the **full additive** cost of every call actually
made (cheap + gate [+ strong]). This is deliberate and honest: an escalated
cascade costs *more* than always-strong (``c_cheap + c_gate + c_strong >
c_strong``) — the losing region of build-spec §8. Refusals never crash the path
(cheap→escalate, gate→escalate, strong→surface honestly).

**Strategy B — Predictive** (split 04): embed the query with a local model, let a
small classifier pick the cheapest sufficient tier, and generate with **only** that
tier — exactly one model call, no cheap-then-strong double spend and no gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from frugalroute.gate import gate
from frugalroute.generate import generate
from frugalroute.llm import DEFAULT_TIERS, cheap_tier, get_client, strong_tier
from frugalroute.models import GateVerdict, RouteResult
from frugalroute.prompts import PROMPT_VERSION

if TYPE_CHECKING:  # pragma: no cover - typing only
    from frugalroute.classifier import PredictiveRouter

CASCADE = "cascade"
PREDICTIVE = "predictive"


def route(
    query: str,
    *,
    strategy: str = CASCADE,
    benchmark: str = "gsm8k",
    tau: float = 0.8,
    theta: float | None = None,
    client: Any = None,
    tiers: Sequence[str] = DEFAULT_TIERS,
    router: PredictiveRouter | None = None,
    embedder: Any = None,
) -> RouteResult:
    """Route ``query`` with the chosen strategy and return a ``RouteResult``.

    ``client`` defaults to :func:`get_client` (resolved lazily, only when actually
    making an API call) so importing this module never needs a key. ``cascade``
    runs cheap → gate → escalate; ``predictive`` needs a trained ``router`` (a
    :class:`~frugalroute.classifier.PredictiveRouter`) and an optional ``embedder``
    (injectable for tests; defaults to the local model). Raises ``ValueError`` on
    an unknown strategy or a missing predictive router.

    The strategy and predictive prerequisites are validated **before** the client
    is resolved, so a bad strategy or a missing router fails fast with a clear
    message rather than first demanding a key. The predictive path also embeds and
    predicts (both local, no key) before resolving the client for its single call.
    """
    if strategy == PREDICTIVE:
        return _route_predictive(client, query, benchmark, router, theta, embedder)
    if strategy != CASCADE:
        raise ValueError(f"Unknown strategy {strategy!r}; expected {CASCADE!r} or {PREDICTIVE!r}.")
    if client is None:
        client = get_client()
    return _route_cascade(client, query, benchmark, tau, tiers)


def _route_predictive(
    client: Any,
    query: str,
    benchmark: str,
    router: PredictiveRouter | None,
    theta: float | None,
    embedder: Any,
) -> RouteResult:
    """Predictive routing: embed → classify → exactly ONE generate call.

    No cheap call and no gate — the classifier picks the tier upfront, so the
    reported cost is that single call's cost only. ``router.predict_tier`` embeds
    locally (no key) and validates the predicted tier and the embedding, raising a
    clear error on a degenerate embedding or an out-of-set tier (split-04 R11). A
    refusal on the single call is surfaced honestly (``refused=True``). ``p_strong``
    (P needs strong) is recorded for the UI decision margin; ``gate`` is ``None``.
    """
    if router is None:
        raise ValueError(
            "Predictive routing requires a trained router; pass route(..., router=...) "
            "(e.g. Router.load(path) / load_router(path))."
        )
    tier, p_strong = router.predict_tier(query, theta, embedder=embedder)
    if client is None:
        client = get_client()
    result = generate(client, tier, query, benchmark)
    return RouteResult(
        query=query,
        strategy=PREDICTIVE,
        tier_used=tier,
        escalated=(tier == router.tiers[-1]),
        answer=result.text,
        correct=None,
        gate=None,
        p_strong=p_strong,
        refused=result.refused,
        cost_usd=result.cost_usd,
        latency_s=result.latency_s,
        prompt_version=PROMPT_VERSION,
    )


def _route_cascade(
    client: Any,
    query: str,
    benchmark: str,
    tau: float,
    tiers: Sequence[str],
) -> RouteResult:
    """Cascade: cheap → gate → accept (conf >= tau) else escalate to strong.

    Cost and latency accumulate additively over exactly the calls made. The
    accept rule is ``sufficient AND confidence >= tau`` (build-spec's "≥ τ"): a
    confidence exactly equal to ``tau`` accepts; doubt (below ``tau``, or not
    sufficient) escalates.
    """
    cheap = cheap_tier(tiers)
    strong = strong_tier(tiers)

    cheap_result = generate(client, cheap, query, benchmark)
    cost = cheap_result.cost_usd
    latency = cheap_result.latency_s

    # Cheap refuses → skip the gate and escalate straight to strong.
    if cheap_result.refused:
        return _escalate(
            client, query, benchmark, strong, cost, latency, gate_verdict=None, refused=True
        )

    outcome = gate(client, query, cheap_result.text, gate_model=cheap)
    cost += outcome.cost_usd
    latency += outcome.latency_s
    verdict = outcome.verdict

    accepted = verdict.sufficient and verdict.confidence >= tau
    if accepted:
        return RouteResult(
            query=query,
            strategy=CASCADE,
            tier_used=cheap,
            escalated=False,
            answer=cheap_result.text,
            correct=None,
            gate=verdict,
            p_strong=None,
            refused=False,
            cost_usd=cost,
            latency_s=latency,
            prompt_version=PROMPT_VERSION,
        )

    return _escalate(
        client,
        query,
        benchmark,
        strong,
        cost,
        latency,
        gate_verdict=verdict,
        refused=outcome.refused,
    )


def _escalate(
    client: Any,
    query: str,
    benchmark: str,
    strong: str,
    cost: float,
    latency: float,
    *,
    gate_verdict: GateVerdict | None,
    refused: bool,
) -> RouteResult:
    """Run the strong tier and assemble the escalated ``RouteResult``.

    ``cost``/``latency`` already include the cheap (+ gate) calls; the strong
    call is added here. A strong-tier refusal is surfaced honestly (answer is the
    empty refusal text, ``refused=True``) — never a crash or a silent downgrade.
    """
    strong_result = generate(client, strong, query, benchmark)
    return RouteResult(
        query=query,
        strategy=CASCADE,
        tier_used=strong,
        escalated=True,
        answer=strong_result.text,
        correct=None,
        gate=gate_verdict,
        p_strong=None,
        refused=refused or strong_result.refused,
        cost_usd=cost + strong_result.cost_usd,
        latency_s=latency + strong_result.latency_s,
        prompt_version=PROMPT_VERSION,
    )
