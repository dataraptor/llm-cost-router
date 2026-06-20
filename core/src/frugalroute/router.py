"""The routing orchestrator (build-spec §5).

``route()`` is the public entry point with the build-spec §13 surface
``route(query, strategy="cascade", tau=0.8)``. ``query`` is first-positional and
``client`` is keyword-only (defaulting to :func:`frugalroute.llm.get_client`),
so live callers stay ergonomic while no-key tests inject a fake client.

**Strategy A — Cascade**:

1. generate on the cheap tier;
2. gate the cheap answer with the cheap structured judge;
3. accept iff ``verdict.sufficient AND verdict.confidence >= tau`` — otherwise
   escalate to the strong tier.

The reported ``cost_usd`` is the **full additive** cost of every call actually
made (cheap + gate [+ strong]). This is deliberate and honest: an escalated
cascade costs *more* than always-strong (``c_cheap + c_gate + c_strong >
c_strong``) — the losing region of build-spec §8. Refusals never crash the path
(cheap→escalate, gate→escalate, strong→surface honestly).

**Strategy B — Predictive**: embed the query with a local model, let a small
classifier pick the cheapest sufficient tier, and generate with **only** that
tier — exactly one model call, no cheap-then-strong double spend and no gate.

**Streaming (split 09).** :func:`route_events` is a generator that yields a
:class:`RouteEvent` at every cascade/predictive boundary and finally yields a
``done`` event carrying the serialized :class:`RouteResult`. :func:`route` simply
**drains** that generator and returns its result, so there is exactly one
implementation of the routing logic — the synchronous result can never diverge
from the streamed one (the ``done`` payload equals ``route()`` for the same
inputs). The events are a presentation-layer narration of the *existing*
boundaries (one event per real call); they add no routing behavior.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from frugalroute.gate import gate
from frugalroute.generate import generate
from frugalroute.llm import DEFAULT_TIERS, cheap_tier, get_client, strong_tier
from frugalroute.models import GateVerdict, RouteResult, route_result_to_dict
from frugalroute.prompts import PROMPT_VERSION

if TYPE_CHECKING:  # pragma: no cover - typing only
    from frugalroute.classifier import PredictiveRouter

CASCADE = "cascade"
PREDICTIVE = "predictive"


@dataclass
class RouteEvent:
    """One ordered boundary event emitted while routing (split 09).

    ``type`` is one of ``"phase"|"candidate"|"gate"|"cost"|"retry"|"refusal"|
    "done"``; ``data`` is the JSON-serializable payload for that type (see the
    split-09 contract). The terminal event is always ``type == "done"`` with
    ``data`` the :func:`~frugalroute.models.route_result_to_dict` of the result.
    """

    type: str
    data: dict[str, Any]


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

    Implemented by **draining** :func:`route_events` and returning the result it
    carries on its terminal ``done`` event — so ``route()`` and ``route_events()``
    share one implementation and can never diverge. ``client`` defaults to
    :func:`get_client` (resolved lazily, only when actually making an API call) so
    importing this module never needs a key. Raises ``ValueError`` on an unknown
    strategy or a missing predictive ``router`` (before any key is demanded).
    """
    events = _route_impl(
        query,
        strategy=strategy,
        benchmark=benchmark,
        tau=tau,
        theta=theta,
        client=client,
        tiers=tiers,
        router=router,
        embedder=embedder,
    )
    result: RouteResult | None = None
    try:
        while True:
            next(events)
    except StopIteration as stop:
        result = stop.value
    if result is None:  # pragma: no cover - the generator always returns a result
        raise RuntimeError("route_events() did not produce a result")
    return result


def route_events(
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
) -> Iterator[RouteEvent]:
    """Yield ordered :class:`RouteEvent`\\ s for the route; last one is ``done``.

    Same routing as :func:`route` (it drains this), same query-first /
    keyword-only ``client`` convention, so the API can inject a shared client.
    The ``done`` event's ``data`` equals what :func:`route` returns (serialized).
    A bad strategy / missing router raises ``ValueError`` on first iteration.
    """
    return _route_impl(
        query,
        strategy=strategy,
        benchmark=benchmark,
        tau=tau,
        theta=theta,
        client=client,
        tiers=tiers,
        router=router,
        embedder=embedder,
    )


def _route_impl(
    query: str,
    *,
    strategy: str,
    benchmark: str,
    tau: float,
    theta: float | None,
    client: Any,
    tiers: Sequence[str],
    router: PredictiveRouter | None,
    embedder: Any,
) -> Generator[RouteEvent, None, RouteResult]:
    """The single routing generator: validates, dispatches, returns the result."""
    if strategy == PREDICTIVE:
        return (yield from _predictive_events(client, query, benchmark, router, theta, embedder))
    if strategy != CASCADE:
        raise ValueError(f"Unknown strategy {strategy!r}; expected {CASCADE!r} or {PREDICTIVE!r}.")
    if client is None:
        client = get_client()
    return (yield from _cascade_events(client, query, benchmark, tau, tiers))


def _refusal_message(tier: str) -> str:
    """An honest, content-free message for a tier refusal (no fabricated answer)."""
    return f"{tier} returned a refusal (stop_reason='refusal')."


def _cascade_events(
    client: Any,
    query: str,
    benchmark: str,
    tau: float,
    tiers: Sequence[str],
) -> Generator[RouteEvent, None, RouteResult]:
    """Cascade: cheap → gate → accept (conf >= tau) else escalate, yielding events.

    Emits (accepted): ``phase:gen → candidate → cost → phase:gate → gate → cost
    → done``; (escalated): ``… → gate → phase:escalate → cost → done``; (cheap
    refusal): ``phase:gen → refusal(cheap) → phase:escalate → cost → done``. Cost
    and latency accumulate additively over exactly the calls made; the accept
    rule is ``sufficient AND confidence >= tau`` (build-spec's "≥ τ").
    """
    cheap = cheap_tier(tiers)
    strong = strong_tier(tiers)

    yield RouteEvent("phase", {"phase": "gen", "tier": cheap})
    cheap_result = generate(client, cheap, query, benchmark)
    cost = cheap_result.cost_usd
    latency = cheap_result.latency_s

    # Cheap refuses → skip the gate and escalate straight to strong.
    if cheap_result.refused:
        yield RouteEvent("refusal", {"stage": "cheap", "message": _refusal_message(cheap)})
        return (
            yield from _escalate_events(
                client,
                query,
                benchmark,
                strong,
                cost,
                latency,
                gate_verdict=None,
                prior_refused=True,
            )
        )

    yield RouteEvent(
        "candidate",
        {"answer": cheap_result.text, "tier": cheap, "cost_usd": cheap_result.cost_usd},
    )
    yield RouteEvent("cost", {"cost_usd_cumulative": cost})

    yield RouteEvent("phase", {"phase": "gate", "tier": cheap})
    outcome = gate(client, query, cheap_result.text, gate_model=cheap)
    cost += outcome.cost_usd
    latency += outcome.latency_s
    verdict = outcome.verdict
    yield RouteEvent(
        "gate",
        {
            "sufficient": verdict.sufficient,
            "confidence": verdict.confidence,
            "reason": verdict.reason,
            "cost_usd": outcome.cost_usd,
        },
    )

    accepted = verdict.sufficient and verdict.confidence >= tau
    if accepted:
        yield RouteEvent("cost", {"cost_usd_cumulative": cost})
        result = RouteResult(
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
        yield RouteEvent("done", route_result_to_dict(result))
        return result

    return (
        yield from _escalate_events(
            client,
            query,
            benchmark,
            strong,
            cost,
            latency,
            gate_verdict=verdict,
            prior_refused=outcome.refused,
        )
    )


def _escalate_events(
    client: Any,
    query: str,
    benchmark: str,
    strong: str,
    cost: float,
    latency: float,
    *,
    gate_verdict: GateVerdict | None,
    prior_refused: bool,
) -> Generator[RouteEvent, None, RouteResult]:
    """Run the strong tier and yield ``phase:escalate → [refusal] → cost → done``.

    ``cost``/``latency`` already include the cheap (+ gate) calls; the strong call
    is added here. A strong-tier refusal is surfaced honestly (a ``refusal`` event
    + ``refused=True`` with the empty refusal text) — never a crash or a silent
    downgrade.
    """
    yield RouteEvent("phase", {"phase": "escalate", "tier": strong})
    strong_result = generate(client, strong, query, benchmark)
    cost += strong_result.cost_usd
    latency += strong_result.latency_s
    if strong_result.refused:
        yield RouteEvent("refusal", {"stage": "strong", "message": _refusal_message(strong)})
    yield RouteEvent("cost", {"cost_usd_cumulative": cost})
    result = RouteResult(
        query=query,
        strategy=CASCADE,
        tier_used=strong,
        escalated=True,
        answer=strong_result.text,
        correct=None,
        gate=gate_verdict,
        p_strong=None,
        refused=prior_refused or strong_result.refused,
        cost_usd=cost,
        latency_s=latency,
        prompt_version=PROMPT_VERSION,
    )
    yield RouteEvent("done", route_result_to_dict(result))
    return result


def _predictive_events(
    client: Any,
    query: str,
    benchmark: str,
    router: PredictiveRouter | None,
    theta: float | None,
    embedder: Any,
) -> Generator[RouteEvent, None, RouteResult]:
    """Predictive routing: embed → classify → exactly ONE generate call.

    Yields ``phase:embed → phase:classify → [refusal] → done`` — no cheap call and
    no gate (the classifier picks the tier upfront), so the reported cost is that
    single call's cost only. ``router.predict_tier`` embeds locally (no key) and
    validates the predicted tier and the embedding, raising a clear ``ValueError``
    on a degenerate embedding or an out-of-set tier (split-04 R11) **before** any
    model call. ``p_strong`` is recorded for the UI decision margin; ``gate`` is
    ``None``.
    """
    if router is None:
        raise ValueError(
            "Predictive routing requires a trained router; pass route(..., router=...) "
            "(e.g. Router.load(path) / load_router(path))."
        )
    yield RouteEvent("phase", {"phase": "embed", "tier": None})
    tier, p_strong = router.predict_tier(query, theta, embedder=embedder)
    yield RouteEvent("phase", {"phase": "classify", "tier": tier})
    if client is None:
        client = get_client()
    result = generate(client, tier, query, benchmark)
    if result.refused:
        stage = "strong" if tier == router.tiers[-1] else "cheap"
        yield RouteEvent("refusal", {"stage": stage, "message": _refusal_message(tier)})
    route_result = RouteResult(
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
    yield RouteEvent("done", route_result_to_dict(route_result))
    return route_result
