"""Streaming router events (split 09, tests 1-4) — no-key, scripted_client.

``route_events()`` yields ordered boundary events and a terminal ``done`` carrying
the serialized ``RouteResult``. It shares one implementation with ``route()`` (the
synchronous result is just the drained stream), so the ``done`` payload must equal
``route()`` for the same scripted responses. The per-call token usage mirrors
``test_cascade.py`` so the costs are the §5/§8 fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from frugalroute.llm import cheap_tier, cost_usd, strong_tier
from frugalroute.models import GateVerdict, route_result_from_dict, route_result_to_dict
from frugalroute.router import route, route_events

CHEAP = cheap_tier()
STRONG = strong_tier()

C_CHEAP = cost_usd(CHEAP, 150, 250)  # 0.0014
C_GATE = cost_usd(CHEAP, 150, 50)  # 0.0004
C_STRONG = cost_usd(STRONG, 150, 250)  # 0.0070


def _cheap_resp(fake_response, fake_usage, *, text="cheap answer", refused=False):
    if refused:
        return fake_response(
            stop_reason="refusal", usage=fake_usage(input_tokens=150, output_tokens=250)
        )
    return fake_response(text=text, usage=fake_usage(input_tokens=150, output_tokens=250))


def _gate_resp(fake_response, fake_usage, *, sufficient, confidence):
    return fake_response(
        parsed_output=GateVerdict(sufficient=sufficient, confidence=confidence, reason="judged"),
        usage=fake_usage(input_tokens=150, output_tokens=50),
    )


def _strong_resp(fake_response, fake_usage, *, text="strong answer", refused=False):
    if refused:
        return fake_response(
            stop_reason="refusal", usage=fake_usage(input_tokens=150, output_tokens=250)
        )
    return fake_response(text=text, usage=fake_usage(input_tokens=150, output_tokens=250))


def _types(events):
    return [e.type for e in events]


def _last(events):
    return events[-1]


# --- 1. Accepted cascade: exact order + done == route() ---------------------


def test_accepted_event_order_and_done_equals_route(
    scripted_client, fake_response, fake_usage
) -> None:
    responses = [
        _cheap_resp(fake_response, fake_usage),
        _gate_resp(fake_response, fake_usage, sufficient=True, confidence=0.91),
    ]
    events = list(
        route_events("Q?", strategy="cascade", tau=0.8, client=scripted_client(responses))
    )

    assert _types(events) == ["phase", "candidate", "cost", "phase", "gate", "cost", "done"]
    # phases in order: gen then gate
    phases = [e.data["phase"] for e in events if e.type == "phase"]
    assert phases == ["gen", "gate"]
    # candidate carries the cheap answer + cheap cost
    cand = next(e for e in events if e.type == "candidate")
    assert cand.data == {
        "answer": "cheap answer",
        "tier": CHEAP,
        "cost_usd": pytest.approx(C_CHEAP),
    }
    # the two cost events tick up: cheap, then cheap+gate
    costs = [e.data["cost_usd_cumulative"] for e in events if e.type == "cost"]
    assert costs == [pytest.approx(C_CHEAP), pytest.approx(C_CHEAP + C_GATE)]

    # done == route() for the same scripted responses (deep-equal of the dict).
    expected = route_result_to_dict(
        route("Q?", strategy="cascade", tau=0.8, client=scripted_client(responses))
    )
    assert _last(events).type == "done"
    assert _last(events).data == expected
    assert _last(events).data["tier_used"] == CHEAP
    assert _last(events).data["escalated"] is False


# --- 2. Escalated cascade: includes phase:escalate; additive cost -----------


def test_escalated_event_order_and_additive_cost(
    scripted_client, fake_response, fake_usage
) -> None:
    responses = [
        _cheap_resp(fake_response, fake_usage),
        _gate_resp(fake_response, fake_usage, sufficient=False, confidence=0.62),
        _strong_resp(fake_response, fake_usage),
    ]
    events = list(
        route_events("Q?", strategy="cascade", tau=0.8, client=scripted_client(responses))
    )

    assert _types(events) == [
        "phase",  # gen
        "candidate",
        "cost",
        "phase",  # gate
        "gate",
        "phase",  # escalate
        "cost",
        "done",
    ]
    phases = [e.data["phase"] for e in events if e.type == "phase"]
    assert phases == ["gen", "gate", "escalate"]
    done = _last(events)
    assert done.data["escalated"] is True
    assert done.data["tier_used"] == STRONG
    assert done.data["cost_usd"] == pytest.approx(C_CHEAP + C_GATE + C_STRONG)
    # the (single, final) cost event equals the full additive cost
    final_cost = [e for e in events if e.type == "cost"][-1]
    assert final_cost.data["cost_usd_cumulative"] == pytest.approx(C_CHEAP + C_GATE + C_STRONG)


# --- 3. Cheap refusal: refusal(cheap) then escalation; done.refused ---------


def test_cheap_refusal_event_order(scripted_client, fake_response, fake_usage) -> None:
    responses = [
        _cheap_resp(fake_response, fake_usage, refused=True),
        _strong_resp(fake_response, fake_usage),
    ]
    events = list(
        route_events("Q?", strategy="cascade", tau=0.8, client=scripted_client(responses))
    )

    assert _types(events) == ["phase", "refusal", "phase", "cost", "done"]
    refusal = next(e for e in events if e.type == "refusal")
    assert refusal.data["stage"] == "cheap"
    phases = [e.data["phase"] for e in events if e.type == "phase"]
    assert phases == ["gen", "escalate"]
    done = _last(events)
    assert done.data["refused"] is True
    assert done.data["escalated"] is True
    assert done.data["gate"] is None  # gate skipped
    assert done.data["cost_usd"] == pytest.approx(C_CHEAP + C_STRONG)


def test_strong_refusal_emits_refusal_event(scripted_client, fake_response, fake_usage) -> None:
    responses = [
        _cheap_resp(fake_response, fake_usage),
        _gate_resp(fake_response, fake_usage, sufficient=False, confidence=0.4),
        _strong_resp(fake_response, fake_usage, refused=True),
    ]
    events = list(
        route_events("Q?", strategy="cascade", tau=0.8, client=scripted_client(responses))
    )

    refusals = [e for e in events if e.type == "refusal"]
    assert len(refusals) == 1 and refusals[0].data["stage"] == "strong"
    done = _last(events)
    assert done.data["refused"] is True
    assert done.data["answer"] == ""  # refusal text discarded, never fabricated


# --- 4. Predictive: phase:embed → phase:classify → done; no gate/candidate --


class _StubClf:
    def __init__(self, classes, proba_row) -> None:
        self.classes_ = np.array(classes)
        self._row = np.asarray(proba_row, dtype=np.float64)

    def predict_proba(self, features):
        return np.tile(self._row, (len(features), 1))


class _FakeEmbedder:
    def __init__(self, vector) -> None:
        self._vector = np.asarray(vector, dtype=np.float32)

    def encode(self, queries, **_kwargs):
        return np.tile(self._vector, (len(list(queries)), 1))


def _router(proba_row):
    from frugalroute.classifier import PredictiveRouter

    return PredictiveRouter(
        clf=_StubClf([CHEAP, STRONG], proba_row),
        tiers=[CHEAP, STRONG],
        embedder_name="fake",
        prompt_version="v1",
        label_run_ids=["labels-x"],
    )


def test_predictive_event_order_no_gate_or_candidate(fake_client, fake_usage) -> None:
    client = fake_client(
        text="The answer is 42.", usage=fake_usage(input_tokens=150, output_tokens=250)
    )
    events = list(
        route_events(
            "Q?",
            strategy="predictive",
            client=client,
            router=_router([0.1, 0.9]),
            embedder=_FakeEmbedder([0.1, 0.2, 0.3, 0.4]),
        )
    )

    assert _types(events) == ["phase", "phase", "done"]
    phases = [e.data["phase"] for e in events if e.type == "phase"]
    assert phases == ["embed", "classify"]
    assert not any(e.type in {"gate", "candidate"} for e in events)
    done = _last(events)
    assert done.data["strategy"] == "predictive"
    assert done.data["gate"] is None
    assert done.data["tier_used"] == STRONG

    # done == route() for the same inputs.
    expected = route_result_to_dict(
        route(
            "Q?",
            strategy="predictive",
            client=fake_client(
                text="The answer is 42.", usage=fake_usage(input_tokens=150, output_tokens=250)
            ),
            router=_router([0.1, 0.9]),
            embedder=_FakeEmbedder([0.1, 0.2, 0.3, 0.4]),
        )
    )
    assert done.data == expected


# --- shared-implementation / fail-fast guarantees ---------------------------


def test_route_drains_route_events_identical_result(
    scripted_client, fake_response, fake_usage
) -> None:
    responses = [
        _cheap_resp(fake_response, fake_usage),
        _gate_resp(fake_response, fake_usage, sufficient=True, confidence=0.9),
    ]
    drained = route("Q?", strategy="cascade", tau=0.8, client=scripted_client(responses))
    streamed = _last(
        list(route_events("Q?", strategy="cascade", tau=0.8, client=scripted_client(responses)))
    )
    assert route_result_to_dict(drained) == streamed.data


def test_route_result_dict_roundtrips(scripted_client, fake_response, fake_usage) -> None:
    # route_result_to_dict ↔ route_result_from_dict is lossless (incl. the gate),
    # which is what lets the API rebuild the full response from a `done` event.
    responses = [
        _cheap_resp(fake_response, fake_usage),
        _gate_resp(fake_response, fake_usage, sufficient=True, confidence=0.9),
    ]
    result = route("Q?", strategy="cascade", tau=0.8, client=scripted_client(responses))
    rebuilt = route_result_from_dict(route_result_to_dict(result))
    assert rebuilt == result  # dataclass equality, incl. the GateVerdict round-trip


def test_unknown_strategy_raises_on_iteration(scripted_client) -> None:
    gen = route_events("Q?", strategy="bogus", client=scripted_client([]))
    with pytest.raises(ValueError, match="Unknown strategy"):
        next(gen)


def test_predictive_without_router_raises_on_iteration(fake_client) -> None:
    gen = route_events("Q?", strategy="predictive", client=fake_client())
    with pytest.raises(ValueError, match="requires a trained router"):
        next(gen)
