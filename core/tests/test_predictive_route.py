"""Predictive routing path — tests 10-13 plus the R11 adversarial cases.

All no-key: a fake embedder (anything with ``.encode``) replaces the local model,
an injected ``PredictiveRouter`` with a stub classifier fixes the prediction, and
the conftest ``fake_client`` stands in for the model call. The single discipline
under test: predictive routing makes **exactly one** generate call — no cheap
call, no gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from frugalroute.classifier import PredictiveRouter
from frugalroute.llm import cost_usd
from frugalroute.prompts import PROMPT_VERSION
from frugalroute.router import route

HAIKU = "claude-haiku-4-5"
OPUS = "claude-opus-4-8"
TIERS = [HAIKU, OPUS]
DIM = 4


class _StubClf:
    def __init__(self, classes: list[str], proba_row: list[float]) -> None:
        self.classes_ = np.array(classes)
        self._row = np.asarray(proba_row, dtype=np.float64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(self._row, (len(features), 1))


class _FakeEmbedder:
    """Returns a fixed vector for every query (no torch)."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = np.asarray(vector, dtype=np.float32)

    def encode(self, queries, **_kwargs):
        return np.tile(self._vector, (len(list(queries)), 1))


def _router(proba_row: list[float], tiers=TIERS, classes=None) -> PredictiveRouter:
    return PredictiveRouter(
        clf=_StubClf(classes if classes is not None else list(tiers), proba_row),
        tiers=list(tiers),
        embedder_name="fake",
        prompt_version="v1",
        label_run_ids=["labels-gsm8k-abc123"],
    )


def _embedder(vector=None) -> _FakeEmbedder:
    return _FakeEmbedder(vector if vector is not None else [0.1, 0.2, 0.3, 0.4])


def test_predictive_makes_exactly_one_call_and_no_gate(fake_client, fake_usage) -> None:
    # 10. One generate call only — no cheap call, no gate.
    client = fake_client(
        text="The answer is 42.", usage=fake_usage(input_tokens=150, output_tokens=250)
    )
    result = route(
        "Q?",
        strategy="predictive",
        benchmark="gsm8k",
        client=client,
        router=_router([0.1, 0.9]),
        embedder=_embedder(),
    )

    assert [method for method, _ in client.messages.calls] == ["create"]
    assert result.gate is None
    assert result.strategy == "predictive"
    assert result.prompt_version == PROMPT_VERSION


def test_predictive_routes_to_strong(fake_client, fake_usage) -> None:
    # 11a. Stub predicts strong (p_strong 0.9 > 0.5) → Opus, escalated.
    client = fake_client(usage=fake_usage(input_tokens=150, output_tokens=250))
    result = route(
        "Q?", strategy="predictive", client=client, router=_router([0.1, 0.9]), embedder=_embedder()
    )
    assert result.tier_used == OPUS
    assert result.escalated is True
    assert result.p_strong == pytest.approx(0.9)
    assert client.last_kwargs["model"] == OPUS


def test_predictive_routes_to_cheap(fake_client, fake_usage) -> None:
    # 11b. Stub predicts cheap (p_strong 0.1) → Haiku, not escalated.
    client = fake_client(usage=fake_usage(input_tokens=150, output_tokens=250))
    result = route(
        "Q?", strategy="predictive", client=client, router=_router([0.9, 0.1]), embedder=_embedder()
    )
    assert result.tier_used == HAIKU
    assert result.escalated is False
    assert result.p_strong == pytest.approx(0.1)
    assert client.last_kwargs["model"] == HAIKU


def test_predictive_cost_is_single_call_only(fake_client, fake_usage) -> None:
    # 12. Cost == that one call's cost; no cheap/gate overhead added.
    client = fake_client(usage=fake_usage(input_tokens=150, output_tokens=250))
    result = route(
        "Q?", strategy="predictive", client=client, router=_router([0.1, 0.9]), embedder=_embedder()
    )
    assert result.cost_usd == pytest.approx(cost_usd(OPUS, 150, 250), abs=1e-12)


def test_predictive_refusal_surfaced(fake_client, fake_usage) -> None:
    # 13. The single call refuses → refused=True, answer surfaced (""), no crash.
    client = fake_client(
        stop_reason="refusal", usage=fake_usage(input_tokens=150, output_tokens=10)
    )
    result = route(
        "Q?", strategy="predictive", client=client, router=_router([0.1, 0.9]), embedder=_embedder()
    )
    assert result.refused is True
    assert result.answer == ""
    assert result.tier_used == OPUS


def test_missing_router_raises_before_key(fake_client) -> None:
    # Fails fast on a missing router (no key demanded, no call made).
    with pytest.raises(ValueError, match="requires a trained router"):
        route("Q?", strategy="predictive", client=fake_client(), embedder=_embedder())


# --- R11 adversarial: fail loudly, never silently mis-route. ---


def test_unknown_tier_raises_and_makes_no_call(fake_client) -> None:
    # R11a. A (multiclass) classifier emitting a tier not in `tiers` → clear error,
    # and the model call is never made.
    client = fake_client()
    three = [HAIKU, "claude-sonnet-4-6", OPUS]
    bogus = _router([1.0], tiers=three, classes=["totally-bogus-tier"])
    with pytest.raises(ValueError, match="not in the router's tiers"):
        route("Q?", strategy="predictive", client=client, router=bogus, embedder=_embedder())
    assert client.messages.calls == []


def test_degenerate_embedding_raises_and_makes_no_call(fake_client) -> None:
    # R11b. NaN embedding → clear error before any prediction/call.
    client = fake_client()
    nan_embedder = _embedder([float("nan"), 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Degenerate embedding"):
        route(
            "Q?",
            strategy="predictive",
            client=client,
            router=_router([0.1, 0.9]),
            embedder=nan_embedder,
        )
    assert client.messages.calls == []
