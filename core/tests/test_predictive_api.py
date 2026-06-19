"""Live predictive smoke (test 16) — train a tiny router and route once.

Trains a small :class:`~frugalroute.classifier.PredictiveRouter` inline (a handful
of items), then routes one query through it and asserts the result is well-formed
and used exactly one tier. Marked ``@pytest.mark.azure`` (the live backend for this
build, gpt-5.5 via the adapter) with a native ``@pytest.mark.api`` variant; both
auto-skip without their key.

The local embedder needs the optional ``embed`` extra (torch). If it cannot load
(extra not installed, or a torch runtime/DLL issue), the test **skips cleanly**
rather than failing — the no-key gates do not depend on it.
"""

from __future__ import annotations

from typing import Any

import pytest

from frugalroute.benchmarks import frozen_split, load_benchmark
from frugalroute.classifier import (
    DEFAULT_EMBEDDER,
    PredictiveRouter,
    generate_labels,
    train,
)
from frugalroute.embed import embed
from frugalroute.llm import DEFAULT_TIERS
from frugalroute.models import RouteResult
from frugalroute.prompts import PROMPT_VERSION
from frugalroute.router import route


def _load_embedder_or_skip() -> Any:
    from frugalroute.embed import get_embedder

    try:
        return get_embedder()
    except (ImportError, OSError, RuntimeError) as exc:  # torch missing / DLL / extra absent
        pytest.skip(f"local embedder unavailable ({type(exc).__name__}): {exc}")


def _train_tiny_router(client: Any, embedder: Any, n: int = 8) -> PredictiveRouter:
    items = load_benchmark("gsm8k", n=n)
    calibration, _test = frozen_split(items)
    tiers = list(DEFAULT_TIERS)
    label_runs = generate_labels(client, calibration, tiers, "gsm8k")
    embeddings = embed([item.question for item in calibration], embedder=embedder)
    clf = train(embeddings, [run.label for run in label_runs], tiers)
    return PredictiveRouter(
        clf=clf,
        tiers=tiers,
        embedder_name=DEFAULT_EMBEDDER,
        prompt_version=PROMPT_VERSION,
        label_run_ids=sorted({run.run_id for run in label_runs}),
    )


def _assert_well_formed(result: RouteResult) -> None:
    assert isinstance(result, RouteResult)
    assert result.strategy == "predictive"
    assert result.tier_used in DEFAULT_TIERS
    assert result.gate is None  # predictive never gates
    assert result.p_strong is not None and 0.0 <= result.p_strong <= 1.0
    assert isinstance(result.answer, str)
    assert isinstance(result.refused, bool)
    assert result.cost_usd > 0
    assert result.latency_s >= 0


@pytest.mark.azure
def test_predictive_route_live_smoke() -> None:
    from frugalroute.azure_client import get_azure_client

    embedder = _load_embedder_or_skip()
    client = get_azure_client()
    router = _train_tiny_router(client, embedder)
    query = load_benchmark("gsm8k", n=1)[0].question
    result = route(
        query,
        strategy="predictive",
        benchmark="gsm8k",
        client=client,
        router=router,
        embedder=embedder,
    )
    _assert_well_formed(result)


@pytest.mark.api
def test_predictive_route_live_smoke_native() -> None:
    # Native-Anthropic variant: skips cleanly without ANTHROPIC_API_KEY.
    from frugalroute.llm import get_client

    embedder = _load_embedder_or_skip()
    client = get_client()
    router = _train_tiny_router(client, embedder)
    query = load_benchmark("gsm8k", n=1)[0].question
    result = route(
        query,
        strategy="predictive",
        benchmark="gsm8k",
        client=client,
        router=router,
        embedder=embedder,
    )
    _assert_well_formed(result)
