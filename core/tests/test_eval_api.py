"""Live eval smoke (test 19 / R13) — runs ``eval --quick`` end-to-end.

Marked ``@pytest.mark.azure`` (the gpt-5.5 adapter, this build's live backend) with
a native ``@pytest.mark.api`` variant; both auto-skip without their key. The
predictive trend additionally needs the local embedder and skips cleanly if it is
unavailable (e.g. the torch DLL is broken on this box) — the no-key gates do not
depend on any of this.
"""

from __future__ import annotations

from typing import Any

import pytest

from frugalroute.harness import format_report, read_run, run_eval, write_run


def _load_embedder_or_skip() -> Any:
    from frugalroute.embed import get_embedder

    try:
        return get_embedder()
    except (ImportError, OSError, RuntimeError) as exc:
        pytest.skip(f"local embedder unavailable ({type(exc).__name__}): {exc}")


def _assert_cascade_smoke(run: Any, tmp_path: Any) -> None:
    report = run.reports["cascade"]
    assert report.points
    strong_cost = report.baselines["always_strong"]["cost"]
    # Cascade can save vs always-strong somewhere on the sweep (cost-reduction > 0).
    assert any(point.cost_usd_per_query < strong_cost for point in report.points)
    text = format_report(run)
    assert "Frontier (cascade)" in text
    assert "frozen split" in text
    assert "free quality" not in text  # honest headline, never "free quality"
    path = write_run(run, tmp_path / "run.jsonl")
    assert "cascade" in read_run(path)["reports"]


@pytest.mark.azure
def test_eval_quick_cascade_live(tmp_path) -> None:
    from frugalroute.azure_client import get_azure_client

    run = run_eval(
        "gsm8k",
        strategy="cascade",
        repeats=1,
        taus=[0.5, 0.8, 1.0],
        client=get_azure_client(),
        n=16,
    )
    _assert_cascade_smoke(run, tmp_path)


@pytest.mark.api
def test_eval_quick_cascade_live_native(tmp_path) -> None:
    from frugalroute.llm import get_client

    run = run_eval(
        "gsm8k", strategy="cascade", repeats=1, taus=[0.5, 0.8, 1.0], client=get_client(), n=16
    )
    _assert_cascade_smoke(run, tmp_path)


@pytest.mark.azure
def test_eval_predictive_cost_le_cascade_trend() -> None:
    # R13 trend: the predictive router (no gate, no double spend) reaches a lower
    # cost floor than the cascade. Needs the embedder; skips cleanly otherwise.
    from frugalroute.azure_client import get_azure_client

    embedder = _load_embedder_or_skip()
    run = run_eval(
        "gsm8k", strategy="both", repeats=1, client=get_azure_client(), embedder=embedder, n=16
    )
    cascade = run.reports["cascade"]
    predictive = run.reports["predictive"]
    cheapest_cascade = min(point.cost_usd_per_query for point in cascade.points)
    cheapest_predictive = min(point.cost_usd_per_query for point in predictive.points)
    assert cheapest_predictive <= cheapest_cascade + 1e-9
