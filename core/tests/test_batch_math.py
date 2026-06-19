"""The Batch API lever (build-spec §10) — tests 17-18.

A fake batch runner stands in for the Anthropic Batches API: the eval generations
cost **half** under batch (the 50% discount), and results returned **out of order**
are re-associated by ``custom_id`` — never by position. Batch is never on the live
router path (that is the harness's concern, not exercised here).
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from frugalroute.harness import (
    BATCH_DISCOUNT,
    GenRequest,
    RawBatchItem,
    run_generations,
)
from frugalroute.llm import cost_usd

MODEL = "claude-haiku-4-5"


def _requests(n: int) -> list[GenRequest]:
    return [GenRequest(f"item{i}::{MODEL}", MODEL, f"q{i}", "gsm8k") for i in range(n)]


def test_batch_halves_the_generation_cost(fake_client, fake_usage) -> None:
    # 17. Same token usage → batch total == 0.5 × the synchronous total.
    requests = _requests(5)
    client = fake_client(
        text="The answer is 7.", usage=fake_usage(input_tokens=100, output_tokens=50)
    )
    sync = run_generations(requests, client=client)

    def runner(reqs: Sequence[GenRequest]) -> list[RawBatchItem]:
        return [
            RawBatchItem(
                req.custom_id, {"input_tokens": 100, "output_tokens": 50}, "The answer is 7."
            )
            for req in reqs
        ]

    batch = run_generations(requests, batch=True, batch_runner=runner)

    sync_total = sum(result.cost_usd for result in sync.values())
    batch_total = sum(result.cost_usd for result in batch.values())
    assert sync_total > 0
    assert batch_total == pytest.approx(BATCH_DISCOUNT * sync_total)
    assert batch_total == pytest.approx(0.5 * sync_total)


def test_batch_results_keyed_by_custom_id_out_of_order() -> None:
    # 18. Distinct usage per id, runner returns them REVERSED → still mapped right.
    requests = _requests(4)
    usages = {
        f"item{i}::{MODEL}": {"input_tokens": 100, "output_tokens": 10 * (i + 1)} for i in range(4)
    }

    def runner(reqs: Sequence[GenRequest]) -> list[RawBatchItem]:
        items = [
            RawBatchItem(req.custom_id, usages[req.custom_id], f"ans-{req.custom_id}")
            for req in reqs
        ]
        return list(reversed(items))  # deliberately out of order

    out = run_generations(requests, batch=True, batch_runner=runner)
    for i in range(4):
        custom_id = f"item{i}::{MODEL}"
        assert out[custom_id].usage["output_tokens"] == 10 * (i + 1)
        assert out[custom_id].text == f"ans-{custom_id}"
        expected = BATCH_DISCOUNT * cost_usd(MODEL, 100, 10 * (i + 1))
        assert out[custom_id].cost_usd == pytest.approx(expected)


def test_batch_refusal_yields_empty_text() -> None:
    requests = _requests(1)

    def runner(reqs: Sequence[GenRequest]) -> list[RawBatchItem]:
        return [
            RawBatchItem(
                req.custom_id, {"input_tokens": 100, "output_tokens": 0}, "partial", refused=True
            )
            for req in reqs
        ]

    out = run_generations(requests, batch=True, batch_runner=runner)
    result = out[f"item0::{MODEL}"]
    assert result.refused is True
    assert result.text == ""


def test_batch_requires_a_runner() -> None:
    with pytest.raises(ValueError, match="batch_runner"):
        run_generations(_requests(1), batch=True)


def test_batch_missing_result_raises() -> None:
    requests = _requests(2)

    def runner(reqs: Sequence[GenRequest]) -> list[RawBatchItem]:
        # Drop one result → the harness must notice (never silently miss an item).
        return [RawBatchItem(reqs[0].custom_id, {"input_tokens": 100, "output_tokens": 10})]

    with pytest.raises(KeyError, match="missing results"):
        run_generations(requests, batch=True, batch_runner=runner)


def test_batch_unknown_custom_id_raises() -> None:
    requests = _requests(1)

    def runner(reqs: Sequence[GenRequest]) -> list[RawBatchItem]:
        return [RawBatchItem("not-a-real-id", {"input_tokens": 100, "output_tokens": 10})]

    with pytest.raises(KeyError, match="unknown custom_id"):
        run_generations(requests, batch=True, batch_runner=runner)
