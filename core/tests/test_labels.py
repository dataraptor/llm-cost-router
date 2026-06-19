"""Label generation (Appendix B) — tests 1-5 plus edges. All no-key.

``label_cheapest_correct`` is the pure core of Strategy B's training labels: pick
the cheapest tier that was correct, or the cheapest tier if none was. A no-key
``generate_labels`` test drives the whole labeling loop with a scripted client.
"""

from __future__ import annotations

import pytest

from frugalroute.benchmarks import BenchItem
from frugalroute.classifier import generate_labels, label_cheapest_correct

HAIKU = "claude-haiku-4-5"
SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-8"
TWO_TIER = [HAIKU, OPUS]
THREE_TIER = [HAIKU, SONNET, OPUS]


def test_cheapest_correct_when_both_correct() -> None:
    # 1. Both tiers correct → the cheapest (Haiku).
    assert label_cheapest_correct({HAIKU: True, OPUS: True}, TWO_TIER) == HAIKU


def test_only_strong_correct() -> None:
    # 2. Only the strong tier correct → Opus.
    assert label_cheapest_correct({HAIKU: False, OPUS: True}, TWO_TIER) == OPUS


def test_none_correct_falls_back_to_cheapest() -> None:
    # 3. No tier correct → the cheapest (escalating wouldn't have helped, §17).
    assert label_cheapest_correct({HAIKU: False, OPUS: False}, TWO_TIER) == HAIKU


def test_three_tier_cheapest_correct_is_middle() -> None:
    # 4. 3-tier: cheapest correct is Sonnet (Haiku wrong, Sonnet+Opus right).
    grades = {HAIKU: False, SONNET: True, OPUS: True}
    assert label_cheapest_correct(grades, THREE_TIER) == SONNET


@pytest.mark.parametrize(
    "grades",
    [
        {HAIKU: True, SONNET: True, OPUS: True},
        {HAIKU: False, SONNET: False, OPUS: False},
        {HAIKU: False, SONNET: True, OPUS: False},
        {},  # nothing graded → still a valid tier (the cheapest)
        {OPUS: True},  # missing keys treated as not-correct
    ],
)
def test_label_is_always_a_member_of_tiers(grades: dict[str, bool]) -> None:
    # 5. Whatever the grades, the label is always one of the configured tiers.
    assert label_cheapest_correct(grades, THREE_TIER) in THREE_TIER


def test_empty_tiers_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        label_cheapest_correct({}, [])


def test_generate_labels_loop_and_provenance(scripted_client, fake_response, fake_usage) -> None:
    # No-key drive of the full labeling loop: 2 items x 2 tiers, scripted answers.
    # item1: both tiers answer 72 (correct) -> label Haiku (cheapest correct).
    # item2: Haiku answers 999 (wrong), Opus answers 10 (correct) -> label Opus.
    items = [
        BenchItem(id="g1", benchmark="gsm8k", question="q1?", gold=72),
        BenchItem(id="g2", benchmark="gsm8k", question="q2?", gold=10),
    ]
    usage = fake_usage(input_tokens=10, output_tokens=10)
    client = scripted_client(
        [
            fake_response(text="The answer is 72.", usage=usage),  # item1, Haiku
            fake_response(text="The answer is 72.", usage=usage),  # item1, Opus
            fake_response(text="The answer is 999.", usage=usage),  # item2, Haiku (wrong)
            fake_response(text="The answer is 10.", usage=usage),  # item2, Opus
        ]
    )

    runs = generate_labels(client, items, TWO_TIER, "gsm8k")

    assert [r.label for r in runs] == [HAIKU, OPUS]
    assert runs[0].per_tier_correct == {HAIKU: True, OPUS: True}
    assert runs[1].per_tier_correct == {HAIKU: False, OPUS: True}
    assert runs[0].item_id == "g1" and runs[1].item_id == "g2"
    # All items in one labeling pass share a single, stable run id.
    assert runs[0].run_id == runs[1].run_id
    assert runs[0].run_id.startswith("labels-gsm8k-")


def test_generate_labels_run_id_is_deterministic(
    scripted_client, fake_response, fake_usage
) -> None:
    items = [BenchItem(id="g1", benchmark="gsm8k", question="q1?", gold=1)]
    usage = fake_usage(input_tokens=10, output_tokens=10)

    def _client():
        return scripted_client(
            [fake_response(text="The answer is 1.", usage=usage) for _ in TWO_TIER]
        )

    first = generate_labels(_client(), items, TWO_TIER, "gsm8k")[0].run_id
    second = generate_labels(_client(), items, TWO_TIER, "gsm8k")[0].run_id
    assert first == second


def test_generate_labels_refusal_counts_as_incorrect(
    scripted_client, fake_response, fake_usage
) -> None:
    # A tier that refuses produced no usable answer → not correct for labeling.
    items = [BenchItem(id="g1", benchmark="gsm8k", question="q1?", gold=72)]
    usage = fake_usage(input_tokens=10, output_tokens=10)
    client = scripted_client(
        [
            fake_response(stop_reason="refusal", usage=usage),  # Haiku refuses
            fake_response(text="The answer is 72.", usage=usage),  # Opus correct
        ]
    )
    runs = generate_labels(client, items, TWO_TIER, "gsm8k")
    assert runs[0].per_tier_correct == {HAIKU: False, OPUS: True}
    assert runs[0].label == OPUS
