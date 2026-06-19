"""No-key tests for the benchmark + examples loaders (split-02 16–18)."""

from __future__ import annotations

import pytest

from frugalroute.benchmarks import load_benchmark
from frugalroute.examples import load_examples

EXPECTED_EXAMPLE_IDS = {
    "gsm8k-1142",
    "gsm8k-882",
    "mmlu-phys-04",
    "mmlu-law-31",
    "gsm8k-1507",
    "refuse-1",
}


def test_load_gsm8k_well_formed() -> None:
    # 16a. gsm8k items: non-empty question + numeric gold.
    items = load_benchmark("gsm8k")
    assert len(items) >= 40
    for it in items:
        assert it.benchmark == "gsm8k"
        assert it.question.strip()
        assert isinstance(it.gold, (int, float))


def test_load_mmlu_well_formed() -> None:
    # 16b. mmlu items: single-letter A–D gold.
    items = load_benchmark("mmlu")
    assert len(items) >= 40
    for it in items:
        assert it.benchmark == "mmlu"
        assert it.question.strip()
        assert it.gold in {"A", "B", "C", "D"}


def test_load_benchmark_cap_is_deterministic() -> None:
    # 17. n caps to exactly n in a stable order.
    five = load_benchmark("gsm8k", n=5)
    assert len(five) == 5
    assert [it.id for it in five] == [it.id for it in load_benchmark("gsm8k")[:5]]


def test_unknown_benchmark_raises() -> None:
    with pytest.raises(ValueError):
        load_benchmark("nonsense")


def test_missing_slice_raises(tmp_path) -> None:
    # A valid benchmark but no slice file in the given dir -> FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        load_benchmark("gsm8k", data_dir=tmp_path)


def test_blank_lines_are_skipped(tmp_path) -> None:
    (tmp_path / "gsm8k.jsonl").write_text(
        '{"id": "a", "question": "q1", "gold": 1}\n\n{"id": "b", "question": "q2", "gold": 2}\n',
        encoding="utf-8",
    )
    items = load_benchmark("gsm8k", data_dir=tmp_path)
    assert [it.id for it in items] == ["a", "b"]


def test_load_examples_matches_mockup() -> None:
    # 18. The 6 demo items; the refuse item has gold null/None.
    examples = load_examples()
    assert len(examples) == 6
    assert {e["id"] for e in examples} == EXPECTED_EXAMPLE_IDS
    by_id = {e["id"]: e for e in examples}
    assert by_id["refuse-1"]["gold"] is None
    assert by_id["gsm8k-1142"]["gold"] == 72
    assert by_id["mmlu-phys-04"]["gold"] == "C"
    for e in examples:
        assert {"id", "benchmark", "label", "query"} <= e.keys()
        assert "gold" in e
