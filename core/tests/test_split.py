"""No-key tests for the deterministic frozen split (split-02 13–15) + manifest."""

from __future__ import annotations

import json

from frugalroute.benchmarks import (
    DEFAULT_DATA_DIR,
    BenchItem,
    frozen_split,
    load_benchmark,
)


def _items(n: int) -> list[BenchItem]:
    return [
        BenchItem(id=f"item-{i:03d}", benchmark="gsm8k", question="q", gold=i) for i in range(n)
    ]


def _ids(items: list[BenchItem]) -> set[str]:
    return {it.id for it in items}


def test_determinism_across_calls_and_reorder() -> None:
    # 13. Same partition across calls and after reordering the input.
    items = _items(100)
    calib_a, test_a = frozen_split(items)
    calib_b, test_b = frozen_split(list(reversed(items)))
    assert _ids(test_a) == _ids(test_b)
    assert _ids(calib_a) == _ids(calib_b)


def test_proportion_is_about_twenty_percent() -> None:
    # 14. Test side ~20% (exact via round() for hash-ordered selection).
    items = _items(100)
    _, test = frozen_split(items)
    assert abs(len(test) - 20) <= 1


def test_no_leakage_partition() -> None:
    # 15. calibration ∩ test == ∅ and union == all.
    items = _items(84)
    calib, test = frozen_split(items)
    assert _ids(calib).isdisjoint(_ids(test))
    assert _ids(calib) | _ids(test) == _ids(items)
    assert len(calib) + len(test) == len(items)


def test_test_frac_parameter_respected() -> None:
    items = _items(100)
    _, test = frozen_split(items, test_frac=0.5)
    assert len(test) == 50


def test_manifest_matches_derivation() -> None:
    # The committed frozen_split.json must reproduce from the code (no drift).
    manifest = json.loads((DEFAULT_DATA_DIR / "frozen_split.json").read_text(encoding="utf-8"))
    for benchmark, recorded in manifest["benchmarks"].items():
        items = load_benchmark(benchmark)
        _, test = frozen_split(items, test_frac=manifest["test_frac"], seed=manifest["seed"])
        assert sorted(it.id for it in test) == recorded["test_ids"]
        assert len(test) == recorded["n_test"]
