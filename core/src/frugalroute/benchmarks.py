"""Objective graders, tolerant answer extraction, loaders, and the frozen split.

This is the rigor of the eval: **no LLM judge**, just deterministic extraction +
comparison against gold. Two rules from the build spec (§17 / Appendix A) are
load-bearing and tested adversarially:

- **Unparseable → wrong.** If no answer can be extracted, the item grades
  ``False`` (a cheap model that emits garbage *loses*); extraction never raises.
- **Frozen split is hash-based**, not an RNG shuffle, so the same item always
  lands on the same side across machines and runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

# eval/data lives at the repo root (core/src/frugalroute/benchmarks.py -> parents:
# frugalroute[0]/src[1]/core[2]/<repo root>[3]).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = _REPO_ROOT / "eval" / "data"

BENCHMARKS = ("gsm8k", "mmlu")

# Numeric/letter equality tolerance for GSM8K (answers are integers or simple
# decimals; this absorbs 72 vs 72.0 without ever calling 72 == 73 equal).
_GSM8K_ABS_TOL = 1e-6

# A number token: optional sign/$, digits with optional thousands commas, optional
# decimal, optional scientific exponent, optional trailing %.
_NUMBER_RE = re.compile(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
# "answer is" / "answer:" marker (the generation prompts end with this line).
_ANSWER_MARKER_RE = re.compile(r"answer\s*(?:is|:)\s*", re.IGNORECASE)
# Letter immediately after the marker, tolerating a wrapping paren: "(C)", "C".
_MMLU_AFTER_MARKER_RE = re.compile(r"answer\s*(?:is|:)\s*\(?\s*([A-D])\)?", re.IGNORECASE)
# A standalone A–D token not glued to other letters: "(C)", "C.", " C ".
_MMLU_STANDALONE_RE = re.compile(r"(?<![A-Za-z])\(?([A-D])\)?(?![A-Za-z])")


# ----------------------------------------------------------------------------
# Extraction + grading (pure, no I/O, no key)
# ----------------------------------------------------------------------------
def _parse_number(token: str) -> float | None:
    """Normalize one matched number token to a float, or ``None`` if unparseable.

    Strips currency/grouping/percent decoration; ``float`` (which also handles
    bare signs/dots and Unicode digits) does the parsing, and any failure → None.
    """
    cleaned = token.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_gsm8k_answer(text: str) -> float | None:
    """Extract the numeric final answer from a GSM8K completion.

    Prefers the number after the last ``answer is`` / ``answer:`` marker; falls
    back to the last number anywhere in the text. Tolerant of ``$``, thousands
    commas, ``.0``, ``%``, scientific notation, and trailing words. Returns
    ``None`` when no number can be parsed (graded wrong; never raises).
    """
    if not text:
        return None
    markers = list(_ANSWER_MARKER_RE.finditer(text))
    if markers:
        match = _NUMBER_RE.search(text, markers[-1].end())
        if match:
            value = _parse_number(match.group())
            if value is not None:
                return value
    for token in reversed(_NUMBER_RE.findall(text)):
        value = _parse_number(token)
        if value is not None:
            return value
    return None


def grade_gsm8k(text: str, gold: float | str) -> bool:
    """True iff the extracted GSM8K answer numerically equals ``gold``.

    Unparseable prediction *or* gold → ``False`` (never an exception).
    """
    pred = extract_gsm8k_answer(text)
    if pred is None:
        return False
    try:
        gold_value = float(gold)
    except (TypeError, ValueError):
        return False
    return math.isclose(pred, gold_value, rel_tol=1e-9, abs_tol=_GSM8K_ABS_TOL)


def extract_mmlu_answer(text: str) -> str | None:
    """Extract the chosen MMLU letter (A–D), uppercased.

    Prefers the letter after the last ``answer is``/``answer:`` marker; falls
    back to the first standalone A–D token. Returns ``None`` if none is found.
    """
    if not text:
        return None
    markers = list(_MMLU_AFTER_MARKER_RE.finditer(text))
    if markers:
        return markers[-1].group(1).upper()
    fallback = _MMLU_STANDALONE_RE.search(text)
    if fallback:
        return fallback.group(1).upper()
    return None


def grade_mmlu(text: str, gold: str) -> bool:
    """True iff the extracted MMLU letter equals ``gold`` (case-insensitive)."""
    pred = extract_mmlu_answer(text)
    if pred is None or gold is None:
        return False
    return pred == str(gold).strip().upper()


def grade(benchmark: str, text: str, gold: float | str) -> bool:
    """Dispatch to the per-benchmark grader. Raises on an unknown benchmark."""
    if benchmark == "gsm8k":
        return grade_gsm8k(text, gold)
    if benchmark == "mmlu":
        return grade_mmlu(text, str(gold))
    raise ValueError(f"Unknown benchmark {benchmark!r}; expected one of {list(BENCHMARKS)}.")


# ----------------------------------------------------------------------------
# Loading + frozen split
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchItem:
    """One benchmark question with its objective gold answer."""

    id: str
    benchmark: str  # "gsm8k" | "mmlu"
    question: str
    gold: float | str  # numeric (gsm8k) or letter A–D (mmlu)
    subject: str | None = None  # optional source metadata (e.g. MMLU subject)


def load_benchmark(
    benchmark: str, n: int | None = None, data_dir: str | Path | None = None
) -> list[BenchItem]:
    """Load benchmark items from the JSONL slice in deterministic (file) order.

    Optional ``n`` caps the number of items returned (first ``n``, stable order).
    Raises on an unknown benchmark or a missing slice file.
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark!r}; expected one of {list(BENCHMARKS)}.")
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = directory / f"{benchmark}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark slice not found: {path}")

    items: list[BenchItem] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            items.append(
                BenchItem(
                    id=str(record["id"]),
                    benchmark=benchmark,
                    question=record["question"],
                    gold=record["gold"],
                    subject=record.get("subject"),
                )
            )
    if n is not None:
        items = items[:n]
    return items


def _split_key(item_id: str, seed: int) -> str:
    """A stable hash key for ``item_id`` (drives the order-independent split)."""
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def frozen_split(
    items: list[BenchItem], test_frac: float = 0.20, seed: int = 0
) -> tuple[list[BenchItem], list[BenchItem]]:
    """Deterministic ``(calibration, test)`` split, hash-based and leakage-free.

    Items are ordered by a stable hash of their id and the lowest ``test_frac``
    fraction goes to the test side. Because the ordering depends only on item
    ids (not input order or an RNG), the *same* items always land on the *same*
    side across calls, machines, and input reorderings. The two sides partition
    the input exactly (disjoint, union == all).
    """
    ordered = sorted(items, key=lambda item: _split_key(item.id, seed))
    n_test = round(len(items) * test_frac)
    test_ids = {item.id for item in ordered[:n_test]}
    # Preserve the caller's original order within each side.
    test = [item for item in items if item.id in test_ids]
    calibration = [item for item in items if item.id not in test_ids]
    return calibration, test
