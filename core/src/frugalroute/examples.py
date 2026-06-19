"""Loader for the bundled demo examples (the 6 worked items from the mockup).

``examples.json`` ships as package data so the UI and CLI can offer the same
curated single-query examples the mockup uses. Each item is
``{id, benchmark, label, query, gold}`` where ``gold`` is the objective answer
(numeric for GSM8K, a letter for MMLU) or ``null`` for the refusal edge item.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_EXAMPLES_PATH = Path(__file__).resolve().parent / "data" / "examples.json"


def load_examples() -> list[dict[str, Any]]:
    """Return the bundled demo examples as a list of dicts (file order)."""
    data: list[dict[str, Any]] = json.loads(_EXAMPLES_PATH.read_text(encoding="utf-8"))
    return data
