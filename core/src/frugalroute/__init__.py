"""FrugalRoute engine — the framework-free core package.

Public surface re-exported here so callers can ``from frugalroute import ...``.
Importing this package never requires ``ANTHROPIC_API_KEY`` (the client is built
lazily in :func:`frugalroute.llm.get_client`).
"""

from __future__ import annotations

from frugalroute.benchmarks import (
    BenchItem,
    extract_gsm8k_answer,
    extract_mmlu_answer,
    frozen_split,
    grade,
    grade_gsm8k,
    grade_mmlu,
    load_benchmark,
)
from frugalroute.economics import break_even_acceptance, cascade_saves, mean_cascade_cost
from frugalroute.examples import load_examples
from frugalroute.gate import GateOutcome, gate
from frugalroute.generate import generate
from frugalroute.llm import DEFAULT_TIERS, PRICING, cost_usd
from frugalroute.models import EvalReport, FrontierPoint, GateVerdict, RouteResult
from frugalroute.prompts import PROMPT_VERSION
from frugalroute.router import route

__all__ = [
    "DEFAULT_TIERS",
    "PRICING",
    "PROMPT_VERSION",
    "BenchItem",
    "EvalReport",
    "FrontierPoint",
    "GateOutcome",
    "GateVerdict",
    "RouteResult",
    "break_even_acceptance",
    "cascade_saves",
    "cost_usd",
    "extract_gsm8k_answer",
    "extract_mmlu_answer",
    "frozen_split",
    "gate",
    "generate",
    "grade",
    "grade_gsm8k",
    "grade_mmlu",
    "load_benchmark",
    "load_examples",
    "mean_cascade_cost",
    "route",
]
