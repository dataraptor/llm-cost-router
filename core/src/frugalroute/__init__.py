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
from frugalroute.classifier import (
    DEFAULT_EMBEDDER,
    LabelRun,
    PredictiveRouter,
    Router,
    generate_labels,
    label_cheapest_correct,
    load_router,
    save_router,
    train,
)
from frugalroute.config import EngineConfig, load_config
from frugalroute.economics import break_even_acceptance, cascade_saves, mean_cascade_cost
from frugalroute.embed import embed, get_embedder
from frugalroute.examples import load_examples
from frugalroute.gate import GateOutcome, gate
from frugalroute.generate import generate
from frugalroute.harness import EvalRun, run_eval
from frugalroute.llm import DEFAULT_TIERS, PRICING, cost_usd
from frugalroute.metrics import (
    accuracy,
    baselines,
    cost_reduction,
    cost_reduction_at_target,
    mean_cost,
    oracle,
    retention,
)
from frugalroute.models import (
    EvalReport,
    FrontierPoint,
    GateVerdict,
    RouteResult,
    route_result_from_dict,
    route_result_to_dict,
)
from frugalroute.obs import concurrency_guard, configure_logging, get_logger, redact
from frugalroute.prompts import PROMPT_VERSION
from frugalroute.router import RouteEvent, route, route_events

__all__ = [
    "DEFAULT_EMBEDDER",
    "DEFAULT_TIERS",
    "PRICING",
    "PROMPT_VERSION",
    "BenchItem",
    "EngineConfig",
    "EvalReport",
    "EvalRun",
    "FrontierPoint",
    "GateOutcome",
    "GateVerdict",
    "LabelRun",
    "PredictiveRouter",
    "RouteEvent",
    "RouteResult",
    "Router",
    "accuracy",
    "baselines",
    "break_even_acceptance",
    "cascade_saves",
    "concurrency_guard",
    "configure_logging",
    "cost_reduction",
    "cost_reduction_at_target",
    "cost_usd",
    "embed",
    "get_logger",
    "extract_gsm8k_answer",
    "extract_mmlu_answer",
    "frozen_split",
    "gate",
    "generate",
    "generate_labels",
    "get_embedder",
    "grade",
    "grade_gsm8k",
    "grade_mmlu",
    "label_cheapest_correct",
    "load_benchmark",
    "load_config",
    "load_examples",
    "load_router",
    "mean_cascade_cost",
    "mean_cost",
    "oracle",
    "redact",
    "retention",
    "route",
    "route_events",
    "route_result_from_dict",
    "route_result_to_dict",
    "run_eval",
    "save_router",
    "train",
]
