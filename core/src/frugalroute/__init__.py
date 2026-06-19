"""FrugalRoute engine — the framework-free core package.

Public surface re-exported here so callers can ``from frugalroute import ...``.
Importing this package never requires ``ANTHROPIC_API_KEY`` (the client is built
lazily in :func:`frugalroute.llm.get_client`).
"""

from __future__ import annotations

from frugalroute.llm import DEFAULT_TIERS, PRICING, cost_usd
from frugalroute.models import EvalReport, FrontierPoint, GateVerdict, RouteResult
from frugalroute.prompts import PROMPT_VERSION

__all__ = [
    "DEFAULT_TIERS",
    "PRICING",
    "PROMPT_VERSION",
    "EvalReport",
    "FrontierPoint",
    "GateVerdict",
    "RouteResult",
    "cost_usd",
]
