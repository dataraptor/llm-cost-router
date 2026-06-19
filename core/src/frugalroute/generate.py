"""Answer generation over the model pool (build-spec §6).

``generate`` is the single way the router/eval produces an answer for a query. It
uses the benchmark's generation system prompt **unchanged across every tier** —
the variable under test is the *model*, not the prompt — and is refusal-safe via
``llm.call``. It adds no sampling/effort/thinking params (those constraints are
enforced in ``llm.call``).
"""

from __future__ import annotations

from typing import Any

from frugalroute.llm import DEFAULT_MAX_TOKENS, CallResult, call
from frugalroute.prompts import GEN_SYSTEM


def generate(
    client: Any,
    model_id: str,
    query: str,
    benchmark: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> CallResult:
    """Produce one plain-text completion for ``query`` on the given tier.

    Uses ``GEN_SYSTEM[benchmark]`` as the system prompt (identical for all
    tiers). Returns a :class:`~frugalroute.llm.CallResult` (text + cost + refused
    + usage + latency). Raises ``ValueError`` for an unknown ``benchmark``.
    """
    try:
        system = GEN_SYSTEM[benchmark]
    except KeyError as exc:
        raise ValueError(
            f"Unknown benchmark {benchmark!r}; expected one of {sorted(GEN_SYSTEM)}."
        ) from exc
    return call(client, model_id, system, query, max_tokens=max_tokens)
