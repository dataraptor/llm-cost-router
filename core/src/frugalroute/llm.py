"""Client, tier config, cache-aware cost engine, and a refusal-safe call wrapper.

This module is the bottom of the engine: pinned pricing keyed by model ID, the
ordered tier list (cheap → strong), the canonical cache-aware per-call cost
formula, and a single ``call()`` that is refusal-safe and sends no forbidden
sampling parameters. The Anthropic client is **injected** into ``call()`` so the
whole module is unit-testable with a fake client and no network.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    import anthropic

# --- Pinned pricing, keyed by model ID (USD per MTok). The gradient is the
#     artifact, so these are pinned exactly (00-INSTRUCTIONS §3 / build-spec §10).
PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    # --- Live-demo backend (split 02). The only API key available is Azure
    #     OpenAI gpt-5.5, not Anthropic, so the engine's Anthropic-shaped client
    #     interface is driven by an adapter (``azure_client.py``) that calls
    #     gpt-5.5. This entry lets ``cost_usd`` price that path. The numbers are
    #     APPROXIMATE public GPT-5-tier list prices, NOT part of the pinned
    #     Anthropic gradient the eval headline is built on (00-INSTRUCTIONS §3).
    "gpt-5.5": {"input": 1.25, "output": 10.0},
}

# --- Config-driven, ordered tier list (cheap → strong). 2-tier default (D2);
#     adding Sonnet is config, not a rewrite — pass a different ``tiers`` list.
DEFAULT_TIERS: list[str] = ["claude-haiku-4-5", "claude-opus-4-8"]

# Cache-cost multipliers applied to the input price (00-INSTRUCTIONS §3).
CACHE_WRITE_MULT = 1.25  # 5-min TTL (the default)
CACHE_READ_MULT = 0.10

# Sane default output cap. Generation tunes this per benchmark in later splits.
DEFAULT_MAX_TOKENS = 1024


def cheap_tier(tiers: Sequence[str] = DEFAULT_TIERS) -> str:
    """The cheapest tier (first in the ordered list)."""
    return tiers[0]


def strong_tier(tiers: Sequence[str] = DEFAULT_TIERS) -> str:
    """The strongest tier (last in the ordered list)."""
    return tiers[-1]


def cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Cache-aware 3-bucket per-call cost in USD (canonical formula, §3).

    Pure function, no I/O. Raises ``KeyError`` on an unknown ``model_id`` (never
    silently returns a zero cost — the cost gradient is the artifact).
    """
    try:
        price = PRICING[model_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model_id {model_id!r}: no pricing entry. Known: {sorted(PRICING)}"
        ) from exc
    in_price = price["input"]
    out_price = price["output"]
    return (
        (input_tokens / 1e6) * in_price
        + (cache_write_tokens / 1e6) * in_price * CACHE_WRITE_MULT
        + (cache_read_tokens / 1e6) * in_price * CACHE_READ_MULT
        + (output_tokens / 1e6) * out_price
    )


def get_client() -> anthropic.Anthropic:
    """Construct the Anthropic client.

    Reads ``ANTHROPIC_API_KEY`` from the environment and raises a clear,
    actionable error if it is unset. ``anthropic`` is imported lazily so that
    importing ``frugalroute`` never requires the key or the dependency at
    import time.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it (see core/.env.example) "
            "before making live Anthropic API calls."
        )
    import anthropic

    return anthropic.Anthropic(api_key=api_key)


@dataclass
class CallResult:
    """Outcome of one completion, with cost computed from ``response.usage``.

    ``text`` is "" on a refusal (any partial content is discarded). ``parsed``
    holds the structured output when ``call(..., parse_model=...)`` was used.
    """

    text: str
    refused: bool  # stop_reason == "refusal"
    stop_reason: str
    cost_usd: float  # computed from usage via cost_usd(...)
    latency_s: float
    usage: dict[str, int]  # raw token buckets, for debugging/eval
    parsed: BaseModel | None = None


# The four token buckets read from ``response.usage`` (build-spec §10 / §3).
_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _usage_to_dict(usage: Any) -> dict[str, int]:
    """Read the four token buckets off a usage object, defaulting missing/None to 0."""
    result: dict[str, int] = {}
    for field_name in _USAGE_FIELDS:
        value = getattr(usage, field_name, 0)
        result[field_name] = int(value) if value is not None else 0
    return result


def _extract_text(response: Any) -> str:
    """Concatenate the text of every ``text`` content block in a response."""
    parts: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def call(
    client: Any,
    model_id: str,
    system: str,
    user: str,
    *,
    parse_model: type[BaseModel] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> CallResult:
    """Refusal-safe single completion against an injected ``client``.

    Sends only ``model`` / ``system`` / ``messages`` / ``max_tokens`` (plus the
    structured-output config when ``parse_model`` is given) — **no** temperature,
    top_p, top_k, seed, thinking, or effort (Anthropic API constraints, §3).

    Checks ``stop_reason`` **before** reading any content: on a refusal it returns
    ``refused=True`` with ``text=""`` and never indexes into ``response.content``.
    Cost is always computed from ``response.usage`` (a mid-stream refusal is still
    billed for what it streamed). Raises ``KeyError`` if ``model_id`` is unpriced.
    """
    messages = [{"role": "user", "content": user}]

    start = time.monotonic()
    if parse_model is not None:
        response = client.messages.parse(
            model=model_id,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_format=parse_model,
        )
    else:
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
    latency_s = time.monotonic() - start

    usage = _usage_to_dict(response.usage)
    cost = cost_usd(
        model_id,
        usage["input_tokens"],
        usage["output_tokens"],
        cache_write_tokens=usage["cache_creation_input_tokens"],
        cache_read_tokens=usage["cache_read_input_tokens"],
    )

    stop_reason = str(response.stop_reason or "")
    if stop_reason == "refusal":
        # Refusal: do NOT read response.content. Discard any partial content.
        return CallResult(
            text="",
            refused=True,
            stop_reason=stop_reason,
            cost_usd=cost,
            latency_s=latency_s,
            usage=usage,
            parsed=None,
        )

    parsed: BaseModel | None = None
    if parse_model is not None:
        parsed = getattr(response, "parsed_output", None)

    return CallResult(
        text=_extract_text(response),
        refused=False,
        stop_reason=stop_reason,
        cost_usd=cost,
        latency_s=latency_s,
        usage=usage,
        parsed=parsed,
    )
