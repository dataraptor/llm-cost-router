"""Azure OpenAI → Anthropic-shape client adapter (live-demo backend).

The FrugalRoute engine is architected against the **Anthropic** client surface:
``llm.call`` only ever touches ``client.messages.create(...)`` /
``client.messages.parse(...)`` and reads ``.usage`` / ``.stop_reason`` /
``.content`` off the response. The only API key available for the live demo is
**Azure OpenAI gpt-5.5**, so this module provides a thin adapter that *presents
that Anthropic shape* while calling Azure OpenAI underneath.

Design notes:
- The engine (``llm.call``, ``generate``) is unchanged and unaware of the
  backend — it just receives an object with a ``.messages.create``.
- The translation (:func:`to_anthropic_response`) is a **pure function** so the
  full mapping (finish-reason → stop-reason, OpenAI usage → the four Anthropic
  token buckets, content extraction, refusals) is unit-testable with no network.
- ``parse`` (structured output for the cascade gate) is intentionally **not**
  implemented here yet; it arrives with the gate in split 03.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# OpenAI ``finish_reason`` → Anthropic ``stop_reason``. Anything unmapped passes
# through unchanged (so a future/unknown reason is still visible, not hidden).
_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}


@dataclass
class _Usage:
    """Anthropic-shaped usage: the four token buckets ``llm.cost_usd`` prices."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class _TextBlock:
    """An Anthropic-shaped ``text`` content block."""

    text: str
    type: str = "text"


@dataclass
class _Response:
    """An Anthropic-shaped message response (the subset ``llm.call`` reads)."""

    stop_reason: str
    content: list[_TextBlock]
    usage: _Usage
    stop_details: Any = None  # populated only on refusal, mirroring the SDK
    parsed_output: Any = field(default=None)


def _map_usage(usage: Any) -> _Usage:
    """Map an OpenAI ``usage`` object onto the four Anthropic token buckets.

    OpenAI reports ``prompt_tokens`` *inclusive* of any cached tokens, so we
    split it into the uncached ``input_tokens`` (full price) and
    ``cache_read_input_tokens`` (0.1× price) the cost engine expects. OpenAI has
    no cache-*write* concept, so ``cache_creation_input_tokens`` is always 0.
    """
    if usage is None:
        return _Usage(input_tokens=0, output_tokens=0)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return _Usage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=completion,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cached,
    )


def to_anthropic_response(raw: Any) -> _Response:
    """Translate an OpenAI chat-completion into an Anthropic-shaped response.

    Pure (no I/O). A content-filter finish reason *or* a populated structured
    ``message.refusal`` becomes ``stop_reason="refusal"`` with empty content, so
    the engine's refusal-safe path (``llm.call``) triggers exactly as it does for
    a native Anthropic refusal.
    """
    choice = raw.choices[0]
    message = choice.message
    finish = choice.finish_reason or ""
    refusal = getattr(message, "refusal", None)

    if refusal or finish == "content_filter":
        return _Response(
            stop_reason="refusal",
            content=[],
            usage=_map_usage(raw.usage),
            stop_details={"reason": "refusal"},
        )

    text = message.content or ""
    blocks = [_TextBlock(text=text)] if text else []
    stop_reason = _FINISH_REASON_MAP.get(finish, finish or "end_turn")
    return _Response(stop_reason=stop_reason, content=blocks, usage=_map_usage(raw.usage))


class _AzureMessages:
    """Anthropic-shaped ``client.messages`` backed by OpenAI chat completions."""

    def __init__(self, openai_client: Any, deployment: str) -> None:
        self._client = openai_client
        self._deployment = deployment

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
        **_ignored: Any,
    ) -> _Response:
        """Mirror ``anthropic.messages.create`` over OpenAI chat completions.

        The Anthropic ``system`` string becomes a leading system message; the
        actual deployment is fixed at construction (gpt-5.5), so the engine's
        per-tier ``model`` id only drives pricing, not which model runs. Sends
        ``max_completion_tokens`` (the gpt-5 family rejects ``max_tokens``) and
        no sampling params, matching the engine's constraints.
        """
        oai_messages = [{"role": "system", "content": system}, *messages]
        raw = self._client.chat.completions.create(
            model=self._deployment,
            messages=oai_messages,
            max_completion_tokens=max_tokens,
        )
        return to_anthropic_response(raw)


class AzureAnthropicClient:
    """A drop-in Anthropic-shaped client whose backend is Azure OpenAI gpt-5.5."""

    def __init__(self, openai_client: Any, deployment: str) -> None:
        self.messages = _AzureMessages(openai_client, deployment)


def get_azure_client(deployment: str | None = None) -> AzureAnthropicClient:
    """Construct the Azure-backed, Anthropic-shaped client from env config.

    Reads ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``,
    ``OPENAI_API_VERSION`` (default ``2025-01-01-preview``) and the deployment
    (``CHAT_LLM_MODEL``). ``openai`` is imported lazily so importing this module
    never requires the dependency or any key. Raises a clear, actionable error
    when configuration is missing.
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("OPENAI_API_VERSION", "2025-01-01-preview")
    deployment = deployment or os.environ.get("CHAT_LLM_MODEL")
    if not (endpoint and api_key and deployment):
        raise RuntimeError(
            "Azure OpenAI config missing: set AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_API_KEY, and CHAT_LLM_MODEL (see .env / .env.example)."
        )
    import openai

    raw = openai.AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    return AzureAnthropicClient(raw, deployment)
