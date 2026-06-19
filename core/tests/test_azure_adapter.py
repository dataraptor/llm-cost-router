"""No-key tests for the Azure OpenAI -> Anthropic-shape adapter.

The whole translation is a pure function, so the finish-reason mapping, usage
bucket mapping, content extraction, and refusal handling are tested with small
fakes and no network. Also proves the adapter is a drop-in for ``llm.call``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frugalroute.azure_client import (
    AzureAnthropicClient,
    get_azure_client,
    to_anthropic_parsed_response,
    to_anthropic_response,
)
from frugalroute.gate import gate
from frugalroute.llm import call, cost_usd
from frugalroute.models import GateVerdict


def _oai_response(
    content, finish_reason="stop", *, refusal=None, prompt=0, completion=0, cached=None
):
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=details
    )
    message = SimpleNamespace(content=content, refusal=refusal)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeOpenAI:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))


def test_normal_completion_maps_to_end_turn() -> None:
    raw = _oai_response("The answer is 7.", "stop", prompt=100, completion=20)
    resp = to_anthropic_response(raw)
    assert resp.stop_reason == "end_turn"
    assert [b.text for b in resp.content] == ["The answer is 7."]
    assert resp.usage.input_tokens == 100
    assert resp.usage.output_tokens == 20
    assert resp.usage.cache_creation_input_tokens == 0
    assert resp.usage.cache_read_input_tokens == 0


def test_length_finish_maps_to_max_tokens() -> None:
    resp = to_anthropic_response(_oai_response("partial", "length"))
    assert resp.stop_reason == "max_tokens"


def test_content_filter_maps_to_refusal() -> None:
    resp = to_anthropic_response(_oai_response("", "content_filter"))
    assert resp.stop_reason == "refusal"
    assert resp.content == []
    assert resp.stop_details is not None


def test_structured_refusal_field_maps_to_refusal() -> None:
    resp = to_anthropic_response(_oai_response(None, "stop", refusal="I can't help with that."))
    assert resp.stop_reason == "refusal"
    assert resp.content == []


def test_cached_tokens_split_into_buckets() -> None:
    # OpenAI prompt_tokens includes cached; we split into uncached + cache_read.
    raw = _oai_response("ok", "stop", prompt=1000, completion=10, cached=400)
    resp = to_anthropic_response(raw)
    assert resp.usage.input_tokens == 600
    assert resp.usage.cache_read_input_tokens == 400


def test_missing_usage_is_zeroed() -> None:
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hi", refusal=None), finish_reason="stop"
            )
        ],
        usage=None,
    )
    resp = to_anthropic_response(raw)
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0


def test_create_sends_max_completion_tokens_and_system_message() -> None:
    fake = _FakeOpenAI(_oai_response("hi", "stop", prompt=10, completion=2))
    client = AzureAnthropicClient(fake, "gpt-5.5")
    client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system="sys",
        messages=[{"role": "user", "content": "u"}],
    )
    sent = fake.chat.completions.calls[-1]
    assert sent["model"] == "gpt-5.5"  # deployment, not the engine's tier id
    assert sent["max_completion_tokens"] == 512
    assert "max_tokens" not in sent  # gpt-5 family rejects it
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["messages"][1] == {"role": "user", "content": "u"}


def test_adapter_is_drop_in_for_llm_call() -> None:
    # The engine's call() works unchanged against the adapter, pricing gpt-5.5.
    fake = _FakeOpenAI(_oai_response("The answer is 4.", "stop", prompt=300, completion=25))
    client = AzureAnthropicClient(fake, "gpt-5.5")
    result = call(client, "gpt-5.5", "sys", "user")
    assert result.text == "The answer is 4."
    assert result.refused is False
    assert result.cost_usd == pytest.approx(cost_usd("gpt-5.5", 300, 25), abs=1e-12)


def test_adapter_refusal_through_llm_call() -> None:
    fake = _FakeOpenAI(_oai_response("", "content_filter", prompt=50, completion=0))
    client = AzureAnthropicClient(fake, "gpt-5.5")
    result = call(client, "gpt-5.5", "sys", "user")
    assert result.refused is True
    assert result.text == ""


# --- Structured-output (gate) path: parse mapping + dispatch (split 03). ---


def _oai_parsed_response(parsed, finish_reason="stop", *, refusal=None, prompt=0, completion=0):
    usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=None
    )
    message = SimpleNamespace(content=None, refusal=refusal, parsed=parsed)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeParseCompletions:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeParseOpenAI:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=_FakeParseCompletions(response))


def test_completions_parse_falls_back_to_beta() -> None:
    # Older SDKs expose parse only at beta.chat.completions.parse — the dispatch
    # must fall back when chat.completions has no parse attribute.
    from frugalroute.azure_client import _completions_parse

    sentinel = object()

    class _BetaCompletions:
        def parse(self, **kwargs):
            return sentinel

    # chat.completions has NO parse; beta.chat.completions.parse exists.
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace()),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=_BetaCompletions())),
    )
    assert _completions_parse(client, model="m", messages=[]) is sentinel


def test_parsed_completion_exposes_parsed_output() -> None:
    verdict = GateVerdict(sufficient=True, confidence=0.9, reason="ok")
    resp = to_anthropic_parsed_response(
        _oai_parsed_response(verdict, "stop", prompt=80, completion=12)
    )
    assert resp.stop_reason == "end_turn"
    assert resp.parsed_output is verdict
    assert resp.usage.input_tokens == 80


def test_parsed_refusal_maps_to_refusal() -> None:
    resp = to_anthropic_parsed_response(
        _oai_parsed_response(None, "stop", refusal="I can't help with that.")
    )
    assert resp.stop_reason == "refusal"
    assert resp.parsed_output is None
    assert resp.content == []


def test_parse_method_sends_response_format_and_returns_parsed() -> None:
    verdict = GateVerdict(sufficient=False, confidence=0.2, reason="hedged")
    fake = _FakeParseOpenAI(_oai_parsed_response(verdict, "stop", prompt=60, completion=8))
    client = AzureAnthropicClient(fake, "gpt-5.5")
    resp = client.messages.parse(
        model="claude-haiku-4-5",
        max_tokens=256,
        system="sys",
        messages=[{"role": "user", "content": "u"}],
        output_format=GateVerdict,
    )
    sent = fake.chat.completions.calls[-1]
    assert sent["model"] == "gpt-5.5"
    assert sent["max_completion_tokens"] == 256
    assert "max_tokens" not in sent
    assert sent["response_format"] is GateVerdict
    assert resp.parsed_output is verdict


def test_gate_runs_through_adapter() -> None:
    # The gate works unchanged against the adapter, pricing gpt-5.5 (no network).
    verdict = GateVerdict(sufficient=True, confidence=0.88, reason="committed")
    fake = _FakeParseOpenAI(_oai_parsed_response(verdict, "stop", prompt=120, completion=10))
    client = AzureAnthropicClient(fake, "gpt-5.5")
    outcome = gate(client, "Q?", "The answer is 7.", gate_model="gpt-5.5")
    assert outcome.refused is False
    assert outcome.verdict.sufficient is True
    assert outcome.verdict.confidence == pytest.approx(0.88)
    assert outcome.cost_usd == pytest.approx(cost_usd("gpt-5.5", 120, 10), abs=1e-12)


def test_get_azure_client_missing_config_raises(monkeypatch) -> None:
    for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "CHAT_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="Azure OpenAI config missing"):
        get_azure_client()


def test_get_azure_client_builds_adapter(monkeypatch) -> None:
    # Construction is offline (no network); proves the env-wiring + adapter shape.
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("CHAT_LLM_MODEL", "gpt-5.5")
    client = get_azure_client()
    assert isinstance(client, AzureAnthropicClient)
    assert hasattr(client.messages, "create")
