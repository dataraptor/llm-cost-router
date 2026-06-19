"""No-key tests for the refusal-safe call wrapper (split-01 cases 10–13 + R10).

All driven by the injected ``fake_client`` fixture — no network.
"""

from __future__ import annotations

import pytest

from frugalroute.llm import call, cost_usd
from frugalroute.models import GateVerdict

FORBIDDEN_PARAMS = ("temperature", "top_p", "top_k", "seed", "thinking", "effort")


def test_normal_completion(fake_client, fake_usage) -> None:
    client = fake_client(
        stop_reason="end_turn",
        text="The answer is 7.",
        usage=fake_usage(input_tokens=150, output_tokens=250),
    )
    result = call(client, "claude-opus-4-8", "sys", "user")

    assert result.text == "The answer is 7."
    assert result.refused is False
    assert result.stop_reason == "end_turn"
    assert result.cost_usd == pytest.approx(cost_usd("claude-opus-4-8", 150, 250), abs=1e-12)
    assert result.cost_usd == pytest.approx(0.0070, abs=1e-9)
    assert result.latency_s >= 0
    assert result.usage["input_tokens"] == 150
    assert result.usage["output_tokens"] == 250
    assert result.parsed is None


def test_refusal_is_safe(fake_client, fake_usage) -> None:
    # content_raises proves call() never indexes content on a refusal.
    client = fake_client(
        stop_reason="refusal",
        content_raises=True,
        usage=fake_usage(input_tokens=120, output_tokens=0),
    )
    result = call(client, "claude-opus-4-8", "sys", "user")

    assert result.refused is True
    assert result.text == ""
    assert result.stop_reason == "refusal"
    # Mid-stream refusals are still billed for what they streamed.
    assert result.cost_usd == pytest.approx(cost_usd("claude-opus-4-8", 120, 0), abs=1e-12)


def test_sends_no_forbidden_params(fake_client) -> None:
    client = fake_client()
    call(client, "claude-haiku-4-5", "sys", "user")

    kwargs = client.last_kwargs
    for param in FORBIDDEN_PARAMS:
        assert param not in kwargs, f"call() must not send {param!r}"
    # Only the allowed request fields are sent.
    assert set(kwargs) == {"model", "max_tokens", "system", "messages"}


def test_structured_path_surfaces_parsed(fake_client, fake_usage) -> None:
    verdict = GateVerdict(sufficient=True, confidence=0.91, reason="commits to one number")
    client = fake_client(
        stop_reason="end_turn",
        parsed_output=verdict,
        usage=fake_usage(input_tokens=250, output_tokens=30),
    )
    result = call(client, "claude-haiku-4-5", "sys", "user", parse_model=GateVerdict)

    assert result.parsed is verdict
    assert result.refused is False
    # The structured path uses messages.parse and passes the output_format.
    method, kwargs = client.messages.calls[-1]
    assert method == "parse"
    assert kwargs["output_format"] is GateVerdict
    for param in FORBIDDEN_PARAMS:
        assert param not in kwargs


def test_adversarial_unknown_model_raises(fake_client, fake_usage) -> None:
    # R10 (a): an unpriced model must raise, not silently produce a 0 cost.
    client = fake_client(usage=fake_usage(input_tokens=100, output_tokens=100))
    with pytest.raises(KeyError):
        call(client, "gpt-5.5", "sys", "user")


def test_adversarial_refusal_content_never_indexed(fake_client) -> None:
    # R10 (b): a refusal whose content accessor raises must not crash call().
    client = fake_client(stop_reason="refusal", content_raises=True)
    result = call(client, "claude-haiku-4-5", "sys", "user")
    assert result.refused is True
    assert result.text == ""
