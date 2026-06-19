"""Tests for generate(): no-key (mocked) 19–20, plus the live @api smoke 21.

The live smoke runs against the **Azure OpenAI gpt-5.5 backend** through the
Anthropic-shaped adapter (the live-demo backend for this build), so it is marked
``@pytest.mark.azure`` and auto-skips without ``AZURE_OPENAI_API_KEY``.
"""

from __future__ import annotations

import pytest

from frugalroute.benchmarks import grade
from frugalroute.generate import generate
from frugalroute.llm import DEFAULT_TIERS, cost_usd
from frugalroute.prompts import GEN_SYSTEM


def test_generate_uses_benchmark_prompt_and_returns_cost(fake_client, fake_usage) -> None:
    # 19. generate sends GEN_SYSTEM[benchmark] and returns text + computed cost.
    client = fake_client(
        text="Work... The answer is 72.",
        usage=fake_usage(input_tokens=200, output_tokens=40),
    )
    result = generate(client, "claude-haiku-4-5", "Q?", "gsm8k")

    assert result.text == "Work... The answer is 72."
    assert client.last_kwargs["system"] == GEN_SYSTEM["gsm8k"]
    assert client.last_kwargs["messages"] == [{"role": "user", "content": "Q?"}]
    assert result.cost_usd == pytest.approx(cost_usd("claude-haiku-4-5", 200, 40), abs=1e-12)
    assert result.refused is False


def test_generate_same_prompt_across_tiers(fake_client) -> None:
    # The prompt does not vary by tier (the model is the variable under test).
    systems = []
    for tier in DEFAULT_TIERS:
        client = fake_client(text="The answer is C")
        generate(client, tier, "Q?", "mmlu")
        systems.append(client.last_kwargs["system"])
    assert len(set(systems)) == 1
    assert systems[0] == GEN_SYSTEM["mmlu"]


def test_generate_unknown_benchmark_raises(fake_client) -> None:
    # 20. Unknown benchmark -> clear error.
    client = fake_client()
    with pytest.raises(ValueError, match="Unknown benchmark"):
        generate(client, "claude-haiku-4-5", "Q?", "trivia")


# --- 21. Live smoke against gpt-5.5 (adapter); auto-skipped without the key. ---
_GSM8K_SAMPLE = "If a box has 12 apples and 5 are removed, how many apples remain?"
_MMLU_SAMPLE = "What is the capital of France?\n(A) Berlin  (B) Paris  (C) Rome  (D) Madrid"
_SMOKE_MAX_TOKENS = 4096  # generous headroom for a reasoning model


@pytest.mark.azure
@pytest.mark.parametrize("tier", DEFAULT_TIERS)
def test_generate_live_each_tier(tier: str) -> None:
    from frugalroute.azure_client import get_azure_client

    client = get_azure_client()

    gsm = generate(client, tier, _GSM8K_SAMPLE, "gsm8k", max_tokens=_SMOKE_MAX_TOKENS)
    assert gsm.refused is False
    assert gsm.text.strip()
    assert isinstance(grade("gsm8k", gsm.text, 7), bool)
    assert gsm.cost_usd > 0

    mc = generate(client, tier, _MMLU_SAMPLE, "mmlu", max_tokens=_SMOKE_MAX_TOKENS)
    assert mc.refused is False
    assert mc.text.strip()
    assert isinstance(grade("mmlu", mc.text, "B"), bool)
