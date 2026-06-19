"""Cascade gate (the cheap structured judge) — tests 1–5 plus malformed-output.

All no-key: the gate is driven by ``fake_client`` returning a canned structured
``parsed_output`` (a ``GateVerdict``) or a refusal. The rigor under test is the
refusal/clamp/malformed safety, not any model behavior.
"""

from __future__ import annotations

import pytest

from frugalroute.gate import gate
from frugalroute.llm import cheap_tier, cost_usd
from frugalroute.models import GateVerdict
from frugalroute.prompts import GATE_SYSTEM, gate_user


def test_gate_sufficient_high_confidence(fake_client, fake_usage) -> None:
    # 1. Sufficient + high confidence is reflected; not refused; cost from usage.
    verdict = GateVerdict(sufficient=True, confidence=0.91, reason="commits to one number")
    client = fake_client(
        parsed_output=verdict, usage=fake_usage(input_tokens=150, output_tokens=50)
    )

    outcome = gate(client, "Q?", "The answer is 72.")

    assert outcome.refused is False
    assert outcome.verdict.sufficient is True
    assert outcome.verdict.confidence == pytest.approx(0.91)
    assert outcome.verdict.reason == "commits to one number"
    assert outcome.cost_usd == pytest.approx(cost_usd(cheap_tier(), 150, 50), abs=1e-12)


def test_gate_insufficient(fake_client) -> None:
    # 2. Insufficient verdict is reflected verbatim.
    verdict = GateVerdict(sufficient=False, confidence=0.6, reason="hedged")
    outcome = gate(fake_client(parsed_output=verdict), "Q?", "maybe 5 or 6")

    assert outcome.refused is False
    assert outcome.verdict.sufficient is False
    assert outcome.verdict.confidence == pytest.approx(0.6)


def test_gate_refusal_is_conservative_and_safe(fake_client) -> None:
    # 3. Gate refusal → refused=True, conservative verdict, content never indexed.
    client = fake_client(stop_reason="refusal", content_raises=True)

    outcome = gate(client, "Q?", "some answer")  # must not raise

    assert outcome.refused is True
    assert outcome.verdict.sufficient is False
    assert outcome.verdict.confidence == 0.0
    assert "escalate" in outcome.verdict.reason


def test_gate_clamps_out_of_range_confidence(fake_client) -> None:
    # 4. Confidence > 1 is clamped to 1.0 before use (schema cannot bound it).
    high = GateVerdict(sufficient=True, confidence=1.4, reason="over-confident")
    assert gate(fake_client(parsed_output=high), "Q?", "a").verdict.confidence == 1.0

    low = GateVerdict(sufficient=True, confidence=-0.3, reason="negative")
    assert gate(fake_client(parsed_output=low), "Q?", "a").verdict.confidence == 0.0


def test_gate_uses_canonical_system_and_user(fake_client) -> None:
    # 5. The gate sends GATE_SYSTEM and gate_user(question, answer) verbatim.
    verdict = GateVerdict(sufficient=True, confidence=0.9, reason="ok")
    client = fake_client(parsed_output=verdict)

    gate(client, "What is 6*7?", "The answer is 42.")

    method, kwargs = client.messages.calls[-1]
    assert method == "parse"  # structured-output path
    assert kwargs["system"] == GATE_SYSTEM
    assert kwargs["messages"] == [
        {"role": "user", "content": gate_user("What is 6*7?", "The answer is 42.")}
    ]
    assert kwargs["model"] == cheap_tier()  # never the strong tier


def test_gate_malformed_output_escalates_without_refusal(fake_client) -> None:
    # R11 (gate level): empty/unparseable structured output → conservative escalate,
    # not flagged as a refusal, no crash.
    outcome = gate(fake_client(parsed_output=None), "Q?", "garbage")

    assert outcome.refused is False
    assert outcome.verdict.sufficient is False
    assert outcome.verdict.confidence == 0.0
    assert "escalate" in outcome.verdict.reason
