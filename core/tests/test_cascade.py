"""Cascade routing — tests 6–12 plus the R11 adversarial path.

All no-key: a ``scripted_client`` returns the cascade's calls in order
(cheap generate → gate parse → strong generate). Token usage is chosen so the
per-call costs are exactly the build-spec §5/§8 fixtures:

  * cheap  generate (Haiku, 150 in / 250 out) = $0.0014
  * gate   judge    (Haiku, 150 in /  50 out) = $0.0004
  * strong generate (Opus,  150 in / 250 out) = $0.0070

so an accepted route costs $0.0018 and an escalated route costs $0.0088 (> the
$0.0070 always-Opus cost — the honest losing case).
"""

from __future__ import annotations

import pytest

from frugalroute.llm import cheap_tier, cost_usd, strong_tier
from frugalroute.models import GateVerdict
from frugalroute.prompts import PROMPT_VERSION
from frugalroute.router import route

CHEAP = cheap_tier()
STRONG = strong_tier()

C_CHEAP = cost_usd(CHEAP, 150, 250)  # 0.0014
C_GATE = cost_usd(CHEAP, 150, 50)  # 0.0004
C_STRONG = cost_usd(STRONG, 150, 250)  # 0.0070


def _cheap_resp(fake_response, fake_usage, *, text="cheap answer", refused=False):
    if refused:
        return fake_response(
            stop_reason="refusal", usage=fake_usage(input_tokens=150, output_tokens=250)
        )
    return fake_response(text=text, usage=fake_usage(input_tokens=150, output_tokens=250))


def _gate_resp(fake_response, fake_usage, *, sufficient, confidence, parsed=True):
    parsed_output = (
        GateVerdict(sufficient=sufficient, confidence=confidence, reason="judged")
        if parsed
        else None
    )
    return fake_response(
        parsed_output=parsed_output, usage=fake_usage(input_tokens=150, output_tokens=50)
    )


def _strong_resp(fake_response, fake_usage, *, text="strong answer", refused=False):
    if refused:
        return fake_response(
            stop_reason="refusal", usage=fake_usage(input_tokens=150, output_tokens=250)
        )
    return fake_response(text=text, usage=fake_usage(input_tokens=150, output_tokens=250))


def test_accepted_path(scripted_client, fake_response, fake_usage) -> None:
    # 6. Cheap answers, gate sufficient @0.91 ≥ τ=0.8 → accept on the cheap tier.
    client = scripted_client(
        [
            _cheap_resp(fake_response, fake_usage),
            _gate_resp(fake_response, fake_usage, sufficient=True, confidence=0.91),
        ]
    )
    result = route("Q?", strategy="cascade", tau=0.8, client=client)

    assert result.escalated is False
    assert result.tier_used == CHEAP
    assert result.answer == "cheap answer"
    assert result.correct is None
    assert result.gate is not None and result.gate.sufficient is True
    assert result.refused is False
    assert result.prompt_version == PROMPT_VERSION
    assert result.cost_usd == pytest.approx(C_CHEAP + C_GATE, abs=1e-12)
    assert result.cost_usd == pytest.approx(0.0018, abs=1e-9)


def test_escalated_path(scripted_client, fake_response, fake_usage) -> None:
    # 7. Gate insufficient @0.62 < τ → escalate to the strong tier; full additive cost.
    client = scripted_client(
        [
            _cheap_resp(fake_response, fake_usage),
            _gate_resp(fake_response, fake_usage, sufficient=False, confidence=0.62),
            _strong_resp(fake_response, fake_usage),
        ]
    )
    result = route("Q?", strategy="cascade", tau=0.8, client=client)

    assert result.escalated is True
    assert result.tier_used == STRONG
    assert result.answer == "strong answer"
    assert result.gate is not None and result.gate.sufficient is False
    assert result.cost_usd == pytest.approx(C_CHEAP + C_GATE + C_STRONG, abs=1e-12)
    assert result.cost_usd == pytest.approx(0.0088, abs=1e-9)


def test_threshold_boundary_accepts_at_tau(scripted_client, fake_response, fake_usage) -> None:
    # 8a. confidence == τ with sufficient=True → accepted (the documented '≥ τ' rule).
    client = scripted_client(
        [
            _cheap_resp(fake_response, fake_usage),
            _gate_resp(fake_response, fake_usage, sufficient=True, confidence=0.8),
        ]
    )
    result = route("Q?", strategy="cascade", tau=0.8, client=client)
    assert result.escalated is False
    assert result.tier_used == CHEAP


def test_threshold_boundary_escalates_just_below_tau(
    scripted_client, fake_response, fake_usage
) -> None:
    # 8b. confidence just below τ → escalated.
    client = scripted_client(
        [
            _cheap_resp(fake_response, fake_usage),
            _gate_resp(fake_response, fake_usage, sufficient=True, confidence=0.79),
            _strong_resp(fake_response, fake_usage),
        ]
    )
    result = route("Q?", strategy="cascade", tau=0.8, client=client)
    assert result.escalated is True
    assert result.tier_used == STRONG


def test_insufficient_high_confidence_still_escalates(
    scripted_client, fake_response, fake_usage
) -> None:
    # 9. sufficient=False gates first, even at confidence 0.99 → escalate.
    client = scripted_client(
        [
            _cheap_resp(fake_response, fake_usage),
            _gate_resp(fake_response, fake_usage, sufficient=False, confidence=0.99),
            _strong_resp(fake_response, fake_usage),
        ]
    )
    result = route("Q?", strategy="cascade", tau=0.8, client=client)
    assert result.escalated is True
    assert result.tier_used == STRONG


def test_cheap_refusal_skips_gate_and_escalates(scripted_client, fake_response, fake_usage) -> None:
    # 10. Cheap refuses → gate skipped, escalate to strong, refused=True.
    client = scripted_client(
        [
            _cheap_resp(fake_response, fake_usage, refused=True),
            _strong_resp(fake_response, fake_usage),
        ]
    )
    result = route("Q?", strategy="cascade", tau=0.8, client=client)

    assert result.escalated is True
    assert result.tier_used == STRONG
    assert result.answer == "strong answer"
    assert result.refused is True
    assert result.gate is None  # gate was skipped
    # No parse call was made — only two create calls (cheap, strong).
    assert [m for m, _ in client.calls] == ["create", "create"]
    # No gate cost: cheap + strong only.
    assert result.cost_usd == pytest.approx(C_CHEAP + C_STRONG, abs=1e-12)


def test_gate_refusal_escalates_with_refused_flag(
    scripted_client, fake_response, fake_usage
) -> None:
    # Refusal matrix (gate row): the gate itself refuses → conservative escalate to
    # strong, refused=True, the (conservative) verdict is attached, cost is additive.
    client = scripted_client(
        [
            _cheap_resp(fake_response, fake_usage),
            fake_response(
                stop_reason="refusal", usage=fake_usage(input_tokens=150, output_tokens=10)
            ),
            _strong_resp(fake_response, fake_usage),
        ]
    )
    result = route("Q?", strategy="cascade", tau=0.8, client=client)

    assert result.escalated is True
    assert result.tier_used == STRONG
    assert result.answer == "strong answer"
    assert result.refused is True
    assert result.gate is not None and result.gate.sufficient is False
    # cheap + gate (billed even on refusal) + strong.
    assert result.cost_usd == pytest.approx(
        C_CHEAP + cost_usd(CHEAP, 150, 10) + C_STRONG, abs=1e-12
    )


def test_strong_refusal_on_escalation_surfaced_honestly(
    scripted_client, fake_response, fake_usage
) -> None:
    # 11. Strong refuses during escalation → refused=True, answer surfaced ("" refusal
    # text), no crash, no silent downgrade.
    client = scripted_client(
        [
            _cheap_resp(fake_response, fake_usage),
            _gate_resp(fake_response, fake_usage, sufficient=False, confidence=0.4),
            _strong_resp(fake_response, fake_usage, refused=True),
        ]
    )
    result = route("Q?", strategy="cascade", tau=0.8, client=client)

    assert result.escalated is True
    assert result.tier_used == STRONG
    assert result.answer == ""  # refusal text discarded by llm.call
    assert result.refused is True


def test_escalated_cost_exceeds_always_opus(scripted_client, fake_response, fake_usage) -> None:
    # 12. The escalated cascade is *more* expensive than always-Opus (§8 made visible).
    client = scripted_client(
        [
            _cheap_resp(fake_response, fake_usage),
            _gate_resp(fake_response, fake_usage, sufficient=False, confidence=0.5),
            _strong_resp(fake_response, fake_usage),
        ]
    )
    result = route("Q?", strategy="cascade", tau=0.8, client=client)
    assert result.cost_usd > cost_usd(STRONG, 150, 250)


def test_adversarial_gate_junk_and_strong_refusal(
    scripted_client, fake_response, fake_usage
) -> None:
    # R11. Gate returns no parseable verdict AND the strong tier refuses → the route
    # still returns a coherent RouteResult and never crashes.
    client = scripted_client(
        [
            _cheap_resp(fake_response, fake_usage),
            _gate_resp(fake_response, fake_usage, sufficient=False, confidence=0.0, parsed=False),
            _strong_resp(fake_response, fake_usage, refused=True),
        ]
    )
    result = route("Q?", strategy="cascade", tau=0.8, client=client)

    assert result.escalated is True
    assert result.tier_used == STRONG
    assert result.answer == ""
    assert result.refused is True
    # Conservative verdict from the junk gate output is still attached.
    assert result.gate is not None and result.gate.sufficient is False
    assert result.cost_usd == pytest.approx(C_CHEAP + C_GATE + C_STRONG, abs=1e-12)


def test_unknown_strategy_raises(scripted_client) -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        route("Q?", strategy="bogus", client=scripted_client([]))


def test_predictive_strategy_not_implemented(scripted_client) -> None:
    with pytest.raises(NotImplementedError, match="split 04"):
        route("Q?", strategy="predictive", client=scripted_client([]))
