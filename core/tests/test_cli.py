"""CLI ``route`` subcommand — tests 17–18 plus output/error coverage.

No-key: a ``scripted_client`` is injected through the ``main(..., client=...)``
test seam, so the CLI is exercised end-to-end without a network or key.
"""

from __future__ import annotations

import json

import pytest

from frugalroute.cli import main
from frugalroute.models import GateVerdict
from frugalroute.router import CASCADE


def _accepted(scripted_client, fake_response, fake_usage):
    return scripted_client(
        [
            fake_response(
                text="The answer is 72.", usage=fake_usage(input_tokens=150, output_tokens=250)
            ),
            fake_response(
                parsed_output=GateVerdict(sufficient=True, confidence=0.9, reason="committed"),
                usage=fake_usage(input_tokens=150, output_tokens=50),
            ),
        ]
    )


def _escalated(scripted_client, fake_response, fake_usage):
    return scripted_client(
        [
            fake_response(text="hmm", usage=fake_usage(input_tokens=150, output_tokens=250)),
            fake_response(
                parsed_output=GateVerdict(sufficient=False, confidence=0.3, reason="hedged"),
                usage=fake_usage(input_tokens=150, output_tokens=50),
            ),
            fake_response(
                text="The answer is 72.", usage=fake_usage(input_tokens=150, output_tokens=250)
            ),
        ]
    )


def test_route_example_json_round_trips(scripted_client, fake_response, fake_usage, capsys) -> None:
    # 17. --json emits valid JSON with the full RouteResult key set.
    client = _accepted(scripted_client, fake_response, fake_usage)
    code = main(
        ["route", "--strategy", "cascade", "--example", "gsm8k-1142", "--json"], client=client
    )
    assert code == 0

    payload = json.loads(capsys.readouterr().out)
    expected_keys = {
        "query",
        "strategy",
        "tier_used",
        "escalated",
        "answer",
        "correct",
        "gate",
        "p_strong",
        "refused",
        "cost_usd",
        "latency_s",
        "prompt_version",
    }
    assert set(payload) == expected_keys
    assert payload["strategy"] == CASCADE
    assert payload["escalated"] is False
    assert payload["gate"] == {"sufficient": True, "confidence": 0.9, "reason": "committed"}
    # The example's own query was used.
    assert payload["query"].startswith("Natalia sold clips")


def test_route_human_output_accepted(scripted_client, fake_response, fake_usage, capsys) -> None:
    client = _accepted(scripted_client, fake_response, fake_usage)
    code = main(["route", "--example", "gsm8k-1142"], client=client)
    out = capsys.readouterr().out
    assert code == 0
    assert "Tier used: claude-haiku-4-5" in out
    assert "Escalated: no" in out
    assert "Breakdown: = Haiku + gate" in out
    assert "sufficient=True" in out


def test_route_human_output_cheap_refusal(
    scripted_client, fake_response, fake_usage, capsys
) -> None:
    # Cheap refuses → gate skipped, escalated, refused: human output reflects all three.
    client = scripted_client(
        [
            fake_response(
                stop_reason="refusal", usage=fake_usage(input_tokens=150, output_tokens=10)
            ),
            fake_response(
                text="The answer is 72.", usage=fake_usage(input_tokens=150, output_tokens=250)
            ),
        ]
    )
    code = main(["route", "--query", "x?", "--benchmark", "gsm8k"], client=client)
    out = capsys.readouterr().out
    assert code == 0
    assert "Gate:      (skipped)" in out
    assert "Refused:   yes" in out
    assert "Breakdown: = Haiku + Opus" in out


def test_route_human_output_escalated_breakdown(
    scripted_client, fake_response, fake_usage, capsys
) -> None:
    client = _escalated(scripted_client, fake_response, fake_usage)
    code = main(["route", "--query", "1+1?", "--benchmark", "gsm8k"], client=client)
    out = capsys.readouterr().out
    assert code == 0
    assert "Escalated: yes" in out
    assert "Breakdown: = Haiku + gate + Opus" in out


def test_predictive_cli_errors_gracefully(
    scripted_client, fake_response, fake_usage, capsys
) -> None:
    # Predictive strategy now needs a trained router (split 04): without --model it
    # errors with a clean message + non-zero exit, not a traceback. (Full predictive
    # CLI behaviour is covered in test_cli_predictive.py.)
    client = scripted_client([])
    code = main(["route", "--strategy", "predictive", "--example", "gsm8k-1142"], client=client)
    err = capsys.readouterr().err
    assert code == 2
    assert "error:" in err
    assert "requires --model" in err


def test_unknown_example_errors_gracefully(scripted_client, capsys) -> None:
    code = main(["route", "--example", "does-not-exist"], client=scripted_client([]))
    err = capsys.readouterr().err
    assert code == 2
    assert "Unknown example id" in err


def test_missing_key_surfaces_clear_message(monkeypatch, capsys) -> None:
    # No client injected and no ANTHROPIC_API_KEY → clear message, exit 1, no trace.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = main(["route", "--query", "hi", "--benchmark", "gsm8k"])  # client=None → get_client()
    err = capsys.readouterr().err
    assert code == 1
    assert "ANTHROPIC_API_KEY is not set" in err


def test_requires_example_or_query(capsys) -> None:
    # --example / --query are a required mutually-exclusive group.
    with pytest.raises(SystemExit):
        main(["route", "--strategy", "cascade"])


def test_no_subcommand_errors(capsys) -> None:
    with pytest.raises(SystemExit):
        main([])
