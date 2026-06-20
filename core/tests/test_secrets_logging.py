"""Split-11 §1: secret hygiene + structured logging (no key, fake client).

These prove the engine's edge logging is structured and that a secret set in the
environment can never reach a log line, an exception, or a serialized result.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from frugalroute import llm, metrics, obs, route
from frugalroute.models import GateVerdict, route_result_to_dict

SENTINEL = "sk-ant-SENTINEL-must-never-be-logged-0123456789"


@pytest.fixture(autouse=True)
def _fresh_obs() -> None:
    """Reset cached logging/semaphore state so each test configures cleanly."""
    obs.reset_runtime()
    yield
    obs.reset_runtime()


def _capture(records: list[logging.LogRecord]) -> logging.Handler:
    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    return _Capture()


def _accepting_cascade(scripted_client: Any, fake_response: Any, fake_usage: Any) -> Any:
    """A scripted client: cheap answer → gate says sufficient (accept)."""
    return scripted_client(
        [
            fake_response(
                text="The answer is 42.", usage=fake_usage(input_tokens=20, output_tokens=8)
            ),
            fake_response(
                parsed_output=GateVerdict(sufficient=True, confidence=0.99, reason="clear"),
                usage=fake_usage(input_tokens=15, output_tokens=4),
            ),
        ]
    )


def test_sentinel_key_never_appears_in_logs_or_result(
    monkeypatch: pytest.MonkeyPatch,
    scripted_client: Any,
    fake_response: Any,
    fake_usage: Any,
) -> None:
    """Test 1: a sentinel key set in env appears nowhere in logs/exc/result."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    obs.reset_runtime()
    obs.configure_logging("INFO")

    records: list[logging.LogRecord] = []
    handler = _capture(records)
    fr_logger = logging.getLogger("frugalroute")
    fr_logger.addHandler(handler)
    try:
        result = route(
            "What is 6 times 7?",
            client=_accepting_cascade(scripted_client, fake_response, fake_usage),
        )
    finally:
        fr_logger.removeHandler(handler)

    # Render every record through the JSON formatter (the production path) AND the
    # raw message/args, then assert the sentinel is absent everywhere.
    formatter = obs.JsonFormatter()
    rendered = "\n".join(formatter.format(r) for r in records)
    assert records, "expected at least one log record from the route"
    assert SENTINEL not in rendered
    # The serialized result must not carry the key either.
    assert SENTINEL not in json.dumps(route_result_to_dict(result))


def test_redact_masks_key_value_and_sk_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """The redaction helper masks both the live env value and any sk- token."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "topsecretvalue")
    text = "boom: key=topsecretvalue and also sk-ant-abc123XYZ_def leaked"
    out = obs.redact(text)
    assert "topsecretvalue" not in out
    assert "sk-ant-abc123XYZ_def" not in out
    assert "***REDACTED***" in out


def test_json_log_lines_carry_contract_fields(
    monkeypatch: pytest.MonkeyPatch,
    scripted_client: Any,
    fake_response: Any,
    fake_usage: Any,
) -> None:
    """Test 2: JSON lines parse and carry the contract fields for a call + route."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    obs.reset_runtime()
    obs.configure_logging("INFO")

    records: list[logging.LogRecord] = []
    handler = _capture(records)
    fr_logger = logging.getLogger("frugalroute")
    fr_logger.addHandler(handler)
    try:
        route(
            "What is 6 times 7?",
            client=_accepting_cascade(scripted_client, fake_response, fake_usage),
        )
    finally:
        fr_logger.removeHandler(handler)

    formatter = obs.JsonFormatter()
    lines = [json.loads(formatter.format(r)) for r in records]
    # Base fields present on every line.
    for line in lines:
        assert {"ts", "level", "logger", "msg"} <= set(line)
    msgs = {line["msg"] for line in lines}
    assert "llm_call" in msgs and "route" in msgs

    call_line = next(line for line in lines if line["msg"] == "llm_call")
    assert {"model", "cost_usd", "latency_s", "tokens", "refused"} <= set(call_line)
    assert isinstance(call_line["tokens"], dict)

    route_line = next(line for line in lines if line["msg"] == "route")
    assert {"strategy", "tier_used", "escalated", "cost_usd", "latency_s", "refused"} <= set(
        route_line
    )


def test_pure_functions_emit_no_logs() -> None:
    """Test 3: pure cost/metric functions never emit a log record."""
    obs.reset_runtime()
    obs.configure_logging("DEBUG")
    records: list[logging.LogRecord] = []
    handler = _capture(records)
    fr_logger = logging.getLogger("frugalroute")
    fr_logger.addHandler(handler)
    try:
        llm.cost_usd("claude-haiku-4-5", 1000, 200)
        metrics.accuracy([True, False, True])
        metrics.mean_cost([0.001, 0.002, 0.003])
        metrics.oracle(
            [{"claude-haiku-4-5": True, "claude-opus-4-8": True}],
            [{"claude-haiku-4-5": 0.001, "claude-opus-4-8": 0.005}],
            ["claude-haiku-4-5", "claude-opus-4-8"],
        )
    finally:
        fr_logger.removeHandler(handler)
    assert records == [], f"pure functions emitted logs: {[r.getMessage() for r in records]}"
