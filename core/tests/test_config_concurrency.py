"""Split-11 §2: env config precedence/validation + the concurrency cap (no key)."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from frugalroute import llm, obs
from frugalroute.config import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_REQUEST_TIMEOUT_S,
    load_config,
)


@pytest.fixture(autouse=True)
def _fresh_obs() -> None:
    obs.reset_runtime()
    yield
    obs.reset_runtime()


def test_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "FRUGALROUTE_MAX_CONCURRENCY",
        "FRUGALROUTE_REQUEST_TIMEOUT_S",
        "FRUGALROUTE_LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config()
    assert cfg.max_concurrency == DEFAULT_MAX_CONCURRENCY
    assert cfg.request_timeout_s == DEFAULT_REQUEST_TIMEOUT_S
    assert cfg.log_level == "INFO"


def test_env_override_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 4a: env override > default."""
    monkeypatch.setenv("FRUGALROUTE_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("FRUGALROUTE_REQUEST_TIMEOUT_S", "12.5")
    monkeypatch.setenv("FRUGALROUTE_LOG_LEVEL", "debug")
    cfg = load_config()
    assert cfg.max_concurrency == 3
    assert cfg.request_timeout_s == 12.5
    assert cfg.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("FRUGALROUTE_MAX_CONCURRENCY", "-1"),
        ("FRUGALROUTE_MAX_CONCURRENCY", "0"),
        ("FRUGALROUTE_MAX_CONCURRENCY", "notanint"),
        ("FRUGALROUTE_REQUEST_TIMEOUT_S", "0"),
        ("FRUGALROUTE_REQUEST_TIMEOUT_S", "-5"),
        ("FRUGALROUTE_LOG_LEVEL", "LOUD"),
    ],
)
def test_invalid_config_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, var: str, value: str
) -> None:
    """Test 4b: an invalid value fails loudly at load, not a silent bad state."""
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError) as excinfo:
        load_config()
    assert var in str(excinfo.value)


class _Tracker:
    """Records peak in-flight concurrency across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def __enter__(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self.current -= 1


class _SlowMessages:
    """A ``client.messages`` stub that holds the call open to expose concurrency."""

    def __init__(self, tracker: _Tracker, response: Any, hold_s: float) -> None:
        self._tracker = tracker
        self._response = response
        self._hold_s = hold_s

    def create(self, **kwargs: object) -> Any:
        with self._tracker:
            time.sleep(self._hold_s)
        return self._response


def test_concurrency_guard_caps_simultaneous_calls(
    monkeypatch: pytest.MonkeyPatch, fake_response: Any
) -> None:
    """Test 5: peak in-flight Anthropic calls never exceeds the configured max."""
    monkeypatch.setenv("FRUGALROUTE_MAX_CONCURRENCY", "2")
    obs.reset_runtime()

    tracker = _Tracker()

    class _Client:
        messages = _SlowMessages(tracker, fake_response(text="ok"), hold_s=0.05)

    client = _Client()

    def worker() -> None:
        llm.call(client, "claude-haiku-4-5", "sys", "user")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tracker.peak <= 2, f"peak in-flight {tracker.peak} exceeded the cap of 2"
    assert tracker.peak >= 2, "expected the cap to actually be reached (otherwise the test is moot)"
