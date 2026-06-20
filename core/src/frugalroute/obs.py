"""Observability: structured JSON logging, secret redaction, concurrency guard.

This is the engine's **edge** instrumentation (split-11). It is imported by the
call/route paths only — the pure functions (cost math, graders, metrics, oracle)
stay completely log-free, per build-spec §12.

Three concerns, each independently testable:

- **Redaction** (:func:`redact`): scrubs the API key (and any ``sk-``-style token)
  from any string so a secret can never reach a log line, an exception, or a
  serialized result. The JSON formatter applies it as a final safety net.
- **Structured logging** (:class:`JsonFormatter`, :func:`configure_logging`,
  :func:`get_logger`): one JSON object per log line carrying the contract fields
  (``ts, level, logger, msg`` + any of ``request_id, model, cost_usd, latency_s,
  tokens, strategy, tier_used, escalated, refused``).
- **Concurrency guard** (:func:`concurrency_guard`): a process-wide bounded
  semaphore (sized by ``FRUGALROUTE_MAX_CONCURRENCY``) that any path fanning out
  Anthropic calls acquires, so simultaneous calls never exceed the configured cap.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from frugalroute.config import load_config

_REDACTED = "***REDACTED***"

# Any ``sk-...`` / ``sk-ant-...`` style token (Anthropic + OpenAI shapes). Kept
# deliberately broad: a run of key-ish characters after the ``sk-`` prefix.
_SK_TOKEN = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")

# Env vars whose *values* must never appear in any log/error/result.
_SECRET_ENV_VARS = ("ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY")

# The contract fields a log record may carry beyond the always-present base set.
_EXTRA_FIELDS = (
    "request_id",
    "model",
    "cost_usd",
    "latency_s",
    "tokens",
    "strategy",
    "tier_used",
    "escalated",
    "refused",
    "method",
    "path",
    "status",
)

_LOGGER_NAME = "frugalroute"


def redact(text: str) -> str:
    """Return ``text`` with any known secret value or ``sk-`` token masked.

    Scrubs (1) the live values of the secret env vars (so a key that is set in
    the environment is removed even if it has no recognizable shape) and (2) any
    ``sk-``-style token by pattern (so a key embedded in an upstream error string
    is removed even when it is not the configured one).
    """
    if not text:
        return text
    for var in _SECRET_ENV_VARS:
        value = os.environ.get(var)
        if value and value in text:
            text = text.replace(value, _REDACTED)
    return _SK_TOKEN.sub(_REDACTED, text)


class JsonFormatter(logging.Formatter):
    """Render each record as a single redacted JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for field in _EXTRA_FIELDS:
            if field in record.__dict__:
                payload[field] = record.__dict__[field]
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Final safety net: redact the whole serialized line so a secret can never
        # slip through, regardless of which field (or message) it landed in.
        return redact(json.dumps(payload, default=str))


_configure_lock = threading.Lock()
_configured = False


def configure_logging(level: str | None = None) -> None:
    """Attach the JSON formatter to the ``frugalroute`` logger (idempotent).

    Uses ``FRUGALROUTE_LOG_LEVEL`` (via :func:`load_config`) unless ``level`` is
    given. Adds a single stream handler and disables propagation so log lines are
    not double-emitted by the root logger.
    """
    global _configured
    resolved = (level or load_config().log_level).upper()
    with _configure_lock:
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(resolved)
        logger.propagate = False
        if not any(getattr(h, "_frugalroute", False) for h in logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(JsonFormatter())
            stream_handler._frugalroute = True  # type: ignore[attr-defined]
            logger.addHandler(stream_handler)
        else:
            for existing in logger.handlers:
                if getattr(existing, "_frugalroute", False):
                    existing.setFormatter(JsonFormatter())
        _configured = True


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a ``frugalroute``-namespaced logger (configuring once if needed)."""
    if not _configured:
        configure_logging()
    if name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


# ----------------------------------------------------------------------------
# Concurrency guard — a process-wide bounded semaphore sized from config.
# ----------------------------------------------------------------------------
_guard_lock = threading.Lock()
_semaphore: threading.BoundedSemaphore | None = None
_semaphore_size: int | None = None


def _get_semaphore() -> threading.BoundedSemaphore:
    """Return the shared semaphore, (re)building it if the configured size changed."""
    global _semaphore, _semaphore_size
    size = load_config().max_concurrency
    with _guard_lock:
        if _semaphore is None or _semaphore_size != size:
            _semaphore = threading.BoundedSemaphore(size)
            _semaphore_size = size
        return _semaphore


@contextmanager
def concurrency_guard() -> Iterator[None]:
    """Bound simultaneous Anthropic calls to ``FRUGALROUTE_MAX_CONCURRENCY``.

    Blocking acquire/release around a single model call. Any path that fans out
    calls (the live route, the eval sweep) acquires this, so the peak number of
    in-flight calls process-wide never exceeds the configured cap.
    """
    sem = _get_semaphore()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def reset_runtime() -> None:
    """Reset cached logging + semaphore state (for tests that change the env)."""
    global _semaphore, _semaphore_size, _configured
    with _guard_lock:
        _semaphore = None
        _semaphore_size = None
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, "_frugalroute", False):
            logger.removeHandler(handler)
    _configured = False
