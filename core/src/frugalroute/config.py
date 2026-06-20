"""Engine runtime settings, env-driven and validated (split-11 §1).

A single, small config surface for the operational knobs the engine respects:
the **concurrency cap** that bounds simultaneous Anthropic calls process-wide, a
per-request **timeout** (enforced at the HTTP edge — see ``frugalroute_api``), and
the **log level**. Everything is read from ``FRUGALROUTE_*`` env vars with a sane
default; an invalid value raises a clear ``ValueError`` at load time rather than
silently degrading to a bad state.

Model/tier/pricing config already lives in :mod:`frugalroute.llm` (config-driven
since split 01); this module only adds the operational settings split 11 needs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --- Defaults (build-spec §17 keeps the worker pool modest — "~6 workers"). ---
DEFAULT_MAX_CONCURRENCY = 6
DEFAULT_REQUEST_TIMEOUT_S = 60.0
DEFAULT_LOG_LEVEL = "INFO"

_VALID_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"})


@dataclass(frozen=True)
class EngineConfig:
    """Validated operational settings for the engine (immutable snapshot)."""

    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    log_level: str = DEFAULT_LOG_LEVEL


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc
    return value


def load_config() -> EngineConfig:
    """Read + validate the engine config from the environment.

    Precedence is **env override > default**. Invalid values fail loudly with a
    clear ``ValueError`` (e.g. a non-positive concurrency or timeout, an unknown
    log level) — never a silent bad state.
    """
    max_concurrency = _env_int("FRUGALROUTE_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
    if max_concurrency < 1:
        raise ValueError(f"FRUGALROUTE_MAX_CONCURRENCY must be >= 1, got {max_concurrency}.")

    request_timeout_s = _env_float("FRUGALROUTE_REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S)
    if request_timeout_s <= 0:
        raise ValueError(f"FRUGALROUTE_REQUEST_TIMEOUT_S must be > 0, got {request_timeout_s}.")

    log_level = os.environ.get("FRUGALROUTE_LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper()
    if log_level not in _VALID_LOG_LEVELS:
        raise ValueError(
            f"FRUGALROUTE_LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {log_level!r}."
        )

    return EngineConfig(
        max_concurrency=max_concurrency,
        request_timeout_s=request_timeout_s,
        log_level=log_level,
    )
