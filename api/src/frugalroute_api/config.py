"""Runtime settings + the few non-engine constants the API serves (split-06 §1).

Everything that is *engine* truth (pricing, tiers, prompt version) is read from
``frugalroute`` at request time and never duplicated here. This module only holds
deployment settings (CORS, paths, backend selection) and the handful of UI defaults
the contract pins (``tau``/``theta`` defaults, the pricing-pinned date, the
always-strong cost reference fallback).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- UI defaults pinned by the contract (split-06 §3). The router's *own* default
#     tau is 0.8 (build-spec §13); theta defaults to 0.6 here and is resolved
#     explicitly before calling route() so the shown theta == the theta used. ---
DEFAULT_TAU = 0.8
DEFAULT_THETA = 0.6

# The §5 always-Opus per-query cost reference. Used only as the fallback when no
# committed sample run is available to source it from (split 10 bundles the real run).
ALWAYS_STRONG_COST_REF_FALLBACK_USD = 0.0070

# Date the pinned pricing gradient was last verified (00-INSTRUCTIONS §3).
PRICING_PINNED_DATE = "2026-06-19"

# Below this the frozen test split is "small" → the UI widens the +/- captions.
SMALL_N_THRESHOLD = 30

# The bundled sample bundle (a dev placeholder until split 10 commits the real run).
_DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "sample_run.json"


class Settings(BaseSettings):
    """Deployment settings, overridable via ``FRUGALROUTE_*`` env vars.

    ``cors_origins`` defaults to ``["*"]`` for local development; **lock it down in
    production** (split 13 serves same-origin so CORS is not needed there at all).
    """

    model_config = SettingsConfigDict(env_prefix="FRUGALROUTE_", extra="ignore")

    cors_origins: list[str] = ["*"]
    api_prefix: str = "/api"
    sample_run_path: Path = _DEFAULT_SAMPLE_PATH
    # Live backend: "azure" injects the gpt-5.5 adapter; anything else uses the
    # native Anthropic client (the engine default).
    backend: str = ""
    # Optional path to a trained predictive router (joblib); needed for live
    # predictive routing. Absent → predictive live returns a clear bad-request.
    router_path: Path | None = None

    # --- Hardening knobs (split-11). The concurrency cap + request timeout come
    #     from the engine config (FRUGALROUTE_MAX_CONCURRENCY / _REQUEST_TIMEOUT_S);
    #     these are the API-specific per-IP rate-limit knobs (a lenient default). ---
    # Token-bucket capacity (the burst a single IP may send before throttling).
    rate_limit_burst: int = 60
    # Sustained refill rate in requests/second (default 60/min = 1/s).
    rate_limit_refill_per_s: float = 1.0
    # Master switch: rate limiting off by default for local dev (lock down in prod).
    rate_limit_enabled: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept a comma-separated string from the env (e.g. ``a.com,b.com``)."""
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process settings (cached). Tests override via ``dependency_overrides``."""
    return Settings()


def always_strong_cost_ref_usd(settings: Settings) -> float:
    """The canonical always-Opus per-query cost the live tally compares against.

    Sourced from the committed sample run's ``always_strong`` baseline mean cost so
    the live "saved vs always-Opus" tally and the Frontier never disagree; falls
    back to the documented §5 constant when no sample run is present. Any read/parse
    problem degrades to the fallback rather than failing ``/api/config``.
    """
    try:
        bundle = json.loads(settings.sample_run_path.read_text(encoding="utf-8"))
        cost = bundle["reports"][0]["baselines"]["always_strong"]["cost"]
        return float(cost)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return ALWAYS_STRONG_COST_REF_FALLBACK_USD
