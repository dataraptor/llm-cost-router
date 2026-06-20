"""The FastAPI app — a thin adapter over ``frugalroute`` (split-06).

Every endpoint validates its input, calls the engine **in-process**, and serializes
the result via :mod:`frugalroute_api.schemas`. No routing/metrics/cost logic lives
here. The engine is referenced via ``frugalroute.route`` / ``frugalroute.run_eval``
attribute access at call time so tests can monkeypatch them without a network.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any

import frugalroute
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from frugalroute import llm
from frugalroute.harness import (
    QUICK_TAUS,
    QUICK_THETAS,
    report_from_dict,
    report_to_dict,
)
from frugalroute.prompts import PROMPT_VERSION

from frugalroute_api import config as cfg
from frugalroute_api import errors, schemas
from frugalroute_api.config import Settings, get_settings

__version__ = "0.1.0"


# ----------------------------------------------------------------------------
# Backend / key helpers (live routing only; full hardening is split 11)
# ----------------------------------------------------------------------------
def has_backend_key(settings: Settings) -> bool:
    """Whether a live model backend is configured (drives the UI's 'View Proof')."""
    if settings.backend.lower() == "azure":
        return bool(os.environ.get("AZURE_OPENAI_API_KEY"))
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def resolve_client(settings: Settings) -> Any:
    """Return an injected client for the configured backend, or ``None``.

    ``None`` lets the engine resolve its native Anthropic client lazily (and raise
    the missing-key error we map to 503). ``backend=azure`` injects the gpt-5.5
    adapter (this build's live backend); a missing Azure config raises the engine's
    own clear error, which the handler maps to 503 ``missing-key`` too.
    """
    if settings.backend.lower() == "azure":
        from frugalroute.azure_client import get_azure_client

        return get_azure_client()
    return None


def _load_router_or_none(settings: Settings) -> Any:
    """Load a configured predictive router (joblib), or ``None`` if not configured."""
    if settings.router_path is None or not settings.router_path.exists():
        return None
    return frugalroute.load_router(str(settings.router_path))


# ----------------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------------
def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="FrugalRoute API",
        version=__version__,
        description="Thin HTTP adapter over the FrugalRoute cost-optimizing router engine.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    errors.register_handlers(app)
    prefix = settings.api_prefix

    # --- Health ---------------------------------------------------------------
    @app.get(f"{prefix}/health", response_model=schemas.HealthResponse, tags=["meta"])
    def health(settings: Settings = Depends(get_settings)) -> schemas.HealthResponse:
        return schemas.HealthResponse(
            status="ok", version=__version__, has_api_key=has_backend_key(settings)
        )

    # --- Config (sourced from core; no duplicated numbers) --------------------
    @app.get(f"{prefix}/config", response_model=schemas.ConfigResponse, tags=["meta"])
    def get_config(settings: Settings = Depends(get_settings)) -> schemas.ConfigResponse:
        pricing = {
            tier: schemas.TierPrice(
                input_per_mtok=llm.PRICING[tier]["input"],
                output_per_mtok=llm.PRICING[tier]["output"],
            )
            for tier in llm.DEFAULT_TIERS
        }
        return schemas.ConfigResponse(
            prompt_version=PROMPT_VERSION,
            model_tiers=list(llm.DEFAULT_TIERS),
            strategies=["cascade", "predictive"],
            pricing=pricing,
            always_strong_cost_ref_usd=cfg.always_strong_cost_ref_usd(settings),
            defaults=schemas.ConfigDefaults(tau=cfg.DEFAULT_TAU, theta=cfg.DEFAULT_THETA),
            pricing_pinned_date=cfg.PRICING_PINNED_DATE,
        )

    # --- Examples (only id/benchmark/label/query — no answers leaked) ---------
    @app.get(f"{prefix}/examples", response_model=list[schemas.ExampleOut], tags=["route"])
    def get_examples() -> list[schemas.ExampleOut]:
        return [
            schemas.ExampleOut(
                id=item["id"],
                benchmark=item["benchmark"],
                label=item["label"],
                query=item["query"],
            )
            for item in frugalroute.load_examples()
        ]

    # --- Route (live single query) -------------------------------------------
    @app.post(f"{prefix}/route", response_model=schemas.RouteResponse, tags=["route"])
    def post_route(
        req: schemas.RouteRequest, settings: Settings = Depends(get_settings)
    ) -> schemas.RouteResponse:
        query, benchmark = _resolve_query(req)
        tau_used = req.tau if req.tau is not None else cfg.DEFAULT_TAU
        theta_used = req.theta if req.theta is not None else cfg.DEFAULT_THETA
        try:
            client = resolve_client(settings)
            router = _load_router_or_none(settings) if req.strategy == "predictive" else None
            result = frugalroute.route(
                query,
                strategy=req.strategy,
                benchmark=benchmark,
                tau=tau_used,
                theta=theta_used,
                client=client,
                router=router,
            )
        except errors.APIError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to a typed structured error
            raise errors.translate_engine_error(exc) from exc
        return schemas.route_response(
            result,
            theta_used=theta_used,
            always_strong_usd=cfg.always_strong_cost_ref_usd(settings),
        )

    # --- Eval: precomputed sample bundle (no key/network) ---------------------
    @app.get(f"{prefix}/eval/sample", tags=["eval"])
    def eval_sample(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
        return _load_sample_bundle(settings)

    # --- Eval: live quick eval (bounded, synchronous) ------------------------
    @app.post(f"{prefix}/eval", tags=["eval"])
    def post_eval(
        req: schemas.EvalRequest, settings: Settings = Depends(get_settings)
    ) -> dict[str, Any]:
        if not req.quick:
            raise errors.bad_request(
                "Only quick eval (quick=true) is supported over HTTP in v1. For a full "
                "sweep (more repeats, batch), run the CLI: `frugalroute eval ...`.",
            )
        repeats = req.repeats if req.repeats is not None else 1
        taus = req.grid if req.grid is not None else QUICK_TAUS
        thetas = req.grid if req.grid is not None else QUICK_THETAS
        try:
            client = resolve_client(settings)
            run = frugalroute.run_eval(
                req.benchmark,
                strategy=req.strategy,
                repeats=repeats,
                taus=taus,
                thetas=thetas,
                client=client,
            )
        except errors.APIError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to a typed structured error
            raise errors.translate_engine_error(exc) from exc
        n_test = int(run.meta.get("n", 0))
        return schemas.bundle_to_json(
            list(run.reports.values()),
            benchmark=req.benchmark,
            n_test=n_test,
            n_calibration=int(run.meta.get("n_calibration", 0)),
            small_n=n_test < cfg.SMALL_N_THRESHOLD,
            generated_at=_now_iso(),
        )

    # --- Streaming placeholder (split 09) ------------------------------------
    @app.get(f"{prefix}/route/stream", tags=["route"])
    def route_stream() -> Any:
        raise errors.not_implemented(
            "Streaming routing (SSE) is implemented in a later split. Use POST "
            f"{prefix}/route for the synchronous result.",
        )

    # --- Root → docs ----------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


# ----------------------------------------------------------------------------
# Request → engine-arg helpers
# ----------------------------------------------------------------------------
def _resolve_query(req: schemas.RouteRequest) -> tuple[str, str]:
    """Resolve the (query, benchmark) to route from the request.

    A direct ``query`` uses ``benchmark`` (or "gsm8k"); an ``example_id`` is looked
    up in the bundled examples (404 if unknown) and supplies the query and, unless
    the request overrides it, the benchmark.
    """
    if req.example_id:
        examples = {item["id"]: item for item in frugalroute.load_examples()}
        example = examples.get(req.example_id)
        if example is None:
            raise errors.not_found(f"Unknown example_id {req.example_id!r}.")
        benchmark = req.benchmark or example["benchmark"]
        return str(example["query"]), str(benchmark)
    # validator guarantees query is present here
    assert req.query is not None
    return req.query, (req.benchmark or "gsm8k")


def _load_sample_bundle(settings: Settings) -> dict[str, Any]:
    """Read + re-serialize the committed sample bundle, enforcing §7 on each report.

    Missing file → 404 ``not-found`` (the UI shows its honest N/A empty state).
    Each report round-trips through the engine's (de)serializers so the served shape
    matches §7 exactly even if the file drifts.
    """
    path = settings.sample_run_path
    if not path.exists():
        raise errors.not_found(
            "No precomputed eval sample is available. Run an eval to populate it "
            f"(expected at {path}).",
        )
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        reports = [report_to_dict(report_from_dict(r)) for r in bundle["reports"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise errors.upstream_api_error(
            "The committed eval sample is malformed.", detail=str(exc)
        ) from exc
    return {
        "reports": reports,
        "benchmark": bundle.get("benchmark"),
        "frozen_split": bundle.get("frozen_split"),
        "generated_at": bundle.get("generated_at"),
    }


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


# Module-level ASGI app for `uvicorn frugalroute_api.app:app`.
app = create_app()
