"""The FastAPI app — a thin adapter over ``frugalroute`` (split-06).

Every endpoint validates its input, calls the engine **in-process**, and serializes
the result via :mod:`frugalroute_api.schemas`. No routing/metrics/cost logic lives
here. The engine is referenced via ``frugalroute.route`` / ``frugalroute.run_eval``
attribute access at call time so tests can monkeypatch them without a network.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
import threading
from collections.abc import Callable, Iterator
from typing import Any, Literal, TypeVar

import frugalroute
from fastapi import Depends, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from frugalroute import llm
from frugalroute.config import load_config
from frugalroute.harness import (
    QUICK_TAUS,
    QUICK_THETAS,
    report_from_dict,
    report_to_dict,
)
from frugalroute.models import RouteResult, route_result_from_dict
from frugalroute.obs import configure_logging
from frugalroute.prompts import PROMPT_VERSION
from pydantic import ValidationError

from frugalroute_api import config as cfg
from frugalroute_api import errors, schemas
from frugalroute_api.config import Settings, get_settings
from frugalroute_api.metrics import Metrics
from frugalroute_api.middleware import HardeningMiddleware
from frugalroute_api.ratelimit import RateLimiter

__version__ = "0.1.0"

_T = TypeVar("_T")


async def _run_with_timeout(fn: Callable[[], _T], timeout_s: float) -> _T:
    """Run a blocking engine call in the threadpool, bounded by ``timeout_s``.

    On timeout the client gets a typed 504 immediately (never a hung connection);
    the abandoned worker finishes in the background. A timeout of 0/None disables
    the bound (used only if explicitly configured non-positive — config validates
    against that, so in practice the bound is always active).
    """
    if timeout_s and timeout_s > 0:
        try:
            return await asyncio.wait_for(run_in_threadpool(fn), timeout=timeout_s)
        except (TimeoutError, asyncio.TimeoutError) as exc:  # noqa: UP041 - explicit for clarity
            raise errors.timeout(
                "The request exceeded the server time limit. Try again, or use a smaller input.",
                detail=f"timeout after {timeout_s}s",
            ) from exc
    return await run_in_threadpool(fn)


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
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else get_settings()
    app = FastAPI(
        title="FrugalRoute API",
        version=__version__,
        description="Thin HTTP adapter over the FrugalRoute cost-optimizing router engine.",
    )

    # --- Hardening setup (split-11): structured logging + per-app runtime state ---
    engine_cfg = load_config()  # validates FRUGALROUTE_* at startup, fails loudly
    configure_logging(engine_cfg.log_level)
    app.state.metrics = Metrics()
    # Back-pressure semaphore bounding concurrent in-flight engine requests.
    app.state.concurrency = threading.BoundedSemaphore(engine_cfg.max_concurrency)
    app.state.timeout_s = engine_cfg.request_timeout_s
    app.state.limiter = (
        RateLimiter(
            capacity=settings.rate_limit_burst,
            refill_per_s=settings.rate_limit_refill_per_s,
        )
        if settings.rate_limit_enabled
        else None
    )

    prefix = settings.api_prefix
    # Order: add CORS first (inner), then Hardening (outermost) so request-id,
    # rate-limit and access logging wrap every request including CORS handling.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        HardeningMiddleware,
        engine_endpoints={("POST", f"{prefix}/route"), ("POST", f"{prefix}/eval")},
    )
    errors.register_handlers(app)

    # --- Health ---------------------------------------------------------------
    @app.get(f"{prefix}/health", response_model=schemas.HealthResponse, tags=["meta"])
    def health(settings: Settings = Depends(get_settings)) -> schemas.HealthResponse:
        return schemas.HealthResponse(
            status="ok", version=__version__, has_api_key=has_backend_key(settings)
        )

    # --- Metrics (process-lifetime counters; reset on restart) ----------------
    @app.get(f"{prefix}/metrics", tags=["meta"])
    def get_metrics(request: Request) -> dict[str, Any]:
        snap: Any = request.app.state.metrics.snapshot()
        return {
            "requests_total": snap.requests_total,
            "cost_usd_total": snap.cost_usd_total,
            "escalation_rate": snap.escalation_rate,
            "refused_total": snap.refused_total,
            "latency_p50_s": snap.latency_p50_s,
            "latency_p95_s": snap.latency_p95_s,
            "since": snap.since,
        }

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
    async def post_route(
        req: schemas.RouteRequest,
        request: Request,
        settings: Settings = Depends(get_settings),
    ) -> schemas.RouteResponse:
        query, benchmark = _resolve_query(req)
        tau_used = req.tau if req.tau is not None else cfg.DEFAULT_TAU
        theta_used = req.theta if req.theta is not None else cfg.DEFAULT_THETA

        def _do() -> RouteResult:
            client = resolve_client(settings)
            router = _load_router_or_none(settings) if req.strategy == "predictive" else None
            return frugalroute.route(
                query,
                strategy=req.strategy,
                benchmark=benchmark,
                tau=tau_used,
                theta=theta_used,
                client=client,
                router=router,
            )

        try:
            result = await _run_with_timeout(_do, request.app.state.timeout_s)
        except errors.APIError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to a typed structured error
            raise errors.translate_engine_error(exc) from exc
        _record_route(request, result)
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
    async def post_eval(
        req: schemas.EvalRequest,
        request: Request,
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        if not req.quick:
            raise errors.bad_request(
                "Only quick eval (quick=true) is supported over HTTP in v1. For a full "
                "sweep (more repeats, batch), run the CLI: `frugalroute eval ...`.",
            )
        repeats = req.repeats if req.repeats is not None else 1
        taus = req.grid if req.grid is not None else QUICK_TAUS
        thetas = req.grid if req.grid is not None else QUICK_THETAS

        def _do() -> Any:
            client = resolve_client(settings)
            return frugalroute.run_eval(
                req.benchmark,
                strategy=req.strategy,
                repeats=repeats,
                taus=taus,
                thetas=thetas,
                client=client,
            )

        try:
            run = await _run_with_timeout(_do, request.app.state.timeout_s)
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

    # --- Route (live single query, streamed as SSE) --------------------------
    @app.get(f"{prefix}/route/stream", tags=["route"])
    def route_stream(
        request: Request,
        strategy: Literal["cascade", "predictive"] = Query("cascade"),
        query: str | None = Query(default=None, max_length=schemas.MAX_QUERY_CHARS),
        example_id: str | None = Query(default=None),
        benchmark: Literal["gsm8k", "mmlu"] | None = Query(default=None),
        tau: float | None = Query(default=None, ge=0.0, le=1.0),
        theta: float | None = Query(default=None, ge=0.0, le=1.0),
        settings: Settings = Depends(get_settings),
    ) -> StreamingResponse:
        # Validate (incl. "exactly one of query|example_id") and resolve BEFORE
        # streaming, so bad input is a clean pre-stream 422/404, not a 200 empty
        # stream. FastAPI already validated the scalar bounds (tau/theta/strategy).
        try:
            req = schemas.RouteRequest(
                strategy=strategy,
                query=query,
                example_id=example_id,
                benchmark=benchmark,
                tau=tau,
                theta=theta,
            )
        except ValidationError as exc:
            raise errors.bad_request(
                "Request validation failed.", detail=str(exc), status_code=422
            ) from exc
        resolved_query, resolved_benchmark = _resolve_query(req)
        tau_used = req.tau if req.tau is not None else cfg.DEFAULT_TAU
        theta_used = req.theta if req.theta is not None else cfg.DEFAULT_THETA
        always_strong_usd = cfg.always_strong_cost_ref_usd(settings)

        def gen() -> Iterator[str]:
            try:
                client = resolve_client(settings)
                router = _load_router_or_none(settings) if req.strategy == "predictive" else None
                for event in frugalroute.route_events(
                    resolved_query,
                    strategy=req.strategy,
                    benchmark=resolved_benchmark,
                    tau=tau_used,
                    theta=theta_used,
                    client=client,
                    router=router,
                ):
                    if event.type == "done":
                        # Re-derive the FULL response (with §7 extras) from the
                        # serialized RouteResult so the terminal `done` body is
                        # byte-identical to POST /api/route for the same inputs.
                        result = route_result_from_dict(event.data)
                        _record_route(request, result)
                        payload = schemas.route_response(
                            result, theta_used=theta_used, always_strong_usd=always_strong_usd
                        ).model_dump()
                        yield _sse_frame("done", payload)
                    else:
                        yield _sse_frame(event.type, event.data)
            except errors.APIError as exc:
                yield _sse_frame("error", {"type": exc.error_type, "message": exc.message})
            except Exception as exc:  # noqa: BLE001 - surfaced as a typed error event
                api_err = errors.translate_engine_error(exc)
                yield _sse_frame("error", {"type": api_err.error_type, "message": api_err.message})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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


def _record_route(request: Request, result: RouteResult) -> None:
    """Feed one completed route into the app's metrics accumulator (split-11)."""
    metrics: Metrics | None = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.record_route(
            cost_usd=result.cost_usd,
            latency_s=result.latency_s,
            escalated=result.escalated,
            refused=result.refused,
        )


def _sse_frame(event_type: str, data: dict[str, Any]) -> str:
    """Render one Server-Sent Event frame: ``event: <type>\\ndata: <json>\\n\\n``."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


# Module-level ASGI app for `uvicorn frugalroute_api.app:app`.
app = create_app()
