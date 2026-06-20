"""The structured error model shared by every endpoint (split-06 §2).

Every non-2xx response carries the **same** body shape so the frontend can render
errors uniformly and never has to parse a stack trace::

    { "error": { "type": "...", "message": "...", "detail": "..."|null } }

The engine surfaces its failures as plain Python exceptions (a missing-key
``RuntimeError``, an Anthropic ``APIError``, a ``ValueError`` for bad input). This
module turns each into a typed :class:`APIError` with the right HTTP status, and
registers FastAPI exception handlers so **no endpoint ever returns an unstructured
500** — even an unexpected exception becomes the structured shape above.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

# The closed set of error types the contract exposes (split-06 §2, split-11 §2).
ErrorType = str  # one of the literals below; kept as str for JSON simplicity.
MISSING_KEY = "missing-key"
API_ERROR = "api-error"
BAD_REQUEST = "bad-request"
NOT_FOUND = "not-found"
BATCH_PENDING = "batch-pending"
BUSY = "busy"  # over the concurrency cap → 503 + Retry-After (split-11)
RATE_LIMITED = "rate-limited"  # per-IP token bucket exhausted → 429 + Retry-After


class ErrorBody(BaseModel):
    """The ``error`` object inside every non-2xx response."""

    type: str
    message: str
    detail: str | None = None


class ErrorResponse(BaseModel):
    """The full non-2xx body: ``{"error": {...}}`` (the only error shape)."""

    error: ErrorBody


class APIError(Exception):
    """A typed application error that renders to :class:`ErrorResponse`.

    Carries the HTTP ``status_code`` and the contract ``error_type`` so the single
    exception handler can serialize it consistently.
    """

    def __init__(
        self,
        status_code: int,
        error_type: str,
        message: str,
        detail: str | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.detail = detail
        self.headers = headers

    def to_response(self) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorBody(type=self.error_type, message=self.message, detail=self.detail)
        )
        return JSONResponse(
            status_code=self.status_code, content=body.model_dump(), headers=self.headers
        )


# --- Constructors for the typed errors (status codes pinned per split-06 §2). ---


def missing_key(message: str, detail: str | None = None) -> APIError:
    return APIError(503, MISSING_KEY, message, detail)


def upstream_api_error(message: str, detail: str | None = None) -> APIError:
    return APIError(502, API_ERROR, message, detail)


def bad_request(message: str, detail: str | None = None, *, status_code: int = 400) -> APIError:
    return APIError(status_code, BAD_REQUEST, message, detail)


def not_found(message: str, detail: str | None = None) -> APIError:
    return APIError(404, NOT_FOUND, message, detail)


def busy(message: str, *, retry_after: int = 1, detail: str | None = None) -> APIError:
    """503: the service is over its concurrency cap; shed load with ``Retry-After``."""
    return APIError(503, BUSY, message, detail, headers={"Retry-After": str(retry_after)})


def rate_limited(message: str, *, retry_after: int = 1, detail: str | None = None) -> APIError:
    """429: the caller exceeded its per-IP rate limit; ``Retry-After`` seconds to wait."""
    return APIError(429, RATE_LIMITED, message, detail, headers={"Retry-After": str(retry_after)})


def timeout(message: str, detail: str | None = None) -> APIError:
    """504: the engine exceeded the per-request timeout (typed ``api-error``, §11)."""
    return APIError(504, API_ERROR, message, detail)


def translate_engine_error(exc: Exception) -> APIError:
    """Map a ``frugalroute`` engine exception onto a typed :class:`APIError`.

    - the missing-``ANTHROPIC_API_KEY`` (or missing Azure config) ``RuntimeError``
      → 503 ``missing-key`` (the UI offers the precomputed "View the Proof" path);
    - an Anthropic SDK/API error → 502 ``api-error``;
    - a ``ValueError`` (unknown strategy, missing predictive router, bad input)
      → 400 ``bad-request``;
    - anything else → 502 ``api-error`` (still structured — never an unhandled 500).
    """
    message = str(exc)
    if _is_missing_key_error(exc):
        return missing_key(
            "No model backend is configured. Set ANTHROPIC_API_KEY (or the Azure "
            "OpenAI credentials) to run live routing, or view the precomputed proof "
            "at GET /api/eval/sample.",
            detail=message,
        )
    if _is_anthropic_error(exc):
        return upstream_api_error("The model backend returned an error.", detail=message)
    if isinstance(exc, ValueError):
        return bad_request(message)
    # Truly-unexpected engine failure: do NOT echo the raw message (it could carry
    # an internal path / detail). Expose only the exception *class* as a hint; the
    # full error is available server-side, never to the client (split-14 hygiene).
    return upstream_api_error("Unexpected engine error while routing.", detail=type(exc).__name__)


def _is_missing_key_error(exc: Exception) -> bool:
    """True for the engine's missing-credentials ``RuntimeError`` (key or Azure cfg)."""
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc)
    return "ANTHROPIC_API_KEY" in message or "Azure OpenAI config missing" in message


def _is_anthropic_error(exc: Exception) -> bool:
    """True for an Anthropic SDK/API exception, without importing it eagerly."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - anthropic is a core dependency
        return False
    return isinstance(exc, anthropic.APIError)


# ----------------------------------------------------------------------------
# FastAPI exception handlers — register on the app so every failure is structured.
# ----------------------------------------------------------------------------
def register_handlers(app: FastAPI) -> None:
    """Wire the handlers that guarantee the structured error body everywhere."""

    @app.exception_handler(APIError)
    async def _handle_api_error(_request: Request, exc: APIError) -> JSONResponse:
        return exc.to_response()

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Framework-raised HTTP errors (unmatched route → 404, wrong method → 405)
        # default to an *unstructured* {"detail": ...} body; reshape them into our
        # envelope so EVERY non-2xx response shares the one error shape (split-14).
        error_type = NOT_FOUND if exc.status_code == 404 else BAD_REQUEST
        message = str(exc.detail) if exc.detail else "HTTP error"
        return APIError(exc.status_code, error_type, message).to_response()

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 body is a list; reshape it into our error model.
        return APIError(
            422,
            BAD_REQUEST,
            "Request validation failed.",
            detail=_summarize_validation(exc.errors()),
        ).to_response()

    @app.exception_handler(Exception)
    async def _handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        # Last-resort safety net: an unhandled exception is still structured. Expose
        # only the exception *class* (never the message — it could carry an internal
        # path / detail); the full error stays server-side (split-14 hygiene).
        return upstream_api_error("Internal server error.", detail=type(exc).__name__).to_response()


def _summarize_validation(errors: Sequence[Any]) -> str:
    """One readable line naming the first few validation problems."""
    parts: list[str] = []
    for err in list(errors)[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        parts.append(f"{loc or '(root)'}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)
