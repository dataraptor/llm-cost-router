"""Cross-cutting request hardening middleware (split-11 §2).

One ASGI middleware ties together the per-request concerns that must apply
uniformly: a **request id** (accepted or generated, echoed on the response and
attached to every log line), a **per-IP rate limit** (429 + ``Retry-After``),
**concurrency back-pressure** on the expensive engine endpoints (503 ``busy`` +
``Retry-After`` instead of an unbounded queue), and structured **access logging**
that never records the API key or a full query body.

State (the rate limiter, the concurrency semaphore, the metrics accumulator) lives
on ``app.state`` so it is per-app — tests build a fresh app and get fresh state.
"""

from __future__ import annotations

import threading
import time
import uuid

from frugalroute.obs import get_logger
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from frugalroute_api import errors

REQUEST_ID_HEADER = "X-Request-ID"

_access_log = get_logger("api.access")


class HardeningMiddleware:
    """Pure-ASGI middleware (avoids BaseHTTPMiddleware's streaming quirks)."""

    def __init__(self, app: ASGIApp, *, engine_endpoints: set[tuple[str, str]]) -> None:
        self.app = app
        self._engine_endpoints = engine_endpoints

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        app = request.app
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex

        method = request.method
        path = request.url.path
        start = time.monotonic()

        async def _send_error(err: errors.APIError) -> None:
            response = err.to_response()
            response.headers[REQUEST_ID_HEADER] = request_id
            self._access(method, path, response.status_code, start, request_id)
            await response(scope, receive, send)

        # --- Per-IP rate limit (cheap, applies to every request when enabled) ---
        limiter = getattr(app.state, "limiter", None)
        if limiter is not None:
            ok, retry_after = limiter.allow(_client_ip(request))
            if not ok:
                await _send_error(
                    errors.rate_limited(
                        "Rate limit exceeded. Slow down and retry after the indicated delay.",
                        retry_after=retry_after,
                    )
                )
                return

        # --- Concurrency back-pressure on the expensive engine endpoints ---
        semaphore: threading.BoundedSemaphore | None = getattr(app.state, "concurrency", None)
        acquired = False
        if semaphore is not None and (method, path) in self._engine_endpoints:
            acquired = semaphore.acquire(blocking=False)
            if not acquired:
                await _send_error(
                    errors.busy(
                        "The service is at capacity. Please retry shortly.",
                        retry_after=1,
                    )
                )
                return

        # --- Run the request, capturing the final status for access logging ---
        status_holder = {"code": 500}

        async def _send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
                headers = message.setdefault("headers", [])
                headers.append((REQUEST_ID_HEADER.encode("latin-1"), request_id.encode("latin-1")))
            await send(message)

        try:
            await self.app(scope, receive, _send_wrapper)
        finally:
            if acquired and semaphore is not None:
                semaphore.release()
            self._access(method, path, status_holder["code"], start, request_id)

    def _access(self, method: str, path: str, status: int, start: float, request_id: str) -> None:
        """Emit one structured access log line — no body, no key, just the metadata."""
        _access_log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,  # path only; the query string (which may carry `query=`) is omitted
                "status": status,
                "latency_s": round(time.monotonic() - start, 6),
            },
        )


def _client_ip(request: Request) -> str:
    """Best-effort client IP for per-IP bucketing (TestClient → 'testclient')."""
    if request.client is not None:
        return request.client.host
    return "unknown"


__all__ = ["REQUEST_ID_HEADER", "HardeningMiddleware"]
