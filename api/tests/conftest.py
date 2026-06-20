"""Shared fixtures for the API tests.

No-key tests drive the app via FastAPI ``TestClient`` and monkeypatch the engine
entrypoints (``frugalroute.route`` / ``frugalroute.run_eval``) with fakes, so they
are deterministic and never touch the network. Live tests (``@pytest.mark.api`` /
``@pytest.mark.azure``) auto-skip without their backend key — mirroring ``core``.

A repo-root ``.env`` is loaded (non-overriding) so the supplied Azure credentials
are picked up automatically for the live ``@azure`` smoke.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from frugalroute.models import EvalReport, GateVerdict, RouteResult


def _load_dotenv() -> None:
    """Load ``KEY=VALUE`` pairs from the repo-root ``.env`` (non-overriding)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip live-API tests when their backend's key is unset."""
    have_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    have_azure = bool(os.environ.get("AZURE_OPENAI_API_KEY"))
    skip_api = pytest.mark.skip(reason="requires ANTHROPIC_API_KEY")
    skip_azure = pytest.mark.skip(reason="requires AZURE_OPENAI_API_KEY")
    # Use the actual marker (not keyword membership): the rootdir is named "api",
    # so ``"api" in item.keywords`` would wrongly match every test's path.
    for item in items:
        if item.get_closest_marker("api") and not have_anthropic:
            item.add_marker(skip_api)
        if item.get_closest_marker("azure") and not have_azure:
            item.add_marker(skip_azure)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A ``TestClient`` over the app; clears settings cache + overrides after."""
    from frugalroute_api.app import app
    from frugalroute_api.config import get_settings

    get_settings.cache_clear()
    # raise_server_exceptions=False so the structured catch-all handler's response
    # reaches the client (rather than the exception re-propagating in the test).
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture
def override_settings() -> Iterator[Callable[..., None]]:
    """Install a ``Settings`` override (e.g. a temp sample path) on the app."""
    from frugalroute_api.app import app
    from frugalroute_api.config import Settings, get_settings

    def _apply(**kwargs: Any) -> None:
        app.dependency_overrides[get_settings] = lambda: Settings(**kwargs)

    yield _apply
    app.dependency_overrides.clear()


@pytest.fixture
def make_route_result() -> Callable[..., RouteResult]:
    """Factory for a canned ``RouteResult`` (sane defaults; override per test)."""

    def _make(
        *,
        query: str = "What is 2+2?",
        strategy: str = "cascade",
        tier_used: str = "claude-haiku-4-5",
        escalated: bool = False,
        answer: str = "The answer is 4.",
        correct: bool | None = None,
        gate: GateVerdict | None = None,
        p_strong: float | None = None,
        refused: bool = False,
        cost_usd: float = 0.0015,
        latency_s: float = 0.42,
        prompt_version: str = "v1",
    ) -> RouteResult:
        if gate is None and strategy == "cascade" and not refused:
            gate = GateVerdict(sufficient=not escalated, confidence=0.9, reason="ok")
        return RouteResult(
            query=query,
            strategy=strategy,
            tier_used=tier_used,
            escalated=escalated,
            answer=answer,
            correct=correct,
            gate=gate,
            p_strong=p_strong,
            refused=refused,
            cost_usd=cost_usd,
            latency_s=latency_s,
            prompt_version=prompt_version,
        )

    return _make


def patch_route(monkeypatch: pytest.MonkeyPatch, fn: Callable[..., RouteResult]) -> None:
    """Replace ``frugalroute.route`` (called via attribute access in the app)."""
    import frugalroute

    monkeypatch.setattr(frugalroute, "route", fn)


def patch_run_eval(monkeypatch: pytest.MonkeyPatch, fn: Callable[..., Any]) -> None:
    """Replace ``frugalroute.run_eval`` (called via attribute access in the app)."""
    import frugalroute

    monkeypatch.setattr(frugalroute, "run_eval", fn)


__all__ = [
    "EvalReport",
    "GateVerdict",
    "RouteResult",
    "patch_route",
    "patch_run_eval",
]
