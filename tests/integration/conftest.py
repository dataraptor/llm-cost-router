"""Fixtures for the repo-root full-stack integration tests (split-10 §2).

These boot the *real* API (FastAPI ``TestClient`` for the contract assertions, and
a real ``uvicorn`` subprocess for the genuine over-HTTP boot smoke) and read the
*committed* sample bundle the Frontier renders. Everything here is no-key and
deterministic; the ``@pytest.mark.api`` / ``@pytest.mark.azure`` live tests
auto-skip without their backend key (mirroring core/api).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_SAMPLE = (
    REPO_ROOT / "api" / "src" / "frugalroute_api" / "data" / "sample_run.json"
)
APP_DIR = REPO_ROOT / "app"


def _load_dotenv() -> None:
    """Load ``KEY=VALUE`` pairs from the repo-root ``.env`` (non-overriding)."""
    env_path = REPO_ROOT / ".env"
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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "api: live native-Anthropic test (skipped without ANTHROPIC_API_KEY)"
    )
    config.addinivalue_line(
        "markers",
        "azure: live gpt-5.5 adapter test (skipped without AZURE_OPENAI_API_KEY)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: Iterable[pytest.Item]
) -> None:
    """Skip live-API tests when their backend's key is unset (mirrors core/api)."""
    have_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    have_azure = bool(os.environ.get("AZURE_OPENAI_API_KEY"))
    skip_api = pytest.mark.skip(reason="requires ANTHROPIC_API_KEY")
    skip_azure = pytest.mark.skip(reason="requires AZURE_OPENAI_API_KEY")
    for item in items:
        if item.get_closest_marker("api") and not have_anthropic:
            item.add_marker(skip_api)
        if item.get_closest_marker("azure") and not have_azure:
            item.add_marker(skip_azure)


@pytest.fixture
def committed_sample() -> Path:
    """Path to the committed sample bundle (`/api/eval/sample` serves this)."""
    return COMMITTED_SAMPLE


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A ``TestClient`` over the real app (no key, native backend by default)."""
    from frugalroute_api.app import app
    from frugalroute_api.config import get_settings

    get_settings.cache_clear()
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture
def override_settings() -> Iterator[Callable[..., None]]:
    """Install a ``Settings`` override on the app (e.g. a temp/empty sample path)."""
    from frugalroute_api.app import app
    from frugalroute_api.config import Settings, get_settings

    def _apply(**kwargs: Any) -> None:
        app.dependency_overrides[get_settings] = lambda: Settings(**kwargs)

    yield _apply
    app.dependency_overrides.clear()


# ----------------------------------------------------------------------------
# Real over-HTTP boot (uvicorn subprocess) — proves the stack starts for real.
# ----------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LiveServer:
    """A running ``uvicorn`` process serving the real API; base URL exposed."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return httpx.get(self.base_url + path, timeout=10.0, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return httpx.post(self.base_url + path, timeout=60.0, **kwargs)


def _boot_uvicorn(backend: str) -> Iterator[LiveServer]:
    """Boot ``uvicorn frugalroute_api.app:app`` on a free port; tear down after."""
    port = _free_port()
    env = {**os.environ, "FRUGALROUTE_BACKEND": backend, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "frugalroute_api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        cwd=str(REPO_ROOT),
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30.0
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"uvicorn exited early (code {proc.returncode}).")
            try:
                if httpx.get(base_url + "/api/health", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.25)
        else:
            raise RuntimeError("uvicorn did not become healthy within 30s.")
        yield LiveServer(base_url)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def live_server() -> Iterator[LiveServer]:
    """A real uvicorn server, native backend (no key) — the missing-key path is
    reachable and the committed sample serves the proof (R3)."""
    yield from _boot_uvicorn(backend="")


@pytest.fixture(scope="session")
def live_server_azure() -> Iterator[LiveServer]:
    """A real uvicorn server on the gpt-5.5 adapter (this box's live backend) for
    the ``@azure`` live route/eval round-trips (R8)."""
    yield from _boot_uvicorn(backend="azure")
