"""FrugalRoute API — a thin FastAPI adapter over the ``frugalroute`` engine.

This package holds **no** routing/metrics/cost logic: it validates HTTP requests,
calls ``frugalroute`` in-process, and serializes the engine's contracts (§7) to
JSON. Delete ``api/`` and the engine is unchanged.
"""

from __future__ import annotations

__version__ = "0.1.0"

from frugalroute_api.app import app, create_app

__all__ = ["app", "create_app"]
