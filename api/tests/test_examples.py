"""Test 3: examples expose only id/benchmark/label/query (no answers leaked)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from frugalroute import load_examples

ALLOWED_KEYS = {"id", "benchmark", "label", "query"}
# Fields that must NEVER leak (they become live /api/route results).
FORBIDDEN_KEYS = {"gold", "haiku", "opus", "conf", "confidence", "sufficient", "pStrong", "answer"}


def test_examples_only_picker_fields(client: TestClient) -> None:
    resp = client.get("/api/examples")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == len(load_examples())
    for entry in body:
        assert set(entry) == ALLOWED_KEYS
        assert not (set(entry) & FORBIDDEN_KEYS)
        assert entry["id"] and entry["query"]
