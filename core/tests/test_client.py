"""No-key test for get_client's missing-key error path."""

from __future__ import annotations

import pytest

from frugalroute.llm import get_client


def test_get_client_raises_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_client()
