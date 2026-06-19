"""Tier-2 smoke test — auto-skipped unless ANTHROPIC_API_KEY is set.

Proves the conftest ``@pytest.mark.api`` skip hook works and that ``get_client``
constructs a client when a key is present. Constructs only; makes no network call.
"""

from __future__ import annotations

import pytest

from frugalroute.llm import get_client


@pytest.mark.api
def test_get_client_constructs_with_key() -> None:
    client = get_client()
    assert hasattr(client, "messages")
