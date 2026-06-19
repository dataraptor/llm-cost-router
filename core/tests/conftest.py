"""Shared test fixtures and the @pytest.mark.api skip hook.

Tier-1 (no-key) unit tests use the injected ``fake_client``; Tier-2 tests marked
``@pytest.mark.api`` are auto-skipped unless ``ANTHROPIC_API_KEY`` is set. No test
in this package touches the network.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip every ``@pytest.mark.api`` test when ANTHROPIC_API_KEY is unset."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    skip_api = pytest.mark.skip(
        reason="requires ANTHROPIC_API_KEY (set it to run @pytest.mark.api tests)"
    )
    for item in items:
        if "api" in item.keywords:
            item.add_marker(skip_api)


# --- Minimal fakes standing in for the Anthropic SDK response surface. ---


class FakeUsage:
    """A stand-in for ``response.usage`` exposing the four token buckets."""

    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class FakeTextBlock:
    """A ``text`` content block."""

    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class FakeResponse:
    """A canned Anthropic message response.

    ``content`` raises if ``content_raises=True`` so tests can prove ``call()``
    never indexes content on a refusal.
    """

    def __init__(
        self,
        *,
        stop_reason: str = "end_turn",
        text: str = "The answer is 42.",
        usage: FakeUsage | None = None,
        parsed_output: Any = None,
        stop_details: Any = None,
        content_raises: bool = False,
    ) -> None:
        self.stop_reason = stop_reason
        self.usage = usage if usage is not None else FakeUsage()
        self.parsed_output = parsed_output
        self.stop_details = stop_details
        self._text = text
        self._content_raises = content_raises

    @property
    def content(self) -> list[FakeTextBlock]:
        if self._content_raises:
            raise AssertionError("response.content must not be accessed on a refusal")
        return [FakeTextBlock(self._text)]


class _FakeMessages:
    """Records the kwargs each call received so tests can inspect them."""

    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(("create", kwargs))
        return self._response

    def parse(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(("parse", kwargs))
        return self._response


class FakeClient:
    """A minimal Anthropic-client stand-in: ``client.messages.create/parse``."""

    def __init__(self, response: FakeResponse) -> None:
        self.messages = _FakeMessages(response)

    @property
    def last_kwargs(self) -> dict[str, Any]:
        """The kwargs passed to the most recent create/parse call."""
        return self.messages.calls[-1][1]


@pytest.fixture
def fake_client() -> Callable[..., FakeClient]:
    """Factory: build a ``FakeClient`` whose response is configured per test.

    Accepts the same keyword arguments as ``FakeResponse`` (``stop_reason``,
    ``text``, ``usage``, ``parsed_output``, ``content_raises``, ...).
    """

    def _make(**response_kwargs: Any) -> FakeClient:
        return FakeClient(FakeResponse(**response_kwargs))

    return _make


@pytest.fixture
def fake_usage() -> type[FakeUsage]:
    """The ``FakeUsage`` class, for building canned ``response.usage`` objects."""
    return FakeUsage
