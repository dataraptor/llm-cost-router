"""Embedder wrapper — no-key tests of the pure array handling + the cache.

The heavy model (torch via sentence-transformers) is never loaded here: a fake
embedder is injected, and the module-level cache is exercised with a stub so the
``embedder=None`` default path is covered without a download.
"""

from __future__ import annotations

import sys

import numpy as np

from frugalroute.embed import embed, get_embedder

# The package re-exports the ``embed`` function as ``frugalroute.embed``, which
# shadows the submodule attribute — reach the real module via sys.modules.
embed_module = sys.modules["frugalroute.embed"]


class _FakeEmbedder:
    def __init__(self, matrix) -> None:
        self._matrix = np.asarray(matrix, dtype=np.float64)

    def encode(self, queries, **_kwargs):
        # Return one row per query (sliced), exercising the normal 2-D path.
        return self._matrix[: len(list(queries))]


def test_embed_returns_2d_float32() -> None:
    fake = _FakeEmbedder([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    out = embed(["a", "b"], embedder=fake)
    assert out.shape == (2, 3)
    assert out.dtype == np.float32


def test_embed_promotes_1d_to_2d() -> None:
    # A 1-D embedding (single vector) is reshaped to (1, d).
    class _OneD:
        def encode(self, queries, **_kwargs):
            return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    out = embed(["only one"], embedder=_OneD())
    assert out.shape == (1, 4)


def test_get_embedder_caches_and_embed_uses_default(monkeypatch) -> None:
    # Pre-seed the cache so get_embedder() returns it without importing torch,
    # and embed(..., embedder=None) picks up that default.
    fake = _FakeEmbedder([[1.0, 2.0]])
    monkeypatch.setitem(embed_module._EMBEDDER_CACHE, "BAAI/bge-small-en-v1.5", fake)

    assert get_embedder() is fake
    out = embed(["q"])  # embedder=None → get_embedder() → cached fake
    assert out.shape == (1, 2)
    assert out.dtype == np.float32
