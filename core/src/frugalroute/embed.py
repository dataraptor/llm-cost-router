"""Local query embeddings for the predictive router (build-spec §16, §19-D).

The Anthropic API has **no embeddings endpoint**, so Strategy B embeds queries
with a *local* sentence-transformer (``bge-small-en-v1.5`` by default) — no API
key, no network at call time beyond a one-time model download.

Two rules keep this no-key-test-friendly (build-spec §12 / split-04 R5):

- the heavy import (``sentence_transformers`` → torch) lives **inside**
  :func:`get_embedder`, so ``import frugalroute`` never loads the model; and
- :func:`embed` accepts an injected ``embedder``, so unit tests pass a tiny fake
  (any object with ``.encode``) and never touch torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"

# Module-level singleton cache, keyed by model name (loaded read-only).
_EMBEDDER_CACHE: dict[str, Any] = {}


def get_embedder(model_name: str = DEFAULT_EMBEDDER) -> Any:
    """Lazily load and cache a local sentence-transformer (singleton per name).

    The model is constructed once and reused. ``sentence_transformers`` is
    imported here (not at module import) so importing ``frugalroute`` requires
    no key, no network, and no torch. Install the optional extra to use it:
    ``pip install -e "core[embed]"``.
    """
    cached = _EMBEDDER_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "sentence-transformers is not installed. Install the embedder extra: "
            'pip install -e "core[embed]" (needed only for the predictive router).'
        ) from exc
    model = SentenceTransformer(model_name)
    _EMBEDDER_CACHE[model_name] = model
    return model


def embed(queries: list[str], embedder: Any = None) -> npt.NDArray[np.float32]:
    """Return an ``(n, d)`` float32 matrix of L2-normalized query embeddings.

    Deterministic for a fixed model and input (no sampling). ``embedder``
    defaults to :func:`get_embedder`; pass a fake (anything with a matching
    ``.encode``) in no-key tests. The result is always a 2-D float32 array, even
    for a single query.
    """
    if embedder is None:
        embedder = get_embedder()
    vectors = embedder.encode(list(queries), normalize_embeddings=True, convert_to_numpy=True)
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array
