"""Local semantic embeddings — meaning-aware recall without an API key.

ReLife runs on the Claude Code Max subscription, not a metered API key, so we
cannot call a hosted embeddings endpoint. Instead we use ``fastembed`` (ONNX,
CPU-only) which runs a small model **entirely offline** after a one-time model
download.

This wrapper is **soft-optional**: if ``fastembed`` isn't installed, the model
can't be fetched, or embeddings are disabled via config, every entry point
degrades gracefully (``available()`` is False, ``embed()`` returns ``None``) and
the rest of the memory system falls back to keyword + activation recall. So the
package is never a hard dependency and the test suite never requires it.
"""

from __future__ import annotations

import math
import threading

from .. import config

_model = None          # lazily constructed TextEmbedding (or None if unavailable)
_tried = False         # have we attempted construction yet?
_lock = threading.Lock()


def _enabled() -> bool:
    mode = (config.EMBEDDINGS_ENABLED or "auto").lower()
    return mode != "off"


def _get_model():
    """Build the embedding model once; return None if it can't be loaded."""
    global _model, _tried
    if _tried:
        return _model
    with _lock:
        if _tried:
            return _model
        _tried = True
        if not _enabled():
            _model = None
            return None
        try:
            from fastembed import TextEmbedding  # type: ignore

            _model = TextEmbedding(model_name=config.EMBED_MODEL)
        except Exception:
            # Not installed, offline on first download, or model unavailable.
            _model = None
        return _model


def available() -> bool:
    """True if semantic embeddings can be produced right now."""
    return _get_model() is not None


def embed(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch of texts. Returns vectors, or None if unavailable."""
    if not texts:
        return []
    model = _get_model()
    if model is None:
        return None
    try:
        return [list(map(float, v)) for v in model.embed(texts)]
    except Exception:
        return None


def embed_one(text: str) -> list[float] | None:
    out = embed([text])
    return out[0] if out else None


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity in [-1, 1]; 0.0 when either vector is missing/degenerate."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
