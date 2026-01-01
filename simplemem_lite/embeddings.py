"""Embedding generation for SimpleMem Lite.

Provides embedding generation via LiteLLM with optional local fallback.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, Callable

from simplemem_lite.config import Config
from simplemem_lite.log_config import get_logger

if TYPE_CHECKING:
    pass

log = get_logger("embeddings")

# Global config reference (set during initialization)
_config: Config | None = None

# Global model cache for local embeddings (avoids 2-3s reload per call)
_local_model: "SentenceTransformer | None" = None  # type: ignore

# Dynamic embedding cache (created with configurable size)
_cached_embed_fn: Callable[[str, str, bool], tuple[float, ...]] | None = None


def _get_local_model() -> "SentenceTransformer":
    """Get or create cached SentenceTransformer model."""
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        log.info("Loading SentenceTransformer model (cached for session)")
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


def _create_cached_embed(cache_size: int) -> Callable[[str, str, bool], tuple[float, ...]]:
    """Create a cached embedding function with configurable cache size."""

    @lru_cache(maxsize=cache_size)
    def cached_embed(text: str, model: str, use_local: bool) -> tuple[float, ...]:
        """Cached embedding generation (returns tuple for hashability)."""
        return tuple(_embed_impl(text, model, use_local))

    return cached_embed


def init_embeddings(config: Config) -> None:
    """Initialize embeddings module with configuration."""
    global _config, _cached_embed_fn
    _config = config
    # Create cache with configured size
    _cached_embed_fn = _create_cached_embed(config.embedding_cache_size)
    log.debug(f"Embeddings initialized: model={config.embedding_model}, local={config.use_local_embeddings}, cache_size={config.embedding_cache_size}")


def _embed_impl(text: str, model: str, use_local: bool) -> list[float]:
    """Actual embedding generation implementation."""
    if use_local:
        return _embed_local(text)
    else:
        return _embed_litellm(text, model)


def _embed_litellm(text: str, model: str) -> list[float]:
    """Generate embedding via LiteLLM."""
    from litellm import embedding

    response = embedding(model=model, input=[text])
    return response.data[0]["embedding"]


def _embed_local(text: str) -> list[float]:
    """Generate embedding via local sentence-transformers."""
    model = _get_local_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed(text: str, config: Config | None = None) -> list[float]:
    """Generate embedding for text.

    Args:
        text: Text to embed
        config: Optional config override (uses global if not provided)

    Returns:
        List of floats representing the embedding vector
    """
    cfg = config or _config
    if cfg is None:
        cfg = Config()

    log.trace(f"Embedding text: {len(text)} chars, model={cfg.embedding_model}")
    # Use cached version if initialized, otherwise generate directly
    if _cached_embed_fn is not None:
        result = _cached_embed_fn(text, cfg.embedding_model, cfg.use_local_embeddings)
    else:
        # Fallback for when init_embeddings hasn't been called
        result = tuple(_embed_impl(text, cfg.embedding_model, cfg.use_local_embeddings))
    log.trace(f"Embedding complete: dim={len(result)}")
    return list(result)


def embed_batch(texts: list[str], config: Config | None = None) -> list[list[float]]:
    """Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        config: Optional config override

    Returns:
        List of embedding vectors
    """
    cfg = config or _config
    if cfg is None:
        cfg = Config()

    if cfg.use_local_embeddings:
        # Local models support batch encoding (using cached model)
        model = _get_local_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]
    else:
        # LiteLLM: batch via API
        from litellm import embedding

        response = embedding(model=cfg.embedding_model, input=texts)
        return [d["embedding"] for d in response.data]
