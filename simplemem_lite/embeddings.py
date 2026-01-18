"""Embedding generation for SimpleMem Lite.

Provides embedding generation via LiteLLM with optional local fallback.
Supports dual embedding pipelines:
- Memory embeddings: text-embedding-3-small via LiteLLM (or local all-MiniLM-L6-v2)
- Code embeddings: Voyage (voyage-code-3) → OpenRouter → Local (jina-embeddings-v2-base-code)
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

from simplemem_lite.config import Config
from simplemem_lite.log_config import get_logger, log_timing

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    import voyageai

log = get_logger("embeddings")

# Global config reference (set during initialization)
_config: Config | None = None

# Global model cache for local embeddings (avoids 2-3s reload per call)
_local_model: "SentenceTransformer | None" = None  # type: ignore

# Dynamic embedding cache (created with configurable size)
_cached_embed_fn: Callable[[str, str, bool, int | None], tuple[float, ...]] | None = None

# =============================================================================
# Code Embedding Caches (Dual Pipeline)
# =============================================================================

# Voyage client cache (avoids re-initialization overhead)
_voyage_client: "voyageai.Client | None" = None  # type: ignore

# Local code embedding model cache (jinaai/jina-embeddings-v2-base-code)
_local_code_model: "SentenceTransformer | None" = None  # type: ignore


@dataclass
class EmbeddingResult:
    """Result of embedding operation with metadata.

    Attributes:
        embeddings: List of embedding vectors
        provider: Provider that generated the embeddings (voyage, openrouter, local)
        model: Model name used
        dimension: Embedding dimension
        elapsed_ms: Time taken in milliseconds
    """
    embeddings: list[list[float]]
    provider: str
    model: str
    dimension: int
    elapsed_ms: float


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


def _create_cached_embed(cache_size: int) -> Callable[[str, str, bool, int | None], tuple[float, ...]]:
    """Create a cached embedding function with configurable cache size."""

    @lru_cache(maxsize=cache_size)
    def cached_embed(text: str, model: str, use_local: bool, dimensions: int | None = None) -> tuple[float, ...]:
        """Cached embedding generation (returns tuple for hashability)."""
        return tuple(_embed_impl(text, model, use_local, dimensions))

    return cached_embed


def init_embeddings(config: Config) -> None:
    """Initialize embeddings module with configuration."""
    global _config, _cached_embed_fn
    _config = config
    # Create cache with configured size
    _cached_embed_fn = _create_cached_embed(config.embedding_cache_size)
    log.debug(f"Embeddings initialized: model={config.embedding_model}, local={config.use_local_embeddings}, cache_size={config.embedding_cache_size}")


def _embed_impl(text: str, model: str, use_local: bool, dimensions: int | None = None) -> list[float]:
    """Actual embedding generation implementation."""
    if use_local:
        return _embed_local(text)
    else:
        return _embed_litellm(text, model, dimensions=dimensions)


def _embed_litellm(text: str, model: str, dimensions: int | None = None) -> list[float]:
    """Generate embedding via LiteLLM."""
    from litellm import embedding

    kwargs = {"model": model, "input": [text]}
    # Pass dimensions for models that support it (e.g., Gemini, OpenAI text-embedding-3-*)
    if dimensions:
        kwargs["dimensions"] = dimensions
    response = embedding(**kwargs)
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

    # Get dimensions for models that support it
    dimensions = cfg.embedding_dim if not cfg.use_local_embeddings else None

    log.trace(f"Embedding text: {len(text)} chars, model={cfg.embedding_model}, dim={dimensions}")
    # Use cached version if initialized, otherwise generate directly
    if _cached_embed_fn is not None:
        result = _cached_embed_fn(text, cfg.embedding_model, cfg.use_local_embeddings, dimensions)
    else:
        # Fallback for when init_embeddings hasn't been called
        result = tuple(_embed_impl(text, cfg.embedding_model, cfg.use_local_embeddings, dimensions))
    log.trace(f"Embedding complete: dim={len(result)}")
    return list(result)


def embed_batch(texts: list[str], config: Config | None = None, batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        config: Optional config override
        batch_size: Max texts per API call (default 100, Gemini's limit)

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
        # LiteLLM: batch via API with chunking to respect API limits
        from litellm import embedding

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            log.debug(f"Embedding batch {i // batch_size + 1}: {len(batch)} texts")
            kwargs = {"model": cfg.embedding_model, "input": batch}
            # Pass dimensions for models that support it
            if cfg.embedding_dim:
                kwargs["dimensions"] = cfg.embedding_dim
            response = embedding(**kwargs)
            all_embeddings.extend([d["embedding"] for d in response.data])
        return all_embeddings


# =============================================================================
# Code Embedding Functions (Dual Pipeline)
# =============================================================================


def _get_voyage_client(api_key: str) -> "voyageai.Client":
    """Get or create cached Voyage AI client.

    Args:
        api_key: Voyage API key

    Returns:
        Cached Voyage client instance
    """
    global _voyage_client
    if _voyage_client is None:
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai not installed. "
                "Install with: pip install voyageai"
            )
        log.info("Initializing Voyage AI client (cached for session)")
        _voyage_client = voyageai.Client(api_key=api_key)
    return _voyage_client


def _get_local_code_model(model_name: str) -> "SentenceTransformer":
    """Get or create cached local code embedding model.

    Args:
        model_name: Model identifier (e.g., jinaai/jina-embeddings-v2-base-code)

    Returns:
        Cached SentenceTransformer model instance
    """
    global _local_code_model
    if _local_code_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        log.info(f"Loading local code embedding model: {model_name} (cached for session)")
        _local_code_model = SentenceTransformer(model_name, trust_remote_code=True)
    return _local_code_model


def _embed_voyage(
    texts: list[str],
    model: str,
    api_key: str,
) -> list[list[float]]:
    """Generate embeddings via Voyage AI API.

    Args:
        texts: List of code strings to embed
        model: Voyage model name (e.g., voyage-code-3)
        api_key: Voyage API key

    Returns:
        List of embedding vectors
    """
    client = _get_voyage_client(api_key)
    # Voyage recommends input_type="document" for code chunks being indexed
    result = client.embed(texts, model=model, input_type="document")
    return result.embeddings


def _embed_code_local(texts: list[str], model_name: str) -> list[list[float]]:
    """Generate code embeddings via local sentence-transformers model.

    Args:
        texts: List of code strings to embed
        model_name: Model identifier

    Returns:
        List of embedding vectors
    """
    model = _get_local_code_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [e.tolist() for e in embeddings]


def _embed_openrouter(texts: list[str], model: str) -> list[list[float]]:
    """Generate embeddings via OpenRouter (LiteLLM passthrough).

    Args:
        texts: List of code strings to embed
        model: OpenRouter model path (e.g., openai/text-embedding-3-large)

    Returns:
        List of embedding vectors

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set
    """
    from litellm import embedding

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    # LiteLLM uses openrouter/ prefix for routing
    response = embedding(
        model=f"openrouter/{model}",
        input=texts,
        api_key=api_key,
    )
    return [d["embedding"] for d in response.data]


def embed_code_batch(
    texts: list[str],
    config: Config | None = None,
    batch_size: int = 128,
    expected_dim: int | None = None,
) -> EmbeddingResult:
    """Generate code-specific embeddings with provider fallback.

    Uses dedicated code embedding models optimized for source code understanding.
    Implements automatic fallback chain for resilience:

    Provider hierarchy:
    1. Voyage (if VOYAGE_API_KEY set) - best quality for code, 1024 dims
    2. OpenRouter (if OPENROUTER_API_KEY set) - good fallback, 3072 dims
    3. Local (always available) - free, no API needed, 768 dims

    IMPORTANT: Providers produce different embedding dimensions. If expected_dim
    is specified, the function will skip providers that produce incompatible
    dimensions rather than producing a dimension mismatch.

    Args:
        texts: List of code strings to embed
        config: Optional config override (uses global if not provided)
        batch_size: Max texts per API call (default 128, Voyage's limit)
        expected_dim: Expected embedding dimension (from existing index).
            If specified, fallback providers with incompatible dimensions
            are skipped. If None, any dimension is accepted.

    Returns:
        EmbeddingResult with embeddings and metadata (provider, model, timing)

    Raises:
        ValueError: If expected_dim is specified and no provider can produce
            compatible embeddings.

    Note:
        Logs every fallback decision with reason for observability.
        Use EmbeddingResult.provider to verify which backend was used.
    """
    from time import perf_counter

    cfg = config or _config
    if cfg is None:
        cfg = Config()

    if not texts:
        return EmbeddingResult(
            embeddings=[],
            provider="none",
            model="none",
            dimension=0,
            elapsed_ms=0.0,
        )

    provider = cfg.code_embedding_provider
    all_embeddings: list[list[float]] = []
    used_provider = provider
    used_model = ""
    start_time = perf_counter()

    # Provider dimension mapping (must match config.code_embedding_dim)
    PROVIDER_DIMS = {
        "voyage": 1024,      # voyage-code-3
        "openrouter": 3072,  # text-embedding-3-large
        "local": 768,        # jina-embeddings-v2-base-code
    }

    def _dimension_compatible(prov: str) -> bool:
        """Check if provider dimension is compatible with expected_dim."""
        if expected_dim is None:
            return True
        return PROVIDER_DIMS.get(prov, 0) == expected_dim

    # ==========================================================================
    # Try Voyage first (default, best quality for code)
    # ==========================================================================
    if provider == "voyage":
        if not _dimension_compatible("voyage"):
            log.debug(f"Skipping Voyage: dimension {PROVIDER_DIMS['voyage']} incompatible with expected {expected_dim}")
            provider = "openrouter"
        elif cfg.voyage_api_key:
            try:
                log.debug(f"Attempting Voyage embeddings: model={cfg.voyage_code_model}, batch_size={len(texts)}")
                with log_timing(f"Voyage embed {len(texts)} texts", log, level="debug"):
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        batch_embeddings = _embed_voyage(batch, cfg.voyage_code_model, cfg.voyage_api_key)
                        all_embeddings.extend(batch_embeddings)
                used_provider = "voyage"
                used_model = cfg.voyage_code_model
                log.info(f"Voyage embedding success: {len(texts)} texts → {len(all_embeddings)} embeddings, dim={len(all_embeddings[0])}")
            except Exception as e:
                log.warning(f"Voyage embedding failed: {e}. Falling back to OpenRouter.")
                provider = "openrouter"  # Trigger fallback
        else:
            log.debug("No VOYAGE_API_KEY configured, falling back to OpenRouter")
            provider = "openrouter"

    # ==========================================================================
    # Try OpenRouter (fallback #1)
    # ==========================================================================
    if provider == "openrouter" and not all_embeddings:
        if not _dimension_compatible("openrouter"):
            log.debug(f"Skipping OpenRouter: dimension {PROVIDER_DIMS['openrouter']} incompatible with expected {expected_dim}")
        else:
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                try:
                    log.debug(f"Attempting OpenRouter embeddings: model={cfg.openrouter_code_model}, batch_size={len(texts)}")
                    with log_timing(f"OpenRouter embed {len(texts)} texts", log, level="debug"):
                        for i in range(0, len(texts), batch_size):
                            batch = texts[i:i + batch_size]
                            batch_embeddings = _embed_openrouter(batch, cfg.openrouter_code_model)
                            all_embeddings.extend(batch_embeddings)
                    used_provider = "openrouter"
                    used_model = cfg.openrouter_code_model
                    log.info(f"OpenRouter embedding success: {len(texts)} texts → {len(all_embeddings)} embeddings, dim={len(all_embeddings[0])}")
                except Exception as e:
                    log.warning(f"OpenRouter embedding failed: {e}. Falling back to local.")
            else:
                log.debug("No OPENROUTER_API_KEY configured, falling back to local")

    # ==========================================================================
    # Local fallback (always available)
    # ==========================================================================
    if not all_embeddings:
        if not _dimension_compatible("local"):
            log.error(f"No compatible provider available: expected dim={expected_dim}, local provides {PROVIDER_DIMS['local']}")
            raise ValueError(
                f"No embedding provider available for expected dimension {expected_dim}. "
                f"Available: voyage={PROVIDER_DIMS['voyage']}, openrouter={PROVIDER_DIMS['openrouter']}, local={PROVIDER_DIMS['local']}"
            )
        log.debug(f"Using local code embeddings: model={cfg.code_local_model}, batch_size={len(texts)}")
        with log_timing(f"Local embed {len(texts)} texts", log, level="debug"):
            all_embeddings = _embed_code_local(texts, cfg.code_local_model)
        used_provider = "local"
        used_model = cfg.code_local_model
        log.info(f"Local embedding success: {len(texts)} texts → {len(all_embeddings)} embeddings, dim={len(all_embeddings[0])}")

    elapsed_ms = (perf_counter() - start_time) * 1000
    dimension = len(all_embeddings[0]) if all_embeddings else 0

    log.debug(f"Code embedding complete: provider={used_provider}, model={used_model}, dim={dimension}, elapsed={elapsed_ms:.1f}ms")

    return EmbeddingResult(
        embeddings=all_embeddings,
        provider=used_provider,
        model=used_model,
        dimension=dimension,
        elapsed_ms=elapsed_ms,
    )
