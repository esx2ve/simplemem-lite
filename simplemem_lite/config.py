"""Configuration for SimpleMem Lite.

Simple dataclass-based configuration with sensible defaults.
Override via environment variables with SIMPLEMEM_LITE_ prefix.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from simplemem_lite.log_config import get_logger

log = get_logger("config")

# Load .env file if present
try:
    from dotenv import load_dotenv
    # Look for .env in package directory and parent
    _pkg_dir = Path(__file__).parent.parent
    _env_loaded = load_dotenv(_pkg_dir / ".env") or load_dotenv(_pkg_dir.parent / ".env")
    log.debug(f"Loaded .env file: {_env_loaded}")
except ImportError:
    log.debug("python-dotenv not installed, using environment variables directly")


def _get_env(key: str, default: str) -> str:
    """Get environment variable with SIMPLEMEM_LITE_ prefix."""
    return os.getenv(f"SIMPLEMEM_LITE_{key}", default)


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable."""
    val = os.getenv(f"SIMPLEMEM_LITE_{key}")
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


@dataclass
class Config:
    """SimpleMem Lite configuration.

    Attributes:
        data_dir: Directory for storing databases (default: ~/.simplemem_lite)
        embedding_model: LiteLLM model for embeddings (default: text-embedding-3-small)
        embedding_dim: Embedding dimension (default: 1536)
        summary_model: Cheap LLM for summarization (default: gemini/gemini-2.5-flash-lite)
        claude_traces_dir: Claude Code traces location (default: ~/.claude/projects)
        use_local_embeddings: Use sentence-transformers instead of API (default: False)
    """

    data_dir: Path = field(
        default_factory=lambda: Path(_get_env("DATA_DIR", str(Path.home() / ".simplemem_lite")))
    )
    embedding_model: str = field(
        default_factory=lambda: _get_env("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    summary_model: str = field(
        default_factory=lambda: _get_env("SUMMARY_MODEL", "gemini/gemini-2.5-flash-lite")
    )
    claude_traces_dir: Path = field(
        default_factory=lambda: Path(_get_env("CLAUDE_TRACES_DIR", str(Path.home() / ".claude" / "projects")))
    )
    use_local_embeddings: bool = field(
        default_factory=lambda: _get_env_bool("USE_LOCAL_EMBEDDINGS", False)
    )
    local_model: str = field(
        default_factory=lambda: _get_env("LOCAL_MODEL", "all-MiniLM-L6-v2")
    )

    # FalkorDB connection settings (use os.getenv directly to support both prefixed and unprefixed)
    falkor_host: str = field(
        default_factory=lambda: os.getenv("SIMPLEMEM_FALKOR_HOST", os.getenv("SIMPLEMEM_LITE_FALKOR_HOST", "localhost"))
    )
    falkor_port: int = field(
        default_factory=lambda: int(os.getenv("SIMPLEMEM_FALKOR_PORT", os.getenv("SIMPLEMEM_LITE_FALKOR_PORT", "6379")))
    )
    falkor_password: str | None = field(
        default_factory=lambda: os.getenv("SIMPLEMEM_FALKOR_PASSWORD", os.getenv("SIMPLEMEM_LITE_FALKOR_PASSWORD"))
    )

    # HTTP server settings (for hook communication)
    http_enabled: bool = field(
        default_factory=lambda: _get_env_bool("HTTP_ENABLED", True)
    )
    http_host: str = field(
        default_factory=lambda: _get_env("HTTP_HOST", "127.0.0.1")
    )
    http_port: int = field(
        default_factory=lambda: int(_get_env("HTTP_PORT", "0"))  # 0 = ephemeral
    )
    http_rate_limit: int = field(
        default_factory=lambda: int(_get_env("HTTP_RATE_LIMIT", "100"))  # requests per minute
    )

    # Code search settings
    code_index_enabled: bool = field(
        default_factory=lambda: _get_env_bool("CODE_INDEX_ENABLED", True)
    )
    code_index_patterns: str = field(
        default_factory=lambda: _get_env("CODE_INDEX_PATTERNS", "**/*.py,**/*.ts,**/*.js,**/*.tsx,**/*.jsx")
    )
    code_chunk_size: int = field(
        default_factory=lambda: int(_get_env("CODE_CHUNK_SIZE", "1200"))
    )
    code_chunk_overlap: int = field(
        default_factory=lambda: int(_get_env("CODE_CHUNK_OVERLAP", "150"))
    )

    # Embedding cache size (number of embeddings to cache in memory)
    embedding_cache_size: int = field(
        default_factory=lambda: int(_get_env("EMBEDDING_CACHE_SIZE", "1000"))
    )

    # Database limits (extracted magic numbers for maintainability)
    memory_content_max_size: int = field(
        default_factory=lambda: int(_get_env("MEMORY_CONTENT_MAX_SIZE", "5000"))
    )
    summary_max_size: int = field(
        default_factory=lambda: int(_get_env("SUMMARY_MAX_SIZE", "500"))
    )
    max_graph_hops: int = field(
        default_factory=lambda: int(_get_env("MAX_GRAPH_HOPS", "3"))
    )
    graph_path_limit: int = field(
        default_factory=lambda: int(_get_env("GRAPH_PATH_LIMIT", "100"))
    )
    cross_session_limit: int = field(
        default_factory=lambda: int(_get_env("CROSS_SESSION_LIMIT", "50"))
    )

    # Auto-indexer settings (P2: Active Session Handling)
    auto_index_enabled: bool = field(
        default_factory=lambda: _get_env_bool("AUTO_INDEX_ENABLED", False)
    )
    auto_index_poll_interval: int = field(
        default_factory=lambda: int(_get_env("AUTO_INDEX_POLL_INTERVAL", "120"))  # 2 minutes
    )
    auto_index_max_file_mb: int = field(
        default_factory=lambda: int(_get_env("AUTO_INDEX_MAX_FILE_MB", "5"))
    )
    auto_index_max_per_day: int = field(
        default_factory=lambda: int(_get_env("AUTO_INDEX_MAX_PER_DAY", "10"))
    )
    auto_index_enabled_at: float = field(
        default_factory=lambda: float(_get_env("AUTO_INDEX_ENABLED_AT", "0.0"))
    )
    auto_index_stability_cycles: int = field(
        default_factory=lambda: int(_get_env("AUTO_INDEX_STABILITY_CYCLES", "2"))
    )

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension based on model."""
        if self.use_local_embeddings:
            # all-MiniLM-L6-v2 produces 384-dim vectors
            return 384
        # Gemini models produce 768-dim vectors (LiteLLM dimensions param not supported)
        if "gemini" in self.embedding_model.lower():
            return 768
        # OpenAI text-embedding-3-small produces 1536-dim vectors
        return 1536

    def __post_init__(self):
        """Ensure paths are Path objects and directories exist."""
        log.trace("Initializing Config")

        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.claude_traces_dir, str):
            self.claude_traces_dir = Path(self.claude_traces_dir)

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Log all configuration
        log.debug(f"HOME={os.environ.get('HOME', 'NOT SET')}")
        log.debug(f"data_dir={self.data_dir}")
        log.debug(f"claude_traces_dir={self.claude_traces_dir}")
        log.debug(f"claude_traces_dir.exists={self.claude_traces_dir.exists()}")
        log.debug(f"embedding_model={self.embedding_model}")
        log.debug(f"summary_model={self.summary_model}")
        log.debug(f"use_local_embeddings={self.use_local_embeddings}")
        log.debug(f"embedding_dim={self.embedding_dim}")
        log.debug(f"falkor_host={self.falkor_host}, falkor_port={self.falkor_port}, falkor_password={'***' if self.falkor_password else 'None'}")
        log.debug(f"code_index_enabled={self.code_index_enabled}")
        log.debug(f"code_index_patterns={self.code_index_patterns}")
        log.info(f"Config initialized: data_dir={self.data_dir}, traces={self.claude_traces_dir}")

    @property
    def graph_dir(self) -> Path:
        """Directory for graph database (legacy, kept for compatibility)."""
        return self.data_dir / "graph"

    @property
    def vectors_dir(self) -> Path:
        """Directory for LanceDB vector database."""
        return self.data_dir / "vectors"

    @property
    def code_patterns_list(self) -> list[str]:
        """Get code patterns as a list."""
        return [p.strip() for p in self.code_index_patterns.split(",") if p.strip()]
