"""Backend configuration with dev/prod security modes."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SecurityMode(str, Enum):
    """Security mode determines default security settings.

    DEV: Relaxed settings for local development and testing
    PROD: Strict settings for production deployment (default)
    """
    DEV = "dev"
    PROD = "prod"


@dataclass
class BackendConfig:
    """Configuration for the backend API server.

    Security settings are determined by SIMPLEMEM_MODE (dev/prod).
    Individual settings can be overridden via environment variables.

    DEV mode:
        - Authentication: disabled
        - Project isolation: disabled
        - Arbitrary Cypher: allowed
        - Verbose errors: enabled
        - Host: 127.0.0.1 (localhost only)

    PROD mode (default):
        - Authentication: required
        - Project isolation: required
        - Arbitrary Cypher: blocked (whitelist only)
        - Verbose errors: disabled
        - Host: 0.0.0.0 (configurable)
    """

    # Security mode - determines defaults for other settings
    mode: SecurityMode = field(default=None)  # type: ignore[assignment]

    # Server settings
    host: str = field(default=None)  # type: ignore[assignment]
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8420")))

    # Data directory (for LanceDB, KuzuDB)
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SIMPLEMEM_DATA_DIR", Path.home() / ".simplemem_lite")
        )
    )

    # Authentication (for cloud mode)
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("SIMPLEMEM_API_KEY")
    )
    require_auth: bool = field(default=None)  # type: ignore[assignment]

    # Project isolation (required in cloud mode)
    require_project_id: bool = field(default=None)  # type: ignore[assignment]

    # Cypher query restrictions
    allow_arbitrary_cypher: bool = field(default=None)  # type: ignore[assignment]

    # Error verbosity
    verbose_errors: bool = field(default=None)  # type: ignore[assignment]

    # Graph backend
    graph_backend: str = field(
        default_factory=lambda: os.environ.get("SIMPLEMEM_GRAPH_BACKEND", "kuzu")
    )

    # LLM settings (for trace summarization, ask_memories)
    llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "SIMPLEMEM_LLM_MODEL", "gemini/gemini-2.0-flash-lite"
        )
    )

    # Decompression limits (always enforced)
    max_decompressed_size_mb: int = field(
        default_factory=lambda: int(os.environ.get("SIMPLEMEM_MAX_DECOMPRESS_MB", "100"))
    )

    def __post_init__(self):
        """Apply mode-based defaults and ensure data directory exists."""
        # Parse mode from environment if not set
        if self.mode is None:
            mode_str = os.environ.get("SIMPLEMEM_MODE", "prod").lower()
            try:
                self.mode = SecurityMode(mode_str)
            except ValueError:
                self.mode = SecurityMode.PROD

        # Apply mode-based defaults for unset values
        if self.mode == SecurityMode.DEV:
            self._apply_dev_defaults()
        else:
            self._apply_prod_defaults()

        # Ensure data directory exists
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _apply_dev_defaults(self) -> None:
        """Apply relaxed defaults for development mode."""
        if self.require_auth is None:
            self.require_auth = self._env_bool("SIMPLEMEM_REQUIRE_AUTH", False)
        if self.require_project_id is None:
            self.require_project_id = self._env_bool("SIMPLEMEM_REQUIRE_PROJECT_ID", False)
        if self.allow_arbitrary_cypher is None:
            self.allow_arbitrary_cypher = self._env_bool("SIMPLEMEM_ALLOW_ARBITRARY_CYPHER", True)
        if self.verbose_errors is None:
            self.verbose_errors = self._env_bool("SIMPLEMEM_VERBOSE_ERRORS", True)
        if self.host is None:
            self.host = os.environ.get("HOST", "127.0.0.1")

    def _apply_prod_defaults(self) -> None:
        """Apply strict defaults for production mode."""
        if self.require_auth is None:
            self.require_auth = self._env_bool("SIMPLEMEM_REQUIRE_AUTH", True)
        if self.require_project_id is None:
            self.require_project_id = self._env_bool("SIMPLEMEM_REQUIRE_PROJECT_ID", True)
        if self.allow_arbitrary_cypher is None:
            self.allow_arbitrary_cypher = self._env_bool("SIMPLEMEM_ALLOW_ARBITRARY_CYPHER", False)
        if self.verbose_errors is None:
            self.verbose_errors = self._env_bool("SIMPLEMEM_VERBOSE_ERRORS", False)
        if self.host is None:
            self.host = os.environ.get("HOST", "0.0.0.0")

    @staticmethod
    def _env_bool(key: str, default: bool) -> bool:
        """Parse boolean from environment variable."""
        val = os.environ.get(key, "").lower()
        if val in ("true", "1", "yes"):
            return True
        if val in ("false", "0", "no"):
            return False
        return default

    @property
    def is_dev_mode(self) -> bool:
        """Check if running in dev mode."""
        return self.mode == SecurityMode.DEV

    @property
    def max_decompressed_size_bytes(self) -> int:
        """Get max decompression size in bytes."""
        return self.max_decompressed_size_mb * 1024 * 1024


# Global config instance
_config: BackendConfig | None = None


def get_config() -> BackendConfig:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = BackendConfig()
    return _config


def reset_config() -> None:
    """Reset config for testing."""
    global _config
    _config = None
