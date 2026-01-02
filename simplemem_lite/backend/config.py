"""Backend configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BackendConfig:
    """Configuration for the backend API server."""

    # Server settings
    host: str = field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
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
    require_auth: bool = field(
        default_factory=lambda: os.environ.get("SIMPLEMEM_REQUIRE_AUTH", "").lower()
        == "true"
    )

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

    def __post_init__(self):
        """Ensure data directory exists."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: BackendConfig | None = None


def get_config() -> BackendConfig:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = BackendConfig()
    return _config
