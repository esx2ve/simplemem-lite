"""Graph backend factory with auto-detection and fallback.

Provides intelligent backend selection:
1. Try FalkorDB first (if Docker available)
2. Fall back to KuzuDB (embedded, works everywhere)

Environment variables for override:
- SIMPLEMEM_GRAPH_BACKEND: Force specific backend ('falkordb', 'kuzu')
- SIMPLEMEM_KUZU_PATH: Override KuzuDB database path
- SIMPLEMEM_FALKOR_HOST: FalkorDB host (default: localhost)
- SIMPLEMEM_FALKOR_PORT: FalkorDB port (default: 6379)
- SIMPLEMEM_FALKOR_PASSWORD: FalkorDB/Redis password for authentication
"""

import os
from pathlib import Path
from typing import Literal

from simplemem_lite.db.graph_protocol import GraphBackend
from simplemem_lite.log_config import get_logger

log = get_logger("graph_factory")

# Type alias for backend names
BackendType = Literal["falkordb", "kuzu", "auto"]


def create_graph_backend(
    backend: BackendType = "auto",
    falkor_host: str | None = None,
    falkor_port: int | None = None,
    falkor_password: str | None = None,
    kuzu_path: str | Path | None = None,
) -> GraphBackend:
    """Create a graph backend with auto-detection and fallback.

    Selection order:
    1. Environment variable SIMPLEMEM_GRAPH_BACKEND if set
    2. Explicit backend parameter if not "auto"
    3. Auto-detection: FalkorDB if available, else KuzuDB

    Args:
        backend: Backend type - "falkordb", "kuzu", or "auto"
        falkor_host: FalkorDB host address (default: localhost, or SIMPLEMEM_FALKOR_HOST)
        falkor_port: FalkorDB port (default: 6379, or SIMPLEMEM_FALKOR_PORT)
        falkor_password: FalkorDB password (or SIMPLEMEM_FALKOR_PASSWORD)
        kuzu_path: Path for KuzuDB database (default: ~/.simplemem/kuzu)

    Returns:
        Initialized GraphBackend instance

    Raises:
        RuntimeError: If no backend can be initialized
    """
    # Check environment override first
    env_backend = os.environ.get("SIMPLEMEM_GRAPH_BACKEND", "").lower()
    if env_backend in ("falkordb", "kuzu"):
        backend = env_backend
        log.info(f"Using backend from environment: {backend}")

    # Override FalkorDB settings from environment
    if falkor_host is None:
        falkor_host = os.environ.get("SIMPLEMEM_FALKOR_HOST", "localhost")
    if falkor_port is None:
        falkor_port = int(os.environ.get("SIMPLEMEM_FALKOR_PORT", "6379"))
    if falkor_password is None:
        falkor_password = os.environ.get("SIMPLEMEM_FALKOR_PASSWORD")

    # Override KuzuDB path from environment
    kuzu_path_override = os.environ.get("SIMPLEMEM_KUZU_PATH")
    if kuzu_path_override:
        kuzu_path = kuzu_path_override

    # Default KuzuDB path
    if kuzu_path is None:
        kuzu_path = Path.home() / ".simplemem" / "kuzu"

    # ══════════════════════════════════════════════════════════════════════════════
    # BACKEND SELECTION
    # ══════════════════════════════════════════════════════════════════════════════

    if backend == "falkordb":
        return _create_falkordb(falkor_host, falkor_port, falkor_password)

    if backend == "kuzu":
        return _create_kuzu(kuzu_path)

    # Auto-detection mode
    log.info("Auto-detecting graph backend...")

    # Try FalkorDB first
    if _is_falkordb_available(falkor_host, falkor_port, falkor_password):
        log.info("FalkorDB detected and healthy, using FalkorDB backend")
        return _create_falkordb(falkor_host, falkor_port, falkor_password)

    # Fall back to KuzuDB
    log.info("FalkorDB not available, falling back to KuzuDB")
    return _create_kuzu(kuzu_path)


def _is_falkordb_available(host: str, port: int, password: str | None = None) -> bool:
    """Check if FalkorDB is available.

    Args:
        host: FalkorDB host
        port: FalkorDB port
        password: Optional Redis password

    Returns:
        True if FalkorDB is reachable
    """
    try:
        from falkordb import FalkorDB

        db = FalkorDB(host=host, port=port, password=password)
        graph = db.select_graph("simplemem_health_check")
        graph.query("RETURN 1")
        log.debug(f"FalkorDB available at {host}:{port}")
        return True

    except ImportError:
        log.debug("FalkorDB package not installed")
        return False

    except Exception as e:
        log.debug(f"FalkorDB not available at {host}:{port}: {e}")
        return False


def _create_falkordb(host: str, port: int, password: str | None = None) -> GraphBackend:
    """Create FalkorDB backend.

    Args:
        host: FalkorDB host
        port: FalkorDB port
        password: Optional Redis password

    Returns:
        FalkorDBBackend instance

    Raises:
        RuntimeError: If connection fails
    """
    try:
        from simplemem_lite.db.falkor_backend import create_falkor_backend

        backend = create_falkor_backend(host, port, password)
        log.info(f"FalkorDB backend initialized at {host}:{port}")
        return backend

    except Exception as e:
        log.error(f"Failed to create FalkorDB backend: {e}")
        raise RuntimeError(f"FalkorDB initialization failed: {e}") from e


def _create_kuzu(db_path: str | Path) -> GraphBackend:
    """Create KuzuDB backend.

    Args:
        db_path: Path to KuzuDB database directory

    Returns:
        KuzuDBBackend instance

    Raises:
        RuntimeError: If initialization fails
    """
    try:
        from simplemem_lite.db.kuzu_backend import create_kuzu_backend

        backend = create_kuzu_backend(db_path)
        log.info(f"KuzuDB backend initialized at {db_path}")
        return backend

    except ImportError as e:
        log.error("KuzuDB package not installed. Install with: pip install kuzu")
        raise RuntimeError(
            "KuzuDB not installed. Run: pip install kuzu"
        ) from e

    except Exception as e:
        log.error(f"Failed to create KuzuDB backend: {e}")
        raise RuntimeError(f"KuzuDB initialization failed: {e}") from e


def get_backend_info() -> dict:
    """Get information about available backends.

    Returns:
        Dict with availability status for each backend
    """
    info = {
        "falkordb": {
            "installed": False,
            "available": False,
            "error": None,
        },
        "kuzu": {
            "installed": False,
            "available": False,
            "error": None,
        },
        "active": None,
        "env_override": os.environ.get("SIMPLEMEM_GRAPH_BACKEND"),
    }

    # Check FalkorDB
    try:
        import falkordb  # noqa: F401
        info["falkordb"]["installed"] = True
        info["falkordb"]["available"] = _is_falkordb_available("localhost", 6379)
    except ImportError as e:
        info["falkordb"]["error"] = str(e)

    # Check KuzuDB
    try:
        import kuzu  # noqa: F401
        info["kuzu"]["installed"] = True
        info["kuzu"]["available"] = True  # Embedded, always available if installed
    except ImportError as e:
        info["kuzu"]["error"] = str(e)

    # Determine which would be active
    if info["falkordb"]["available"]:
        info["active"] = "falkordb"
    elif info["kuzu"]["available"]:
        info["active"] = "kuzu"

    return info
