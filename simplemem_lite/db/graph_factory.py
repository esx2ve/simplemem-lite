"""Graph backend factory with auto-detection and fallback.

Provides intelligent backend selection:
1. Try Memgraph first (preferred - native C++ engine, no SIGSEGV issues)
2. Try FalkorDB if Memgraph unavailable
3. Fall back to KuzuDB (embedded, works everywhere)

Environment variables for override:
- SIMPLEMEM_GRAPH_BACKEND: Force specific backend ('memgraph', 'falkordb', 'kuzu')
- SIMPLEMEM_KUZU_PATH: Override KuzuDB database path
- SIMPLEMEM_MEMGRAPH_HOST: Memgraph host (default: localhost)
- SIMPLEMEM_MEMGRAPH_PORT: Memgraph Bolt port (default: 7687)
- SIMPLEMEM_MEMGRAPH_USERNAME: Memgraph username for authentication
- SIMPLEMEM_MEMGRAPH_PASSWORD: Memgraph password for authentication
- SIMPLEMEM_FALKOR_HOST: FalkorDB host (default: localhost)
- SIMPLEMEM_FALKOR_PORT: FalkorDB port (default: 6379)
- SIMPLEMEM_FALKOR_PASSWORD: FalkorDB/Redis password for authentication
"""

import os
import time
from pathlib import Path
from typing import Literal

from simplemem_lite.db.graph_protocol import GraphBackend
from simplemem_lite.log_config import get_logger

log = get_logger("graph_factory")

# FalkorDB startup can take time when loading AOF data
# These settings control how long we wait for it to be ready
# 120s default handles large AOF files (consensus: GPT-5.2 + DeepSeek recommended 120-300s)
FALKORDB_READY_TIMEOUT = int(os.environ.get("SIMPLEMEM_FALKOR_READY_TIMEOUT", "120"))  # seconds
FALKORDB_READY_INTERVAL = float(os.environ.get("SIMPLEMEM_FALKOR_READY_INTERVAL", "0.5"))  # seconds
FALKORDB_MAX_INTERVAL = float(os.environ.get("SIMPLEMEM_FALKOR_MAX_INTERVAL", "5.0"))  # seconds (for backoff)

# Memgraph startup settings
MEMGRAPH_READY_TIMEOUT = int(os.environ.get("SIMPLEMEM_MEMGRAPH_READY_TIMEOUT", "60"))  # seconds
MEMGRAPH_READY_INTERVAL = float(os.environ.get("SIMPLEMEM_MEMGRAPH_READY_INTERVAL", "0.5"))  # seconds

# Type alias for backend names
BackendType = Literal["memgraph", "falkordb", "kuzu", "auto"]


def create_graph_backend(
    backend: BackendType = "auto",
    memgraph_host: str | None = None,
    memgraph_port: int | None = None,
    memgraph_username: str | None = None,
    memgraph_password: str | None = None,
    falkor_host: str | None = None,
    falkor_port: int | None = None,
    falkor_password: str | None = None,
    kuzu_path: str | Path | None = None,
) -> GraphBackend:
    """Create a graph backend with auto-detection and fallback.

    Selection order:
    1. Environment variable SIMPLEMEM_GRAPH_BACKEND if set
    2. Explicit backend parameter if not "auto"
    3. Auto-detection: Memgraph first, then FalkorDB, else KuzuDB

    Args:
        backend: Backend type - "memgraph", "falkordb", "kuzu", or "auto"
        memgraph_host: Memgraph host address (default: localhost, or SIMPLEMEM_MEMGRAPH_HOST)
        memgraph_port: Memgraph Bolt port (default: 7687, or SIMPLEMEM_MEMGRAPH_PORT)
        memgraph_username: Memgraph username (or SIMPLEMEM_MEMGRAPH_USERNAME)
        memgraph_password: Memgraph password (or SIMPLEMEM_MEMGRAPH_PASSWORD)
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
    if env_backend in ("memgraph", "falkordb", "kuzu"):
        backend = env_backend
        log.info(f"Using backend from environment: {backend}")

    # Override Memgraph settings from environment
    if memgraph_host is None:
        memgraph_host = os.environ.get("SIMPLEMEM_MEMGRAPH_HOST", "localhost")
    if memgraph_port is None:
        memgraph_port = int(os.environ.get("SIMPLEMEM_MEMGRAPH_PORT", "7687"))
    if memgraph_username is None:
        memgraph_username = os.environ.get("SIMPLEMEM_MEMGRAPH_USERNAME", "")
    if memgraph_password is None:
        memgraph_password = os.environ.get("SIMPLEMEM_MEMGRAPH_PASSWORD", "")

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

    if backend == "memgraph":
        return _create_memgraph(memgraph_host, memgraph_port, memgraph_username, memgraph_password)

    if backend == "falkordb":
        return _create_falkordb(falkor_host, falkor_port, falkor_password)

    if backend == "kuzu":
        return _create_kuzu(kuzu_path)

    # Auto-detection mode
    log.info("Auto-detecting graph backend...")

    # Try Memgraph first (preferred - no SIGSEGV issues)
    if _is_memgraph_available(memgraph_host, memgraph_port, memgraph_username, memgraph_password):
        log.info("Memgraph detected and healthy, using Memgraph backend")
        return _create_memgraph(memgraph_host, memgraph_port, memgraph_username, memgraph_password)

    # Try FalkorDB second
    if _is_falkordb_available(falkor_host, falkor_port, falkor_password):
        log.info("FalkorDB detected and healthy, using FalkorDB backend")
        return _create_falkordb(falkor_host, falkor_port, falkor_password)

    # Fall back to KuzuDB
    log.info("Memgraph and FalkorDB not available, falling back to KuzuDB")
    return _create_kuzu(kuzu_path)


def _wait_for_redis_ready(host: str, port: int, password: str | None = None) -> bool:
    """Wait for Redis/FalkorDB to finish loading and be ready.

    CRITICAL: Uses Redis-level PING command instead of graph queries.
    Graph queries during AOF loading can cause SIGSEGV crashes in FalkorDB.
    This function is safe to call during the loading phase.

    Uses exponential backoff with jitter to reduce load during long AOF replays.

    Args:
        host: Redis/FalkorDB host
        port: Redis/FalkorDB port
        password: Optional Redis password

    Returns:
        True if Redis is ready, False if timeout or unavailable
    """
    import redis

    start_time = time.time()
    last_error = None
    attempts = 0
    current_interval = FALKORDB_READY_INTERVAL

    while (time.time() - start_time) < FALKORDB_READY_TIMEOUT:
        attempts += 1
        try:
            # Use raw Redis client for PING - safe during loading
            r = redis.Redis(host=host, port=port, password=password, socket_timeout=5)
            r.ping()  # Returns True if ready, raises if loading

            elapsed = time.time() - start_time
            if attempts > 1:
                log.info(f"Redis ready after {elapsed:.1f}s ({attempts} attempts)")
            else:
                log.debug(f"Redis available at {host}:{port}")
            return True

        except redis.exceptions.BusyLoadingError:
            # Redis is loading AOF/RDB - this is expected, wait and retry
            if attempts == 1:
                log.info(f"Redis is loading data at {host}:{port}, waiting...")
            time.sleep(current_interval)
            # Exponential backoff with cap
            current_interval = min(current_interval * 1.5, FALKORDB_MAX_INTERVAL)
            continue

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Case-insensitive check for loading state (consensus fix)
            if "loading" in error_msg or "busy" in error_msg:
                if attempts == 1:
                    log.info(f"Redis is loading data at {host}:{port}, waiting...")
                time.sleep(current_interval)
                current_interval = min(current_interval * 1.5, FALKORDB_MAX_INTERVAL)
                continue

            # Connection errors - Redis not started yet
            if "connection refused" in error_msg or "connect" in error_msg:
                if attempts == 1:
                    log.debug(f"Waiting for Redis to start at {host}:{port}...")
                time.sleep(current_interval)
                current_interval = min(current_interval * 1.5, FALKORDB_MAX_INTERVAL)
                continue

            # Other errors - don't retry
            log.debug(f"Redis not available at {host}:{port}: {e}")
            return False

    # Timeout reached
    elapsed = time.time() - start_time
    log.warning(
        f"Redis not ready after {elapsed:.1f}s ({attempts} attempts): {last_error}"
    )
    return False


def _is_falkordb_available(host: str, port: int, password: str | None = None) -> bool:
    """Check if FalkorDB is available and ready to accept queries.

    Two-phase check (consensus recommendation):
    1. Wait for Redis to be ready (using safe PING command)
    2. Verify FalkorDB graph module is functional

    Args:
        host: FalkorDB host
        port: FalkorDB port
        password: Optional Redis password

    Returns:
        True if FalkorDB is reachable and ready
    """
    try:
        from falkordb import FalkorDB
    except ImportError:
        log.debug("FalkorDB package not installed")
        return False

    # Phase 1: Wait for Redis to be ready (safe during loading)
    if not _wait_for_redis_ready(host, port, password):
        return False

    # Phase 2: Verify FalkorDB module is functional (Redis is now ready)
    try:
        db = FalkorDB(host=host, port=port, password=password)
        graph = db.select_graph("simplemem_health_check")
        graph.query("RETURN 1")
        log.debug(f"FalkorDB module verified at {host}:{port}")
        return True

    except Exception as e:
        log.debug(f"FalkorDB module not available at {host}:{port}: {e}")
        return False


def _create_falkordb(host: str, port: int, password: str | None = None) -> GraphBackend:
    """Create FalkorDB backend with safe startup synchronization.

    Two-phase approach (consensus recommendation):
    1. Wait for Redis to be ready using PING (safe during AOF loading)
    2. Create backend and verify with health check

    This prevents SIGSEGV crashes that occur when graph queries are
    sent while FalkorDB is still loading AOF data.

    Args:
        host: FalkorDB host
        port: FalkorDB port
        password: Optional Redis password

    Returns:
        FalkorDBBackend instance

    Raises:
        RuntimeError: If connection fails after timeout
    """
    from simplemem_lite.db.falkor_backend import create_falkor_backend

    # Phase 1: Wait for Redis to be ready (safe during loading)
    # This is CRITICAL - graph queries during loading cause SIGSEGV
    if not _wait_for_redis_ready(host, port, password):
        raise RuntimeError(
            f"FalkorDB/Redis not ready at {host}:{port} after {FALKORDB_READY_TIMEOUT}s"
        )

    # Phase 2: Create backend (Redis is now ready)
    try:
        backend = create_falkor_backend(host, port, password)

        # Phase 3: Explicit health check after creation (handles lazy init)
        # FalkorDB client may be lazy - force a query to verify connection
        if hasattr(backend, 'health_check'):
            if not backend.health_check():
                raise RuntimeError("FalkorDB health check failed after backend creation")

        log.info(f"FalkorDB backend initialized and verified at {host}:{port}")
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


def _wait_for_memgraph_ready(
    host: str,
    port: int,
    username: str = "",
    password: str = "",
) -> bool:
    """Wait for Memgraph to be ready.

    Uses Bolt protocol connectivity check with exponential backoff.

    Args:
        host: Memgraph host
        port: Memgraph Bolt port
        username: Optional username
        password: Optional password

    Returns:
        True if Memgraph is ready, False if timeout
    """
    start_time = time.time()
    last_error = None
    attempts = 0
    current_interval = MEMGRAPH_READY_INTERVAL

    while (time.time() - start_time) < MEMGRAPH_READY_TIMEOUT:
        attempts += 1
        try:
            from neo4j import GraphDatabase

            uri = f"bolt://{host}:{port}"
            if username or password:
                driver = GraphDatabase.driver(uri, auth=(username, password))
            else:
                driver = GraphDatabase.driver(uri)

            driver.verify_connectivity()
            driver.close()

            elapsed = time.time() - start_time
            if attempts > 1:
                log.info(f"Memgraph ready after {elapsed:.1f}s ({attempts} attempts)")
            else:
                log.debug(f"Memgraph available at {host}:{port}")
            return True

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Connection errors - Memgraph not started yet
            if "connection refused" in error_msg or "connect" in error_msg:
                if attempts == 1:
                    log.debug(f"Waiting for Memgraph to start at {host}:{port}...")
                time.sleep(current_interval)
                current_interval = min(current_interval * 1.5, 5.0)
                continue

            # Other errors - don't retry
            log.debug(f"Memgraph not available at {host}:{port}: {e}")
            return False

    # Timeout reached
    elapsed = time.time() - start_time
    log.warning(
        f"Memgraph not ready after {elapsed:.1f}s ({attempts} attempts): {last_error}"
    )
    return False


def _is_memgraph_available(
    host: str,
    port: int,
    username: str = "",
    password: str = "",
    timeout: float = 2.0,
) -> bool:
    """Check if Memgraph is available and ready to accept queries.

    Uses socket-level pre-check before neo4j driver to avoid long hangs.

    Args:
        host: Memgraph host
        port: Memgraph Bolt port
        username: Optional username
        password: Optional password
        timeout: Connection timeout in seconds (default: 2.0)

    Returns:
        True if Memgraph is reachable and ready
    """
    import socket

    try:
        from neo4j import GraphDatabase
    except ImportError:
        log.debug("neo4j package not installed")
        return False

    # Quick socket-level check first (avoids neo4j driver's long timeout)
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
    except (socket.timeout, socket.error, OSError) as e:
        log.debug(f"Memgraph socket check failed at {host}:{port}: {e}")
        return False

    # Socket is open, now verify with neo4j driver
    try:
        uri = f"bolt://{host}:{port}"
        if username or password:
            driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                connection_timeout=timeout,
            )
        else:
            driver = GraphDatabase.driver(uri, connection_timeout=timeout)

        driver.verify_connectivity()

        # Verify with a simple query
        with driver.session() as session:
            session.run("RETURN 1").consume()

        driver.close()
        log.debug(f"Memgraph verified at {host}:{port}")
        return True

    except Exception as e:
        log.debug(f"Memgraph not available at {host}:{port}: {e}")
        return False


def _create_memgraph(
    host: str,
    port: int,
    username: str = "",
    password: str = "",
) -> GraphBackend:
    """Create Memgraph backend.

    Args:
        host: Memgraph host
        port: Memgraph Bolt port
        username: Optional username
        password: Optional password

    Returns:
        MemgraphBackend instance

    Raises:
        RuntimeError: If connection fails
    """
    from simplemem_lite.db.memgraph_backend import create_memgraph_backend

    # Wait for Memgraph to be ready
    if not _wait_for_memgraph_ready(host, port, username, password):
        raise RuntimeError(
            f"Memgraph not ready at {host}:{port} after {MEMGRAPH_READY_TIMEOUT}s"
        )

    try:
        backend = create_memgraph_backend(host, port, username, password)

        # Explicit health check after creation
        if hasattr(backend, 'health_check'):
            if not backend.health_check():
                raise RuntimeError("Memgraph health check failed after backend creation")

        log.info(f"Memgraph backend initialized and verified at {host}:{port}")
        return backend

    except Exception as e:
        log.error(f"Failed to create Memgraph backend: {e}")
        raise RuntimeError(f"Memgraph initialization failed: {e}") from e


def get_backend_info() -> dict:
    """Get information about available backends.

    Returns:
        Dict with availability status for each backend
    """
    info = {
        "memgraph": {
            "installed": False,
            "available": False,
            "error": None,
        },
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

    # Check Memgraph (preferred)
    try:
        import neo4j  # noqa: F401
        info["memgraph"]["installed"] = True
        info["memgraph"]["available"] = _is_memgraph_available("localhost", 7687)
    except ImportError as e:
        info["memgraph"]["error"] = str(e)

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

    # Determine which would be active (priority: memgraph > falkordb > kuzu)
    if info["memgraph"]["available"]:
        info["active"] = "memgraph"
    elif info["falkordb"]["available"]:
        info["active"] = "falkordb"
    elif info["kuzu"]["available"]:
        info["active"] = "kuzu"

    return info
