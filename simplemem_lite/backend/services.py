"""Service layer for SimpleMem-Lite backend.

Provides lazy-initialized service instances for API endpoints.
Services are singletons that persist for the application lifetime.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from simplemem_lite.backend.config import get_config
from simplemem_lite.log_config import get_logger

if TYPE_CHECKING:
    from simplemem_lite.code_index import CodeIndexer
    from simplemem_lite.config import Config
    from simplemem_lite.db import DatabaseManager
    from simplemem_lite.job_manager import JobManager
    from simplemem_lite.memory import MemoryStore
    from simplemem_lite.traces import HierarchicalIndexer

log = get_logger("backend.services")


@lru_cache(maxsize=1)
def get_simplemem_config() -> "Config":
    """Get the SimpleMem Config instance (cached).

    Uses the backend config's data_dir for storage.
    """
    from simplemem_lite.config import Config

    backend_config = get_config()
    return Config(data_dir=str(backend_config.data_dir))


@lru_cache(maxsize=1)
def get_database_manager() -> "DatabaseManager":
    """Get the DatabaseManager instance (cached).

    Manages both LanceDB (vectors) and KuzuDB (graph).
    """
    from simplemem_lite.db import DatabaseManager
    from simplemem_lite.embeddings import init_embeddings

    config = get_simplemem_config()
    init_embeddings(config)
    log.info("Initializing DatabaseManager")
    return DatabaseManager(config)


@lru_cache(maxsize=1)
def get_memory_store() -> "MemoryStore":
    """Get the MemoryStore instance (cached).

    Provides memory storage with hybrid vector + graph search.
    """
    from simplemem_lite.memory import MemoryStore

    config = get_simplemem_config()
    log.info("Initializing MemoryStore")
    return MemoryStore(config)


@lru_cache(maxsize=1)
def get_code_indexer() -> "CodeIndexer":
    """Get the CodeIndexer instance (cached).

    Provides semantic code search capabilities.
    """
    from simplemem_lite.code_index import CodeIndexer

    db = get_database_manager()
    config = get_simplemem_config()
    log.info("Initializing CodeIndexer")
    return CodeIndexer(db, config)


@lru_cache(maxsize=1)
def get_hierarchical_indexer() -> "HierarchicalIndexer":
    """Get the HierarchicalIndexer instance (cached).

    Provides trace processing and summarization.
    """
    from simplemem_lite.traces import HierarchicalIndexer

    memory_store = get_memory_store()
    config = get_simplemem_config()
    log.info("Initializing HierarchicalIndexer")
    return HierarchicalIndexer(memory_store, config)


@lru_cache(maxsize=1)
def get_job_manager() -> "JobManager":
    """Get the JobManager instance (cached).

    Provides background job management and status tracking.
    """
    from simplemem_lite.job_manager import JobManager

    config = get_simplemem_config()
    log.info("Initializing JobManager")
    return JobManager(data_dir=config.data_dir)


def clear_service_caches() -> None:
    """Clear all service caches (for testing)."""
    get_simplemem_config.cache_clear()
    get_database_manager.cache_clear()
    get_memory_store.cache_clear()
    get_code_indexer.cache_clear()
    get_hierarchical_indexer.cache_clear()
    get_job_manager.cache_clear()
    log.info("Service caches cleared")


def shutdown_services() -> None:
    """Gracefully shutdown all services.

    Closes database connections and flushes pending writes.
    Critical for preventing LanceDB corruption on Fly.io auto-suspend.
    """
    log.info("Shutting down services...")

    # Close database manager if initialized
    # Check cache to see if it was ever created
    cache_info = get_database_manager.cache_info()
    if cache_info.hits > 0 or cache_info.currsize > 0:
        try:
            db_manager = get_database_manager()
            db_manager.close()
            log.info("Database manager closed")
        except Exception as e:
            log.error(f"Error closing database manager: {e}")

    # Clear caches after shutdown
    clear_service_caches()
    log.info("Services shutdown complete")
