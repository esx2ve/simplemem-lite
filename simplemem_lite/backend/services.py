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


def clear_service_caches() -> None:
    """Clear all service caches (for testing)."""
    get_simplemem_config.cache_clear()
    get_database_manager.cache_clear()
    get_memory_store.cache_clear()
    get_code_indexer.cache_clear()
    get_hierarchical_indexer.cache_clear()
    log.info("Service caches cleared")
