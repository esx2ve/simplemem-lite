"""Database management for SimpleMem Lite.

This package provides DatabaseManager which handles graph database (FalkorDB/KuzuDB)
and LanceDB (vectors) with two-phase commit for consistency.

Graph Backend Selection:
- FalkorDB: Preferred when Docker is available (full Cypher support, PageRank)
- KuzuDB: Embedded fallback for HPC/serverless environments (no Docker required)
- Auto-detection: Tries FalkorDB first, falls back to KuzuDB

Environment Variables:
- SIMPLEMEM_GRAPH_BACKEND: Force "falkordb" or "kuzu"
- SIMPLEMEM_KUZU_PATH: Override KuzuDB database path

Module Structure:
- manager.py: DatabaseManager class with all database operations
- graph_protocol.py: Abstract protocol for graph backends
- graph_factory.py: Backend selection and auto-detection
- falkor_backend.py: FalkorDB implementation
- kuzu_backend.py: KuzuDB implementation

Example:
    from simplemem_lite.db import DatabaseManager
    from simplemem_lite.config import Config

    db = DatabaseManager(Config())
    print(f"Using backend: {db.backend_name}")  # "falkordb" or "kuzu"
    db.add_memory_node(uuid="...", content="...", ...)

Backend Info:
    from simplemem_lite.db import get_backend_info
    info = get_backend_info()
    print(info)  # Shows which backends are available
"""

from simplemem_lite.db.manager import DatabaseManager
from simplemem_lite.db.graph_factory import create_graph_backend, get_backend_info
from simplemem_lite.db.graph_protocol import GraphBackend, QueryResult

__all__ = [
    "DatabaseManager",
    "GraphBackend",
    "QueryResult",
    "create_graph_backend",
    "get_backend_info",
]
