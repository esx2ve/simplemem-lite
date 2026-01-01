"""Database management for SimpleMem Lite.

This package provides DatabaseManager which handles FalkorDB (graph) and LanceDB (vectors)
with two-phase commit for consistency.

Module Structure:
- manager.py: DatabaseManager class with all database operations

The package is organized to support future modular splitting:
- Core: Connections, health checks, base operations (lines 1-230)
- Memory: Node CRUD, relationships, entity linking (lines 234-795)
- Code Search: Code indexing and semantic search (lines 800-1200)
- Graph: Graph traversal and path operations (lines 1200-1400)
- Analytics: PageRank, stats, insights (lines 1400+)

Example:
    from simplemem_lite.db import DatabaseManager
    from simplemem_lite.config import Config

    db = DatabaseManager(Config())
    db.add_memory_node(uuid="...", content="...", ...)
"""

from simplemem_lite.db.manager import DatabaseManager

__all__ = ["DatabaseManager"]
