"""Database management for SimpleMem Lite.

Handles graph database (FalkorDB or KuzuDB) and LanceDB (vectors) with two-phase commit.

Graph Backend Selection:
- FalkorDB: Preferred when Docker is available (full Cypher support, PageRank)
- KuzuDB: Fallback for HPC/embedded environments (no Docker required)

The backend is auto-detected at startup, or can be forced via SIMPLEMEM_GRAPH_BACKEND env var.
"""

import gc
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from simplemem_lite.config import Config
from simplemem_lite.db.graph_factory import create_graph_backend, get_backend_info
from simplemem_lite.db.graph_protocol import GraphBackend
from simplemem_lite.log_config import get_logger

log = get_logger("db")


class DatabaseManager:
    """Manages graph database and LanceDB connections with consistency guarantees.

    Implements a simple two-phase commit pattern:
    1. Write to graph (FalkorDB/KuzuDB) first - source of truth
    2. Write to vectors (LanceDB) second
    3. Rollback graph if vector write fails

    Graph Backend Selection:
    - Auto-detects FalkorDB if Docker is available
    - Falls back to KuzuDB for HPC/embedded environments
    - Can be forced via SIMPLEMEM_GRAPH_BACKEND environment variable

    CONCURRENCY WARNING:
    This class uses a threading.Lock for write operations which only protects
    against concurrent access within a single process. Running multiple server
    instances against the same database is NOT supported and may cause data
    corruption. For multi-instance deployments, use a single server with the
    daemon mode, or implement external distributed locking.

    Node Types:
    - Memory: session_summary, chunk_summary, message
    - Entity: file, tool, error, concept (for cross-session linking)

    Relationship Types:
    - CONTAINS: session -> chunk, chunk -> message
    - CHILD_OF: reverse of contains
    - FOLLOWS: temporal sequence
    - REFERENCES: memory -> entity
    - SUPPORTS, MENTIONS, RELATES: semantic relationships
    """

    VECTOR_TABLE_NAME = "memories"
    CODE_TABLE_NAME = "code_chunks"
    GRAPH_NAME = "simplemem"

    def __init__(self, config: Config):
        """Initialize database connections.

        Args:
            config: SimpleMem Lite configuration
        """
        log.trace("DatabaseManager.__init__ starting")
        self.config = config
        # Use RLock (reentrant lock) to allow nested acquisition from same thread
        # This is needed because MemoryStore.store() acquires write_lock externally
        # and then calls add_memory_vector() which also needs the lock
        self._write_lock = threading.RLock()

        # Initialize graph backend with auto-detection
        # Falls back from FalkorDB → KuzuDB if Docker not available
        log.debug("Initializing graph backend with auto-detection")
        self._graph_backend: GraphBackend = create_graph_backend(
            backend="auto",
            falkor_host=self.config.falkor_host,
            falkor_port=self.config.falkor_port,
            falkor_password=self.config.falkor_password,
            kuzu_path=self.config.data_dir / "kuzu",
        )
        log.info(f"Graph backend initialized: {self._graph_backend.backend_name}")

        # For backward compatibility, expose graph attribute
        # FalkorDBBackend has a .graph property that returns the underlying graph
        if hasattr(self._graph_backend, "graph"):
            self.graph = self._graph_backend.graph
        else:
            # KuzuDB uses the backend directly
            self.graph = self._graph_backend

        # Initialize LanceDB (it creates its own directory)
        log.debug(f"Initializing LanceDB at {self.config.vectors_dir}")
        self.lance_db = lancedb.connect(str(self.config.vectors_dir))
        self._init_lance_table()
        log.info(f"LanceDB initialized at {self.config.vectors_dir}")

        # Track connection health
        self._graph_healthy = True
        self._last_health_check = 0

    def close(self) -> None:
        """Close database connections gracefully.

        Ensures all pending LanceDB writes are flushed to disk.
        Critical for preventing corruption on Fly.io auto-suspend.
        """
        log.info("Closing database connections...")

        # Close LanceDB - this flushes pending writes
        if hasattr(self, 'lance_db') and self.lance_db is not None:
            try:
                # LanceDB connections don't have explicit close(),
                # but we can optimize tables to ensure writes are flushed
                with self._write_lock:  # Ensure no concurrent ops during shutdown
                    if hasattr(self, 'lance_table') and self.lance_table is not None:
                        try:
                            self.lance_table.optimize()
                            log.debug("Optimized memories LanceDB table")
                        except Exception as e:
                            log.warning(f"Failed to optimize memories table: {e}")

                    if hasattr(self, 'code_table') and self.code_table is not None:
                        try:
                            self.code_table.optimize()
                            log.debug("Optimized code_chunks LanceDB table")
                        except Exception as e:
                            log.warning(f"Failed to optimize code_chunks table: {e}")

                log.info("LanceDB tables optimized and flushed")
            except Exception as e:
                log.error(f"Error closing LanceDB: {e}")

        # Close graph backend if it has a close method
        if hasattr(self, '_graph_backend') and hasattr(self._graph_backend, 'close'):
            try:
                self._graph_backend.close()
                log.info("Graph backend closed")
            except Exception as e:
                log.warning(f"Error closing graph backend: {e}")

        log.info("Database connections closed")

    @property
    def graph_backend(self) -> GraphBackend:
        """Get the underlying graph backend.

        Returns:
            The GraphBackend instance (FalkorDB or KuzuDB)
        """
        return self._graph_backend

    @property
    def backend_name(self) -> str:
        """Get the name of the active graph backend.

        Returns:
            "falkordb" or "kuzu"
        """
        return self._graph_backend.backend_name

    def health_check(self) -> dict[str, Any]:
        """Check health of database connections.

        Returns:
            Dict with health status for each database component
        """
        import time

        result = {
            "graph": {"healthy": False, "error": None, "backend": self.backend_name},
            "lancedb": {"healthy": False, "error": None},
            "timestamp": time.time(),
        }

        # Check graph backend (FalkorDB or KuzuDB)
        try:
            result["graph"]["healthy"] = self._graph_backend.health_check()
            self._graph_healthy = result["graph"]["healthy"]
        except Exception as e:
            result["graph"]["error"] = str(e)
            self._graph_healthy = False
            log.warning(f"Graph health check failed: {e}")

        # Check LanceDB
        try:
            # Verify table is accessible
            with self._write_lock:  # LanceDB not thread-safe for concurrent ops
                _ = self.lance_table.count_rows()
            result["lancedb"]["healthy"] = True
        except Exception as e:
            result["lancedb"]["error"] = str(e)
            log.warning(f"LanceDB health check failed: {e}")

        self._last_health_check = result["timestamp"]
        return result

    def is_healthy(self) -> bool:
        """Quick check if databases are healthy.

        Returns:
            True if all databases are healthy
        """
        health = self.health_check()
        return health["graph"]["healthy"] and health["lancedb"]["healthy"]

    def reconnect_graph(self) -> bool:
        """Attempt to reconnect to the graph backend.

        Returns:
            True if reconnection successful
        """
        # Only FalkorDB supports reconnection
        if hasattr(self._graph_backend, "reconnect"):
            try:
                result = self._graph_backend.reconnect()
                if result and hasattr(self._graph_backend, "graph"):
                    self.graph = self._graph_backend.graph
                self._graph_healthy = result
                return result
            except Exception as e:
                log.error(f"Graph reconnection failed: {e}")
                self._graph_healthy = False
                return False
        else:
            log.warning("Current backend does not support reconnection")
            return self._graph_backend.health_check()

    # Backward compatibility alias
    def reconnect_falkordb(self) -> bool:
        """Attempt to reconnect to graph backend (deprecated, use reconnect_graph)."""
        return self.reconnect_graph()

    def _init_lance_table(self) -> None:
        """Initialize LanceDB table if not exists."""
        log.trace("Checking LanceDB table")
        if self.VECTOR_TABLE_NAME not in self.lance_db.table_names():
            log.debug(f"Creating LanceDB table with embedding_dim={self.config.embedding_dim}")
            # Create table with schema
            schema = pa.schema([
                pa.field("uuid", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.config.embedding_dim)),
                pa.field("content", pa.string()),
                pa.field("type", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("metadata", pa.string()),  # JSON-serialized metadata
            ])

            # Create empty table with schema
            self.lance_db.create_table(
                self.VECTOR_TABLE_NAME,
                schema=schema,
            )
            log.info(f"LanceDB table '{self.VECTOR_TABLE_NAME}' created")
        else:
            log.debug(f"LanceDB table '{self.VECTOR_TABLE_NAME}' already exists")

        self.lance_table = self.lance_db.open_table(self.VECTOR_TABLE_NAME)

        # Initialize code search table if enabled
        if self.config.code_index_enabled:
            self._init_code_table()

    def _init_code_table(self) -> None:
        """Initialize LanceDB table for code chunks.

        Includes corruption recovery: if the table exists but is corrupted,
        it will be dropped and recreated automatically.
        """
        log.trace("Checking code_chunks table")

        schema = pa.schema([
            pa.field("uuid", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.config.embedding_dim)),
            pa.field("content", pa.string()),
            pa.field("filepath", pa.string()),
            pa.field("project_id", pa.string()),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
        ])

        if self.CODE_TABLE_NAME not in self.lance_db.table_names():
            log.debug(f"Creating code_chunks table with embedding_dim={self.config.embedding_dim}")
            self.lance_db.create_table(self.CODE_TABLE_NAME, schema=schema)
            log.info(f"LanceDB table '{self.CODE_TABLE_NAME}' created")
        else:
            log.debug(f"LanceDB table '{self.CODE_TABLE_NAME}' already exists")

        # Open table and verify it's not corrupted
        try:
            self.code_table = self.lance_db.open_table(self.CODE_TABLE_NAME)
            # Verify table is readable with a simple operation
            _ = self.code_table.count_rows()
        except Exception as e:
            log.warning(f"Code table corrupted, recreating: {e}")
            # Table is corrupted - drop and recreate
            try:
                self.lance_db.drop_table(self.CODE_TABLE_NAME)
            except Exception:
                pass  # May fail if partially corrupted
            self.lance_db.create_table(self.CODE_TABLE_NAME, schema=schema)
            self.code_table = self.lance_db.open_table(self.CODE_TABLE_NAME)
            log.info(f"LanceDB table '{self.CODE_TABLE_NAME}' recreated after corruption")

    # Keywords that indicate mutation queries (used for validation)
    MUTATION_KEYWORDS = frozenset({"CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DETACH"})

    def execute_graph(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a graph query with optional parameters.

        Args:
            query: Cypher query string
            params: Optional query parameters

        Returns:
            Query result
        """
        if params:
            return self.graph.query(query, params)
        return self.graph.query(query)

    def _serialize_graph_value(self, value: Any) -> Any:
        """Serialize graph objects to JSON-safe Python types.

        Handles Node, Edge, Path objects from both FalkorDB and neo4j/Memgraph.

        FalkorDB objects have:
        - Node: .properties (dict), .labels
        - Edge: .properties (dict), .relation
        - Path: .nodes() method, .edges() method

        Neo4j/Memgraph objects have:
        - Node: dict-like access (keys(), items()), .labels
        - Relationship: dict-like access, .type
        - Path: .nodes (property), .relationships (property)

        Args:
            value: Any value from a query result

        Returns:
            JSON-serializable Python value
        """
        # Get class info for explicit type detection (neo4j objects need this)
        value_type = type(value)
        type_name = value_type.__name__
        module_name = value_type.__module__ if hasattr(value_type, '__module__') else ''

        # ══════════════════════════════════════════════════════════════════════════════
        # EXPLICIT NEO4J/MEMGRAPH TYPE DETECTION (must come first!)
        # The hasattr-based checks below can fail for neo4j objects because:
        # 1. neo4j 5.x removed .properties attribute (now dict-like only)
        # 2. Mapping ABC inheritance may not be detected reliably by hasattr
        # ══════════════════════════════════════════════════════════════════════════════

        if module_name.startswith('neo4j.graph'):
            if type_name == 'Node':
                # Neo4j Node: dict-like with .labels (frozenset)
                return {
                    "_type": "node",
                    "labels": list(value.labels) if hasattr(value, 'labels') and value.labels else [],
                    "properties": dict(value),  # Node is dict-like in neo4j 5.x
                }
            elif type_name == 'Relationship':
                # Neo4j Relationship: dict-like with .type (string)
                return {
                    "_type": "edge",
                    "relation": value.type if hasattr(value, 'type') else "RELATED",
                    "properties": dict(value),  # Relationship is dict-like
                }
            elif type_name == 'Path':
                # Neo4j Path: has .nodes and .relationships properties
                return {
                    "_type": "path",
                    "nodes": [self._serialize_graph_value(n) for n in value.nodes],
                    "edges": [self._serialize_graph_value(r) for r in value.relationships],
                }

        # ══════════════════════════════════════════════════════════════════════════════
        # FALLBACK: HASATTR-BASED DETECTION FOR FALKORDB AND OTHER BACKENDS
        # ══════════════════════════════════════════════════════════════════════════════

        # Check for FalkorDB Node (has .properties attribute)
        if hasattr(value, 'properties') and hasattr(value, 'labels'):
            return {
                "_type": "node",
                "labels": list(value.labels) if value.labels else [],
                "properties": dict(value.properties) if value.properties else {},
            }

        # Check for Neo4j/Memgraph Node (dict-like with .labels, no .properties)
        # This is a fallback for neo4j versions not caught by module check
        if hasattr(value, 'labels') and hasattr(value, 'keys') and not hasattr(value, 'properties'):
            return {
                "_type": "node",
                "labels": list(value.labels) if value.labels else [],
                "properties": dict(value),  # neo4j Node is dict-like
            }

        # Check for FalkorDB Edge (has .properties and .relation)
        if hasattr(value, 'properties') and hasattr(value, 'relation'):
            return {
                "_type": "edge",
                "relation": value.relation,
                "properties": dict(value.properties) if value.properties else {},
            }

        # Check for Neo4j/Memgraph Relationship (dict-like with .type, no .relation)
        # This is a fallback for neo4j versions not caught by module check
        if hasattr(value, 'type') and hasattr(value, 'keys') and not hasattr(value, 'relation'):
            return {
                "_type": "edge",
                "relation": value.type,  # neo4j uses .type instead of .relation
                "properties": dict(value),  # neo4j Relationship is dict-like
            }

        # Check for FalkorDB Path (has .nodes() and .edges() methods)
        if hasattr(value, 'nodes') and hasattr(value, 'edges') and callable(getattr(value, 'nodes', None)):
            return {
                "_type": "path",
                "nodes": [self._serialize_graph_value(n) for n in value.nodes()],
                "edges": [self._serialize_graph_value(e) for e in value.edges()],
            }

        # Check for Neo4j/Memgraph Path (has .nodes and .relationships properties)
        # This is a fallback for neo4j versions not caught by module check
        if hasattr(value, 'nodes') and hasattr(value, 'relationships') and not callable(getattr(value, 'nodes', None)):
            return {
                "_type": "path",
                "nodes": [self._serialize_graph_value(n) for n in value.nodes],
                "edges": [self._serialize_graph_value(e) for e in value.relationships],
            }

        # Handle lists recursively
        if isinstance(value, list):
            return [self._serialize_graph_value(v) for v in value]

        # Handle dicts recursively
        if isinstance(value, dict):
            return {k: self._serialize_graph_value(v) for k, v in value.items()}

        # Primitive types pass through
        return value

    def execute_validated_cypher(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        timeout_ms: int = 5000,
        max_results: int = 100,
        allow_mutations: bool = False,
    ) -> dict[str, Any]:
        """Execute Cypher with validation and resource limits.

        Provides safe access to arbitrary Cypher queries with:
        - Mutation blocking (default): Rejects CREATE, MERGE, DELETE, SET, REMOVE
        - LIMIT injection: Adds LIMIT if not present
        - Result truncation: Caps output size

        Args:
            query: Cypher query string
            params: Optional query parameters (prevents injection)
            timeout_ms: Query timeout in milliseconds (default: 5000)
            max_results: Maximum results to return (default: 100)
            allow_mutations: If False, blocks mutation queries (default: False)

        Returns:
            {results: [...], truncated: bool, execution_time_ms: float, row_count: int}

        Raises:
            ValueError: If mutation detected and allow_mutations=False
        """
        import re
        import time

        log.info(f"execute_validated_cypher: allow_mutations={allow_mutations}, timeout={timeout_ms}ms")
        log.debug(f"Query preview: {query[:100]}...")

        # Security: Check for mutation keywords
        if not allow_mutations:
            query_upper = query.upper()
            for keyword in self.MUTATION_KEYWORDS:
                # Match keyword as a word boundary to avoid false positives
                # e.g., "DELETED" shouldn't match "DELETE"
                if re.search(rf'\b{keyword}\b', query_upper):
                    log.warning(f"Mutation keyword '{keyword}' detected in query, rejecting")
                    raise ValueError(
                        f"Mutation queries not allowed (found '{keyword}'). "
                        "Use allow_mutations=True for admin access or use dedicated tools "
                        "(store_memory, relate_memories) for data modifications."
                    )

        # Inject LIMIT if not present (match LIMIT <number> or LIMIT $param)
        query_stripped = query.strip().rstrip(';')
        if not re.search(r'\bLIMIT\s+(\d+|\$\w+)', query_upper):
            query_stripped = f"{query_stripped} LIMIT {max_results}"
            log.debug(f"Injected LIMIT {max_results}")

        # Execute with timing
        start_time = time.time()
        try:
            if params:
                result = self.graph.query(query_stripped, params)
            else:
                result = self.graph.query(query_stripped)
            execution_time_ms = (time.time() - start_time) * 1000
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            log.error(f"Cypher execution failed after {execution_time_ms:.1f}ms: {e}")
            raise

        # Process results
        rows = []
        if result.result_set:
            # Get column names from header if available
            header = result.header if hasattr(result, 'header') else None
            column_names = [col[1] if isinstance(col, tuple) else str(col) for col in header] if header else None

            for row in result.result_set:
                # Serialize FalkorDB objects (Node, Edge, Path) to JSON-safe dicts
                serialized_row = [self._serialize_graph_value(val) for val in row]

                if column_names and len(column_names) == len(serialized_row):
                    rows.append(dict(zip(column_names, serialized_row)))
                else:
                    # Fallback to positional indexing
                    rows.append({f"col_{i}": val for i, val in enumerate(serialized_row)})

        truncated = len(rows) >= max_results
        log.info(f"Cypher executed: {len(rows)} rows in {execution_time_ms:.1f}ms, truncated={truncated}")

        return {
            "results": rows,
            "truncated": truncated,
            "execution_time_ms": round(execution_time_ms, 2),
            "row_count": len(rows),
        }

    def get_schema(self) -> dict[str, Any]:
        """Get the graph schema for query generation.

        Returns a complete schema including:
        - node_labels: Dict of labels with their properties and indexes
        - relationship_types: List of relationship type names
        - common_queries: Example Cypher templates for common operations

        Returns:
            Complete schema dict for AI query generation
        """
        log.info("Getting graph schema")

        # Hardcoded schema based on codebase knowledge
        # This is more reliable than dynamic introspection and provides
        # better documentation for the AI agent
        schema = {
            "node_labels": {
                "Memory": {
                    "description": "Stored memories from sessions, facts, and learnings",
                    "properties": {
                        "uuid": "string - Unique identifier",
                        "content": "string - Memory text content",
                        "type": "string - Memory type: fact, session_summary, chunk_summary, message, lesson_learned",
                        "source": "string - Source: user, claude_trace, extracted",
                        "session_id": "string - Associated session UUID (optional)",
                        "created_at": "integer - Unix timestamp",
                    },
                    "indexes": ["uuid", "type", "session_id"],
                },
                "Entity": {
                    "description": "Extracted entities for cross-session linking",
                    "properties": {
                        "name": "string - Canonicalized entity name",
                        "type": "string - Entity type: file, tool, error, command, concept",
                        "version": "integer - Version number (incremented on MODIFIES)",
                        "created_at": "integer - Unix timestamp",
                        "last_modified": "integer - Last modification timestamp",
                    },
                    "indexes": ["name", "type"],
                },
                "CodeChunk": {
                    "description": "Indexed code snippets from project files",
                    "properties": {
                        "uuid": "string - Unique identifier",
                        "filepath": "string - Relative file path",
                        "project_id": "string - Canonical project identifier (e.g., git:github.com/user/repo)",
                        "start_line": "integer - Starting line number",
                        "end_line": "integer - Ending line number",
                        "created_at": "integer - Unix timestamp",
                    },
                    "indexes": ["uuid", "filepath"],
                },
                "Project": {
                    "description": "Tracked project roots for session scoping",
                    "properties": {
                        "id": "string - Project identifier (derived from path)",
                        "path": "string - Original project path",
                        "session_count": "integer - Number of sessions",
                        "created_at": "integer - Unix timestamp",
                        "last_updated": "integer - Last update timestamp",
                    },
                    "indexes": ["id"],
                },
                "ProjectIndex": {
                    "description": "Code index metadata for staleness detection",
                    "properties": {
                        "project_id": "string - Canonical project identifier (e.g., git:github.com/user/repo)",
                        "last_commit_hash": "string - Git commit hash when indexed",
                        "indexed_at": "integer - Unix timestamp of last index",
                        "file_count": "integer - Number of files indexed",
                        "chunk_count": "integer - Number of chunks created",
                    },
                    "indexes": ["project_id"],
                },
                "Goal": {
                    "description": "User objectives for intent-based retrieval",
                    "properties": {
                        "id": "string - Goal identifier",
                        "intent": "string - User's objective description",
                        "session_id": "string - Associated session UUID",
                        "status": "string - Goal status: active, completed, abandoned",
                        "created_at": "integer - Unix timestamp",
                    },
                    "indexes": ["id"],
                },
            },
            "relationship_types": {
                "RELATES_TO": {
                    "description": "Generic semantic relationship between memories",
                    "from": "Memory",
                    "to": "Memory",
                    "properties": ["relation_type", "weight"],
                },
                "CONTAINS": {
                    "description": "Hierarchical containment (session -> chunk -> message)",
                    "from": "Memory",
                    "to": "Memory",
                    "properties": [],
                },
                "CHILD_OF": {
                    "description": "Reverse of CONTAINS",
                    "from": "Memory",
                    "to": "Memory",
                    "properties": [],
                },
                "FOLLOWS": {
                    "description": "Temporal sequence between memories",
                    "from": "Memory",
                    "to": "Memory",
                    "properties": [],
                },
                "READS": {
                    "description": "Memory reads an entity (file, etc.)",
                    "from": "Memory",
                    "to": "Entity",
                    "properties": ["timestamp", "implicit"],
                },
                "MODIFIES": {
                    "description": "Memory modifies an entity",
                    "from": "Memory",
                    "to": "Entity",
                    "properties": ["timestamp", "change_summary"],
                },
                "EXECUTES": {
                    "description": "Memory executes a tool/command",
                    "from": "Memory",
                    "to": "Entity",
                    "properties": ["timestamp"],
                },
                "TRIGGERED": {
                    "description": "Memory triggered an error",
                    "from": "Memory",
                    "to": "Entity",
                    "properties": ["timestamp"],
                },
                "REFERENCES": {
                    "description": "Generic reference to an entity",
                    "from": "Memory|CodeChunk",
                    "to": "Entity",
                    "properties": ["relation", "created_at"],
                },
                "BELONGS_TO": {
                    "description": "Session summary belongs to a project",
                    "from": "Memory",
                    "to": "Project",
                    "properties": [],
                },
                "HAS_GOAL": {
                    "description": "Session summary has a goal",
                    "from": "Memory",
                    "to": "Goal",
                    "properties": [],
                },
                "ACHIEVES": {
                    "description": "Memory contributes to achieving a goal",
                    "from": "Memory",
                    "to": "Goal",
                    "properties": [],
                },
                "SUPERSEDES": {
                    "description": "Newer memory supersedes (replaces/updates) older memory",
                    "from": "Memory",
                    "to": "Memory",
                    "properties": ["confidence", "supersession_type", "created_at"],
                },
                "MERGED_INTO": {
                    "description": "Memory was merged into another (soft delete marker)",
                    "from": "Memory",
                    "to": "Memory",
                    "properties": ["merged_at"],
                },
            },
            "common_queries": [
                {
                    "name": "recent_memories",
                    "description": "Get recent memories",
                    "cypher": "MATCH (m:Memory) RETURN m.uuid, m.content, m.type, m.created_at ORDER BY m.created_at DESC LIMIT $limit",
                    "params": {"limit": 10},
                },
                {
                    "name": "entity_frequency",
                    "description": "Find most referenced entities",
                    "cypher": "MATCH (e:Entity)<-[r]-(m:Memory) RETURN e.name, e.type, count(r) as refs ORDER BY refs DESC LIMIT $limit",
                    "params": {"limit": 20},
                },
                {
                    "name": "cross_session_entities",
                    "description": "Find entities appearing in multiple sessions",
                    "cypher": "MATCH (m:Memory)-[r]->(e:Entity) WHERE m.session_id IS NOT NULL WITH e, collect(DISTINCT m.session_id) AS sessions WHERE size(sessions) >= $min_sessions RETURN e.name, e.type, size(sessions) as session_count ORDER BY session_count DESC",
                    "params": {"min_sessions": 2},
                },
                {
                    "name": "file_history",
                    "description": "Get history of a specific file",
                    "cypher": "MATCH (m:Memory)-[r:READS|MODIFIES]->(e:Entity {name: $filename, type: 'file'}) RETURN m.uuid, m.content, type(r) as action, r.timestamp ORDER BY r.timestamp DESC LIMIT $limit",
                    "params": {"filename": "example.py", "limit": 20},
                },
                {
                    "name": "error_solutions",
                    "description": "Find memories related to an error type",
                    "cypher": "MATCH (m:Memory)-[:TRIGGERED]->(e:Entity {type: 'error'}) WHERE e.name CONTAINS $error_pattern RETURN m.uuid, m.content, e.name as error, m.session_id LIMIT $limit",
                    "params": {"error_pattern": "error:", "limit": 10},
                },
                {
                    "name": "session_summary",
                    "description": "Get session with its chunks",
                    "cypher": "MATCH (s:Memory {type: 'session_summary', session_id: $session_id})-[:CONTAINS]->(c:Memory {type: 'chunk_summary'}) RETURN s.uuid, s.content, collect({uuid: c.uuid, content: c.content}) as chunks",
                    "params": {"session_id": "example-uuid"},
                },
                {
                    "name": "shortest_path",
                    "description": "Find shortest path between two entities",
                    "cypher": "MATCH path = shortestPath((a:Entity {name: $from_entity})-[*..5]-(b:Entity {name: $to_entity})) RETURN path",
                    "params": {"from_entity": "redis", "to_entity": "auth"},
                },
                {
                    "name": "graph_stats",
                    "description": "Get graph statistics",
                    "cypher": "MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC",
                    "params": {},
                },
                {
                    "name": "session_goals",
                    "description": "Get goals linked to sessions, optionally filtered by session_id or goal_id",
                    "cypher": """
                        MATCH (s:Memory {type: 'session_summary'})-[:HAS_GOAL]->(g:Goal)
                        WHERE ($session_id IS NULL OR s.session_id = $session_id)
                          AND ($goal_id IS NULL OR g.id = $goal_id)
                        RETURN g.id as goal_id, g.intent, g.status, s.session_id, s.uuid as session_uuid
                        ORDER BY s.created_at DESC
                        LIMIT $limit
                    """,
                    "params": {"session_id": None, "goal_id": None, "limit": 10},
                },
                {
                    "name": "all_goals",
                    "description": "Get all goals across all sessions",
                    "cypher": """
                        MATCH (g:Goal)
                        OPTIONAL MATCH (s:Memory)-[:HAS_GOAL]->(g)
                        RETURN g.id as goal_id, g.intent, g.status, g.created_at, s.session_id
                        ORDER BY g.created_at DESC
                        LIMIT $limit
                    """,
                    "params": {"limit": 20},
                },
            ],
        }

        log.debug(f"Schema returned: {len(schema['node_labels'])} labels, {len(schema['relationship_types'])} rel types")
        return schema

    def add_memory_node(
        self,
        uuid: str,
        content: str,
        mem_type: str,
        source: str,
        session_id: str | None,
        created_at: int,
        project_id: str | None = None,
    ) -> None:
        """Add a memory node to the graph.

        Args:
            uuid: Unique identifier
            content: Memory content
            mem_type: Memory type (fact, session_summary, chunk_summary, message)
            source: Source of memory (claude_trace, user, extracted)
            session_id: Optional session identifier
            created_at: Unix timestamp
            project_id: Optional project identifier for isolation
        """
        log.trace(f"Adding memory node: uuid={uuid[:8]}..., type={mem_type}, project={project_id}")
        # Parameterized queries handle escaping automatically
        self.graph.query(
            """
            CREATE (m:Memory {
                uuid: $uuid,
                content: $content,
                type: $type,
                source: $source,
                session_id: $session_id,
                created_at: $created_at,
                project_id: $project_id
            })
            """,
            {
                "uuid": uuid,
                "content": content[:self.config.memory_content_max_size],
                "type": mem_type,
                "source": source,
                "session_id": session_id or "",
                "created_at": created_at,
                "project_id": project_id or "",
            },
        )

    def add_entity_node(
        self,
        name: str,
        entity_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add or get an Entity node (for cross-session linking).

        Args:
            name: Entity name (e.g., "src/main.py", "Read", "ImportError")
            entity_type: Entity type (file, tool, error, concept)
            metadata: Optional additional metadata

        Returns:
            Entity name (used as identifier)
        """
        log.trace(f"Adding/getting entity: {entity_type}:{name}")
        # MERGE creates if not exists, returns existing if exists
        self.graph.query(
            """
            MERGE (e:Entity {name: $name, type: $type})
            ON CREATE SET e.created_at = timestamp()
            """,
            {
                "name": name,
                "type": entity_type,
            },
        )
        return name

    def add_memory_vector(
        self,
        uuid: str,
        vector: list[float],
        content: str,
        mem_type: str,
        session_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a memory vector to LanceDB.

        Args:
            uuid: Unique identifier (foreign key to graph)
            vector: Embedding vector
            content: Memory content (for retrieval)
            mem_type: Memory type
            session_id: Optional session identifier
            metadata: Additional metadata (stored as JSON)
        """
        import json
        with self._write_lock:  # LanceDB not thread-safe for concurrent writes
            self.lance_table.add([
                {
                    "uuid": uuid,
                    "vector": vector,
                    "content": content,
                    "type": mem_type,
                    "session_id": session_id or "",
                    "metadata": json.dumps(metadata or {}),
                }
            ])

    def add_relationship(
        self,
        from_uuid: str,
        to_uuid: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        """Add a relationship between two memories.

        Args:
            from_uuid: Source memory UUID
            to_uuid: Target memory UUID
            relation_type: Type of relationship
            weight: Relationship weight (default: 1.0)
        """
        self.graph.query(
            """
            MATCH (from:Memory {uuid: $from_uuid}), (to:Memory {uuid: $to_uuid})
            CREATE (from)-[:RELATES_TO {relation_type: $rel_type, weight: $weight}]->(to)
            """,
            {
                "from_uuid": from_uuid,
                "to_uuid": to_uuid,
                "rel_type": relation_type,
                "weight": weight,
            },
        )

    def would_create_supersession_cycle(
        self,
        newer_uuid: str,
        older_uuid: str,
        max_depth: int = 10,
    ) -> bool:
        """Check if creating a SUPERSEDES edge would create a cycle.

        DAG enforcement for supersession graph: prevents situations like
        A supersedes B supersedes C supersedes A (cycle).

        Args:
            newer_uuid: UUID of the newer/superseding memory
            older_uuid: UUID of the older/superseded memory
            max_depth: Maximum traversal depth to check (default: 10)

        Returns:
            True if creating the edge would create a cycle, False otherwise
        """
        # Check if older_uuid can reach newer_uuid via SUPERSEDES edges
        # If so, creating newer->older would complete a cycle
        # Note: Cypher doesn't support parameters for path length bounds,
        # so we inject max_depth directly (safe: typed as int)
        result = self.graph.query(
            f"""
            MATCH path = (older:Memory {{uuid: $older_uuid}})-[:SUPERSEDES*1..{max_depth}]->(newer:Memory {{uuid: $newer_uuid}})
            RETURN count(path) > 0 AS would_cycle
            """,
            {
                "newer_uuid": newer_uuid,
                "older_uuid": older_uuid,
            },
        )

        if result.result_set:
            would_cycle = result.result_set[0][0]
            if would_cycle:
                log.warning(
                    f"Supersession would create cycle: {newer_uuid[:8]}... -> {older_uuid[:8]}..."
                )
            return bool(would_cycle)
        return False

    def add_supersession(
        self,
        newer_uuid: str,
        older_uuid: str,
        confidence: float,
        supersession_type: str = "full_replace",
        reason: str | None = None,
    ) -> bool:
        """Mark that a newer memory supersedes an older one.

        Used during consolidation when a newer memory provides updated
        information that replaces an older memory.

        Enforces DAG constraint: will not create the edge if it would
        result in a cycle in the supersession graph.

        Args:
            newer_uuid: UUID of the newer/superseding memory
            older_uuid: UUID of the older/superseded memory
            confidence: Confidence score (0.0-1.0) from LLM classifier
            supersession_type: "full_replace" or "partial_update"
            reason: Optional explanation of why this supersession occurred

        Returns:
            True if edge was created, False if blocked (cycle or same node)
        """
        import time

        # Self-supersession is not allowed
        if newer_uuid == older_uuid:
            log.warning(f"Cannot supersede self: {newer_uuid[:8]}...")
            return False

        # DAG enforcement: check for cycles
        if self.would_create_supersession_cycle(newer_uuid, older_uuid):
            log.warning(
                f"Supersession blocked (would create cycle): "
                f"{newer_uuid[:8]}... -> {older_uuid[:8]}..."
            )
            return False

        self.graph.query(
            """
            MATCH (newer:Memory {uuid: $newer_uuid})
            MATCH (older:Memory {uuid: $older_uuid})
            MERGE (newer)-[r:SUPERSEDES]->(older)
            ON CREATE SET r.confidence = $confidence,
                          r.supersession_type = $type,
                          r.reason = $reason,
                          r.created_at = $now
            ON MATCH SET r.confidence = $confidence,
                         r.supersession_type = $type,
                         r.reason = $reason
            """,
            {
                "newer_uuid": newer_uuid,
                "older_uuid": older_uuid,
                "confidence": confidence,
                "type": supersession_type,
                "reason": reason or "",
                "now": int(time.time()),
            },
        )

        log.info(
            f"Supersession created: {newer_uuid[:8]}... -> {older_uuid[:8]}... "
            f"(confidence={confidence:.2f})"
        )
        return True

    def mark_merged(
        self,
        source_uuid: str,
        target_uuid: str,
    ) -> None:
        """Mark a memory as merged into another (soft delete).

        The source memory is kept but marked with MERGED_INTO relationship.
        This preserves history while indicating the memory is superseded.

        Args:
            source_uuid: UUID of memory that was merged (will be marked)
            target_uuid: UUID of memory it was merged into
        """
        import time
        self.graph.query(
            """
            MATCH (source:Memory {uuid: $source_uuid})
            MATCH (target:Memory {uuid: $target_uuid})
            MERGE (source)-[r:MERGED_INTO]->(target)
            ON CREATE SET r.merged_at = $now
            SET source.merged_into = $target_uuid,
                source.merged_at = $now
            """,
            {
                "source_uuid": source_uuid,
                "target_uuid": target_uuid,
                "now": int(time.time()),
            },
        )

    def get_superseded_memories(
        self,
        project_id: str | None = None,
    ) -> list[dict]:
        """Get all superseded memory UUIDs for exclusion from search.

        Args:
            project_id: Optional project filter

        Returns:
            List of dicts with older_uuid and superseding_uuid
        """
        if project_id:
            query = """
            MATCH (newer:Memory)-[r:SUPERSEDES]->(older:Memory)
            WHERE older.project_id = $project_id OR newer.project_id = $project_id
            RETURN older.uuid AS superseded_uuid,
                   newer.uuid AS superseding_uuid,
                   r.confidence AS confidence,
                   r.supersession_type AS type
            """
            result = self.graph.query(query, {"project_id": project_id})
        else:
            query = """
            MATCH (newer:Memory)-[r:SUPERSEDES]->(older:Memory)
            RETURN older.uuid AS superseded_uuid,
                   newer.uuid AS superseding_uuid,
                   r.confidence AS confidence,
                   r.supersession_type AS type
            """
            result = self.graph.query(query)

        return [
            {
                "superseded_uuid": row[0],
                "superseding_uuid": row[1],
                "confidence": row[2],
                "type": row[3],
            }
            for row in (result.result_set or [])
        ]

    def get_merged_memories(
        self,
        project_id: str | None = None,
    ) -> list[dict]:
        """Get all merged memory UUIDs for exclusion from search.

        Args:
            project_id: Optional project filter

        Returns:
            List of dicts with source_uuid and target_uuid
        """
        if project_id:
            query = """
            MATCH (source:Memory)-[r:MERGED_INTO]->(target:Memory)
            WHERE source.project_id = $project_id OR target.project_id = $project_id
            RETURN source.uuid AS merged_uuid,
                   target.uuid AS merged_into_uuid,
                   r.merged_at AS merged_at
            """
            result = self.graph.query(query, {"project_id": project_id})
        else:
            query = """
            MATCH (source:Memory)-[r:MERGED_INTO]->(target:Memory)
            RETURN source.uuid AS merged_uuid,
                   target.uuid AS merged_into_uuid,
                   r.merged_at AS merged_at
            """
            result = self.graph.query(query)

        return [
            {
                "merged_uuid": row[0],
                "merged_into_uuid": row[1],
                "merged_at": row[2],
            }
            for row in (result.result_set or [])
        ]

    # Allowed verb edge types for security validation
    ALLOWED_VERB_EDGES = {"READS", "MODIFIES", "EXECUTES", "TRIGGERED", "REFERENCES"}

    # Pre-defined Cypher queries for each edge type (avoids f-string interpolation)
    # Security: These are hardcoded to prevent any possibility of injection
    _EDGE_QUERIES_WITH_SUMMARY = {
        "READS": """
            MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
            CREATE (m)-[:READS {timestamp: $ts, change_summary: $summary}]->(e)
            """,
        "MODIFIES": """
            MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
            CREATE (m)-[:MODIFIES {timestamp: $ts, change_summary: $summary}]->(e)
            """,
        "EXECUTES": """
            MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
            CREATE (m)-[:EXECUTES {timestamp: $ts, change_summary: $summary}]->(e)
            """,
        "TRIGGERED": """
            MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
            CREATE (m)-[:TRIGGERED {timestamp: $ts, change_summary: $summary}]->(e)
            """,
        "REFERENCES": """
            MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
            CREATE (m)-[:REFERENCES {timestamp: $ts, change_summary: $summary}]->(e)
            """,
    }

    _EDGE_QUERIES_NO_SUMMARY = {
        "READS": """
            MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
            CREATE (m)-[:READS {timestamp: $ts}]->(e)
            """,
        "MODIFIES": """
            MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
            CREATE (m)-[:MODIFIES {timestamp: $ts}]->(e)
            """,
        "EXECUTES": """
            MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
            CREATE (m)-[:EXECUTES {timestamp: $ts}]->(e)
            """,
        "TRIGGERED": """
            MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
            CREATE (m)-[:TRIGGERED {timestamp: $ts}]->(e)
            """,
        "REFERENCES": """
            MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
            CREATE (m)-[:REFERENCES {timestamp: $ts}]->(e)
            """,
    }

    def add_verb_edge(
        self,
        memory_uuid: str,
        entity_name: str,
        entity_type: str,
        action: str,
        timestamp: int | None = None,
        change_summary: str | None = None,
    ) -> None:
        """Add a verb-specific edge between memory and entity.

        Supports semantic edge types: READS, MODIFIES, EXECUTES, TRIGGERED.
        Automatically increments entity version on MODIFIES.

        Args:
            memory_uuid: Memory UUID
            entity_name: Entity name (will be canonicalized)
            entity_type: Entity type (file, tool, command, error)
            action: Action type (reads, modifies, executes, triggered)
            timestamp: Optional timestamp for the action
            change_summary: Optional summary of changes (for modifies)
        """
        import time

        # Canonicalize entity name
        canonical_name = self._canonicalize_entity(entity_name, entity_type)

        # Map action to edge type with security validation
        action_to_edge = {
            "reads": "READS",
            "modifies": "MODIFIES",
            "executes": "EXECUTES",
            "triggered": "TRIGGERED",
        }
        edge_type = action_to_edge.get(action.lower(), "REFERENCES")

        # Security: validate edge type to prevent Cypher injection
        # Edge type is used in f-string but is validated against allow-list
        if edge_type not in self.ALLOWED_VERB_EDGES:
            log.error(f"Invalid edge type attempted: {edge_type}")
            raise ValueError(f"Invalid edge type: {edge_type}")

        # Ensure entity exists with version property
        if action.lower() == "modifies":
            # Increment version on MODIFIES
            self.graph.query(
                """
                MERGE (e:Entity {name: $name, type: $type})
                ON CREATE SET e.created_at = timestamp(), e.version = 1
                ON MATCH SET e.version = COALESCE(e.version, 0) + 1, e.last_modified = timestamp()
                """,
                {"name": canonical_name, "type": entity_type},
            )
        else:
            # Just ensure entity exists
            self.graph.query(
                """
                MERGE (e:Entity {name: $name, type: $type})
                ON CREATE SET e.created_at = timestamp(), e.version = 1
                """,
                {"name": canonical_name, "type": entity_type},
            )

        # Create the verb-specific edge using hardcoded queries (no f-string interpolation)
        ts = timestamp or int(time.time())
        if change_summary:
            query = self._EDGE_QUERIES_WITH_SUMMARY[edge_type]
            self.graph.query(
                query,
                {"uuid": memory_uuid, "name": canonical_name, "type": entity_type, "ts": ts, "summary": change_summary[:self.config.summary_max_size]},
            )
        else:
            query = self._EDGE_QUERIES_NO_SUMMARY[edge_type]
            self.graph.query(
                query,
                {"uuid": memory_uuid, "name": canonical_name, "type": entity_type, "ts": ts},
            )

        log.trace(f"Added {edge_type} edge: memory={memory_uuid[:8]}... -> {entity_type}:{canonical_name}")

        # Infer READS from MODIFIES: you can't modify what you haven't read
        # Only for files - tools/commands don't have this semantic
        if edge_type == "MODIFIES" and entity_type == "file":
            # Check if READS edge already exists for this memory->entity
            check_result = self.graph.query(
                """
                MATCH (m:Memory {uuid: $uuid})-[r:READS]->(e:Entity {name: $name, type: $type})
                RETURN count(r) as cnt
                """,
                {"uuid": memory_uuid, "name": canonical_name, "type": entity_type},
            )
            reads_exists = check_result.result_set and check_result.result_set[0][0] > 0

            if not reads_exists:
                # Create implicit READS edge (before the modify)
                self.graph.query(
                    """
                    MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
                    CREATE (m)-[:READS {timestamp: $ts, implicit: true}]->(e)
                    """,
                    {"uuid": memory_uuid, "name": canonical_name, "type": entity_type, "ts": ts - 1},
                )
                log.trace(f"Added implicit READS edge for MODIFIES: {entity_type}:{canonical_name}")

    def _canonicalize_entity(self, name: str, entity_type: str) -> str:
        """Canonicalize entity name for consistent deduplication.

        Args:
            name: Raw entity name
            entity_type: Entity type

        Returns:
            Canonicalized entity name
        """
        import os
        import hashlib

        if entity_type == "file":
            # Normalize file paths for consistent deduplication
            try:
                # Clean up the path
                clean_name = name.strip().strip("'\"")
                normalized = os.path.normpath(clean_name)

                # Strip home directory prefix to make paths project-relative
                # This helps deduplicate: /Users/foo/repo/myproj/main.py → myproj/main.py
                home = os.path.expanduser("~")
                if normalized.startswith(home):
                    # Remove home prefix, keep from first meaningful directory
                    relative = normalized[len(home):].lstrip(os.sep)
                    # Skip only definitive container directories (not project dirs like "src")
                    # Require at least 3 parts (container/project/file) to avoid over-stripping
                    parts = relative.split(os.sep)
                    if len(parts) > 2 and parts[0] in ("repo", "repos", "projects", "dev", "work"):
                        return os.sep.join(parts[1:])  # Skip the container prefix
                    return relative

                return normalized
            except Exception:
                return name

        elif entity_type == "tool":
            # Normalize tool names - lowercase, strip common prefixes
            normalized = name.lower()
            normalized = normalized.replace("mcp__", "").replace("__", ":")
            return normalized

        elif entity_type == "command":
            # Keep 2-word for common driver commands (git commit, npm install, etc.)
            parts = name.strip().split()
            if not parts:
                return name
            base = parts[0].lower()
            # Common drivers where subcommand is semantically important
            command_drivers = {"git", "npm", "pip", "docker", "kubectl", "python", "node", "yarn", "cargo", "go"}
            if base in command_drivers and len(parts) > 1:
                return f"{base} {parts[1].lower()}"
            return base

        elif entity_type == "error":
            # Extract exception type for better deduplication
            # "ValueError: invalid input" → "error:valueerror"
            # "TypeError: cannot..." → "error:typeerror"
            import re
            # Match common exception patterns
            match = re.match(r"^(\w+(?:Error|Exception|Warning|Failure))", name, re.IGNORECASE)
            if match:
                return f"error:{match.group(1).lower()}"
            # Fallback: hash for unrecognized error formats
            error_hash = hashlib.sha256(name[:200].encode()).hexdigest()[:12]
            return f"error:{error_hash}"

        return name

    def _cleanup_orphan_entities(self) -> int:
        """Delete Entity nodes that have no incoming edges.

        Called after code chunk deletion to clean up entities that are no longer
        referenced by any CodeChunk or Memory node.

        Returns:
            Number of orphaned entities deleted
        """
        # Find and delete entities with no incoming edges
        # Use OPTIONAL MATCH + WHERE null pattern for FalkorDB compatibility
        try:
            result = self.graph.query(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (other)-[r]->(e)
                WITH e, other
                WHERE other IS NULL
                DETACH DELETE e
                RETURN count(*) AS deleted_count
                """
            )

            # Handle None/empty results safely
            if result.result_set and len(result.result_set) > 0:
                deleted = result.result_set[0][0]
                if deleted is not None and deleted > 0:
                    log.debug(f"Deleted {deleted} orphaned Entity nodes")
                    return int(deleted)
            return 0
        except Exception as e:
            log.warning(f"Orphan entity cleanup failed: {e}")
            return 0

    def add_goal_node(
        self,
        goal_id: str,
        intent: str,
        session_id: str,
        status: str = "active",
    ) -> None:
        """Add a Goal node for intent-based retrieval.

        Args:
            goal_id: Unique goal identifier
            intent: User's objective description
            session_id: Associated session ID
            status: Goal status (active, completed, abandoned)
        """
        self.graph.query(
            """
            CREATE (g:Goal {
                id: $goal_id,
                intent: $intent,
                session_id: $session_id,
                status: $status,
                created_at: timestamp()
            })
            """,
            {
                "goal_id": goal_id,
                "intent": intent[:self.config.memory_content_max_size],
                "session_id": session_id,
                "status": status,
            },
        )
        log.trace(f"Created Goal node: {goal_id[:8]}... intent={intent[:50]}...")

    def link_session_to_goal(
        self,
        session_summary_uuid: str,
        goal_id: str,
    ) -> None:
        """Link a session summary to a goal with HAS_GOAL edge.

        Args:
            session_summary_uuid: Session summary memory UUID
            goal_id: Goal node ID
        """
        self.graph.query(
            """
            MATCH (m:Memory {uuid: $uuid}), (g:Goal {id: $goal_id})
            CREATE (m)-[:HAS_GOAL]->(g)
            """,
            {"uuid": session_summary_uuid, "goal_id": goal_id},
        )

    def link_memory_to_goal(
        self,
        memory_uuid: str,
        goal_id: str,
    ) -> None:
        """Link a memory to a goal with ACHIEVES edge.

        Args:
            memory_uuid: Memory UUID that contributed to the goal
            goal_id: Goal node ID
        """
        self.graph.query(
            """
            MATCH (m:Memory {uuid: $uuid}), (g:Goal {id: $goal_id})
            CREATE (m)-[:ACHIEVES]->(g)
            """,
            {"uuid": memory_uuid, "goal_id": goal_id},
        )

    def add_project_node(
        self,
        project_id: str,
        project_path: str,
    ) -> None:
        """Add or update a Project node for session scoping.

        Projects enable cross-session linking even without shared files.

        Args:
            project_id: Unique project identifier (derived from path)
            project_path: Original project path
        """
        # Parameterized queries handle escaping automatically
        self.graph.query(
            """
            MERGE (p:Project {id: $project_id})
            ON CREATE SET p.path = $path, p.created_at = timestamp(), p.session_count = 1
            ON MATCH SET p.session_count = p.session_count + 1, p.last_updated = timestamp()
            """,
            {"project_id": project_id, "path": project_path[:self.config.summary_max_size]},
        )
        log.trace(f"Created/updated Project node: {project_id}")

    def link_session_to_project(
        self,
        session_summary_uuid: str,
        project_id: str,
    ) -> None:
        """Link a session summary to a project with BELONGS_TO edge.

        Args:
            session_summary_uuid: Session summary memory UUID
            project_id: Project node ID
        """
        self.graph.query(
            """
            MATCH (m:Memory {uuid: $uuid}), (p:Project {id: $project_id})
            CREATE (m)-[:BELONGS_TO]->(p)
            """,
            {"uuid": session_summary_uuid, "project_id": project_id},
        )
        log.trace(f"Linked session {session_summary_uuid[:8]}... to project {project_id}")

    def delete_memory_node(self, uuid: str) -> None:
        """Delete a memory node from the graph (for rollback).

        Args:
            uuid: Memory UUID to delete
        """
        self.graph.query(
            "MATCH (m:Memory {uuid: $uuid}) DETACH DELETE m",
            {"uuid": uuid},
        )

    def delete_memory_vector(self, uuid: str) -> None:
        """Delete a memory vector from LanceDB (for rollback).

        Args:
            uuid: Memory UUID to delete
        """
        with self._write_lock:  # LanceDB not thread-safe for concurrent writes
            self.lance_table.delete(f'uuid = "{uuid}"')

    def delete_session_memories(self, session_id: str) -> dict:
        """Delete all memories for a session (UPSERT semantics).

        Removes all Memory nodes, their edges, and vectors for a given session_id.
        Used to enable re-indexing without duplicates.

        Args:
            session_id: Session UUID to clean up

        Returns:
            dict with counts of deleted items
        """
        log.info(f"Deleting session memories: {session_id}")

        # Get all memory UUIDs for this session from graph
        result = self.graph.query(
            "MATCH (m:Memory {session_id: $session_id}) RETURN m.uuid",
            {"session_id": session_id},
        )
        uuids = [row[0] for row in result.result_set if row[0]]
        log.debug(f"Found {len(uuids)} memories to delete for session {session_id}")

        if not uuids:
            return {"memories_deleted": 0, "vectors_deleted": 0}

        # Delete from graph (with all relationships via DETACH DELETE)
        self.graph.query(
            "MATCH (m:Memory {session_id: $session_id}) DETACH DELETE m",
            {"session_id": session_id},
        )
        log.debug(f"Deleted {len(uuids)} memory nodes from graph")

        # Delete from vectors using batch operation (avoids N+1 pattern)
        # LanceDB delete requires SQL-style filter with IN clause
        try:
            # Escape UUIDs and build IN clause for batch delete
            escaped_uuids = [uid.replace('"', '\\"') for uid in uuids]
            uuid_list = ", ".join(f'"{uid}"' for uid in escaped_uuids)
            with self._write_lock:  # LanceDB not thread-safe for concurrent writes
                self.lance_table.delete(f"uuid IN ({uuid_list})")
            log.debug(f"Batch deleted {len(uuids)} vectors")
        except Exception as e:
            log.warning(f"Batch vector delete failed: {e}")

        log.info(f"Session cleanup complete: deleted {len(uuids)} memories")
        return {"memories_deleted": len(uuids), "vectors_deleted": len(uuids)}

    def search_vectors(
        self,
        query_vector: list[float],
        limit: int = 10,
        type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding
            limit: Maximum results
            type_filter: Optional filter by memory type

        Returns:
            List of matching memories with distance scores
        """
        log.trace(f"Searching vectors: limit={limit}, type_filter={type_filter}")

        with self._write_lock:  # LanceDB not thread-safe for concurrent ops
            search = self.lance_table.search(query_vector).limit(limit)

            if type_filter:
                search = search.where(f"type = '{type_filter}'")

            results = search.to_list()

        log.debug(f"Vector search returned {len(results)} results")
        return results

    # ═══════════════════════════════════════════════════════════════════════════════
    # MEMORY REINDEXING METHODS
    # ═══════════════════════════════════════════════════════════════════════════════

    def get_memories_for_reindex(
        self,
        project_id: str,
        batch_size: int = 100,
    ) -> list[list[dict[str, Any]]]:
        """Fetch memories from graph for re-indexing.

        Retrieves all memories for a project from the graph database
        to enable re-generation of their embeddings.

        Args:
            project_id: Project to fetch memories for
            batch_size: Number of memories per batch

        Returns:
            List of batches, each containing dicts with uuid, content, type, session_id
        """
        log.info(f"Fetching memories for reindex: project={project_id}, batch_size={batch_size}")

        # Get all memories for this project
        result = self.graph.query(
            """
            MATCH (m:Memory)
            WHERE m.project_id = $project_id
            RETURN m.uuid, m.content, m.type, m.session_id
            """,
            {"project_id": project_id},
        )

        if not result.result_set:
            log.info(f"No memories found for project {project_id}")
            return []

        # Convert to list of dicts
        all_memories = [
            {
                "uuid": row[0],
                "content": row[1],
                "type": row[2],
                "session_id": row[3] if row[3] else None,
            }
            for row in result.result_set
            if row[0] and row[1]  # Ensure uuid and content exist
        ]

        log.info(f"Found {len(all_memories)} memories to reindex")

        # Split into batches
        batches = []
        for i in range(0, len(all_memories), batch_size):
            batches.append(all_memories[i : i + batch_size])

        log.debug(f"Split into {len(batches)} batches")
        return batches

    def upsert_memory_vectors(
        self,
        memories: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> int:
        """Write/update vectors in LanceDB for given memory UUIDs.

        Strategy: Delete existing by UUID, then add new.
        LanceDB lacks native upsert, so we implement it manually.

        Args:
            memories: List of memory dicts with uuid, content, type, session_id
            vectors: Corresponding embedding vectors

        Returns:
            Number of vectors written
        """
        import json

        if not memories or not vectors:
            return 0

        if len(memories) != len(vectors):
            raise ValueError(f"Mismatched counts: {len(memories)} memories, {len(vectors)} vectors")

        log.debug(f"Upserting {len(memories)} memory vectors")

        with self._write_lock:
            # Step 1: Delete existing vectors for these UUIDs
            uuids = [m["uuid"] for m in memories]
            escaped_uuids = [uid.replace('"', '\\"') for uid in uuids]
            uuid_list = ", ".join(f'"{uid}"' for uid in escaped_uuids)
            try:
                self.lance_table.delete(f"uuid IN ({uuid_list})")
                log.trace(f"Deleted existing vectors for {len(uuids)} UUIDs")
            except Exception as e:
                log.warning(f"Delete step failed (may be empty): {e}")

            # Step 2: Add new vectors
            records = []
            for mem, vec in zip(memories, vectors):
                records.append({
                    "uuid": mem["uuid"],
                    "vector": vec,
                    "content": mem["content"],
                    "type": mem.get("type", "fact"),
                    "session_id": mem.get("session_id") or "",
                    "metadata": json.dumps({"reindexed": True}),
                })

            self.lance_table.add(records)
            log.debug(f"Added {len(records)} vectors to LanceDB")

        return len(records)

    def get_memory_count(self, project_id: str) -> int:
        """Get count of memories for a project in the graph.

        Args:
            project_id: Project to count memories for

        Returns:
            Number of memories in the graph
        """
        result = self.graph.query(
            """
            MATCH (m:Memory)
            WHERE m.project_id = $project_id
            RETURN count(m) AS cnt
            """,
            {"project_id": project_id},
        )

        if result.result_set and result.result_set[0]:
            return result.result_set[0][0] or 0
        return 0

    # ═══════════════════════════════════════════════════════════════════════════════
    # CODE SEARCH METHODS
    # ═══════════════════════════════════════════════════════════════════════════════

    def add_code_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Add code chunks to the code index.

        Args:
            chunks: List of dicts with keys: uuid, vector, content, filepath, project_id, start_line, end_line

        Returns:
            Number of chunks added
        """
        if not self.config.code_index_enabled:
            log.warning("Code indexing disabled, skipping add_code_chunks")
            return 0

        if not chunks:
            return 0

        log.info(f"Adding {len(chunks)} code chunks to index")

        with self._write_lock:  # LanceDB not thread-safe for concurrent ops
            self.code_table.add(chunks)

        log.debug(f"Code chunks added successfully")
        return len(chunks)

    def search_code(
        self,
        query_vector: list[float],
        limit: int = 10,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar code chunks.

        Args:
            query_vector: Query embedding
            limit: Maximum results
            project_id: Optional filter by project ID (canonical identifier)

        Returns:
            List of matching code chunks with distance scores
        """
        if not self.config.code_index_enabled:
            log.warning("Code indexing disabled, returning empty results")
            return []

        log.trace(f"Searching code: limit={limit}, project_id={project_id}")

        with self._write_lock:  # LanceDB not thread-safe for concurrent ops
            search = self.code_table.search(query_vector).limit(limit)

            if project_id:
                safe_project_id = project_id.replace("'", "''")
                search = search.where(f"project_id = '{safe_project_id}'")

            results = search.to_list()

        log.debug(f"Code search returned {len(results)} results")
        return results

    def clear_code_index(self, project_id: str | None = None) -> int:
        """Clear code index for a project or all projects.

        Clears both LanceDB vectors and graph CodeChunk nodes, then
        cleans up any orphaned Entity nodes.

        Args:
            project_id: If provided, only clear chunks from this project

        Returns:
            Number of chunks deleted
        """
        if not self.config.code_index_enabled:
            return 0

        with self._write_lock:  # LanceDB not thread-safe for concurrent ops
            if project_id:
                log.info(f"Clearing code index for project: {project_id}")
                safe_project_id = project_id.replace("'", "''")
                # Count before delete
                try:
                    count = len(self.code_table.search([0.0] * self.config.embedding_dim)
                               .where(f"project_id = '{safe_project_id}'")
                               .limit(100000).to_list())
                except Exception:
                    count = 0

                # Delete with corruption recovery
                try:
                    self.code_table.delete(f"project_id = '{safe_project_id}'")
                except Exception as e:
                    log.warning(f"Code table corrupted during delete, recreating: {e}")
                    # Table is corrupted - drop and recreate (losing all data)
                    try:
                        self.lance_db.drop_table(self.CODE_TABLE_NAME)
                    except Exception:
                        pass
                    self._init_code_table()

                # Also clear CodeChunk nodes from graph for this project
                self.graph.query(
                    "MATCH (c:CodeChunk {project_id: $project_id}) DETACH DELETE c",
                    {"project_id": project_id},
                )
                # Reinitialize indexes to prevent FalkorDB SIGSEGV on subsequent inserts
                self._graph_backend.reinit_code_chunk_indexes()
            else:
                log.info("Clearing entire code index")
                try:
                    count = self.code_table.count_rows()
                except Exception:
                    count = 0
                # Drop and recreate table (with corruption recovery)
                try:
                    self.lance_db.drop_table(self.CODE_TABLE_NAME)
                except Exception as e:
                    log.warning(f"Failed to drop code table (may be corrupted): {e}")
                self._init_code_table()

                # Also clear ALL CodeChunk nodes from graph
                self.graph.query("MATCH (c:CodeChunk) DETACH DELETE c")
                # Reinitialize indexes to prevent FalkorDB SIGSEGV on subsequent inserts
                self._graph_backend.reinit_code_chunk_indexes()

            # Clean up orphaned Entity nodes
            orphans_deleted = self._cleanup_orphan_entities()
            if orphans_deleted > 0:
                log.info(f"Cleaned up {orphans_deleted} orphaned entity nodes")

        log.info(f"Cleared {count} code chunks")
        return count

    def delete_chunks_by_filepath(
        self,
        project_id: str,
        filepath: str,
    ) -> int:
        """Delete all chunks for a specific file (for incremental updates).

        Args:
            project_id: Project ID (canonical identifier)
            filepath: File path (relative or absolute)

        Returns:
            Number of chunks deleted
        """
        if not self.config.code_index_enabled:
            return 0

        log.info(f"Deleting chunks for file: {filepath} in {project_id}")

        # Sanitize inputs to prevent issues with special characters
        safe_project_id = project_id.replace("'", "''")
        safe_filepath = filepath.replace("'", "''")
        where_clause = f"project_id = '{safe_project_id}' AND filepath = '{safe_filepath}'"

        with self._write_lock:
            # Collect all chunk UUIDs using pagination (no arbitrary limit)
            chunk_uuids: list[str] = []
            try:
                offset = 0
                batch_size = 1000
                while True:
                    matches = self.code_table.search([0.0] * self.config.embedding_dim).where(
                        where_clause
                    ).limit(batch_size).to_list()
                    if not matches:
                        break
                    chunk_uuids.extend(m.get("uuid") for m in matches if m.get("uuid"))
                    if len(matches) < batch_size:
                        break
                    offset += batch_size
            except Exception:
                chunk_uuids = []

            count = len(chunk_uuids)
            if count == 0:
                log.debug(f"No chunks found for {filepath}")
                return 0

            # Delete from LanceDB
            self.code_table.delete(where_clause)

            # Delete CodeChunk nodes and their edges from FalkorDB
            for uuid in chunk_uuids:
                self.graph.query(
                    "MATCH (c:CodeChunk {uuid: $uuid}) DETACH DELETE c",
                    {"uuid": uuid},
                )

            # Clean up orphaned Entity nodes that have no remaining references
            # (neither from CodeChunks nor from Memories)
            orphans_deleted = self._cleanup_orphan_entities()
            if orphans_deleted > 0:
                log.info(f"Cleaned up {orphans_deleted} orphaned entity nodes")

            log.info(f"Deleted {count} chunks for {filepath}")
            return count

    def get_code_stats(self, project_id: str | None = None) -> dict[str, Any]:
        """Get statistics about the code index.

        Args:
            project_id: Optional filter by project ID

        Returns:
            Dict with chunk_count and unique files
        """
        if not self.config.code_index_enabled:
            return {"enabled": False, "chunk_count": 0}

        try:
            with self._write_lock:  # LanceDB not thread-safe for concurrent ops
                total = self.code_table.count_rows()
                # Get sample to count unique files (approximate)
                sample = self.code_table.search([0.0] * self.config.embedding_dim).limit(1000).to_list()
            unique_files = len(set(r.get("filepath", "") for r in sample))
            return {
                "enabled": True,
                "chunk_count": total,
                "unique_files_sample": unique_files,
                "project_id": project_id,
            }
        except Exception as e:
            log.error(f"Failed to get code stats: {e}")
            return {"enabled": True, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════════
    # P1: ENTITY LINKING (CODE ↔ MEMORY BRIDGE)
    # ═══════════════════════════════════════════════════════════════════════════════

    def add_code_chunk_node(
        self,
        uuid: str,
        filepath: str,
        project_id: str,
        start_line: int,
        end_line: int,
    ) -> None:
        """Add a CodeChunk node to the graph for entity linking.

        Args:
            uuid: Unique identifier (matches LanceDB)
            filepath: File path
            project_id: Project ID (canonical identifier)
            start_line: Starting line number
            end_line: Ending line number
        """
        log.trace(f"Adding CodeChunk node: {filepath}:{start_line}-{end_line}")
        self.graph.query(
            """
            MERGE (c:CodeChunk {uuid: $uuid})
            ON CREATE SET
                c.filepath = $filepath,
                c.project_id = $project_id,
                c.start_line = $start_line,
                c.end_line = $end_line,
                c.created_at = timestamp()
            """,
            {
                "uuid": uuid,
                "filepath": filepath,
                "project_id": project_id,
                "start_line": start_line,
                "end_line": end_line,
            },
        )

    def link_code_to_entity(
        self,
        chunk_uuid: str,
        entity_name: str,
        entity_type: str,
        relation: str = "DEFINES",
    ) -> None:
        """Link a CodeChunk to an Entity node.

        Args:
            chunk_uuid: CodeChunk UUID
            entity_name: Entity name (function, class, import, etc.)
            entity_type: Entity type (function, class, import, file)
            relation: Relationship type (DEFINES, IMPORTS, CALLS, etc.)
        """
        log.trace(f"Linking code {chunk_uuid[:8]}... to {entity_type}:{entity_name}")
        # Ensure entity exists
        self.add_entity_node(entity_name, entity_type)
        # Create relationship
        self.graph.query(
            """
            MATCH (c:CodeChunk {uuid: $chunk_uuid}), (e:Entity {name: $entity_name, type: $entity_type})
            MERGE (c)-[r:REFERENCES {relation: $relation}]->(e)
            ON CREATE SET r.created_at = timestamp()
            """,
            {
                "chunk_uuid": chunk_uuid,
                "entity_name": entity_name,
                "entity_type": entity_type,
                "relation": relation,
            },
        )

    def get_code_related_memories(
        self,
        chunk_uuid: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find memories that share entities with a code chunk.

        Args:
            chunk_uuid: CodeChunk UUID
            limit: Maximum results

        Returns:
            List of related memories with shared entity info
        """
        log.trace(f"Finding memories related to code chunk: {chunk_uuid[:8]}...")
        result = self.graph.query(
            """
            MATCH (c:CodeChunk {uuid: $uuid})-[:REFERENCES]->(e:Entity)<-[:REFERENCES|READS|MODIFIES]-(m:Memory)
            RETURN DISTINCT
                m.uuid AS uuid,
                m.content AS content,
                m.type AS type,
                m.session_id AS session_id,
                collect(DISTINCT e.name) AS shared_entities
            ORDER BY size(shared_entities) DESC
            LIMIT $limit
            """,
            {"uuid": chunk_uuid, "limit": limit},
        )

        rows = []
        for record in result.result_set:
            rows.append({
                "uuid": record[0],
                "content": record[1],
                "type": record[2],
                "session_id": record[3],
                "shared_entities": record[4],
            })
        log.debug(f"Found {len(rows)} memories related to code chunk")
        return rows

    def get_memory_related_code(
        self,
        memory_uuid: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find code chunks that share entities with a memory.

        Args:
            memory_uuid: Memory UUID
            limit: Maximum results

        Returns:
            List of related code chunks with shared entity info
        """
        log.trace(f"Finding code related to memory: {memory_uuid[:8]}...")
        result = self.graph.query(
            """
            MATCH (m:Memory {uuid: $uuid})-[:REFERENCES|READS|MODIFIES]->(e:Entity)<-[:REFERENCES]-(c:CodeChunk)
            RETURN DISTINCT
                c.uuid AS uuid,
                c.filepath AS filepath,
                c.project_id AS project_id,
                c.start_line AS start_line,
                c.end_line AS end_line,
                collect(DISTINCT e.name) AS shared_entities
            ORDER BY size(shared_entities) DESC
            LIMIT $limit
            """,
            {"uuid": memory_uuid, "limit": limit},
        )

        rows = []
        for record in result.result_set:
            rows.append({
                "uuid": record[0],
                "filepath": record[1],
                "project_id": record[2],
                "start_line": record[3],
                "end_line": record[4],
                "shared_entities": record[5],
            })
        log.debug(f"Found {len(rows)} code chunks related to memory")
        return rows

    # ═══════════════════════════════════════════════════════════════════════════════
    # P2: STALENESS DETECTION (PROJECT INDEX METADATA)
    # ═══════════════════════════════════════════════════════════════════════════════

    def set_project_index_metadata(
        self,
        project_id: str,
        commit_hash: str | None,
        file_count: int,
        chunk_count: int,
    ) -> None:
        """Store/update project index metadata in graph.

        Args:
            project_id: Canonical project identifier (e.g., "git:github.com/user/repo")
            commit_hash: Git commit hash at time of indexing (None if not a git repo)
            file_count: Number of files indexed
            chunk_count: Number of chunks created
        """
        log.info(f"Setting project index metadata: {project_id} (hash={commit_hash[:8] if commit_hash else 'N/A'})")
        self.graph.query(
            """
            MERGE (p:ProjectIndex {project_id: $project_id})
            SET p.last_commit_hash = $commit_hash,
                p.indexed_at = timestamp(),
                p.file_count = $file_count,
                p.chunk_count = $chunk_count
            """,
            {
                "project_id": project_id,
                "commit_hash": commit_hash,
                "file_count": file_count,
                "chunk_count": chunk_count,
            },
        )

    def get_project_index_metadata(self, project_id: str) -> dict[str, Any] | None:
        """Get project index metadata from graph.

        Args:
            project_id: Canonical project identifier (e.g., "git:github.com/user/repo")

        Returns:
            Dict with project_id, last_commit_hash, indexed_at, file_count, chunk_count
            or None if not indexed
        """
        log.trace(f"Getting project index metadata: {project_id}")
        result = self.graph.query(
            """
            MATCH (p:ProjectIndex {project_id: $project_id})
            RETURN p.project_id, p.last_commit_hash, p.indexed_at, p.file_count, p.chunk_count
            """,
            {"project_id": project_id},
        )

        if not result.result_set:
            return None

        row = result.result_set[0]
        return {
            "project_id": row[0],
            "last_commit_hash": row[1],
            "indexed_at": row[2],
            "file_count": row[3],
            "chunk_count": row[4],
        }

    def get_related_nodes(
        self,
        uuid: str,
        hops: int = 1,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Get related memories via graph traversal.

        Args:
            uuid: Starting memory UUID
            hops: Number of hops to traverse (1-3)
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of related memories
        """
        log.trace(f"Getting related nodes: uuid={uuid[:8]}..., hops={hops}, direction={direction}")
        hops = min(max(hops, 1), self.config.max_graph_hops)

        if direction == "outgoing":
            pattern = f"(start:Memory {{uuid: $uuid}})-[r*1..{hops}]->(connected:Memory)"
        elif direction == "incoming":
            pattern = f"(start:Memory {{uuid: $uuid}})<-[r*1..{hops}]-(connected:Memory)"
        else:
            pattern = f"(start:Memory {{uuid: $uuid}})-[r*1..{hops}]-(connected:Memory)"

        result = self.graph.query(
            f"""
            MATCH {pattern}
            RETURN DISTINCT
                connected.uuid AS uuid,
                connected.content AS content,
                connected.type AS type,
                connected.session_id AS session_id,
                connected.created_at AS created_at
            """,
            {"uuid": uuid},
        )

        rows = []
        for record in result.result_set:
            rows.append({
                "uuid": record[0],
                "content": record[1],
                "type": record[2],
                "session_id": record[3],
                "created_at": record[4],
                "hops": 1,  # Simplified
            })

        log.debug(f"Graph traversal returned {len(rows)} related nodes")
        return rows

    def get_paths(
        self,
        from_uuid: str,
        max_hops: int = 2,
        direction: str = "outgoing",
    ) -> list[dict[str, Any]]:
        """Get paths from a node with full edge metadata.

        Returns paths with node and edge information for scoring.
        Includes paths through Entity nodes for cross-session reasoning.

        Args:
            from_uuid: Starting memory UUID
            max_hops: Maximum path length (1-3)
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of paths with nodes, edge_types, and metadata
        """
        log.trace(f"Getting paths: from={from_uuid[:8]}..., max_hops={max_hops}")
        max_hops = min(max(max_hops, 1), self.config.max_graph_hops)

        # FalkorDB Cypher for variable-length paths
        if direction == "outgoing":
            pattern = f"(start:Memory {{uuid: $uuid}})-[r*1..{max_hops}]->(target:Memory)"
        elif direction == "incoming":
            pattern = f"(start:Memory {{uuid: $uuid}})<-[r*1..{max_hops}]-(target:Memory)"
        else:
            pattern = f"(start:Memory {{uuid: $uuid}})-[r*1..{max_hops}]-(target:Memory)"

        result = self.graph.query(
            f"""
            MATCH path = {pattern}
            WHERE start <> target
            RETURN DISTINCT
                target.uuid AS end_uuid,
                target.content AS end_content,
                target.type AS end_type,
                target.session_id AS session_id,
                target.created_at AS created_at,
                [rel in relationships(path) | type(rel)] AS rel_types,
                [rel in relationships(path) | rel.relation_type] AS relation_types,
                [rel in relationships(path) | rel.weight] AS weights,
                length(path) AS hops
            LIMIT $limit
            """,
            {"uuid": from_uuid, "limit": self.config.graph_path_limit},
        )

        paths = []
        for record in result.result_set:
            # Extract relationship info
            rel_types = record[5] or []  # RELATES_TO, REFERENCES, etc.
            relation_types = record[6] or []  # contains, follows, etc.
            weights = record[7] or []

            # Combine into edge_types (prefer relation_type if available)
            edge_types = []
            for i, rt in enumerate(rel_types):
                if i < len(relation_types) and relation_types[i]:
                    edge_types.append(relation_types[i])
                elif rt == "REFERENCES":
                    edge_types.append("references")
                else:
                    edge_types.append("relates")

            paths.append({
                "end_uuid": record[0],
                "end_content": record[1],
                "end_type": record[2],
                "session_id": record[3],
                "created_at": record[4] or 0,
                "edge_types": edge_types,
                "edge_weights": [w or 1.0 for w in (weights or [])],
                "hops": record[8] or len(edge_types),
            })

        log.debug(f"Found {len(paths)} paths from {from_uuid[:8]}...")
        return paths

    def get_cross_session_paths(
        self,
        from_uuid: str,
        max_hops: int = 3,
    ) -> list[dict[str, Any]]:
        """Get paths that cross sessions via shared Entity nodes.

        This enables reasoning like "what other sessions touched this file?"

        Args:
            from_uuid: Starting memory UUID
            max_hops: Maximum path length

        Returns:
            List of cross-session paths with entity bridge info
        """
        log.trace(f"Getting cross-session paths from {from_uuid[:8]}...")

        result = self.graph.query(
            """
            MATCH (start:Memory {uuid: $uuid})-[:REFERENCES]->(e:Entity)<-[:REFERENCES]-(other:Memory)
            WHERE start.session_id <> other.session_id
            RETURN DISTINCT
                other.uuid AS end_uuid,
                other.content AS end_content,
                other.type AS end_type,
                other.session_id AS session_id,
                other.created_at AS created_at,
                e.name AS entity_name,
                e.type AS entity_type
            LIMIT $limit
            """,
            {"uuid": from_uuid, "limit": self.config.cross_session_limit},
        )

        paths = []
        for record in result.result_set:
            paths.append({
                "end_uuid": record[0],
                "end_content": record[1],
                "end_type": record[2],
                "session_id": record[3],
                "created_at": record[4] or 0,
                "edge_types": ["references", "references"],
                "edge_weights": [0.7, 0.7],
                "hops": 2,
                "bridge_entity": {
                    "name": record[5],
                    "type": record[6],
                },
            })

        log.debug(f"Found {len(paths)} cross-session paths")
        return paths

    @property
    def write_lock(self) -> threading.Lock:
        """Get the write lock for two-phase commit.

        IMPORTANT LIMITATION: This lock only provides protection for concurrent
        operations within a single process. It does NOT protect against:

        1. Multiple MCP server instances running simultaneously
        2. Multiple processes accessing the same database files
        3. External tools modifying LanceDB/FalkorDB directly

        For production deployments with multiple writers, consider:
        - Using a single server instance (recommended)
        - External distributed locking (e.g., Redis-based)
        - Database-level transactions (FalkorDB supports ACID)

        LanceDB specifically is NOT thread-safe for concurrent write operations,
        hence this lock is required even for single-process operation.
        """
        return self._write_lock

    def reset_all(self) -> dict[str, int]:
        """Reset all data - delete all memories and relationships.

        WARNING: This is destructive and irreversible. For debug purposes only.

        Returns:
            Dictionary with counts of deleted items
        """
        log.warning("RESET_ALL: Starting complete data wipe")

        with self._write_lock:
            # Count before deletion for reporting
            result = self.graph.query("MATCH (m:Memory) RETURN count(m)")
            memories_count = result.result_set[0][0] if result.result_set else 0

            result = self.graph.query("MATCH ()-[r]->() RETURN count(r)")
            relations_count = result.result_set[0][0] if result.result_set else 0

            # Delete everything in the graph
            log.debug("RESET_ALL: Deleting all graph data")
            self.graph.query("MATCH (n) DETACH DELETE n")

            # Recreate indexes
            self._graph_backend.init_schema()

            # Drop and recreate LanceDB table
            log.debug("RESET_ALL: Dropping LanceDB table")
            if self.VECTOR_TABLE_NAME in self.lance_db.table_names():
                self.lance_db.drop_table(self.VECTOR_TABLE_NAME)

            # Recreate table with schema
            log.debug("RESET_ALL: Recreating LanceDB table")
            self._init_lance_table()

        log.warning(f"RESET_ALL: Complete. Deleted {memories_count} memories, {relations_count} relations")
        return {
            "memories_deleted": memories_count,
            "relations_deleted": relations_count,
        }

    def _detach_lancedb_handles(self) -> None:
        """Best-effort disposal of LanceDB handles before directory rotation.

        LanceDB python APIs don't reliably expose a close() method that guarantees
        file handles are released. The critical piece is the global lock + dropping refs.
        """
        self.lance_table = None
        self.code_table = None
        self.lance_db = None
        gc.collect()

    def _rotate_vectors_dir(self, vectors_dir: Path) -> Path | None:
        """Atomically move vectors_dir aside and recreate it empty.

        Uses os.replace() for atomic rename on same filesystem.
        Falls back to shutil.rmtree() if rename fails.

        Args:
            vectors_dir: Path to the vectors directory

        Returns:
            Path to backup directory if rotation happened, None otherwise
        """
        vectors_dir = vectors_dir.resolve()
        parent = vectors_dir.parent
        parent.mkdir(parents=True, exist_ok=True)

        bak_dir: Path | None = None
        if vectors_dir.exists():
            ts = time.strftime("%Y%m%d-%H%M%S")
            bak_dir = parent / f"{vectors_dir.name}.bak.{ts}.{uuid.uuid4().hex[:8]}"
            try:
                os.replace(str(vectors_dir), str(bak_dir))  # atomic on same filesystem
                log.info(f"WIPE: Rotated vectors_dir to {bak_dir}")
            except OSError as e:
                # If atomic move fails (different filesystem, permissions, open handles),
                # fall back to delete-in-place (less ideal but better than failing wipe)
                log.warning(f"WIPE: Atomic rename failed ({e}), falling back to rmtree")
                shutil.rmtree(vectors_dir, ignore_errors=True)
                bak_dir = None

        vectors_dir.mkdir(parents=True, exist_ok=True)
        return bak_dir

    def _delete_dir_async(self, path: Path) -> None:
        """Fire-and-forget deletion of a directory.

        Safe for dev-mode wipes; avoids blocking the HTTP response.
        """
        def _worker(p: Path) -> None:
            try:
                shutil.rmtree(p, ignore_errors=True)
                log.debug(f"WIPE: Async deleted {p}")
            except Exception as e:
                log.warning(f"WIPE: Async delete failed for {p}: {e}")

        t = threading.Thread(target=_worker, args=(path,), name="vectors-bak-cleanup", daemon=True)
        t.start()

    def wipe_all_data(self, include_logs: bool = False) -> dict[str, Any]:
        """Wipe ALL data including databases, vectors, and metadata files.

        DEV MODE ONLY. This is a complete factory reset.

        Unlike reset_all() which only clears memories and vectors, this method
        also clears:
        - Code chunks table
        - Metadata files (projects.json, session_state.db, etc.)
        - Jobs directory
        - Optionally logs

        Args:
            include_logs: If True, also wipe the logs directory (default: False)

        Returns:
            Dictionary with detailed stats of what was deleted
        """
        log.warning("=" * 70)
        log.warning("  WIPE_ALL_DATA: Starting COMPLETE data wipe")
        log.warning("=" * 70)

        stats: dict[str, Any] = {
            "tables_dropped": [],
            "files_deleted": [],
            "directories_deleted": [],
            "graph_cleared": False,
            "memories_deleted": 0,
            "relations_deleted": 0,
            "code_chunks_deleted": 0,
            "wipe_mode": None,  # "soft" or "hard"
        }

        with self._write_lock:
            # 1. Count and clear graph
            try:
                result = self.graph.query("MATCH (m:Memory) RETURN count(m)")
                stats["memories_deleted"] = result.result_set[0][0] if result.result_set else 0

                result = self.graph.query("MATCH ()-[r]->() RETURN count(r)")
                stats["relations_deleted"] = result.result_set[0][0] if result.result_set else 0

                log.debug("WIPE: Deleting all graph data")
                self.graph.query("MATCH (n) DETACH DELETE n")
                self._graph_backend.init_schema()
                stats["graph_cleared"] = True
                log.info(f"WIPE: Graph cleared ({stats['memories_deleted']} memories, {stats['relations_deleted']} relations)")
            except Exception as e:
                log.error(f"WIPE: Failed to clear graph: {e}")
                stats["graph_error"] = str(e)

            # 2. Two-tier LanceDB wipe (soft + hard)
            # This handles corruption gracefully - if soft wipe fails, do a hard reset
            vectors_dir = Path(self.config.vectors_dir)
            tables_to_drop = [self.VECTOR_TABLE_NAME, self.CODE_TABLE_NAME]

            # Always detach handles first so they can't be used during wipe
            self._detach_lancedb_handles()

            # Tier A: Soft wipe - try drop_table directly (don't call table_names)
            soft_wipe_failed = False
            soft_error: str | None = None
            try:
                db = lancedb.connect(str(vectors_dir))

                for table_name in tables_to_drop:
                    try:
                        db.drop_table(table_name)
                        stats["tables_dropped"].append(table_name)
                        log.info(f"WIPE: Soft-dropped table '{table_name}'")
                    except Exception as e:
                        # Accept "not found"-like cases; anything else triggers hard wipe
                        msg = str(e).lower()
                        if "not found" in msg or "does not exist" in msg or "no such" in msg:
                            log.debug(f"WIPE: Table '{table_name}' not found (OK)")
                        else:
                            raise  # Re-raise to trigger hard wipe

                # Reconnect and init tables
                self.lance_db = lancedb.connect(str(vectors_dir))
                self._init_lance_table()
                if self.config.code_index_enabled:
                    self._init_code_table()
                stats["wipe_mode"] = "soft"
                log.info("WIPE: Soft wipe succeeded")

            except Exception as e:
                soft_wipe_failed = True
                soft_error = repr(e)
                log.warning(f"WIPE: Soft wipe failed ({soft_error}), falling back to hard wipe")

            # Tier B: Hard wipe - rotate entire vectors_dir and recreate fresh
            if soft_wipe_failed:
                bak_dir = self._rotate_vectors_dir(vectors_dir)

                # Fresh connect + init
                self.lance_db = lancedb.connect(str(vectors_dir))
                self._init_lance_table()
                if self.config.code_index_enabled:
                    self._init_code_table()

                stats["wipe_mode"] = "hard"
                stats["soft_error"] = soft_error
                if bak_dir:
                    stats["bak_dir"] = str(bak_dir)
                    stats["directories_deleted"].append(f"vectors/ (backed up to {bak_dir.name})")
                    # Schedule async deletion of backup
                    self._delete_dir_async(bak_dir)
                else:
                    stats["directories_deleted"].append("vectors/ (deleted in-place)")
                log.info(f"WIPE: Hard wipe succeeded (bak_dir={bak_dir})")

            # 4. Delete metadata files
            metadata_files = [
                "projects.json",
                "session_state.db",
                "pending_session.json",
                "status.json",
            ]
            for filename in metadata_files:
                filepath = self.config.data_dir / filename
                if filepath.exists():
                    try:
                        filepath.unlink()
                        stats["files_deleted"].append(filename)
                        log.info(f"WIPE: Deleted {filename}")
                    except Exception as e:
                        log.error(f"WIPE: Failed to delete {filename}: {e}")

            # 5. Clear jobs directory
            jobs_dir = self.config.data_dir / "jobs"
            if jobs_dir.exists():
                try:
                    shutil.rmtree(jobs_dir)
                    jobs_dir.mkdir(exist_ok=True)
                    stats["directories_deleted"].append("jobs/")
                    log.info("WIPE: Cleared jobs directory")
                except Exception as e:
                    log.error(f"WIPE: Failed to clear jobs directory: {e}")

            # 6. Optionally clear logs
            if include_logs:
                logs_dir = self.config.data_dir / "logs"
                if logs_dir.exists():
                    try:
                        shutil.rmtree(logs_dir)
                        logs_dir.mkdir(exist_ok=True)
                        stats["directories_deleted"].append("logs/")
                        log.info("WIPE: Cleared logs directory")
                    except Exception as e:
                        log.error(f"WIPE: Failed to clear logs directory: {e}")

        log.warning("=" * 70)
        log.warning(f"  WIPE_ALL_DATA: Complete (mode={stats['wipe_mode']})")
        log.warning(f"  Tables dropped: {stats['tables_dropped']}")
        log.warning(f"  Files deleted: {stats['files_deleted']}")
        log.warning(f"  Directories cleared: {stats['directories_deleted']}")
        if stats.get("soft_error"):
            log.warning(f"  Soft wipe error (recovered): {stats['soft_error']}")
        log.warning("=" * 70)

        return stats

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics including entity and verb edge counts.

        Returns:
            Dictionary with memory, entity, goal, and relationship counts
        """
        result = self.graph.query("MATCH (m:Memory) RETURN count(m)")
        memory_count = result.result_set[0][0] if result.result_set else 0

        result = self.graph.query("MATCH (e:Entity) RETURN count(e)")
        entity_count = result.result_set[0][0] if result.result_set else 0

        result = self.graph.query("MATCH (g:Goal) RETURN count(g)")
        goal_count = result.result_set[0][0] if result.result_set else 0

        result = self.graph.query("MATCH (p:Project) RETURN count(p)")
        project_count = result.result_set[0][0] if result.result_set else 0

        result = self.graph.query("MATCH ()-[r]->() RETURN count(r)")
        relation_count = result.result_set[0][0] if result.result_set else 0

        # Entity type breakdown
        result = self.graph.query(
            "MATCH (e:Entity) RETURN e.type, count(e) AS cnt ORDER BY cnt DESC"
        )
        entity_breakdown = {}
        for record in result.result_set:
            entity_breakdown[record[0]] = record[1]

        # Verb edge breakdown
        verb_edges = {}
        for edge_type in ["READS", "MODIFIES", "EXECUTES", "TRIGGERED"]:
            result = self.graph.query(f"MATCH ()-[r:{edge_type}]->() RETURN count(r)")
            verb_edges[edge_type.lower()] = result.result_set[0][0] if result.result_set else 0

        return {
            "memories": memory_count,
            "entities": entity_count,
            "goals": goal_count,
            "projects": project_count,
            "relations": relation_count,
            "entity_types": entity_breakdown,
            "verb_edges": verb_edges,
        }

    def get_sync_health(self, project_id: str | None = None) -> dict[str, Any]:
        """Check synchronization health between graph and vector stores.

        Detects memories that exist in the graph but are missing from LanceDB
        (the "desync" problem caused by non-atomic writes).

        Args:
            project_id: Optional project filter. If provided, only checks
                        memories belonging to that project.

        Returns:
            Dictionary with:
            - graph_count: Number of memories in graph
            - vector_count: Number of memories in LanceDB
            - missing_count: Number of memories in graph but not in vectors
            - missing_uuids: List of UUIDs missing from vectors (max 100)
            - sync_ratio: Ratio of synced memories (0.0 to 1.0)
            - healthy: True if sync_ratio >= 0.99
        """
        log.info(f"Checking sync health (project={project_id})")

        # Count and get UUIDs from graph
        if project_id:
            result = self.graph.query(
                "MATCH (m:Memory {project_id: $project_id}) RETURN m.uuid",
                {"project_id": project_id},
            )
        else:
            result = self.graph.query("MATCH (m:Memory) RETURN m.uuid")

        graph_uuids = set()
        if result.result_set:
            for row in result.result_set:
                if row[0]:  # Skip null UUIDs
                    graph_uuids.add(row[0])

        graph_count = len(graph_uuids)
        log.debug(f"Graph has {graph_count} memories")

        # Get UUIDs from LanceDB
        with self._write_lock:
            try:
                # Use to_pandas for efficient bulk read
                df = self.lance_table.to_pandas(columns=["uuid"])
                vector_uuids = set(df["uuid"].tolist())
            except Exception as e:
                log.warning(f"Failed to read LanceDB: {e}")
                vector_uuids = set()

        vector_count = len(vector_uuids)
        log.debug(f"LanceDB has {vector_count} vectors")

        # Find missing (in graph but not in vectors)
        missing_uuids = list(graph_uuids - vector_uuids)
        missing_count = len(missing_uuids)

        # Calculate sync ratio
        if graph_count == 0:
            sync_ratio = 1.0  # Empty is considered healthy
        else:
            sync_ratio = (graph_count - missing_count) / graph_count

        healthy = sync_ratio >= 0.99

        log.info(
            f"Sync health: graph={graph_count}, vectors={vector_count}, "
            f"missing={missing_count}, ratio={sync_ratio:.3f}, healthy={healthy}"
        )

        return {
            "graph_count": graph_count,
            "vector_count": vector_count,
            "missing_count": missing_count,
            "missing_uuids": missing_uuids[:100],  # Cap at 100 to avoid huge responses
            "sync_ratio": round(sync_ratio, 4),
            "healthy": healthy,
        }

    def repair_sync(
        self,
        project_id: str | None = None,
        dry_run: bool = True,
        batch_size: int = 50,
    ) -> dict[str, Any]:
        """Repair graph/vector desync by regenerating missing embeddings.

        Finds memories that exist in graph but are missing from LanceDB,
        generates embeddings, and writes them to the vector store.

        Args:
            project_id: Optional project filter
            dry_run: If True, only report what would be repaired (default: True)
            batch_size: Number of memories to embed per batch

        Returns:
            Dictionary with:
            - dry_run: Whether this was a dry run
            - missing_count: Number of memories needing repair
            - repaired_count: Number of memories repaired (0 if dry_run)
            - errors: List of UUIDs that failed to repair
            - missing_uuids: UUIDs that need repair (if dry_run)
        """
        from simplemem_lite.embeddings import embed_batch

        log.info(f"Starting repair_sync (project={project_id}, dry_run={dry_run})")

        # Get sync health to find missing UUIDs
        health = self.get_sync_health(project_id)
        missing_uuids = health["missing_uuids"]

        if not missing_uuids:
            log.info("No desync detected, nothing to repair")
            return {
                "dry_run": dry_run,
                "missing_count": 0,
                "repaired_count": 0,
                "errors": [],
            }

        # If sync_health capped at 100, get full list for repair
        if health["missing_count"] > len(missing_uuids):
            log.debug("Fetching full list of missing UUIDs for repair")
            if project_id:
                result = self.graph.query(
                    "MATCH (m:Memory {project_id: $project_id}) RETURN m.uuid",
                    {"project_id": project_id},
                )
            else:
                result = self.graph.query("MATCH (m:Memory) RETURN m.uuid")

            graph_uuids = {row[0] for row in result.result_set if row[0]}

            with self._write_lock:
                df = self.lance_table.to_pandas(columns=["uuid"])
                vector_uuids = set(df["uuid"].tolist())

            missing_uuids = list(graph_uuids - vector_uuids)

        missing_count = len(missing_uuids)
        log.info(f"Found {missing_count} memories needing repair")

        if dry_run:
            return {
                "dry_run": True,
                "missing_count": missing_count,
                "repaired_count": 0,
                "errors": [],
                "missing_uuids": missing_uuids[:100],  # Cap preview
            }

        # Actually repair: fetch content, generate embeddings, write vectors
        repaired_count = 0
        errors = []

        # Process in batches
        for i in range(0, missing_count, batch_size):
            batch_uuids = missing_uuids[i:i + batch_size]
            log.debug(f"Repairing batch {i // batch_size + 1}: {len(batch_uuids)} memories")

            try:
                # Fetch memory content from graph
                uuid_list = list(batch_uuids)
                result = self.graph.query(
                    """
                    MATCH (m:Memory)
                    WHERE m.uuid IN $uuids
                    RETURN m.uuid, m.content, m.type, m.session_id
                    """,
                    {"uuids": uuid_list},
                )

                if not result.result_set:
                    log.warning(f"No results for UUIDs: {uuid_list[:3]}...")
                    errors.extend(uuid_list)
                    continue

                # Prepare for batch embedding
                memories = []
                for row in result.result_set:
                    memories.append({
                        "uuid": row[0],
                        "content": row[1],
                        "type": row[2] or "fact",
                        "session_id": row[3] or "",
                    })

                if not memories:
                    continue

                # Generate embeddings in batch
                contents = [m["content"] for m in memories]
                embeddings = embed_batch(contents, self.config)

                if len(embeddings) != len(memories):
                    log.warning(f"Embedding count mismatch: {len(embeddings)} vs {len(memories)}")
                    # Process what we got
                    memories = memories[:len(embeddings)]

                # Write to LanceDB
                import json
                with self._write_lock:
                    records = [
                        {
                            "uuid": m["uuid"],
                            "vector": embeddings[i],
                            "content": m["content"],
                            "type": m["type"],
                            "session_id": m["session_id"],
                            "metadata": json.dumps({}),
                        }
                        for i, m in enumerate(memories)
                    ]
                    self.lance_table.add(records)

                repaired_count += len(memories)
                log.debug(f"Repaired {len(memories)} memories in batch")

            except Exception as e:
                log.error(f"Batch repair failed: {e}")
                errors.extend(batch_uuids)

        log.info(f"Repair complete: {repaired_count} repaired, {len(errors)} errors")

        return {
            "dry_run": False,
            "missing_count": missing_count,
            "repaired_count": repaired_count,
            "errors": errors[:100],  # Cap error list
        }

    def get_pagerank_scores(
        self,
        uuids: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute PageRank scores for Memory nodes.

        Delegates to the graph backend's PageRank implementation:
        - FalkorDB: algo.pageRank()
        - Memgraph: pagerank.get() via MAGE
        - KuzuDB: degree-based fallback

        Args:
            uuids: Optional list of UUIDs to get scores for (None = all)

        Returns:
            Dictionary mapping UUID -> PageRank score (0.0 to 1.0)
        """
        log.trace("Computing PageRank scores")

        try:
            # Delegate to graph backend (handles FalkorDB/Memgraph/KuzuDB differences)
            scores = self._graph_backend.get_pagerank_scores()

            log.debug(f"PageRank computed for {len(scores)} nodes")

            # Filter to requested UUIDs if specified
            if uuids:
                scores = {uuid: scores.get(uuid, 0.0) for uuid in uuids}

            return scores

        except Exception as e:
            log.warning(f"PageRank computation failed (may not be supported): {e}")
            # Return empty dict if PageRank not available
            return {}

    def get_pagerank_for_nodes(
        self,
        uuids: list[str],
    ) -> dict[str, float]:
        """Get PageRank scores for specific nodes.

        Computes PageRank on the full graph and filters to requested nodes.
        Falls back to degree-based scoring if PageRank is unavailable.

        Args:
            uuids: List of UUIDs to get scores for

        Returns:
            Dictionary mapping UUID -> PageRank score
        """
        log.trace(f"Getting PageRank for {len(uuids)} nodes")

        try:
            # Get PageRank for the whole graph
            all_scores = self.get_pagerank_scores()

            if all_scores:
                # Filter for the requested UUIDs
                scores = {uuid: all_scores.get(uuid, 0.0) for uuid in uuids}
                log.debug(f"PageRank retrieved for {len(scores)} of {len(uuids)} nodes")
                return scores

            # Empty result, fall through to fallback
            log.debug("PageRank returned no scores, using fallback")
            return self._get_degree_scores(uuids)

        except Exception as e:
            log.warning(f"PageRank failed, using degree-based fallback: {e}")
            # Fallback: use in-degree as a proxy for importance
            return self._get_degree_scores(uuids)

    def _get_degree_scores(self, uuids: list[str]) -> dict[str, float]:
        """Fallback: compute importance scores based on in-degree.

        Args:
            uuids: List of UUIDs

        Returns:
            Normalized in-degree scores
        """
        result = self.graph.query(
            """
            MATCH (m:Memory)
            WHERE m.uuid IN $uuids
            OPTIONAL MATCH (other)-[r]->(m)
            WITH m.uuid AS uuid, count(r) AS in_degree
            RETURN uuid, in_degree
            """,
            {"uuids": uuids},
        )

        # Get raw degrees
        degrees = {}
        max_degree = 1
        for record in result.result_set:
            degrees[record[0]] = record[1]
            max_degree = max(max_degree, record[1])

        # Normalize to 0-1 range
        scores = {uuid: degrees.get(uuid, 0) / max_degree for uuid in uuids}
        return scores

    # ═══════════════════════════════════════════════════════════════════════════════
    # ENTITY-CENTRIC QUERIES (P1 Resources Support)
    # ═══════════════════════════════════════════════════════════════════════════════

    def get_entities_by_type(
        self,
        entity_type: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all entities of a specific type with action counts.

        Args:
            entity_type: Type of entity (file, tool, error, command)
            limit: Maximum results to return

        Returns:
            List of entities with name, type, version, session_count, action counts
        """
        log.trace(f"Getting entities by type: {entity_type}")

        result = self.graph.query(
            """
            MATCH (e:Entity {type: $type})
            OPTIONAL MATCH (m:Memory)-[r:READS]->(e)
            WITH e, count(DISTINCT r) AS reads_count, collect(DISTINCT m.session_id) AS read_sessions
            OPTIONAL MATCH (m2:Memory)-[w:MODIFIES]->(e)
            WITH e, reads_count, read_sessions, count(DISTINCT w) AS modifies_count, collect(DISTINCT m2.session_id) AS modify_sessions
            OPTIONAL MATCH (m3:Memory)-[x:EXECUTES]->(e)
            WITH e, reads_count, read_sessions, modifies_count, modify_sessions,
                 count(DISTINCT x) AS executes_count, collect(DISTINCT m3.session_id) AS exec_sessions
            RETURN
                e.name AS name,
                e.type AS type,
                e.version AS version,
                e.created_at AS created_at,
                e.last_modified AS last_modified,
                reads_count,
                modifies_count,
                executes_count,
                read_sessions + modify_sessions + exec_sessions AS all_sessions
            ORDER BY reads_count + modifies_count + executes_count DESC
            LIMIT $limit
            """,
            {"type": entity_type, "limit": limit},
        )

        entities = []
        for record in result.result_set:
            # Deduplicate sessions
            all_sessions = list(set(s for s in (record[8] or []) if s))
            entities.append({
                "name": record[0],
                "type": record[1],
                "version": record[2] or 1,
                "created_at": record[3],
                "last_modified": record[4],
                "reads": record[5] or 0,
                "modifies": record[6] or 0,
                "executes": record[7] or 0,
                "sessions_count": len(all_sessions),
            })

        log.debug(f"Found {len(entities)} entities of type {entity_type}")
        return entities

    def get_entity_history(
        self,
        name: str,
        entity_type: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get complete history of a specific entity.

        Args:
            name: Entity name (will be canonicalized)
            entity_type: Entity type
            limit: Maximum memories to return

        Returns:
            Entity details with all linked memories and actions
        """
        log.trace(f"Getting entity history: {entity_type}:{name}")

        # Canonicalize the entity name for lookup
        canonical_name = self._canonicalize_entity(name, entity_type)

        # Get entity details
        entity_result = self.graph.query(
            """
            MATCH (e:Entity {name: $name, type: $type})
            RETURN e.name, e.type, e.version, e.created_at, e.last_modified
            """,
            {"name": canonical_name, "type": entity_type},
        )

        if not entity_result.result_set:
            log.debug(f"Entity not found: {entity_type}:{canonical_name}")
            return {"error": "Entity not found", "name": name, "type": entity_type}

        record = entity_result.result_set[0]
        entity = {
            "name": record[0],
            "type": record[1],
            "version": record[2] or 1,
            "created_at": record[3],
            "last_modified": record[4],
        }

        # Get all memories linked to this entity with their actions
        memories_result = self.graph.query(
            """
            MATCH (m:Memory)-[r]->(e:Entity {name: $name, type: $type})
            WHERE type(r) IN ['READS', 'MODIFIES', 'EXECUTES', 'TRIGGERED', 'REFERENCES']
            RETURN
                m.uuid AS uuid,
                m.content AS content,
                m.type AS mem_type,
                m.session_id AS session_id,
                m.created_at AS created_at,
                type(r) AS action,
                r.timestamp AS action_timestamp,
                r.change_summary AS change_summary
            ORDER BY m.created_at DESC
            LIMIT $limit
            """,
            {"name": canonical_name, "type": entity_type, "limit": limit},
        )

        memories = []
        sessions = set()
        for record in memories_result.result_set:
            memories.append({
                "uuid": record[0],
                "content": record[1][:300] if record[1] else "",  # Truncate for display
                "type": record[2],
                "session_id": record[3],
                "created_at": record[4],
                "action": record[5],
                "action_timestamp": record[6],
                "change_summary": record[7],
            })
            if record[3]:
                sessions.add(record[3])

        # Get linked errors (for files/tools)
        errors_result = self.graph.query(
            """
            MATCH (m:Memory)-[:READS|MODIFIES|EXECUTES]->(e:Entity {name: $name, type: $type})
            MATCH (m)-[:TRIGGERED]->(err:Entity {type: 'error'})
            RETURN DISTINCT err.name AS error_name, count(m) AS occurrences
            ORDER BY occurrences DESC
            LIMIT 10
            """,
            {"name": canonical_name, "type": entity_type},
        )

        errors = []
        for record in errors_result.result_set:
            errors.append({
                "error": record[0],
                "occurrences": record[1],
            })

        log.debug(f"Entity history: {len(memories)} memories, {len(sessions)} sessions, {len(errors)} errors")

        return {
            "entity": entity,
            "memories": memories,
            "sessions": list(sessions),
            "sessions_count": len(sessions),
            "related_errors": errors,
        }

    def get_cross_session_entities(
        self,
        min_sessions: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get entities that appear across multiple sessions (bridge entities).

        These are valuable for cross-session insights as they link
        different work sessions together.

        Args:
            min_sessions: Minimum number of sessions the entity must appear in
            limit: Maximum results to return

        Returns:
            List of entities with session count and linked sessions
        """
        log.trace(f"Getting cross-session entities (min_sessions={min_sessions})")

        result = self.graph.query(
            """
            MATCH (m:Memory)-[r]->(e:Entity)
            WHERE type(r) IN ['READS', 'MODIFIES', 'EXECUTES', 'TRIGGERED', 'REFERENCES']
              AND m.session_id IS NOT NULL AND m.session_id <> ''
            WITH e, collect(DISTINCT m.session_id) AS sessions
            WHERE size(sessions) >= $min_sessions
            RETURN
                e.name AS name,
                e.type AS type,
                e.version AS version,
                sessions AS session_ids,
                size(sessions) AS sessions_count
            ORDER BY size(sessions) DESC
            LIMIT $limit
            """,
            {"min_sessions": min_sessions, "limit": limit},
        )

        entities = []
        for record in result.result_set:
            entities.append({
                "name": record[0],
                "type": record[1],
                "version": record[2] or 1,
                "session_ids": record[3] or [],
                "sessions_count": record[4] or 0,
            })

        log.debug(f"Found {len(entities)} cross-session entities")
        return entities

    def get_project_insights(
        self,
        project_id: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get all learnings and insights for a specific project.

        Args:
            project_id: Project identifier
            limit: Maximum items per category

        Returns:
            Project overview with sessions, entities, errors, and learnings
        """
        log.trace(f"Getting project insights: {project_id}")

        # Get project node
        project_result = self.graph.query(
            """
            MATCH (p:Project {id: $project_id})
            RETURN p.id, p.path, p.session_count, p.created_at, p.last_updated
            """,
            {"project_id": project_id},
        )

        if not project_result.result_set:
            return {"error": "Project not found", "project_id": project_id}

        record = project_result.result_set[0]
        project = {
            "id": record[0],
            "path": record[1],
            "session_count": record[2] or 0,
            "created_at": record[3],
            "last_updated": record[4],
        }

        # Get sessions belonging to this project
        sessions_result = self.graph.query(
            """
            MATCH (m:Memory)-[:BELONGS_TO]->(p:Project {id: $project_id})
            WHERE m.type = 'session_summary'
            RETURN m.uuid, m.content, m.session_id, m.created_at
            ORDER BY m.created_at DESC
            LIMIT $limit
            """,
            {"project_id": project_id, "limit": limit},
        )

        sessions = []
        for record in sessions_result.result_set:
            sessions.append({
                "uuid": record[0],
                "summary": record[1][:500] if record[1] else "",
                "session_id": record[2],
                "created_at": record[3],
            })

        # Get frequently touched files in this project
        files_result = self.graph.query(
            """
            MATCH (m:Memory)-[:BELONGS_TO]->(p:Project {id: $project_id})
            MATCH (m)-[:CONTAINS*0..2]->(chunk:Memory)
            MATCH (chunk)-[:READS|MODIFIES]->(e:Entity {type: 'file'})
            WITH e.name AS file_name, count(*) AS touch_count
            RETURN file_name, touch_count
            ORDER BY touch_count DESC
            LIMIT $limit
            """,
            {"project_id": project_id, "limit": limit},
        )

        files = []
        for record in files_result.result_set:
            files.append({
                "file": record[0],
                "touches": record[1],
            })

        # Get errors encountered in this project
        errors_result = self.graph.query(
            """
            MATCH (m:Memory)-[:BELONGS_TO]->(p:Project {id: $project_id})
            MATCH (m)-[:CONTAINS*0..2]->(chunk:Memory)
            MATCH (chunk)-[:TRIGGERED]->(e:Entity {type: 'error'})
            WITH e.name AS error_name, count(*) AS occurrences
            RETURN error_name, occurrences
            ORDER BY occurrences DESC
            LIMIT $limit
            """,
            {"project_id": project_id, "limit": limit},
        )

        errors = []
        for record in errors_result.result_set:
            errors.append({
                "error": record[0],
                "occurrences": record[1],
            })

        log.debug(f"Project insights: {len(sessions)} sessions, {len(files)} files, {len(errors)} errors")

        return {
            "project": project,
            "sessions": sessions,
            "top_files": files,
            "errors": errors,
        }

    def get_error_patterns(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get common errors and their resolution patterns.

        Returns errors with occurrence counts and any linked solutions.

        Args:
            limit: Maximum results

        Returns:
            List of errors with occurrences and sessions
        """
        log.trace("Getting error patterns")

        result = self.graph.query(
            """
            MATCH (m:Memory)-[r:TRIGGERED]->(e:Entity {type: 'error'})
            WITH e, collect(DISTINCT m.session_id) AS sessions, count(r) AS trigger_count
            RETURN
                e.name AS error_name,
                trigger_count,
                sessions,
                size(sessions) AS sessions_count
            ORDER BY trigger_count DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )

        errors = []
        for record in result.result_set:
            errors.append({
                "error": record[0],
                "occurrences": record[1],
                "sessions": [s for s in (record[2] or []) if s],
                "sessions_count": record[3] or 0,
            })

        log.debug(f"Found {len(errors)} error patterns")
        return errors

    def get_tool_usage(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get tool usage patterns across sessions.

        Args:
            limit: Maximum results

        Returns:
            List of tools with execution counts and sessions
        """
        log.trace("Getting tool usage patterns")

        result = self.graph.query(
            """
            MATCH (m:Memory)-[r:EXECUTES]->(e:Entity {type: 'tool'})
            WITH e, collect(DISTINCT m.session_id) AS sessions, count(r) AS exec_count
            RETURN
                e.name AS tool_name,
                exec_count,
                sessions,
                size(sessions) AS sessions_count
            ORDER BY exec_count DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )

        tools = []
        for record in result.result_set:
            tools.append({
                "tool": record[0],
                "executions": record[1],
                "sessions": [s for s in (record[2] or []) if s],
                "sessions_count": record[3] or 0,
            })

        log.debug(f"Found {len(tools)} tool usage patterns")
        return tools

    def get_memories_in_project(
        self,
        project_id: str,
        uuids: list[str],
    ) -> set[str]:
        """Filter memory UUIDs to only those belonging to a specific project.

        A memory belongs to a project if:
        1. It's a session_summary directly linked to the project (BELONGS_TO)
        2. It's a child of a session_summary linked to the project (CONTAINS chain)

        Args:
            project_id: Project identifier
            uuids: List of memory UUIDs to filter

        Returns:
            Set of UUIDs that belong to the project
        """
        log.trace(f"Filtering {len(uuids)} memories for project: {project_id}")

        if not uuids:
            return set()

        result = self.graph.query(
            """
            // First, find memories with direct project_id property match
            UNWIND $uuids AS uuid
            MATCH (m:Memory {uuid: uuid})
            WHERE m.project_id = $project_id
            RETURN DISTINCT m.uuid

            UNION

            // Also find memories via session-based relationships
            MATCH (p:Project {id: $project_id})
            OPTIONAL MATCH (session:Memory)-[:BELONGS_TO]->(p)
            WITH collect(session.uuid) AS session_uuids
            UNWIND $uuids AS uuid
            MATCH (m:Memory {uuid: uuid})
            OPTIONAL MATCH (parent:Memory)-[:CONTAINS*1..3]->(m)
            WHERE parent.uuid IN session_uuids
            WITH m, session_uuids, parent
            WHERE m.uuid IN session_uuids OR parent IS NOT NULL
            RETURN DISTINCT m.uuid
            """,
            {"project_id": project_id, "uuids": uuids},
        )

        project_uuids = {record[0] for record in result.result_set}
        log.debug(f"Project filter: {len(project_uuids)}/{len(uuids)} memories belong to {project_id}")
        return project_uuids

    def get_projects(self, limit: int = 50) -> list[dict]:
        """Get all tracked projects sorted by session count.

        Args:
            limit: Maximum number of projects to return

        Returns:
            List of project dicts with id, path, session_count, created_at, last_updated
        """
        log.trace(f"Fetching projects, limit={limit}")

        result = self.graph.query(
            """
            MATCH (p:Project)
            RETURN p.id, p.path, p.session_count, p.created_at, p.last_updated
            ORDER BY p.session_count DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )

        projects = []
        for record in result.result_set:
            projects.append({
                "id": record[0],
                "path": record[1],
                "session_count": record[2] or 0,
                "created_at": record[3],
                "last_updated": record[4],
            })

        log.debug(f"Retrieved {len(projects)} projects")
        return projects

    def cleanup_orphan_entities(self) -> dict[str, Any]:
        """Manually run orphan entity cleanup.

        Finds and deletes Entity nodes with no incoming edges
        (not referenced by any Memory or CodeChunk).

        Returns:
            Dict with orphans_deleted count and any errors
        """
        log.info("Running manual orphan entity cleanup")
        try:
            with self._write_lock:
                deleted = self._cleanup_orphan_entities()
            return {
                "orphans_deleted": deleted,
                "status": "success",
            }
        except Exception as e:
            log.error(f"Orphan cleanup failed: {e}")
            return {
                "orphans_deleted": 0,
                "status": "error",
                "error": str(e),
            }

    # =========================================================================
    # Review Queue Methods (Consolidation Candidates)
    # =========================================================================

    def make_dedupe_key(
        self,
        project_id: str,
        candidate_type: str,
        involved_ids: list[str],
    ) -> str:
        """Create deterministic key for candidate deduplication.

        Args:
            project_id: Project scope
            candidate_type: entity_dedup | memory_merge | supersession
            involved_ids: UUIDs or entity names involved in the candidate

        Returns:
            16-character hex hash for idempotent MERGE operations
        """
        import hashlib
        sorted_ids = sorted(involved_ids)
        raw = f"{project_id}|{candidate_type}|{','.join(sorted_ids)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def add_review_candidate(
        self,
        project_id: str,
        candidate_type: str,
        confidence: float,
        reason: str,
        decision_data: dict[str, Any],
        involved_ids: list[str],
        source_id: str,
        target_id: str,
        similarity: float,
    ) -> dict[str, Any]:
        """Persist a consolidation candidate for human review.

        Uses MERGE with dedupe_key for idempotency - same candidate won't be
        duplicated across multiple consolidation runs.

        Args:
            project_id: Project scope
            candidate_type: entity_dedup | memory_merge | supersession
            confidence: LLM confidence score (typically 0.7-0.9 for review queue)
            reason: LLM reasoning for the suggested action
            decision_data: Type-specific decision details (stored as JSON)
            involved_ids: UUIDs or entity names for dedupe_key generation
            source_id: First involved ID (for filtering)
            target_id: Second involved ID (for filtering)
            similarity: Embedding similarity score (for filtering)

        Returns:
            {"uuid": str, "created": bool} - created=False if existing candidate
        """
        import json
        import time
        import uuid as uuid_lib

        dedupe_key = self.make_dedupe_key(project_id, candidate_type, involved_ids)

        # Check if already exists (including rejected)
        existing = self.graph.query(
            """
            MATCH (c:ReviewCandidate {dedupe_key: $dedupe_key})
            RETURN c.uuid, c.status
            """,
            {"dedupe_key": dedupe_key},
        )

        if existing.result_set:
            existing_uuid, existing_status = existing.result_set[0]
            # If rejected, don't recreate - honor the rejection
            if existing_status == "rejected":
                return {"uuid": existing_uuid, "created": False, "skipped": "rejected"}
            # If pending or approved, return existing
            return {"uuid": existing_uuid, "created": False}

        # Create new candidate
        candidate_uuid = str(uuid_lib.uuid4())
        now = int(time.time())

        self.graph.query(
            """
            CREATE (c:ReviewCandidate {
                uuid: $uuid,
                dedupe_key: $dedupe_key,
                project_id: $project_id,
                type: $type,
                status: 'pending',
                confidence: $confidence,
                reason: $reason,
                source_id: $source_id,
                target_id: $target_id,
                similarity: $similarity,
                decision_data: $decision_data,
                schema_version: 1,
                created_at: $now,
                resolved_at: NULL
            })
            """,
            {
                "uuid": candidate_uuid,
                "dedupe_key": dedupe_key,
                "project_id": project_id,
                "type": candidate_type,
                "confidence": confidence,
                "reason": reason,
                "source_id": source_id,
                "target_id": target_id,
                "similarity": similarity,
                "decision_data": json.dumps(decision_data),
                "now": now,
            },
        )

        log.debug(f"Created ReviewCandidate {candidate_uuid} for {candidate_type}")
        return {"uuid": candidate_uuid, "created": True}

    def get_review_candidates(
        self,
        project_id: str,
        status: str = "pending",
        type_filter: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch review candidates with optional filters.

        Args:
            project_id: Project scope
            status: Filter by status (pending, approved, rejected)
            type_filter: Filter by type (entity_dedup, memory_merge, supersession)
            limit: Maximum candidates to return

        Returns:
            List of candidate dicts with all fields
        """
        import json

        query = """
            MATCH (c:ReviewCandidate {project_id: $project_id, status: $status})
        """
        params: dict[str, Any] = {
            "project_id": project_id,
            "status": status,
            "limit": limit,
        }

        if type_filter:
            query += " WHERE c.type = $type_filter"
            params["type_filter"] = type_filter

        query += """
            RETURN c.uuid, c.dedupe_key, c.project_id, c.type, c.status,
                   c.confidence, c.reason, c.source_id, c.target_id,
                   c.similarity, c.decision_data, c.schema_version,
                   c.created_at, c.resolved_at
            ORDER BY c.created_at DESC
            LIMIT $limit
        """

        result = self.graph.query(query, params)

        candidates = []
        for record in result.result_set:
            decision_data_str = record[10]
            try:
                decision_data = json.loads(decision_data_str) if decision_data_str else {}
            except (json.JSONDecodeError, TypeError):
                decision_data = {}

            candidates.append({
                "uuid": record[0],
                "dedupe_key": record[1],
                "project_id": record[2],
                "type": record[3],
                "status": record[4],
                "confidence": record[5],
                "reason": record[6],
                "source_id": record[7],
                "target_id": record[8],
                "similarity": record[9],
                "decision_data": decision_data,
                "schema_version": record[11],
                "created_at": record[12],
                "resolved_at": record[13],
            })

        log.debug(f"Retrieved {len(candidates)} review candidates for {project_id}")
        return candidates

    def get_review_candidate(self, uuid: str) -> dict[str, Any] | None:
        """Fetch a single review candidate by UUID.

        Args:
            uuid: Candidate UUID

        Returns:
            Candidate dict or None if not found
        """
        import json

        result = self.graph.query(
            """
            MATCH (c:ReviewCandidate {uuid: $uuid})
            RETURN c.uuid, c.dedupe_key, c.project_id, c.type, c.status,
                   c.confidence, c.reason, c.source_id, c.target_id,
                   c.similarity, c.decision_data, c.schema_version,
                   c.created_at, c.resolved_at
            """,
            {"uuid": uuid},
        )

        if not result.result_set:
            return None

        record = result.result_set[0]
        decision_data_str = record[10]
        try:
            decision_data = json.loads(decision_data_str) if decision_data_str else {}
        except (json.JSONDecodeError, TypeError):
            decision_data = {}

        return {
            "uuid": record[0],
            "dedupe_key": record[1],
            "project_id": record[2],
            "type": record[3],
            "status": record[4],
            "confidence": record[5],
            "reason": record[6],
            "source_id": record[7],
            "target_id": record[8],
            "similarity": record[9],
            "decision_data": decision_data,
            "schema_version": record[11],
            "created_at": record[12],
            "resolved_at": record[13],
        }

    def update_candidate_status(
        self,
        uuid: str,
        status: str,
        resolved_at: int | None = None,
    ) -> bool:
        """Update a review candidate's status.

        Idempotent - updating an already-resolved candidate returns False
        without making changes.

        Args:
            uuid: Candidate UUID
            status: New status (approved | rejected)
            resolved_at: Timestamp when resolved (defaults to now)

        Returns:
            True if updated, False if already resolved or not found
        """
        import time

        if resolved_at is None:
            resolved_at = int(time.time())

        # Only update if currently pending
        result = self.graph.query(
            """
            MATCH (c:ReviewCandidate {uuid: $uuid, status: 'pending'})
            SET c.status = $status, c.resolved_at = $resolved_at
            RETURN c.uuid
            """,
            {
                "uuid": uuid,
                "status": status,
                "resolved_at": resolved_at,
            },
        )

        updated = bool(result.result_set)
        if updated:
            log.debug(f"Updated ReviewCandidate {uuid} to status={status}")
        else:
            log.debug(f"ReviewCandidate {uuid} not updated (already resolved or not found)")
        return updated

    def add_rejected_pair(
        self,
        uuid1: str,
        uuid2: str,
        candidate_uuid: str,
    ) -> None:
        """Create REJECTED_PAIR edge between two entities/memories.

        Used to skip pairs in future consolidation runs after human rejection.
        Stores pair in normalized order (pair_a=min, pair_b=max) for consistent lookup.

        Args:
            uuid1: First UUID/entity name
            uuid2: Second UUID/entity name
            candidate_uuid: UUID of the rejected ReviewCandidate
        """
        import time

        # Normalize order for consistent lookup
        pair_a = min(uuid1, uuid2)
        pair_b = max(uuid1, uuid2)
        now = int(time.time())

        # Create relationship between the two items
        # We store as a standalone node since the entities might be Memory or Entity
        self.graph.query(
            """
            MERGE (r:RejectedPair {pair_a: $pair_a, pair_b: $pair_b})
            ON CREATE SET r.rejected_at = $now,
                          r.candidate_uuid = $candidate_uuid
            """,
            {
                "pair_a": pair_a,
                "pair_b": pair_b,
                "candidate_uuid": candidate_uuid,
                "now": now,
            },
        )
        log.debug(f"Added RejectedPair: {pair_a} <-> {pair_b}")

    def is_rejected_pair(self, uuid1: str, uuid2: str) -> bool:
        """Check if a pair was previously rejected.

        Uses normalized lookup (pair_a=min, pair_b=max) so order doesn't matter.

        Args:
            uuid1: First UUID/entity name
            uuid2: Second UUID/entity name

        Returns:
            True if pair was rejected, False otherwise
        """
        pair_a = min(uuid1, uuid2)
        pair_b = max(uuid1, uuid2)

        result = self.graph.query(
            """
            MATCH (r:RejectedPair {pair_a: $pair_a, pair_b: $pair_b})
            RETURN r.pair_a
            """,
            {"pair_a": pair_a, "pair_b": pair_b},
        )

        return bool(result.result_set)
