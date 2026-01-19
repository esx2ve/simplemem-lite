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
from simplemem_lite.db.consolidation import ConsolidationManager
from simplemem_lite.db.entity_repository import EntityRepository
from simplemem_lite.db.graph_analytics import GraphAnalytics
from simplemem_lite.db.graph_factory import create_graph_backend, get_backend_info
from simplemem_lite.db.graph_protocol import GraphBackend
from simplemem_lite.db.relationship_manager import RelationshipManager
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

        # Initialize extracted managers (god class decomposition)
        self._consolidation = ConsolidationManager(self.graph)
        self._graph_analytics = GraphAnalytics(
            graph=self.graph,
            pagerank_backend=self._graph_backend,
            max_graph_hops=config.max_graph_hops,
            graph_path_limit=config.graph_path_limit,
            cross_session_limit=config.cross_session_limit,
        )
        self._entity_repository = EntityRepository(
            graph=self.graph,
            summary_max_size=config.summary_max_size,
        )
        self._relationship_manager = RelationshipManager(
            graph=self.graph,
            max_supersession_depth=10,
        )

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
        """Initialize LanceDB table for code chunks with AST metadata.

        Schema includes:
        - Core fields: uuid, vector, content, filepath, project_id, start_line, end_line
        - AST metadata: function_name, class_name, language, node_type, signature
        - Provenance: indexed_at, embedding_model, embedding_dim

        NOTE: Vector dimension is determined by the embedding provider at runtime.
        The configured embedding_dim is used for the schema, but actual vectors
        may have different dimensions based on which provider is used (Voyage=1024,
        OpenRouter=3072, Local=768). The embed_code_batch function handles
        dimension validation to prevent mismatches.

        Includes corruption recovery: if the table exists but is corrupted,
        it will be dropped and recreated automatically.
        """
        log.trace("Checking code_chunks table")

        # Use code embedding dimension from config (1024 for Voyage)
        code_dim = self.config.code_embedding_dim

        schema = pa.schema([
            # Core fields
            pa.field("uuid", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), code_dim)),
            pa.field("content", pa.string()),
            pa.field("filepath", pa.string()),
            pa.field("project_id", pa.string()),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
            # AST metadata fields (from Tree-sitter chunking)
            pa.field("function_name", pa.string()),  # Name of function/method
            pa.field("class_name", pa.string()),     # Name of containing class
            pa.field("language", pa.string()),       # Programming language
            pa.field("node_type", pa.string()),      # function|class|method|module|interface|type
            pa.field("signature", pa.string()),      # Function/class signature for skeleton mode
            # Timestamp and embedding provenance
            pa.field("indexed_at", pa.string()),     # ISO timestamp
            pa.field("embedding_model", pa.string()), # Model used (voyage-code-3, etc.)
            pa.field("embedding_dim", pa.int32()),   # Dimension for validation
        ])

        if self.CODE_TABLE_NAME not in self.lance_db.table_names():
            log.debug(f"Creating code_chunks table with code_embedding_dim={code_dim}")
            self.lance_db.create_table(self.CODE_TABLE_NAME, schema=schema)
            log.info(f"LanceDB table '{self.CODE_TABLE_NAME}' created")
        else:
            log.debug(f"LanceDB table '{self.CODE_TABLE_NAME}' already exists")

        # Open table and verify it's not corrupted
        try:
            self.code_table = self.lance_db.open_table(self.CODE_TABLE_NAME)
            # Verify table is readable with a simple operation
            _ = self.code_table.count_rows()

            # Schema migration: check for missing columns and add them
            existing_fields = {f.name for f in self.code_table.schema}
            required_fields = {f.name for f in schema}
            missing_fields = required_fields - existing_fields

            if missing_fields:
                log.warning(f"Code table missing columns: {missing_fields}. Recreating table.")
                # LanceDB doesn't support ALTER TABLE, so we must recreate
                # This is acceptable for code index as it can be rebuilt
                try:
                    self.lance_db.drop_table(self.CODE_TABLE_NAME)
                except Exception:
                    pass
                self.lance_db.create_table(self.CODE_TABLE_NAME, schema=schema)
                self.code_table = self.lance_db.open_table(self.CODE_TABLE_NAME)
                log.info(f"LanceDB table '{self.CODE_TABLE_NAME}' recreated with updated schema")

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

        Delegates to EntityRepository.
        """
        return self._entity_repository.add_entity_node(name, entity_type, metadata)

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

        Delegates to RelationshipManager.

        Args:
            from_uuid: Source memory UUID
            to_uuid: Target memory UUID
            relation_type: Type of relationship
            weight: Relationship weight (default: 1.0)
        """
        self._relationship_manager.add_relationship(from_uuid, to_uuid, relation_type, weight)

    def would_create_supersession_cycle(
        self,
        newer_uuid: str,
        older_uuid: str,
        max_depth: int = 10,
    ) -> bool:
        """Check if creating a SUPERSEDES edge would create a cycle.

        Delegates to RelationshipManager.

        Args:
            newer_uuid: UUID of the newer/superseding memory
            older_uuid: UUID of the older/superseded memory
            max_depth: Maximum traversal depth to check (default: 10)

        Returns:
            True if creating the edge would create a cycle, False otherwise
        """
        return self._relationship_manager.would_create_supersession_cycle(newer_uuid, older_uuid, max_depth)

    def add_supersession(
        self,
        newer_uuid: str,
        older_uuid: str,
        confidence: float,
        supersession_type: str = "full_replace",
        reason: str | None = None,
    ) -> bool:
        """Mark that a newer memory supersedes an older one.

        Delegates to RelationshipManager.

        Args:
            newer_uuid: UUID of the newer/superseding memory
            older_uuid: UUID of the older/superseded memory
            confidence: Confidence score (0.0-1.0) from LLM classifier
            supersession_type: "full_replace" or "partial_update"
            reason: Optional explanation of why this supersession occurred

        Returns:
            True if edge was created, False if blocked (cycle or same node)
        """
        return self._relationship_manager.add_supersession(
            newer_uuid, older_uuid, confidence, supersession_type, reason
        )

    def mark_merged(
        self,
        source_uuid: str,
        target_uuid: str,
    ) -> None:
        """Mark a memory as merged into another (soft delete).

        Delegates to RelationshipManager.

        Args:
            source_uuid: UUID of memory that was merged (will be marked)
            target_uuid: UUID of memory it was merged into
        """
        self._relationship_manager.mark_merged(source_uuid, target_uuid)

    def get_superseded_memories(
        self,
        project_id: str | None = None,
    ) -> list[dict]:
        """Get all superseded memory UUIDs for exclusion from search.

        Delegates to RelationshipManager.

        Args:
            project_id: Optional project filter

        Returns:
            List of dicts with older_uuid and superseding_uuid
        """
        return self._relationship_manager.get_superseded_memories(project_id)

    def get_merged_memories(
        self,
        project_id: str | None = None,
    ) -> list[dict]:
        """Get all merged memory UUIDs for exclusion from search.

        Delegates to RelationshipManager.

        Args:
            project_id: Optional project filter

        Returns:
            List of dicts with source_uuid and target_uuid
        """
        return self._relationship_manager.get_merged_memories(project_id)

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

        Delegates to EntityRepository.
        """
        self._entity_repository.add_verb_edge(
            memory_uuid, entity_name, entity_type, action, timestamp, change_summary
        )

    def _canonicalize_entity(self, name: str, entity_type: str) -> str:
        """Canonicalize entity name for consistent deduplication.

        Delegates to EntityRepository.
        """
        return self._entity_repository._canonicalize_entity(name, entity_type)

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
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding
            limit: Maximum results (before project filtering)
            type_filter: Optional filter by memory type
            project_id: Optional filter by project ID (Python-side filtering)

        Returns:
            List of matching memories with distance scores
        """
        log.trace(f"Searching vectors: limit={limit}, type_filter={type_filter}, project_id={project_id}")

        # Fetch more results if we need to filter by project_id in Python
        fetch_limit = limit * 3 if project_id else limit

        with self._write_lock:  # LanceDB not thread-safe for concurrent ops
            try:
                # Check for empty table first to avoid LanceDB search-on-empty bug
                row_count = self.lance_table.count_rows()
                if row_count == 0:
                    log.debug("Memory table is empty, returning no results")
                    return []

                # LanceDB 0.1 pattern with explicit vector_column_name and query_type
                search = self.lance_table.search(
                    query_vector,
                    vector_column_name="vector",
                    query_type="auto"
                ).limit(fetch_limit)

                if type_filter:
                    search = search.where(f"type = '{type_filter}'")

                results = search.to_list()
            except Exception as e:
                # Handle LanceDB corruption/empty table errors gracefully
                error_msg = str(e)
                if "Invalid range" in error_msg or "empty" in error_msg.lower():
                    log.warning(f"LanceDB table appears corrupted or empty: {e}")
                    return []
                raise  # Re-raise unexpected errors

        log.debug(f"Vector search returned {len(results)} results before filtering")

        # Python-side filtering for project_id (LanceDB 0.1 doesn't support JSON querying)
        if project_id and results:
            import json
            filtered_results = []
            for result in results:
                try:
                    metadata = json.loads(result.get("metadata", "{}"))
                    if metadata.get("project_id") == project_id:
                        filtered_results.append(result)
                        if len(filtered_results) >= limit:
                            break
                except (json.JSONDecodeError, AttributeError):
                    continue
            results = filtered_results
            log.debug(f"After project_id filter: {len(results)} results")

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
        project_id: str | None = None,
    ) -> int:
        """Write/update vectors in LanceDB for given memory UUIDs.

        Strategy: Delete existing by UUID, then add new.
        LanceDB lacks native upsert, so we implement it manually.

        Args:
            memories: List of memory dicts with uuid, content, type, session_id
            vectors: Corresponding embedding vectors
            project_id: Project ID to store in metadata for filtering

        Returns:
            Number of vectors written
        """
        import json

        if not memories or not vectors:
            return 0

        if len(memories) != len(vectors):
            raise ValueError(f"Mismatched counts: {len(memories)} memories, {len(vectors)} vectors")

        log.debug(f"Upserting {len(memories)} memory vectors for project={project_id}")

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
                # Build metadata with project_id for search filtering
                metadata = {"reindexed": True}
                if project_id:
                    metadata["project_id"] = project_id
                records.append({
                    "uuid": mem["uuid"],
                    "vector": vec,
                    "content": mem["content"],
                    "type": mem.get("type", "fact"),
                    "session_id": mem.get("session_id") or "",
                    "metadata": json.dumps(metadata),
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
            chunks: List of dicts with keys: uuid, vector, content, filepath, project_id,
                    start_line, end_line, function_name, class_name, language, node_type,
                    signature (optional), indexed_at, embedding_model, embedding_dim

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
            try:
                # Check for empty/uninitialized table first
                if self.code_table is None:
                    log.debug("Code table not initialized, returning no results")
                    return []
                row_count = self.code_table.count_rows()
                if row_count == 0:
                    log.debug("Code table is empty, returning no results")
                    return []

                search = self.code_table.search(query_vector).limit(limit)

                if project_id:
                    safe_project_id = project_id.replace("'", "''")
                    search = search.where(f"project_id = '{safe_project_id}'")

                results = search.to_list()
            except Exception as e:
                # Handle LanceDB corruption/empty table errors gracefully
                error_msg = str(e)
                if "Invalid range" in error_msg or "empty" in error_msg.lower():
                    log.warning(f"Code table appears corrupted or empty: {e}")
                    return []
                raise  # Re-raise unexpected errors

        log.debug(f"Code search returned {len(results)} results")
        return results

    def get_code_chunk_by_uuid(self, uuid: str) -> dict | None:
        """Retrieve a specific code chunk by UUID.

        Args:
            uuid: Code chunk UUID

        Returns:
            Code chunk dict if found, None otherwise
        """
        if not self.config.code_index_enabled:
            return None

        log.trace(f"Getting code chunk: uuid={uuid[:8]}...")

        with self._write_lock:  # LanceDB not thread-safe for concurrent ops
            try:
                safe_uuid = uuid.replace("'", "''")
                results = self.code_table.search([0.0] * self.config.code_embedding_dim).where(
                    f"uuid = '{safe_uuid}'"
                ).limit(1).to_list()

                if results:
                    log.trace(f"Found code chunk: filepath={results[0].get('filepath', '')[:50]}")
                    return results[0]

            except Exception as e:
                log.warning(f"Failed to get code chunk by UUID: {e}")

        log.trace(f"Code chunk not found: uuid={uuid[:8]}...")
        return None

    def get_code_embedding_dimension(self, project_id: str | None = None) -> int | None:
        """Get the embedding dimension of existing code chunks for a project.

        Used for dimension mismatch protection: when adding new chunks, we need
        to ensure the embedding dimension matches existing data to prevent
        LanceDB errors and inconsistent search results.

        Args:
            project_id: Optional filter by project ID. If None, checks any project.

        Returns:
            Embedding dimension if chunks exist, None if no chunks found.
            Returns None if code indexing is disabled.
        """
        if not self.config.code_index_enabled:
            return None

        log.trace(f"Checking code embedding dimension for project: {project_id}")

        with self._write_lock:
            try:
                # Query for one existing chunk to get its dimension
                search = self.code_table.search(
                    [0.0] * self.config.code_embedding_dim
                ).limit(1)

                if project_id:
                    safe_project_id = project_id.replace("'", "''")
                    search = search.where(f"project_id = '{safe_project_id}'")

                results = search.to_list()

                if results and "embedding_dim" in results[0]:
                    dim = results[0]["embedding_dim"]
                    log.debug(f"Found existing code chunks with dim={dim} for project={project_id}")
                    return dim
                elif results and "vector" in results[0]:
                    # Fallback: infer dimension from vector length
                    dim = len(results[0]["vector"])
                    log.debug(f"Inferred embedding dim={dim} from vector length for project={project_id}")
                    return dim

            except Exception as e:
                log.warning(f"Failed to get code embedding dimension: {e}")

        log.trace(f"No existing code chunks found for project: {project_id}")
        return None

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
                # Count before delete - use count_rows with filter to avoid search-on-empty bug
                try:
                    # LanceDB search fails on empty tables, use filter + count instead
                    total_rows = self.code_table.count_rows()
                    if total_rows == 0:
                        count = 0
                    else:
                        # Table has rows, safe to search
                        count = len(self.code_table.search([0.0] * self.config.code_embedding_dim)
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
                    matches = self.code_table.search([0.0] * self.config.code_embedding_dim).where(
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
                sample = self.code_table.search([0.0] * self.config.code_embedding_dim).limit(1000).to_list()
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

        Delegates to EntityRepository.
        """
        self._entity_repository.link_code_to_entity(chunk_uuid, entity_name, entity_type, relation)

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

        Delegates to GraphAnalytics.
        """
        return self._graph_analytics.get_related_nodes(uuid, hops, direction)

    def get_paths(
        self,
        from_uuid: str,
        max_hops: int = 2,
        direction: str = "outgoing",
    ) -> list[dict[str, Any]]:
        """Get paths from a node with full edge metadata.

        Delegates to GraphAnalytics.
        """
        return self._graph_analytics.get_paths(from_uuid, max_hops, direction)

    def get_cross_session_paths(
        self,
        from_uuid: str,
        max_hops: int = 3,
    ) -> list[dict[str, Any]]:
        """Get paths that cross sessions via shared Entity nodes.

        Delegates to GraphAnalytics.
        """
        return self._graph_analytics.get_cross_session_paths(from_uuid, max_hops)

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
                # Use to_arrow for efficient bulk read (no pandas dependency)
                arrow_table = self.lance_table.to_arrow()
                vector_uuids = set(arrow_table.column("uuid").to_pylist())
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
                arrow_table = self.lance_table.to_arrow()
                vector_uuids = set(arrow_table.column("uuid").to_pylist())

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

        Delegates to GraphAnalytics.
        """
        return self._graph_analytics.get_pagerank_scores(uuids)

    def get_pagerank_for_nodes(
        self,
        uuids: list[str],
    ) -> dict[str, float]:
        """Get PageRank scores for specific nodes.

        Delegates to GraphAnalytics.
        """
        return self._graph_analytics.get_pagerank_for_nodes(uuids)

    def _get_degree_scores(self, uuids: list[str]) -> dict[str, float]:
        """Fallback: compute importance scores based on in-degree.

        Delegates to GraphAnalytics.
        """
        return self._graph_analytics._get_degree_scores(uuids)

    # ═══════════════════════════════════════════════════════════════════════════════
    # ENTITY-CENTRIC QUERIES (P1 Resources Support)
    # ═══════════════════════════════════════════════════════════════════════════════

    def get_entities_by_type(
        self,
        entity_type: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all entities of a specific type with action counts.

        Delegates to EntityRepository.

        Args:
            entity_type: Type of entity (file, tool, error, command)
            limit: Maximum results to return

        Returns:
            List of entities with name, type, version, session_count, action counts
        """
        return self._entity_repository.get_entities_by_type(entity_type, limit)

    def get_entity_history(
        self,
        name: str,
        entity_type: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get complete history of a specific entity.

        Delegates to EntityRepository.

        Args:
            name: Entity name (will be canonicalized)
            entity_type: Entity type
            limit: Maximum memories to return

        Returns:
            Entity details with all linked memories and actions
        """
        return self._entity_repository.get_entity_history(name, entity_type, limit)

    def get_cross_session_entities(
        self,
        min_sessions: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get entities that appear across multiple sessions (bridge entities).

        Delegates to GraphAnalytics.
        """
        return self._graph_analytics.get_cross_session_entities(min_sessions, limit)

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
    # Delegated to ConsolidationManager - kept for backward compatibility
    # =========================================================================

    def make_dedupe_key(
        self,
        project_id: str,
        candidate_type: str,
        involved_ids: list[str],
    ) -> str:
        """Create deterministic key for candidate deduplication.

        Delegates to ConsolidationManager.
        """
        return self._consolidation.make_dedupe_key(project_id, candidate_type, involved_ids)

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

        Delegates to ConsolidationManager.
        """
        return self._consolidation.add_review_candidate(
            project_id=project_id,
            candidate_type=candidate_type,
            confidence=confidence,
            reason=reason,
            decision_data=decision_data,
            involved_ids=involved_ids,
            source_id=source_id,
            target_id=target_id,
            similarity=similarity,
        )

    def get_review_candidates(
        self,
        project_id: str,
        status: str = "pending",
        type_filter: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch review candidates with optional filters.

        Delegates to ConsolidationManager.
        """
        return self._consolidation.get_review_candidates(
            project_id=project_id,
            status=status,
            type_filter=type_filter,
            limit=limit,
        )

    def get_review_candidate(self, uuid: str) -> dict[str, Any] | None:
        """Fetch a single review candidate by UUID.

        Delegates to ConsolidationManager.
        """
        return self._consolidation.get_review_candidate(uuid)

    def update_candidate_status(
        self,
        uuid: str,
        status: str,
        resolved_at: int | None = None,
    ) -> bool:
        """Update a review candidate's status.

        Delegates to ConsolidationManager.
        """
        return self._consolidation.update_candidate_status(uuid, status, resolved_at)

    def add_rejected_pair(
        self,
        uuid1: str,
        uuid2: str,
        candidate_uuid: str,
    ) -> None:
        """Create REJECTED_PAIR edge between two entities/memories.

        Delegates to ConsolidationManager.
        """
        return self._consolidation.add_rejected_pair(uuid1, uuid2, candidate_uuid)

    def is_rejected_pair(self, uuid1: str, uuid2: str) -> bool:
        """Check if a pair was previously rejected.

        Delegates to ConsolidationManager.
        """
        return self._consolidation.is_rejected_pair(uuid1, uuid2)

    # =========================================================================
    # Graph-Enhanced Scoring Methods (MAGE Integration)
    # =========================================================================

    def get_memory_degrees(
        self,
        uuids: list[str],
        project_id: str | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get in-degree and out-degree for a list of memory UUIDs.

        Delegates to GraphAnalytics.
        """
        return self._graph_analytics.get_memory_degrees(uuids, project_id)

    def get_graph_normalization_stats(
        self,
        project_id: str | None = None,
    ) -> dict[str, float]:
        """Get statistics for normalizing graph scores.

        Delegates to GraphAnalytics.
        """
        return self._graph_analytics.get_graph_normalization_stats(project_id)

    def compute_and_cache_pagerank(
        self,
        project_id: str | None = None,
        max_iterations: int = 100,
        damping_factor: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank using MAGE and cache scores on Memory nodes.

        Delegates to GraphAnalytics.
        """
        return self._graph_analytics.compute_and_cache_pagerank(
            project_id, max_iterations, damping_factor
        )

    def get_memory_pageranks(
        self,
        uuids: list[str],
    ) -> dict[str, float]:
        """Get cached PageRank scores for a list of memory UUIDs.

        Delegates to GraphAnalytics.
        """
        return self._graph_analytics.get_memory_pageranks(uuids)
