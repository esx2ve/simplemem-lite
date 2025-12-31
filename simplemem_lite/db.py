"""Database management for SimpleMem Lite.

Handles FalkorDB (graph) and LanceDB (vectors) with two-phase commit.
"""

import threading
from typing import Any

import lancedb
import pyarrow as pa
from falkordb import FalkorDB

from simplemem_lite.config import Config
from simplemem_lite.logging import get_logger

log = get_logger("db")


class DatabaseManager:
    """Manages FalkorDB and LanceDB connections with consistency guarantees.

    Implements a simple two-phase commit pattern:
    1. Write to graph (FalkorDB) first - source of truth
    2. Write to vectors (LanceDB) second
    3. Rollback graph if vector write fails

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
        self._write_lock = threading.Lock()

        # Initialize FalkorDB (requires running instance)
        log.debug(f"Connecting to FalkorDB at {self.config.falkor_host}:{self.config.falkor_port}")
        self.falkor_db = FalkorDB(
            host=self.config.falkor_host,
            port=self.config.falkor_port,
        )
        self.graph = self.falkor_db.select_graph(self.GRAPH_NAME)
        self._init_graph_indexes()
        log.info(f"FalkorDB connected: graph={self.GRAPH_NAME}")

        # Initialize LanceDB (it creates its own directory)
        log.debug(f"Initializing LanceDB at {self.config.vectors_dir}")
        self.lance_db = lancedb.connect(str(self.config.vectors_dir))
        self._init_lance_table()
        log.info(f"LanceDB initialized at {self.config.vectors_dir}")

    def _init_graph_indexes(self) -> None:
        """Create indexes for efficient lookups."""
        log.trace("Creating FalkorDB indexes")
        try:
            # Create indexes for Memory nodes
            self.graph.query("CREATE INDEX FOR (m:Memory) ON (m.uuid)")
            self.graph.query("CREATE INDEX FOR (m:Memory) ON (m.type)")
            self.graph.query("CREATE INDEX FOR (m:Memory) ON (m.session_id)")
            # Create indexes for Entity nodes
            self.graph.query("CREATE INDEX FOR (e:Entity) ON (e.name)")
            self.graph.query("CREATE INDEX FOR (e:Entity) ON (e.type)")
            # Create indexes for CodeChunk nodes (P1: entity linking)
            self.graph.query("CREATE INDEX FOR (c:CodeChunk) ON (c.uuid)")
            self.graph.query("CREATE INDEX FOR (c:CodeChunk) ON (c.filepath)")
            # Create index for ProjectIndex nodes (P2: staleness detection)
            self.graph.query("CREATE INDEX FOR (p:ProjectIndex) ON (p.project_root)")
            log.debug("FalkorDB indexes created")
        except Exception as e:
            # Indexes may already exist
            log.trace(f"Index creation (may already exist): {e}")

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
        """Initialize LanceDB table for code chunks."""
        log.trace("Checking code_chunks table")
        if self.CODE_TABLE_NAME not in self.lance_db.table_names():
            log.debug(f"Creating code_chunks table with embedding_dim={self.config.embedding_dim}")
            schema = pa.schema([
                pa.field("uuid", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.config.embedding_dim)),
                pa.field("content", pa.string()),
                pa.field("filepath", pa.string()),
                pa.field("project_root", pa.string()),
                pa.field("start_line", pa.int32()),
                pa.field("end_line", pa.int32()),
            ])
            self.lance_db.create_table(self.CODE_TABLE_NAME, schema=schema)
            log.info(f"LanceDB table '{self.CODE_TABLE_NAME}' created")
        else:
            log.debug(f"LanceDB table '{self.CODE_TABLE_NAME}' already exists")

        self.code_table = self.lance_db.open_table(self.CODE_TABLE_NAME)

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

    def add_memory_node(
        self,
        uuid: str,
        content: str,
        mem_type: str,
        source: str,
        session_id: str | None,
        created_at: int,
    ) -> None:
        """Add a memory node to the graph.

        Args:
            uuid: Unique identifier
            content: Memory content
            mem_type: Memory type (fact, session_summary, chunk_summary, message)
            source: Source of memory (claude_trace, user, extracted)
            session_id: Optional session identifier
            created_at: Unix timestamp
        """
        log.trace(f"Adding memory node: uuid={uuid[:8]}..., type={mem_type}")
        # Parameterized queries handle escaping automatically
        self.graph.query(
            """
            CREATE (m:Memory {
                uuid: $uuid,
                content: $content,
                type: $type,
                source: $source,
                session_id: $session_id,
                created_at: $created_at
            })
            """,
            {
                "uuid": uuid,
                "content": content[:5000],  # Limit content size
                "type": mem_type,
                "source": source,
                "session_id": session_id or "",
                "created_at": created_at,
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
    ) -> None:
        """Add a memory vector to LanceDB.

        Args:
            uuid: Unique identifier (foreign key to graph)
            vector: Embedding vector
            content: Memory content (for retrieval)
            mem_type: Memory type
            session_id: Optional session identifier
        """
        self.lance_table.add([
            {
                "uuid": uuid,
                "vector": vector,
                "content": content,
                "type": mem_type,
                "session_id": session_id or "",
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

    # Allowed verb edge types for security validation
    ALLOWED_VERB_EDGES = {"READS", "MODIFIES", "EXECUTES", "TRIGGERED", "REFERENCES"}

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

        # Create the verb-specific edge
        ts = timestamp or int(time.time())
        if change_summary:
            self.graph.query(
                f"""
                MATCH (m:Memory {{uuid: $uuid}}), (e:Entity {{name: $name, type: $type}})
                CREATE (m)-[:{edge_type} {{timestamp: $ts, change_summary: $summary}}]->(e)
                """,
                {"uuid": memory_uuid, "name": canonical_name, "type": entity_type, "ts": ts, "summary": change_summary[:500]},
            )
        else:
            self.graph.query(
                f"""
                MATCH (m:Memory {{uuid: $uuid}}), (e:Entity {{name: $name, type: $type}})
                CREATE (m)-[:{edge_type} {{timestamp: $ts}}]->(e)
                """,
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
                {"uuid": memory_uuid, "name": safe_name, "type": entity_type},
            )
            reads_exists = check_result.result_set and check_result.result_set[0][0] > 0

            if not reads_exists:
                # Create implicit READS edge (before the modify)
                self.graph.query(
                    """
                    MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
                    CREATE (m)-[:READS {timestamp: $ts, implicit: true}]->(e)
                    """,
                    {"uuid": memory_uuid, "name": safe_name, "type": entity_type, "ts": ts - 1},
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
                "intent": intent[:1000],
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
            {"project_id": project_id, "path": project_path[:500]},
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

        # Delete from vectors
        # LanceDB delete requires SQL-style filter
        for uuid in uuids:
            try:
                self.lance_table.delete(f'uuid = "{uuid}"')
            except Exception as e:
                log.trace(f"Vector delete failed (may not exist): {e}")

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
        search = self.lance_table.search(query_vector).limit(limit)

        if type_filter:
            search = search.where(f"type = '{type_filter}'")

        results = search.to_list()
        log.debug(f"Vector search returned {len(results)} results")
        return results

    # ═══════════════════════════════════════════════════════════════════════════════
    # CODE SEARCH METHODS
    # ═══════════════════════════════════════════════════════════════════════════════

    def add_code_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Add code chunks to the code index.

        Args:
            chunks: List of dicts with keys: uuid, vector, content, filepath, project_root, start_line, end_line

        Returns:
            Number of chunks added
        """
        if not self.config.code_index_enabled:
            log.warning("Code indexing disabled, skipping add_code_chunks")
            return 0

        if not chunks:
            return 0

        log.info(f"Adding {len(chunks)} code chunks to index")
        self.code_table.add(chunks)
        log.debug(f"Code chunks added successfully")
        return len(chunks)

    def search_code(
        self,
        query_vector: list[float],
        limit: int = 10,
        project_root: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar code chunks.

        Args:
            query_vector: Query embedding
            limit: Maximum results
            project_root: Optional filter by project root path

        Returns:
            List of matching code chunks with distance scores
        """
        if not self.config.code_index_enabled:
            log.warning("Code indexing disabled, returning empty results")
            return []

        log.trace(f"Searching code: limit={limit}, project_root={project_root}")
        search = self.code_table.search(query_vector).limit(limit)

        if project_root:
            search = search.where(f"project_root = '{project_root}'")

        results = search.to_list()
        log.debug(f"Code search returned {len(results)} results")
        return results

    def clear_code_index(self, project_root: str | None = None) -> int:
        """Clear code index for a project or all projects.

        Args:
            project_root: If provided, only clear chunks from this project

        Returns:
            Number of chunks deleted
        """
        if not self.config.code_index_enabled:
            return 0

        if project_root:
            log.info(f"Clearing code index for project: {project_root}")
            # Count before delete
            try:
                count = len(self.code_table.search([0.0] * self.config.embedding_dim)
                           .where(f"project_root = '{project_root}'")
                           .limit(100000).to_list())
            except Exception:
                count = 0
            self.code_table.delete(f"project_root = '{project_root}'")
        else:
            log.info("Clearing entire code index")
            try:
                count = self.code_table.count_rows()
            except Exception:
                count = 0
            # Drop and recreate table
            self.lance_db.drop_table(self.CODE_TABLE_NAME)
            self._init_code_table()

        log.info(f"Cleared {count} code chunks")
        return count

    def delete_chunks_by_filepath(
        self,
        project_root: str,
        filepath: str,
    ) -> int:
        """Delete all chunks for a specific file (for incremental updates).

        Args:
            project_root: Project root path
            filepath: Relative file path within the project

        Returns:
            Number of chunks deleted
        """
        if not self.config.code_index_enabled:
            return 0

        log.info(f"Deleting chunks for file: {filepath} in {project_root}")

        # Sanitize inputs to prevent issues with special characters
        safe_project_root = project_root.replace("'", "''")
        safe_filepath = filepath.replace("'", "''")
        where_clause = f"project_root = '{safe_project_root}' AND filepath = '{safe_filepath}'"

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

            log.info(f"Deleted {count} chunks for {filepath}")
            return count

    def get_code_stats(self, project_root: str | None = None) -> dict[str, Any]:
        """Get statistics about the code index.

        Args:
            project_root: Optional filter by project

        Returns:
            Dict with chunk_count and unique files
        """
        if not self.config.code_index_enabled:
            return {"enabled": False, "chunk_count": 0}

        try:
            total = self.code_table.count_rows()
            # Get sample to count unique files (approximate)
            sample = self.code_table.search([0.0] * self.config.embedding_dim).limit(1000).to_list()
            unique_files = len(set(r.get("filepath", "") for r in sample))
            return {
                "enabled": True,
                "chunk_count": total,
                "unique_files_sample": unique_files,
                "project_root": project_root,
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
        project_root: str,
        start_line: int,
        end_line: int,
    ) -> None:
        """Add a CodeChunk node to the graph for entity linking.

        Args:
            uuid: Unique identifier (matches LanceDB)
            filepath: Relative file path
            project_root: Project root path
            start_line: Starting line number
            end_line: Ending line number
        """
        log.trace(f"Adding CodeChunk node: {filepath}:{start_line}-{end_line}")
        self.graph.query(
            """
            MERGE (c:CodeChunk {uuid: $uuid})
            ON CREATE SET
                c.filepath = $filepath,
                c.project_root = $project_root,
                c.start_line = $start_line,
                c.end_line = $end_line,
                c.created_at = timestamp()
            """,
            {
                "uuid": uuid,
                "filepath": filepath,
                "project_root": project_root,
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
                c.project_root AS project_root,
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
                "project_root": record[2],
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
        project_root: str,
        commit_hash: str | None,
        file_count: int,
        chunk_count: int,
    ) -> None:
        """Store/update project index metadata in graph.

        Args:
            project_root: Absolute path to project root
            commit_hash: Git commit hash at time of indexing (None if not a git repo)
            file_count: Number of files indexed
            chunk_count: Number of chunks created
        """
        log.info(f"Setting project index metadata: {project_root} (hash={commit_hash[:8] if commit_hash else 'N/A'})")
        self.graph.query(
            """
            MERGE (p:ProjectIndex {project_root: $project_root})
            SET p.last_commit_hash = $commit_hash,
                p.indexed_at = timestamp(),
                p.file_count = $file_count,
                p.chunk_count = $chunk_count
            """,
            {
                "project_root": project_root,
                "commit_hash": commit_hash,
                "file_count": file_count,
                "chunk_count": chunk_count,
            },
        )

    def get_project_index_metadata(self, project_root: str) -> dict[str, Any] | None:
        """Get project index metadata from graph.

        Args:
            project_root: Absolute path to project root

        Returns:
            Dict with project_root, last_commit_hash, indexed_at, file_count, chunk_count
            or None if not indexed
        """
        log.trace(f"Getting project index metadata: {project_root}")
        result = self.graph.query(
            """
            MATCH (p:ProjectIndex {project_root: $project_root})
            RETURN p.project_root, p.last_commit_hash, p.indexed_at, p.file_count, p.chunk_count
            """,
            {"project_root": project_root},
        )

        if not result.result_set:
            return None

        row = result.result_set[0]
        return {
            "project_root": row[0],
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
        hops = min(max(hops, 1), 3)  # Clamp to 1-3

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
        max_hops = min(max(max_hops, 1), 3)

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
            LIMIT 100
            """,
            {"uuid": from_uuid},
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
            LIMIT 50
            """,
            {"uuid": from_uuid},
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
        """Get the write lock for two-phase commit."""
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
            self._init_graph_indexes()

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
            "MATCH (e:Entity) RETURN e.type, count(e) ORDER BY count(e) DESC"
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

    def get_pagerank_scores(
        self,
        uuids: list[str] | None = None,
        max_iterations: int = 20,
        damping_factor: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank scores for Memory nodes.

        Uses FalkorDB's built-in PageRank algorithm to compute node importance
        based on graph structure. Nodes with more high-quality incoming edges
        get higher scores.

        Args:
            uuids: Optional list of UUIDs to get scores for (None = all)
            max_iterations: PageRank iterations (default: 20)
            damping_factor: PageRank damping factor (default: 0.85)

        Returns:
            Dictionary mapping UUID -> PageRank score (0.0 to 1.0)
        """
        log.trace("Computing PageRank scores")

        try:
            # Call FalkorDB's PageRank algorithm
            # Note: FalkorDB uses algo.pageRank procedure
            result = self.graph.query(
                """
                CALL algo.pageRank('Memory', 'RELATES_TO', {
                    maxIterations: $max_iter,
                    dampingFactor: $damping
                })
                YIELD node, score
                MATCH (m:Memory) WHERE id(m) = id(node)
                RETURN m.uuid AS uuid, score
                """,
                {
                    "max_iter": max_iterations,
                    "damping": damping_factor,
                },
            )

            scores = {}
            for record in result.result_set:
                scores[record[0]] = record[1]

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
            MATCH (p:Project {id: $project_id})
            OPTIONAL MATCH (session:Memory)-[:BELONGS_TO]->(p)
            WITH collect(session.uuid) AS session_uuids

            // Get all memories that are session summaries or children of session summaries
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
