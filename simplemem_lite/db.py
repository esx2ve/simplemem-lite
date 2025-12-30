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
        # Escape content for Cypher (replace quotes and backslashes)
        safe_content = content.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

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
                "content": safe_content[:5000],  # Limit content size
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
        safe_name = name.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

        # MERGE creates if not exists, returns existing if exists
        self.graph.query(
            """
            MERGE (e:Entity {name: $name, type: $type})
            ON CREATE SET e.created_at = timestamp()
            """,
            {
                "name": safe_name,
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

    def add_entity_reference(
        self,
        memory_uuid: str,
        entity_name: str,
        entity_type: str,
        weight: float = 0.7,
    ) -> None:
        """Link a memory to an entity (creates entity if not exists).

        DEPRECATED: Use add_verb_edge() for more specific relationship types.

        Args:
            memory_uuid: Memory UUID
            entity_name: Entity name
            entity_type: Entity type (file, tool, error, concept)
            weight: Relationship weight
        """
        safe_name = entity_name.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

        # First ensure entity exists
        self.add_entity_node(entity_name, entity_type)

        # Then create the reference
        self.graph.query(
            """
            MATCH (m:Memory {uuid: $uuid}), (e:Entity {name: $name, type: $type})
            CREATE (m)-[:REFERENCES {weight: $weight}]->(e)
            """,
            {
                "uuid": memory_uuid,
                "name": safe_name,
                "type": entity_type,
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
        safe_name = canonical_name.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

        # Map action to edge type with security validation
        action_to_edge = {
            "reads": "READS",
            "modifies": "MODIFIES",
            "executes": "EXECUTES",
            "triggered": "TRIGGERED",
        }
        edge_type = action_to_edge.get(action.lower(), "REFERENCES")

        # Security: validate edge type to prevent Cypher injection
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
                {"name": safe_name, "type": entity_type},
            )
        else:
            # Just ensure entity exists
            self.graph.query(
                """
                MERGE (e:Entity {name: $name, type: $type})
                ON CREATE SET e.created_at = timestamp(), e.version = 1
                """,
                {"name": safe_name, "type": entity_type},
            )

        # Create the verb-specific edge
        ts = timestamp or int(time.time())
        if change_summary:
            safe_summary = change_summary.replace("\\", "\\\\").replace("'", "\\'")[:500]
            self.graph.query(
                f"""
                MATCH (m:Memory {{uuid: $uuid}}), (e:Entity {{name: $name, type: $type}})
                CREATE (m)-[:{edge_type} {{timestamp: $ts, change_summary: $summary}}]->(e)
                """,
                {"uuid": memory_uuid, "name": safe_name, "type": entity_type, "ts": ts, "summary": safe_summary},
            )
        else:
            self.graph.query(
                f"""
                MATCH (m:Memory {{uuid: $uuid}}), (e:Entity {{name: $name, type: $type}})
                CREATE (m)-[:{edge_type} {{timestamp: $ts}}]->(e)
                """,
                {"uuid": memory_uuid, "name": safe_name, "type": entity_type, "ts": ts},
            )

        log.trace(f"Added {edge_type} edge: memory={memory_uuid[:8]}... -> {entity_type}:{canonical_name}")

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
            # Normalize file paths - resolve to absolute, handle symlinks
            try:
                if os.path.isabs(name):
                    return os.path.normpath(name)
                # For relative paths, just normalize
                return os.path.normpath(name)
            except Exception:
                return name

        elif entity_type == "tool":
            # Normalize tool names - lowercase, strip common prefixes
            normalized = name.lower()
            normalized = normalized.replace("mcp__", "").replace("__", ":")
            return normalized

        elif entity_type == "command":
            # Extract base command (first word)
            parts = name.strip().split()
            return parts[0].lower() if parts else name

        elif entity_type == "error":
            # Hash error type + message prefix for deduplication
            error_hash = hashlib.sha256(name[:200].encode()).hexdigest()[:16]
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
        safe_intent = intent.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')[:1000]

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
                "intent": safe_intent,
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

    def migrate_references_to_verbs(self) -> dict[str, int]:
        """Migrate existing REFERENCES edges to verb-specific edges.

        Converts based on entity type:
        - file -> READS (safe default)
        - tool -> EXECUTES
        - command -> EXECUTES
        - error -> TRIGGERED

        Returns:
            Dictionary with migration counts
        """
        log.info("Starting migration of REFERENCES edges to verb-specific edges")

        counts = {"reads": 0, "executes": 0, "triggered": 0, "total": 0}

        # Get all REFERENCES edges with their entity types
        result = self.graph.query(
            """
            MATCH (m:Memory)-[r:REFERENCES]->(e:Entity)
            RETURN m.uuid, e.name, e.type, r.weight
            """
        )

        for record in result.result_set:
            memory_uuid, entity_name, entity_type, weight = record
            counts["total"] += 1

            # Determine target edge type
            if entity_type == "file":
                edge_type = "READS"
                counts["reads"] += 1
            elif entity_type in ("tool", "command"):
                edge_type = "EXECUTES"
                counts["executes"] += 1
            elif entity_type == "error":
                edge_type = "TRIGGERED"
                counts["triggered"] += 1
            else:
                edge_type = "READS"
                counts["reads"] += 1

            # Create new edge
            safe_name = entity_name.replace("\\", "\\\\").replace("'", "\\'")
            self.graph.query(
                f"""
                MATCH (m:Memory {{uuid: $uuid}}), (e:Entity {{name: $name, type: $type}})
                CREATE (m)-[:{edge_type} {{timestamp: timestamp(), migrated: true}}]->(e)
                """,
                {"uuid": memory_uuid, "name": safe_name, "type": entity_type},
            )

        # Delete old REFERENCES edges
        self.graph.query("MATCH ()-[r:REFERENCES]->() DELETE r")

        log.info(f"Migration complete: {counts}")
        return counts

    def delete_memory_node(self, uuid: str) -> None:
        """Delete a memory node from the graph (for rollback).

        Args:
            uuid: Memory UUID to delete
        """
        self.graph.query(
            "MATCH (m:Memory {uuid: $uuid}) DETACH DELETE m",
            {"uuid": uuid},
        )

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
        """Get graph statistics including entity counts.

        Returns:
            Dictionary with memory, entity, and relationship counts
        """
        result = self.graph.query("MATCH (m:Memory) RETURN count(m)")
        memory_count = result.result_set[0][0] if result.result_set else 0

        result = self.graph.query("MATCH (e:Entity) RETURN count(e)")
        entity_count = result.result_set[0][0] if result.result_set else 0

        result = self.graph.query("MATCH ()-[r]->() RETURN count(r)")
        relation_count = result.result_set[0][0] if result.result_set else 0

        result = self.graph.query(
            "MATCH (e:Entity) RETURN e.type, count(e) ORDER BY count(e) DESC"
        )
        entity_breakdown = {}
        for record in result.result_set:
            entity_breakdown[record[0]] = record[1]

        return {
            "memories": memory_count,
            "entities": entity_count,
            "relations": relation_count,
            "entity_types": entity_breakdown,
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
