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
