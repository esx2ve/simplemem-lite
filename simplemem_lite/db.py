"""Database management for SimpleMem Lite.

Handles KuzuDB (graph) and LanceDB (vectors) with two-phase commit.
"""

import threading
from typing import Any

import kuzu
import lancedb
import pyarrow as pa

from simplemem_lite.config import Config
from simplemem_lite.logging import get_logger

log = get_logger("db")


class DatabaseManager:
    """Manages KuzuDB and LanceDB connections with consistency guarantees.

    Implements a simple two-phase commit pattern:
    1. Write to graph (KuzuDB) first - source of truth
    2. Write to vectors (LanceDB) second
    3. Rollback graph if vector write fails
    """

    VECTOR_TABLE_NAME = "memories"

    def __init__(self, config: Config):
        """Initialize database connections.

        Args:
            config: SimpleMem Lite configuration
        """
        log.trace("DatabaseManager.__init__ starting")
        self.config = config
        self._write_lock = threading.Lock()

        # Initialize KuzuDB (it creates its own directory)
        log.debug(f"Initializing KuzuDB at {self.config.graph_dir}")
        self.kuzu_db = kuzu.Database(str(self.config.graph_dir))
        self.kuzu_conn = kuzu.Connection(self.kuzu_db)
        self._init_kuzu_schema()
        log.info(f"KuzuDB initialized at {self.config.graph_dir}")

        # Initialize LanceDB (it creates its own directory)
        log.debug(f"Initializing LanceDB at {self.config.vectors_dir}")
        self.lance_db = lancedb.connect(str(self.config.vectors_dir))
        self._init_lance_table()
        log.info(f"LanceDB initialized at {self.config.vectors_dir}")

    def _init_kuzu_schema(self) -> None:
        """Initialize KuzuDB schema if not exists."""
        log.trace("Checking KuzuDB schema")
        # Check if Memory table exists
        try:
            self.kuzu_conn.execute("MATCH (m:Memory) RETURN m LIMIT 1")
            log.debug("KuzuDB schema already exists")
        except Exception as e:
            log.debug(f"Creating KuzuDB schema (table check failed: {e})")
            # Create schema
            self.kuzu_conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Memory (
                    uuid STRING PRIMARY KEY,
                    content STRING,
                    type STRING,
                    source STRING,
                    session_id STRING,
                    created_at INT64
                )
            """)

            self.kuzu_conn.execute("""
                CREATE REL TABLE IF NOT EXISTS RELATES_TO (
                    FROM Memory TO Memory,
                    relation_type STRING,
                    weight DOUBLE
                )
            """)
            log.info("KuzuDB schema created")

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
            return self.kuzu_conn.execute(query, params)
        return self.kuzu_conn.execute(query)

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
        self.kuzu_conn.execute(
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
                "content": content,
                "type": mem_type,
                "source": source,
                "session_id": session_id or "",
                "created_at": created_at,
            },
        )

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
        self.kuzu_conn.execute(
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

    def delete_memory_node(self, uuid: str) -> None:
        """Delete a memory node from the graph (for rollback).

        Args:
            uuid: Memory UUID to delete
        """
        self.kuzu_conn.execute(
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
            pattern = f"(start:Memory {{uuid: $uuid}})-[r:RELATES_TO*1..{hops}]->(connected:Memory)"
        elif direction == "incoming":
            pattern = f"(start:Memory {{uuid: $uuid}})<-[r:RELATES_TO*1..{hops}]-(connected:Memory)"
        else:
            pattern = f"(start:Memory {{uuid: $uuid}})-[r:RELATES_TO*1..{hops}]-(connected:Memory)"

        result = self.kuzu_conn.execute(
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
        while result.has_next():
            row = result.get_next()
            rows.append({
                "uuid": row[0],
                "content": row[1],
                "type": row[2],
                "session_id": row[3],
                "created_at": row[4],
                "hops": 1,  # Simplified: assume 1 hop for scoring
            })

        log.debug(f"Graph traversal returned {len(rows)} related nodes")
        return rows

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
            result = self.kuzu_conn.execute("MATCH (m:Memory) RETURN count(m)")
            memories_count = result.get_next()[0] if result.has_next() else 0

            result = self.kuzu_conn.execute("MATCH ()-[r:RELATES_TO]->() RETURN count(r)")
            relations_count = result.get_next()[0] if result.has_next() else 0

            # Delete all relationships first (must be done before nodes)
            log.debug("RESET_ALL: Deleting all relationships")
            self.kuzu_conn.execute("MATCH ()-[r:RELATES_TO]->() DELETE r")

            # Delete all memory nodes
            log.debug("RESET_ALL: Deleting all memory nodes")
            self.kuzu_conn.execute("MATCH (m:Memory) DELETE m")

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
