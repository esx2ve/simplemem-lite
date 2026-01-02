"""KuzuDB backend implementation for SimpleMem Lite.

KuzuDB is an embedded graph database that works on systems without Docker.
It supports most Cypher features but requires schema upfront (unlike FalkorDB).

Key differences from FalkorDB:
- Schema must be created before data insertion (CREATE NODE TABLE, CREATE REL TABLE)
- No shortestPath() function - use variable-length paths with LIMIT
- Entity PK must be single column - use composite ID like "name:type"
- No algo.pageRank - use degree-based fallback
"""

import re
from pathlib import Path
from typing import Any

from simplemem_lite.db.graph_protocol import BaseGraphBackend, QueryResult
from simplemem_lite.log_config import get_logger

log = get_logger("kuzu")


class KuzuDBBackend(BaseGraphBackend):
    """KuzuDB-based graph backend for environments without Docker.

    Uses KuzuDB's embedded database for graph operations. Schema is
    created upfront since Kuzu requires explicit table definitions.
    """

    def __init__(self, db_path: str | Path):
        """Initialize KuzuDB connection.

        Args:
            db_path: Path to the database directory (will be created by KuzuDB)
        """
        import kuzu

        self.db_path = Path(db_path)
        # Note: KuzuDB creates the directory itself - don't create it beforehand
        # Just ensure the parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Initializing KuzuDB at {self.db_path}")
        self._db = kuzu.Database(str(self.db_path))
        self._conn = kuzu.Connection(self._db)

        # Track if schema has been initialized
        self._schema_initialized = False

    @property
    def backend_name(self) -> str:
        return "kuzu"

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> QueryResult:
        """Execute a Cypher query with KuzuDB dialect handling.

        Args:
            cypher: Cypher query string
            params: Optional query parameters

        Returns:
            QueryResult with result_set, header, and stats
        """
        # Translate Cypher dialect differences
        translated = self._translate_cypher(cypher)

        log.trace(f"KuzuDB query: {translated[:100]}...")

        try:
            if params:
                result = self._conn.execute(translated, params)
            else:
                result = self._conn.execute(translated)

            # Extract results
            result_set = []
            header = None

            if result.has_next():
                # Get column names from first iteration
                while result.has_next():
                    row = result.get_next()
                    result_set.append(list(row))

            # Try to get column names
            try:
                header = result.get_column_names()
            except Exception:
                header = None

            # KuzuDB doesn't provide detailed stats like FalkorDB
            stats = {"backend": "kuzu"}

            return QueryResult(
                result_set=result_set,
                header=header,
                stats=stats,
            )

        except Exception as e:
            log.error(f"KuzuDB query failed: {e}")
            log.debug(f"Query was: {translated}")
            raise

    def health_check(self) -> bool:
        """Check if KuzuDB is healthy.

        Returns:
            True if connection is operational
        """
        try:
            result = self._conn.execute("RETURN 1")
            return result.has_next()
        except Exception as e:
            log.warning(f"KuzuDB health check failed: {e}")
            return False

    def close(self) -> None:
        """Close the KuzuDB connection."""
        log.info("Closing KuzuDB connection")
        # KuzuDB Connection doesn't need explicit close
        # Database will be closed when object is garbage collected
        pass

    def init_schema(self) -> None:
        """Initialize the graph schema for SimpleMem.

        Creates all node tables and relationship tables upfront.
        KuzuDB requires schema before data can be inserted.
        """
        if self._schema_initialized:
            log.debug("Schema already initialized, skipping")
            return

        log.info("Initializing KuzuDB schema")

        # ══════════════════════════════════════════════════════════════════════════════
        # NODE TABLES
        # ══════════════════════════════════════════════════════════════════════════════

        # Memory node table
        self._create_table_if_not_exists(
            """
            CREATE NODE TABLE Memory(
                uuid STRING,
                content STRING,
                type STRING,
                source STRING,
                session_id STRING,
                created_at INT64,
                PRIMARY KEY (uuid)
            )
            """
        )

        # Entity node table - KuzuDB requires single PK, so we use composite ID "name:type"
        self._create_table_if_not_exists(
            """
            CREATE NODE TABLE Entity(
                id STRING,
                name STRING,
                type STRING,
                version INT64,
                created_at INT64,
                last_modified INT64,
                PRIMARY KEY (id)
            )
            """
        )

        # CodeChunk node table
        self._create_table_if_not_exists(
            """
            CREATE NODE TABLE CodeChunk(
                uuid STRING,
                filepath STRING,
                project_root STRING,
                start_line INT32,
                end_line INT32,
                created_at INT64,
                PRIMARY KEY (uuid)
            )
            """
        )

        # ProjectIndex node table
        self._create_table_if_not_exists(
            """
            CREATE NODE TABLE ProjectIndex(
                project_root STRING,
                last_commit_hash STRING,
                indexed_at INT64,
                file_count INT32,
                chunk_count INT32,
                PRIMARY KEY (project_root)
            )
            """
        )

        # Goal node table
        self._create_table_if_not_exists(
            """
            CREATE NODE TABLE Goal(
                id STRING,
                intent STRING,
                session_id STRING,
                status STRING,
                created_at INT64,
                PRIMARY KEY (id)
            )
            """
        )

        # Project node table
        self._create_table_if_not_exists(
            """
            CREATE NODE TABLE Project(
                id STRING,
                path STRING,
                session_count INT64,
                created_at INT64,
                last_updated INT64,
                PRIMARY KEY (id)
            )
            """
        )

        # ══════════════════════════════════════════════════════════════════════════════
        # RELATIONSHIP TABLES
        # ══════════════════════════════════════════════════════════════════════════════

        # Memory-to-Memory relationships
        self._create_rel_table_if_not_exists(
            "RELATES_TO",
            "Memory",
            "Memory",
            "relation_type STRING, weight DOUBLE",
        )
        self._create_rel_table_if_not_exists("CONTAINS", "Memory", "Memory", "")
        self._create_rel_table_if_not_exists("CHILD_OF", "Memory", "Memory", "")
        self._create_rel_table_if_not_exists("FOLLOWS", "Memory", "Memory", "")

        # Memory-to-Entity verb relationships
        self._create_rel_table_if_not_exists(
            "READS",
            "Memory",
            "Entity",
            "timestamp INT64, implicit BOOLEAN, change_summary STRING",
        )
        self._create_rel_table_if_not_exists(
            "MODIFIES",
            "Memory",
            "Entity",
            "timestamp INT64, change_summary STRING",
        )
        self._create_rel_table_if_not_exists(
            "EXECUTES",
            "Memory",
            "Entity",
            "timestamp INT64, change_summary STRING",
        )
        self._create_rel_table_if_not_exists(
            "TRIGGERED",
            "Memory",
            "Entity",
            "timestamp INT64, change_summary STRING",
        )

        # CodeChunk-to-Entity relationship
        self._create_rel_table_if_not_exists(
            "REFERENCES_CODE",
            "CodeChunk",
            "Entity",
            "relation STRING, created_at INT64",
        )

        # Memory-to-Entity references (generic)
        self._create_rel_table_if_not_exists(
            "REFERENCES",
            "Memory",
            "Entity",
            "relation STRING, timestamp INT64",
        )

        # Memory-to-Project relationship
        self._create_rel_table_if_not_exists("BELONGS_TO", "Memory", "Project", "")

        # Memory-to-Goal relationships
        self._create_rel_table_if_not_exists("HAS_GOAL", "Memory", "Goal", "")
        self._create_rel_table_if_not_exists("ACHIEVES", "Memory", "Goal", "")

        self._schema_initialized = True
        log.info("KuzuDB schema initialization complete")

    def _create_table_if_not_exists(self, create_statement: str) -> None:
        """Create a node table if it doesn't exist.

        Args:
            create_statement: CREATE NODE TABLE statement
        """
        try:
            self._conn.execute(create_statement)
            log.debug(f"Created table: {create_statement[:50]}...")
        except Exception as e:
            if "already exists" in str(e).lower():
                log.trace(f"Table already exists: {create_statement[:30]}...")
            else:
                log.warning(f"Table creation error: {e}")

    def _create_rel_table_if_not_exists(
        self,
        name: str,
        from_table: str,
        to_table: str,
        properties: str,
    ) -> None:
        """Create a relationship table if it doesn't exist.

        Args:
            name: Relationship type name
            from_table: Source node table
            to_table: Target node table
            properties: Property definitions (comma-separated)
        """
        if properties:
            stmt = f"CREATE REL TABLE {name}(FROM {from_table} TO {to_table}, {properties})"
        else:
            stmt = f"CREATE REL TABLE {name}(FROM {from_table} TO {to_table})"

        try:
            self._conn.execute(stmt)
            log.debug(f"Created rel table: {name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                log.trace(f"Rel table already exists: {name}")
            else:
                log.warning(f"Rel table creation error ({name}): {e}")

    def _translate_cypher(self, cypher: str) -> str:
        """Translate FalkorDB Cypher dialect to KuzuDB dialect.

        Handles key differences:
        - shortestPath() not supported -> variable-length path with LIMIT 1
        - timestamp() function -> use parameter
        - Entity lookup by name,type -> lookup by composite id

        Args:
            cypher: FalkorDB-style Cypher query

        Returns:
            KuzuDB-compatible Cypher query
        """
        result = cypher

        # Replace shortestPath with variable-length path
        # shortestPath((a)-[*..5]-(b)) -> (a)-[*1..5]-(b) LIMIT 1
        if "shortestPath" in result.lower():
            log.debug("Translating shortestPath to variable-length path")
            # This is a simple heuristic - complex cases may need manual handling
            result = re.sub(
                r"shortestPath\s*\(\s*\((\w+)\)\s*-\s*\[\*\.\.(\d+)\]\s*-\s*\((\w+)\)\s*\)",
                r"(\1)-[*1..\2]-(\3)",
                result,
                flags=re.IGNORECASE,
            )

        # Replace timestamp() with a placeholder (caller should provide as param)
        # KuzuDB doesn't have a built-in timestamp() function
        if "timestamp()" in result.lower():
            log.debug("timestamp() function not supported - should use parameter")
            # The caller should provide timestamp as a parameter instead

        # Handle Entity matching by name,type -> use composite id
        # MATCH (e:Entity {name: $name, type: $type}) -> MATCH (e:Entity {id: $entity_id})
        # This requires the caller to construct the composite ID

        return result

    def get_entity_id(self, name: str, entity_type: str) -> str:
        """Generate composite Entity ID from name and type.

        KuzuDB requires single-column primary key, so we use a composite ID.

        Args:
            name: Entity name
            entity_type: Entity type

        Returns:
            Composite ID in format "type:name"
        """
        # Use type:name format for consistent ordering
        return f"{entity_type}:{name}"

    # ══════════════════════════════════════════════════════════════════════════════════
    # HELPER METHODS FOR COMMON OPERATIONS
    # ══════════════════════════════════════════════════════════════════════════════════

    def merge_memory(
        self,
        uuid: str,
        content: str,
        mem_type: str,
        source: str,
        session_id: str | None,
        created_at: int,
    ) -> None:
        """Insert or update a Memory node.

        Args:
            uuid: Unique identifier
            content: Memory content
            mem_type: Memory type
            source: Source of memory
            session_id: Optional session identifier
            created_at: Unix timestamp
        """
        # KuzuDB uses MERGE syntax similar to Neo4j
        self._conn.execute(
            """
            MERGE (m:Memory {uuid: $uuid})
            SET m.content = $content,
                m.type = $type,
                m.source = $source,
                m.session_id = $session_id,
                m.created_at = $created_at
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

    def merge_entity(
        self,
        name: str,
        entity_type: str,
        created_at: int,
    ) -> str:
        """Insert or update an Entity node.

        Args:
            name: Entity name
            entity_type: Entity type
            created_at: Unix timestamp

        Returns:
            Entity composite ID
        """
        entity_id = self.get_entity_id(name, entity_type)

        self._conn.execute(
            """
            MERGE (e:Entity {id: $id})
            ON CREATE SET
                e.name = $name,
                e.type = $type,
                e.version = 1,
                e.created_at = $created_at
            """,
            {
                "id": entity_id,
                "name": name,
                "type": entity_type,
                "created_at": created_at,
            },
        )

        return entity_id

    def increment_entity_version(self, entity_id: str, modified_at: int) -> None:
        """Increment entity version on modification.

        Args:
            entity_id: Composite entity ID
            modified_at: Unix timestamp
        """
        self._conn.execute(
            """
            MATCH (e:Entity {id: $id})
            SET e.version = e.version + 1,
                e.last_modified = $modified_at
            """,
            {"id": entity_id, "modified_at": modified_at},
        )


def create_kuzu_backend(db_path: str | Path) -> KuzuDBBackend:
    """Factory function to create and initialize a KuzuDB backend.

    Args:
        db_path: Path to database directory

    Returns:
        Initialized KuzuDBBackend instance
    """
    backend = KuzuDBBackend(db_path)
    backend.init_schema()
    return backend
