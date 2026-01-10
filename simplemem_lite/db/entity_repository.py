"""Entity Repository for SimpleMem Lite.

Manages Entity nodes and their relationships to Memory nodes.
Extracted from DatabaseManager as part of god class decomposition.

This module handles:
- Entity node CRUD operations
- Entity name canonicalization
- Verb-specific edges (READS, MODIFIES, EXECUTES, TRIGGERED)
- Code-to-entity linking
- Entity history and listing
"""

import os
import time
from typing import Any, Protocol

from simplemem_lite.log_config import get_logger

log = get_logger("db.entity_repository")


class GraphQueryProtocol(Protocol):
    """Protocol for graph query capability."""

    def query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a Cypher query."""
        ...


class EntityRepository:
    """Repository for Entity node operations.

    Manages Entity nodes which represent external objects like files, tools,
    commands, and errors. Entities enable cross-session linking and semantic
    relationships between memories.

    Uses dependency injection for the graph backend to enable testing
    and flexibility in database selection.
    """

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

    def __init__(
        self,
        graph: GraphQueryProtocol,
        summary_max_size: int = 500,
    ):
        """Initialize EntityRepository.

        Args:
            graph: Graph backend with query capability
            summary_max_size: Maximum length for change summaries (default: 500)
        """
        self.graph = graph
        self.summary_max_size = summary_max_size

    # =========================================================================
    # Entity CRUD Operations
    # =========================================================================

    def add_entity_node(
        self,
        name: str,
        entity_type: str,
        metadata: dict[str, Any] | None = None,  # noqa: ARG002 - reserved for future use
    ) -> str:
        """Add or get an Entity node (for cross-session linking).

        Args:
            name: Entity name (e.g., "src/main.py", "Read", "ImportError")
            entity_type: Entity type (file, tool, error, concept)
            metadata: Optional additional metadata (reserved for future use)

        Returns:
            Entity name (used as identifier)
        """
        log.trace(f"Adding/getting entity: {entity_type}:{name}")
        # MERGE creates if not exists, returns existing if exists
        # Note: metadata parameter reserved for future use (e.g., entity attributes)
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

    def _canonicalize_entity(self, name: str, entity_type: str) -> str:
        """Canonicalize entity name for consistent deduplication.

        Args:
            name: Raw entity name
            entity_type: Entity type

        Returns:
            Canonicalized entity name
        """
        import hashlib
        import re

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
            # Match common exception patterns
            match = re.match(r"^(\w+(?:Error|Exception|Warning|Failure))", name, re.IGNORECASE)
            if match:
                return f"error:{match.group(1).lower()}"
            # Fallback: hash for unrecognized error formats
            error_hash = hashlib.sha256(name[:200].encode()).hexdigest()[:12]
            return f"error:{error_hash}"

        return name

    # =========================================================================
    # Verb Edge Operations
    # =========================================================================

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
                {"uuid": memory_uuid, "name": canonical_name, "type": entity_type, "ts": ts, "summary": change_summary[:self.summary_max_size]},
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

    # =========================================================================
    # Code-Entity Linking
    # =========================================================================

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

    # =========================================================================
    # Entity Query Operations
    # =========================================================================

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
