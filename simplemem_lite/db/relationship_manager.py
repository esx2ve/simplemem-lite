"""Relationship Manager for SimpleMem Lite.

Manages memory-to-memory relationships in the graph:
- RELATES_TO: General relationship between memories
- SUPERSEDES: Newer memory replaces older one (DAG enforced)
- MERGED_INTO: Memory was merged into another

Extracted from DatabaseManager as part of god class decomposition.
"""

import time
from typing import Any, Protocol

from simplemem_lite.log_config import get_logger

log = get_logger("db.relationships")


class GraphQueryProtocol(Protocol):
    """Protocol for graph query capability."""

    def query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a Cypher query."""
        ...


class RelationshipManager:
    """Manages memory-to-memory relationships in the knowledge graph.

    Handles three types of relationships:
    - RELATES_TO: General semantic relationship between memories
    - SUPERSEDES: DAG-enforced replacement (newer info supersedes older)
    - MERGED_INTO: Soft-delete for consolidated memories

    Uses dependency injection for the graph backend to enable testing
    and flexibility in database selection.
    """

    def __init__(self, graph: GraphQueryProtocol, max_supersession_depth: int = 10):
        """Initialize RelationshipManager.

        Args:
            graph: Graph backend implementing the query method
            max_supersession_depth: Max depth for cycle detection (default: 10)
        """
        self.graph = graph
        self.max_supersession_depth = max_supersession_depth

    # =========================================================================
    # Basic Relationships (RELATES_TO)
    # =========================================================================

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
            relation_type: Type of relationship (e.g., "similar", "follows", "supports")
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
        log.trace(f"Added RELATES_TO edge: {from_uuid[:8]}... -> {to_uuid[:8]}... ({relation_type})")

    # =========================================================================
    # Supersession Management (DAG-enforced)
    # =========================================================================

    def would_create_supersession_cycle(
        self,
        newer_uuid: str,
        older_uuid: str,
        max_depth: int | None = None,
    ) -> bool:
        """Check if creating a SUPERSEDES edge would create a cycle.

        DAG enforcement for supersession graph: prevents situations like
        A supersedes B supersedes C supersedes A (cycle).

        Args:
            newer_uuid: UUID of the newer/superseding memory
            older_uuid: UUID of the older/superseded memory
            max_depth: Maximum traversal depth to check (default: self.max_supersession_depth)

        Returns:
            True if creating the edge would create a cycle, False otherwise
        """
        depth = max_depth or self.max_supersession_depth

        # Check if older_uuid can reach newer_uuid via SUPERSEDES edges
        # If so, creating newer->older would complete a cycle
        # Note: Cypher doesn't support parameters for path length bounds,
        # so we inject max_depth directly (safe: typed as int)
        result = self.graph.query(
            f"""
            MATCH path = (older:Memory {{uuid: $older_uuid}})-[:SUPERSEDES*1..{depth}]->(newer:Memory {{uuid: $newer_uuid}})
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

    def get_superseded_memories(
        self,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
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

    # =========================================================================
    # Merge Tracking (Soft Delete)
    # =========================================================================

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
        log.debug(f"Marked memory {source_uuid[:8]}... as merged into {target_uuid[:8]}...")

    def get_merged_memories(
        self,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
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
