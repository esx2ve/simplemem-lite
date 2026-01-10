"""Consolidation Manager for SimpleMem Lite.

Manages review candidates and rejected pairs for memory consolidation operations.
Extracted from DatabaseManager as part of god class decomposition.

This module handles:
- Review candidate lifecycle (create, read, update)
- Rejected pair tracking for skipping previously-rejected consolidations
- Dedupe key generation for idempotent operations
"""

import hashlib
import json
import time
import uuid as uuid_lib
from typing import Any, Protocol

from simplemem_lite.log_config import get_logger

log = get_logger("db.consolidation")


class GraphBackendProtocol(Protocol):
    """Protocol for graph backend dependency injection."""

    def query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a Cypher query."""
        ...


class ConsolidationManager:
    """Manages consolidation review candidates and rejected pairs.

    This class handles the review queue for memory consolidation operations,
    including entity deduplication, memory merging, and supersession tracking.

    Uses dependency injection for the graph backend to enable testing
    and flexibility in database selection.
    """

    def __init__(self, graph: GraphBackendProtocol):
        """Initialize ConsolidationManager.

        Args:
            graph: Graph backend implementing the query method
        """
        self.graph = graph

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
