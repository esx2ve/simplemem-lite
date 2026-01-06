"""Phase 3: Execute consolidation decisions.

Applies merges/supersessions based on confidence thresholds:
- confidence >= 0.9: Auto-execute
- 0.7 <= confidence < 0.9: Add to review queue
- confidence < 0.7: Skip (log as rejected)

All operations preserve history for auditability.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from simplemem_lite.backend.services import get_memory_store
from simplemem_lite.log_config import get_logger

from . import ConsolidationConfig
from .scorer import EntityDecision, MemoryDecision, SupersessionDecision

log = get_logger("consolidation.executor")


async def execute_entity_merges(
    decisions: list[EntityDecision],
    config: ConsolidationConfig,
    project_id: str,
) -> dict[str, Any]:
    """Execute entity merge decisions.

    For same_entity=True with high confidence:
    - Redirect all edges from deprecated entity to canonical entity
    - Delete the deprecated entity

    Args:
        decisions: List of EntityDecision from scorer
        config: Consolidation configuration
        project_id: Project scope for persisting review candidates

    Returns:
        Dict with execution stats and review items
    """
    store = get_memory_store()
    executed = 0
    queued = 0
    review_items = []

    for decision in decisions:
        if not decision.same_entity:
            continue  # Not a match, skip

        entity_a_name = decision.pair.entity_a["name"]
        entity_b_name = decision.pair.entity_b["name"]

        if decision.confidence >= config.confidence_threshold:
            # Auto-execute merge
            try:
                canonical = decision.canonical_name
                if not canonical:
                    canonical = entity_a_name

                # Determine which entity to keep and which to deprecate
                if entity_a_name == canonical:
                    deprecated_name = entity_b_name
                else:
                    deprecated_name = entity_a_name

                entity_type = decision.pair.entity_type

                log.info(
                    f"Merging entity: {deprecated_name} -> {canonical} (type: {entity_type})"
                )

                # Redirect all edges from deprecated to canonical
                store.db.graph.query(
                    """
                    MATCH (m)-[r]->(e:Entity {name: $deprecated_name, type: $type})
                    MATCH (canonical:Entity {name: $canonical_name, type: $type})
                    CREATE (m)-[r2:REFERENCES]->(canonical)
                    DELETE r
                    """,
                    {
                        "deprecated_name": deprecated_name,
                        "canonical_name": canonical,
                        "type": entity_type,
                    },
                )

                # Delete deprecated entity (only if no remaining edges)
                store.db.graph.query(
                    """
                    MATCH (e:Entity {name: $name, type: $type})
                    WHERE NOT EXISTS((e)-[]-())
                    DELETE e
                    """,
                    {"name": deprecated_name, "type": entity_type},
                )

                executed += 1

            except Exception as e:
                log.error(f"Failed to merge entity: {e}")
                review_items.append(
                    {
                        "type": "entity_dedup",
                        "error": str(e),
                        "decision": _serialize_decision(decision),
                    }
                )

        elif decision.confidence >= 0.7:
            # Check if this pair was previously rejected
            if store.db.is_rejected_pair(entity_a_name, entity_b_name):
                log.debug(f"Skipping rejected pair: {entity_a_name} <-> {entity_b_name}")
                continue

            # Persist to review queue
            decision_data = {
                "entity_a_name": entity_a_name,
                "entity_b_name": entity_b_name,
                "entity_type": decision.pair.entity_type,
                "canonical_name": decision.canonical_name,
                "similarity": decision.pair.similarity,
            }

            result = store.db.add_review_candidate(
                project_id=project_id,
                candidate_type="entity_dedup",
                confidence=decision.confidence,
                reason=decision.reason or "",
                decision_data=decision_data,
                involved_ids=[entity_a_name, entity_b_name],
                source_id=entity_a_name,
                target_id=entity_b_name,
                similarity=decision.pair.similarity,
            )

            if result.get("created") or not result.get("skipped"):
                queued += 1
                review_items.append(
                    {
                        "type": "entity_dedup",
                        "action": "review_required",
                        "candidate_uuid": result["uuid"],
                        "decision": _serialize_decision(decision),
                    }
                )

        # else: confidence < 0.7, skip silently

    log.info(f"Entity merges: {executed} executed, {queued} queued for review")
    return {
        "executed": executed,
        "queued": queued,
        "review_items": review_items,
    }


async def execute_memory_merges(
    decisions: list[MemoryDecision],
    config: ConsolidationConfig,
    project_id: str,
) -> dict[str, Any]:
    """Execute memory merge decisions.

    For should_merge=True with high confidence:
    - Keep the newer memory, update its content with merged content
    - Mark older memory with MERGED_INTO relationship

    Args:
        decisions: List of MemoryDecision from scorer
        config: Consolidation configuration
        project_id: Project scope for persisting review candidates

    Returns:
        Dict with execution stats and review items
    """
    store = get_memory_store()
    executed = 0
    queued = 0
    review_items = []

    for decision in decisions:
        if not decision.should_merge:
            continue  # Not a merge candidate, skip

        mem_a = decision.pair.memory_a
        mem_b = decision.pair.memory_b
        mem_a_uuid = mem_a.get("uuid", "")
        mem_b_uuid = mem_b.get("uuid", "")

        if decision.confidence >= config.confidence_threshold:
            # Auto-execute merge
            try:
                # Determine which memory is newer
                time_a = mem_a.get("created_at") or 0
                time_b = mem_b.get("created_at") or 0

                if time_a >= time_b:
                    newer, older = mem_a, mem_b
                else:
                    newer, older = mem_b, mem_a

                log.info(
                    f"Merging memory: {older['uuid'][:8]}... -> {newer['uuid'][:8]}..."
                )

                # Update newer memory with merged content (if provided)
                if decision.merged_content:
                    store.db.graph.query(
                        """
                        MATCH (m:Memory {uuid: $uuid})
                        SET m.content = $content,
                            m.merged_from = $older_uuid
                        """,
                        {
                            "uuid": newer["uuid"],
                            "content": decision.merged_content,
                            "older_uuid": older["uuid"],
                        },
                    )

                # Mark older memory as merged
                store.db.mark_merged(older["uuid"], newer["uuid"])

                executed += 1

            except Exception as e:
                log.error(f"Failed to merge memory: {e}")
                review_items.append(
                    {
                        "type": "memory_merge",
                        "error": str(e),
                        "decision": _serialize_decision(decision),
                    }
                )

        elif decision.confidence >= 0.7:
            # Check if this pair was previously rejected
            if store.db.is_rejected_pair(mem_a_uuid, mem_b_uuid):
                log.debug(f"Skipping rejected pair: {mem_a_uuid[:8]}... <-> {mem_b_uuid[:8]}...")
                continue

            # Persist to review queue
            decision_data = {
                "memory_a_uuid": mem_a_uuid,
                "memory_b_uuid": mem_b_uuid,
                "merged_content": decision.merged_content,
                "similarity": decision.pair.similarity,
                "shared_entities": decision.pair.shared_entities,
            }

            result = store.db.add_review_candidate(
                project_id=project_id,
                candidate_type="memory_merge",
                confidence=decision.confidence,
                reason=decision.reason or "",
                decision_data=decision_data,
                involved_ids=[mem_a_uuid, mem_b_uuid],
                source_id=mem_a_uuid,
                target_id=mem_b_uuid,
                similarity=decision.pair.similarity,
            )

            if result.get("created") or not result.get("skipped"):
                queued += 1
                review_items.append(
                    {
                        "type": "memory_merge",
                        "action": "review_required",
                        "candidate_uuid": result["uuid"],
                        "decision": _serialize_decision(decision),
                    }
                )

    log.info(f"Memory merges: {executed} executed, {queued} queued for review")
    return {
        "executed": executed,
        "queued": queued,
        "review_items": review_items,
    }


async def execute_supersessions(
    decisions: list[SupersessionDecision],
    config: ConsolidationConfig,
    project_id: str,
) -> dict[str, Any]:
    """Execute supersession decisions.

    For supersedes=True with high confidence:
    - Create SUPERSEDES relationship from newer to older

    Args:
        decisions: List of SupersessionDecision from scorer
        config: Consolidation configuration
        project_id: Project scope for persisting review candidates

    Returns:
        Dict with execution stats and review items
    """
    store = get_memory_store()
    executed = 0
    queued = 0
    review_items = []

    for decision in decisions:
        if not decision.supersedes:
            continue  # Not a supersession, skip

        if decision.supersession_type == "none":
            continue  # Explicitly marked as not superseding

        newer_uuid = decision.pair.newer.get("uuid", "")
        older_uuid = decision.pair.older.get("uuid", "")

        if decision.confidence >= config.confidence_threshold:
            # Auto-execute supersession
            try:
                log.info(
                    f"Marking supersession: {newer_uuid[:8]}... supersedes {older_uuid[:8]}..."
                )

                # Create SUPERSEDES relationship
                store.db.add_supersession(
                    newer_uuid=newer_uuid,
                    older_uuid=older_uuid,
                    confidence=decision.confidence,
                    supersession_type=decision.supersession_type,
                )

                executed += 1

            except Exception as e:
                log.error(f"Failed to mark supersession: {e}")
                review_items.append(
                    {
                        "type": "supersession",
                        "error": str(e),
                        "decision": _serialize_decision(decision),
                    }
                )

        elif decision.confidence >= 0.7:
            # Check if this pair was previously rejected
            if store.db.is_rejected_pair(newer_uuid, older_uuid):
                log.debug(f"Skipping rejected pair: {newer_uuid[:8]}... <-> {older_uuid[:8]}...")
                continue

            # Persist to review queue
            decision_data = {
                "newer_uuid": newer_uuid,
                "older_uuid": older_uuid,
                "supersession_type": decision.supersession_type,
                "entity": decision.pair.entity,
                "time_delta_days": decision.pair.time_delta_days,
            }

            result = store.db.add_review_candidate(
                project_id=project_id,
                candidate_type="supersession",
                confidence=decision.confidence,
                reason=decision.reason or "",
                decision_data=decision_data,
                involved_ids=[newer_uuid, older_uuid],
                source_id=newer_uuid,
                target_id=older_uuid,
                similarity=0.0,  # Supersession doesn't use similarity
            )

            if result.get("created") or not result.get("skipped"):
                queued += 1
                review_items.append(
                    {
                        "type": "supersession",
                        "action": "review_required",
                        "candidate_uuid": result["uuid"],
                        "decision": _serialize_decision(decision),
                    }
                )

    log.info(f"Supersessions: {executed} executed, {queued} queued for review")
    return {
        "executed": executed,
        "queued": queued,
        "review_items": review_items,
    }


def _serialize_decision(decision: EntityDecision | MemoryDecision | SupersessionDecision) -> dict[str, Any]:
    """Serialize a decision for the review queue."""
    if isinstance(decision, EntityDecision):
        return {
            "same_entity": decision.same_entity,
            "confidence": decision.confidence,
            "canonical_name": decision.canonical_name,
            "reason": decision.reason,
            "entity_a": decision.pair.entity_a,
            "entity_b": decision.pair.entity_b,
            "entity_type": decision.pair.entity_type,
            "similarity": decision.pair.similarity,
        }
    elif isinstance(decision, MemoryDecision):
        return {
            "should_merge": decision.should_merge,
            "confidence": decision.confidence,
            "merged_content": decision.merged_content,
            "reason": decision.reason,
            "memory_a_uuid": decision.pair.memory_a.get("uuid"),
            "memory_b_uuid": decision.pair.memory_b.get("uuid"),
            "similarity": decision.pair.similarity,
            "shared_entities": decision.pair.shared_entities,
        }
    elif isinstance(decision, SupersessionDecision):
        return {
            "supersedes": decision.supersedes,
            "confidence": decision.confidence,
            "supersession_type": decision.supersession_type,
            "reason": decision.reason,
            "newer_uuid": decision.pair.newer.get("uuid"),
            "older_uuid": decision.pair.older.get("uuid"),
            "entity": decision.pair.entity,
            "time_delta_days": decision.pair.time_delta_days,
        }
    else:
        return {}


__all__ = [
    "execute_entity_merges",
    "execute_memory_merges",
    "execute_supersessions",
]
