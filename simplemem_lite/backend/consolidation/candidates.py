"""Phase 1: Candidate generation for graph consolidation.

Uses embeddings + cosine similarity to find potential duplicates.
No LLM calls - this is the cheap, fast pre-filtering step.

Complexity: O(n) per type due to blocking strategy:
1. Project isolation (reduces n to project scope)
2. Type-based blocking (only compare same types)
3. Embedding similarity for candidate pairs
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from simplemem_lite.backend.services import get_memory_store
from simplemem_lite.embeddings import embed_batch
from simplemem_lite.log_config import get_logger

log = get_logger("consolidation.candidates")

# =============================================================================
# Phase 1 Filtering Constants (prevent false positive candidates)
# =============================================================================

# Entity types too generic for supersession detection
# File entities create too many false matches (every session touches common files)
GENERIC_ENTITY_TYPES = {"file", "module", "directory", "path"}

# Memory type pairs that should NEVER supersede each other
# Summaries describe sessions; insights describe learnings - different purposes
TYPE_INCOMPATIBLE = {
    frozenset({"session_summary", "lesson_learned"}),
    frozenset({"session_summary", "decision"}),
    frozenset({"session_summary", "pattern"}),
    frozenset({"session_summary", "fact"}),
    frozenset({"chunk_summary", "lesson_learned"}),
    frozenset({"chunk_summary", "decision"}),
    frozenset({"chunk_summary", "pattern"}),
    frozenset({"chunk_summary", "fact"}),
    frozenset({"message", "lesson_learned"}),  # Session messages ≠ insights
    frozenset({"message", "decision"}),
    frozenset({"message", "pattern"}),
}


@dataclass
class EntityPair:
    """Candidate pair for entity deduplication."""

    entity_a: dict[str, Any]  # {name, type}
    entity_b: dict[str, Any]
    similarity: float
    entity_type: str


@dataclass
class MemoryPair:
    """Candidate pair for memory merging."""

    memory_a: dict[str, Any]  # {uuid, content, type, created_at, ...}
    memory_b: dict[str, Any]
    similarity: float
    shared_entities: list[str]


@dataclass
class SupersessionPair:
    """Candidate pair for supersession detection."""

    newer: dict[str, Any]  # More recent memory
    older: dict[str, Any]  # Older memory
    entity: str  # Shared entity name
    similarity: float
    time_delta_days: int


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(dot / (norm_a * norm_b))


def find_similar_pairs(
    items: list[dict[str, Any]],
    embeddings: list[list[float]],
    threshold: float,
    id_key: str = "name",
) -> list[tuple[int, int, float]]:
    """Find pairs of items with similarity above threshold.

    Uses brute-force O(n²) but with small n due to type blocking.
    For >1000 items, would need LSH/ANN index.

    Args:
        items: List of items
        embeddings: Corresponding embeddings
        threshold: Minimum similarity threshold
        id_key: Key to use for identifying items

    Returns:
        List of (idx_a, idx_b, similarity) tuples
    """
    n = len(items)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                pairs.append((i, j, sim))

    return pairs


async def get_entities_by_type(project_id: str) -> dict[str, list[dict[str, Any]]]:
    """Get all entities for a project, grouped by type.

    Args:
        project_id: Project identifier

    Returns:
        Dict mapping entity_type -> list of entity dicts
    """
    store = get_memory_store()

    # Query entities that have edges from memories in this project
    result = store.db.graph.query(
        """
        MATCH (m:Memory)-[]->(e:Entity)
        WHERE m.project_id = $project_id
        RETURN DISTINCT e.name AS name, e.type AS type
        """,
        {"project_id": project_id},
    )

    entities_by_type: dict[str, list[dict[str, Any]]] = {}
    for record in result.result_set or []:
        name, entity_type = record[0], record[1]
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append({"name": name, "type": entity_type})

    return entities_by_type


async def get_memories_by_type(project_id: str) -> dict[str, list[dict[str, Any]]]:
    """Get all memories for a project, grouped by type.

    Args:
        project_id: Project identifier

    Returns:
        Dict mapping memory_type -> list of memory dicts
    """
    store = get_memory_store()

    # Query memories with their metadata
    result = store.db.graph.query(
        """
        MATCH (m:Memory)
        WHERE m.project_id = $project_id
        RETURN m.uuid, m.content, m.type, m.created_at, m.session_id
        """,
        {"project_id": project_id},
    )

    memories_by_type: dict[str, list[dict[str, Any]]] = {}
    for record in result.result_set or []:
        uuid, content, mem_type, created_at, session_id = record
        mem_type = mem_type or "fact"
        if mem_type not in memories_by_type:
            memories_by_type[mem_type] = []
        memories_by_type[mem_type].append({
            "uuid": uuid,
            "content": content,
            "type": mem_type,
            "created_at": created_at,
            "session_id": session_id,
        })

    return memories_by_type


async def get_memories_by_entity(project_id: str) -> dict[str, dict[str, Any]]:
    """Get memories grouped by the entities they reference.

    Args:
        project_id: Project identifier

    Returns:
        Dict mapping entity_name -> {entity_type, memories: [...]}
    """
    store = get_memory_store()

    result = store.db.graph.query(
        """
        MATCH (m:Memory)-[r]->(e:Entity)
        WHERE m.project_id = $project_id
        RETURN e.name AS entity,
               e.type AS entity_type,
               m.uuid AS uuid,
               m.content AS content,
               m.type AS type,
               m.created_at AS created_at
        ORDER BY e.name, m.created_at DESC
        """,
        {"project_id": project_id},
    )

    memories_by_entity: dict[str, dict[str, Any]] = {}
    for record in result.result_set or []:
        entity, entity_type, uuid, content, mem_type, created_at = record
        if entity not in memories_by_entity:
            memories_by_entity[entity] = {
                "entity_type": entity_type or "unknown",
                "memories": [],
            }
        memories_by_entity[entity]["memories"].append({
            "uuid": uuid,
            "content": content,
            "type": mem_type or "fact",
            "created_at": created_at,
        })

    return memories_by_entity


async def find_entity_candidates(
    project_id: str,
    threshold: float = 0.85,
) -> list[EntityPair]:
    """Find potentially duplicate entities using embedding similarity.

    Strategy:
    1. Group entities by type (blocking)
    2. Embed entity names
    3. Find pairs with similarity > threshold

    Args:
        project_id: Project to analyze
        threshold: Similarity threshold (default: 0.85)

    Returns:
        List of EntityPair candidates for deduplication
    """
    log.info(f"Finding entity candidates for project: {project_id}")

    entities_by_type = await get_entities_by_type(project_id)
    all_candidates: list[EntityPair] = []

    for entity_type, entities in entities_by_type.items():
        if len(entities) < 2:
            continue

        log.debug(f"Processing {len(entities)} entities of type: {entity_type}")

        # Embed entity names (batch call)
        names = [e["name"] for e in entities]
        try:
            embeddings = embed_batch(names)
        except Exception as e:
            log.warning(f"Failed to embed entities of type {entity_type}: {e}")
            continue

        # Find similar pairs
        pairs = find_similar_pairs(entities, embeddings, threshold, id_key="name")

        for i, j, sim in pairs:
            all_candidates.append(
                EntityPair(
                    entity_a=entities[i],
                    entity_b=entities[j],
                    similarity=sim,
                    entity_type=entity_type,
                )
            )

    log.info(f"Found {len(all_candidates)} entity dedup candidates")
    return all_candidates


async def find_memory_candidates(
    project_id: str,
    threshold: float = 0.90,
) -> list[MemoryPair]:
    """Find potentially duplicate/mergeable memories.

    Strategy:
    1. Group memories by type (blocking)
    2. Embed memory content
    3. Find pairs with similarity > threshold
    4. Check for shared entities (additional signal)

    Args:
        project_id: Project to analyze
        threshold: Similarity threshold (default: 0.90, high for memories)

    Returns:
        List of MemoryPair candidates for merging
    """
    log.info(f"Finding memory candidates for project: {project_id}")

    memories_by_type = await get_memories_by_type(project_id)
    all_candidates: list[MemoryPair] = []

    # Pre-fetch entity relationships for shared entity detection
    store = get_memory_store()

    for mem_type, memories in memories_by_type.items():
        if len(memories) < 2:
            continue

        log.debug(f"Processing {len(memories)} memories of type: {mem_type}")

        # Embed memory content (batch call)
        contents = [m["content"] for m in memories]
        try:
            embeddings = embed_batch(contents)
        except Exception as e:
            log.warning(f"Failed to embed memories of type {mem_type}: {e}")
            continue

        # Find similar pairs
        pairs = find_similar_pairs(memories, embeddings, threshold, id_key="uuid")

        for i, j, sim in pairs:
            # Check for shared entities
            try:
                result = store.db.graph.query(
                    """
                    MATCH (m1:Memory {uuid: $uuid1})-[]->(e:Entity)<-[]-(m2:Memory {uuid: $uuid2})
                    RETURN DISTINCT e.name
                    """,
                    {"uuid1": memories[i]["uuid"], "uuid2": memories[j]["uuid"]},
                )
                shared = [r[0] for r in (result.result_set or [])]
            except Exception:
                shared = []

            all_candidates.append(
                MemoryPair(
                    memory_a=memories[i],
                    memory_b=memories[j],
                    similarity=sim,
                    shared_entities=shared,
                )
            )

    log.info(f"Found {len(all_candidates)} memory merge candidates")
    return all_candidates


async def find_supersession_candidates(
    project_id: str,
    min_days_apart: int = 1,
    similarity_threshold: float = 0.75,  # Raised from 0.6 to reduce false positives
) -> list[SupersessionPair]:
    """Find memories that may supersede older ones.

    Strategy:
    1. Group memories by shared entity
    2. Filter out generic entities (file paths create too many false matches)
    3. Check memory type compatibility (summaries shouldn't supersede insights)
    4. Compare newer to older memories by content similarity

    Args:
        project_id: Project to analyze
        min_days_apart: Minimum days between memories (default: 1)
        similarity_threshold: Content similarity threshold (default: 0.75, raised from 0.6)

    Returns:
        List of SupersessionPair candidates
    """
    log.info(f"Finding supersession candidates for project: {project_id}")

    memories_by_entity = await get_memories_by_entity(project_id)
    all_candidates: list[SupersessionPair] = []
    skipped_generic = 0
    skipped_type_incompatible = 0

    for entity_name, entity_data in memories_by_entity.items():
        entity_type = entity_data.get("entity_type", "unknown")
        memories = entity_data.get("memories", [])

        # FILTER 1: Skip generic entity types (file paths create false positives)
        if entity_type in GENERIC_ENTITY_TYPES:
            skipped_generic += len(memories)
            log.debug(f"Skipping generic entity: {entity_name} (type: {entity_type})")
            continue

        if len(memories) < 2:
            continue

        # Sort by created_at descending (newest first)
        sorted_memories = sorted(
            memories,
            key=lambda m: m["created_at"] or 0,
            reverse=True,
        )

        # Embed content for similarity check
        contents = [m["content"] for m in sorted_memories]
        try:
            embeddings = embed_batch(contents)
        except Exception as e:
            log.warning(f"Failed to embed memories for entity {entity_name}: {e}")
            continue

        # Compare newer to older
        for i, newer in enumerate(sorted_memories[:-1]):
            for j, older in enumerate(sorted_memories[i + 1:], start=i + 1):
                # FILTER 2: Check memory type compatibility
                newer_type = newer.get("type", "fact")
                older_type = older.get("type", "fact")
                type_pair = frozenset({newer_type, older_type})

                if type_pair in TYPE_INCOMPATIBLE:
                    skipped_type_incompatible += 1
                    continue

                newer_time = newer["created_at"] or 0
                older_time = older["created_at"] or 0

                # Check time delta
                if isinstance(newer_time, datetime):
                    newer_ts = newer_time.timestamp()
                else:
                    newer_ts = float(newer_time)
                if isinstance(older_time, datetime):
                    older_ts = older_time.timestamp()
                else:
                    older_ts = float(older_time)

                days_apart = (newer_ts - older_ts) / 86400
                if days_apart < min_days_apart:
                    continue

                # Check content similarity (raised threshold)
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim < similarity_threshold:
                    continue  # Too different, not related

                if sim > 0.95:
                    continue  # Too similar, probably duplicate (handled by memory_merge)

                all_candidates.append(
                    SupersessionPair(
                        newer=newer,
                        older=older,
                        entity=entity_name,
                        similarity=sim,
                        time_delta_days=int(days_apart),
                    )
                )

    log.info(
        f"Found {len(all_candidates)} supersession candidates "
        f"(skipped: {skipped_generic} generic entities, {skipped_type_incompatible} type-incompatible pairs)"
    )
    return all_candidates


__all__ = [
    "EntityPair",
    "MemoryPair",
    "SupersessionPair",
    "find_entity_candidates",
    "find_memory_candidates",
    "find_supersession_candidates",
]
