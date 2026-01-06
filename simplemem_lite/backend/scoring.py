"""Temporal decay and importance scoring for memory retrieval.

Transforms pure vector similarity into multi-factor scoring that
incorporates temporal decay, type-based importance, and access patterns.

Key insight: Vector similarity scores are NOT comparable across queries
(cosine similarity scale varies). Must normalize per-request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, log
from typing import Any

# Half-life per memory type (days)
# Decisions persist longer, session summaries decay faster
HALF_LIVES: dict[str, float] = {
    "decision": 180.0,
    "lesson_learned": 90.0,
    "pattern": 120.0,
    "fact": 60.0,
    "session_summary": 45.0,
    "chunk_summary": 30.0,
}

# Type-based importance prior
# Decisions and lessons are more valuable than raw summaries
IMPORTANCE_PRIOR: dict[str, float] = {
    "decision": 0.9,
    "lesson_learned": 0.7,
    "pattern": 0.6,
    "fact": 0.5,
    "session_summary": 0.4,
    "chunk_summary": 0.3,
}

# Decay floors - minimum temporal factor to prevent important old memories
# from completely disappearing. Decisions stay relevant even when old.
DECAY_FLOORS: dict[str, float] = {
    "decision": 0.4,
    "lesson_learned": 0.3,
    "pattern": 0.35,
    "fact": 0.4,
    "session_summary": 0.2,
    "chunk_summary": 0.1,
}

# Default weights for score combination
DEFAULT_VECTOR_WEIGHT: float = 0.6
DEFAULT_TEMPORAL_WEIGHT: float = 0.25
DEFAULT_IMPORTANCE_WEIGHT: float = 0.15


@dataclass
class ScoringWeights:
    """Configuration for score combination weights."""

    vector: float = DEFAULT_VECTOR_WEIGHT
    temporal: float = DEFAULT_TEMPORAL_WEIGHT
    importance: float = DEFAULT_IMPORTANCE_WEIGHT

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0 (within tolerance)."""
        total = self.vector + self.temporal + self.importance
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class ScoredMemory:
    """Memory with detailed scoring breakdown."""

    uuid: str
    content: str
    type: str
    created_at: datetime
    score_raw: float  # Original vector similarity
    score_normalized: float  # Normalized to [0,1] within result set
    temporal_decay: float  # Decay factor [floor, 1.0]
    importance_prior: float  # Type-based importance [0, 1]
    final_score: float  # Combined score
    metadata: dict[str, Any] | None = None


def compute_temporal_decay(
    created_at: datetime,
    memory_type: str,
    reference_time: datetime | None = None,
) -> float:
    """Compute temporal decay factor using half-life formula.

    Uses exponential decay: factor = max(floor, e^(-ln(2) * age / half_life))

    This models the "forgetting curve" while ensuring important old
    memories don't completely disappear (via decay floor).

    Args:
        created_at: When the memory was created
        memory_type: Type of memory for half-life lookup
        reference_time: Current time for age calculation (defaults to now)

    Returns:
        Decay factor in [floor, 1.0] where 1.0 = brand new
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    # Ensure timezone-aware comparison
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    age_days = (reference_time - created_at).total_seconds() / 86400

    # Clamp negative ages (future timestamps) to 0
    if age_days < 0:
        age_days = 0

    half_life = HALF_LIVES.get(memory_type, 90.0)
    floor = DECAY_FLOORS.get(memory_type, 0.1)

    # Exponential decay: e^(-ln(2) * t / half_life)
    decay = exp(-log(2) * age_days / half_life)

    # Apply floor to prevent complete decay
    return max(floor, decay)


def get_importance_prior(memory_type: str) -> float:
    """Get importance prior for a memory type.

    Args:
        memory_type: Type of memory

    Returns:
        Importance score in [0, 1]
    """
    return IMPORTANCE_PRIOR.get(memory_type, 0.5)


def normalize_scores(scores: list[float], method: str = "rank_preserve") -> list[float]:
    """Normalize scores to [0, 1] range while preserving relative differences.

    CRITICAL: Vector similarity scores are not comparable across queries.
    Must normalize within each result set before combining with other factors.

    Args:
        scores: Raw vector similarity scores
        method: Normalization method:
            - "rank_preserve": Preserves relative differences using max-based scaling
            - "minmax": Traditional min-max (can distort with small result sets)

    Returns:
        Scores normalized to [0, 1] where 1 = highest in set
    """
    if not scores:
        return []

    max_score = max(scores)
    min_score = min(scores)

    if max_score < 1e-9:
        # All scores are zero
        return [0.0] * len(scores)

    if method == "rank_preserve":
        # Divide by max to preserve relative ratios
        # e.g., 0.92 and 0.78 become 1.0 and 0.848 (ratio preserved)
        # This is better for small result sets where min-max would distort
        return [s / max_score for s in scores]
    else:
        # Traditional min-max normalization
        score_range = max_score - min_score
        if score_range < 1e-9:
            return [1.0] * len(scores)
        return [(s - min_score) / score_range for s in scores]


def compute_final_score(
    score_normalized: float,
    temporal_decay: float,
    importance_prior: float,
    weights: ScoringWeights | None = None,
) -> float:
    """Combine scoring factors into final score.

    Args:
        score_normalized: Normalized vector similarity [0, 1]
        temporal_decay: Temporal decay factor [floor, 1]
        importance_prior: Type-based importance [0, 1]
        weights: Weight configuration (uses defaults if None)

    Returns:
        Combined score in [0, 1]
    """
    if weights is None:
        weights = ScoringWeights()

    return (
        weights.vector * score_normalized
        + weights.temporal * temporal_decay
        + weights.importance * importance_prior
    )


def score_memories(
    memories: list[dict[str, Any]],
    weights: ScoringWeights | None = None,
    reference_time: datetime | None = None,
) -> list[ScoredMemory]:
    """Apply temporal decay and importance scoring to memories.

    Takes raw vector search results and re-ranks them using multi-factor
    scoring that considers recency and importance.

    Args:
        memories: List of memory dicts from vector search with 'score' field
        weights: Scoring weight configuration
        reference_time: Reference time for decay calculation

    Returns:
        List of ScoredMemory objects sorted by final_score descending
    """
    if not memories:
        return []

    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    if weights is None:
        weights = ScoringWeights()

    # Extract raw scores for normalization
    raw_scores = [m.get("score", 0.0) for m in memories]
    normalized = normalize_scores(raw_scores)

    scored: list[ScoredMemory] = []
    for i, m in enumerate(memories):
        # Parse created_at timestamp
        created_at = m.get("created_at")
        if isinstance(created_at, (int, float)):
            created_dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
        elif isinstance(created_at, datetime):
            created_dt = (
                created_at
                if created_at.tzinfo
                else created_at.replace(tzinfo=timezone.utc)
            )
        else:
            # Default to reference time if no timestamp
            created_dt = reference_time

        memory_type = m.get("type", "fact")

        # Compute scoring components
        temporal = compute_temporal_decay(created_dt, memory_type, reference_time)
        importance = get_importance_prior(memory_type)
        final = compute_final_score(normalized[i], temporal, importance, weights)

        scored.append(
            ScoredMemory(
                uuid=m.get("uuid", ""),
                content=m.get("content", ""),
                type=memory_type,
                created_at=created_dt,
                score_raw=raw_scores[i],
                score_normalized=normalized[i],
                temporal_decay=temporal,
                importance_prior=importance,
                final_score=final,
                metadata=m.get("metadata"),
            )
        )

    # Sort by final score descending
    return sorted(scored, key=lambda x: -x.final_score)


def apply_temporal_scoring(
    memories: list[dict[str, Any]],
    weights: ScoringWeights | None = None,
    return_details: bool = False,
) -> list[dict[str, Any]]:
    """Apply temporal scoring to memory results (dict interface).

    Convenience function that preserves the original dict format while
    adding scoring information.

    Args:
        memories: List of memory dicts from vector search
        weights: Scoring weight configuration
        return_details: Include scoring breakdown in each result

    Returns:
        Re-ranked list of memory dicts with added 'final_score' field
    """
    scored = score_memories(memories, weights)

    results: list[dict[str, Any]] = []
    for sm in scored:
        # Find original memory dict
        original = next((m for m in memories if m.get("uuid") == sm.uuid), {})

        result = dict(original)
        result["final_score"] = sm.final_score

        if return_details:
            result["scoring_details"] = {
                "score_normalized": sm.score_normalized,
                "temporal_decay": sm.temporal_decay,
                "importance_prior": sm.importance_prior,
            }

        results.append(result)

    return results


# Supersession handling (Phase 2 - placeholder for now)

SUPERSESSION_PENALTY: float = 0.2


def apply_supersession_penalty(
    memories: list[dict[str, Any]],
    superseded_ids: set[str],
) -> list[dict[str, Any]]:
    """Apply penalty to superseded memories.

    Superseded memories should still be retrievable but ranked lower.

    Args:
        memories: List of scored memory dicts with 'final_score'
        superseded_ids: Set of UUIDs that have been superseded

    Returns:
        Memories with superseded ones penalized
    """
    results = []
    for m in memories:
        result = dict(m)
        if m.get("uuid") in superseded_ids:
            result["final_score"] = result.get("final_score", 0.0) * SUPERSESSION_PENALTY
            result["superseded"] = True
        results.append(result)

    return sorted(results, key=lambda x: -x.get("final_score", 0.0))
