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

# LLM Reranking constants
RERANK_CONTENT_PREVIEW_LEN: int = 200  # Max chars of content to show LLM for reranking


def _escape_xml_chars(text: str) -> str:
    """Escape XML special characters to prevent prompt injection.

    Memory content could contain XML-like tags that might confuse the LLM
    or attempt prompt injection. This escapes them to literal text.

    Args:
        text: Raw text that may contain XML characters

    Returns:
        Text with <, >, & escaped
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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


# LLM Reranking (Phase 2)

async def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = 10,
    rerank_pool: int = 20,
    model: str | None = None,
) -> dict[str, Any]:
    """Rerank search results using LLM for improved precision.

    Takes the top results from vector search and uses an LLM to reorder
    them by semantic relevance to the query. Also detects potential
    conflicts between memories.

    Args:
        query: Original search query
        results: List of memory dicts from search (with uuid, type, content)
        top_k: Number of top results to return after reranking
        rerank_pool: Number of results to consider for reranking (default: 20)
        model: LLM model to use (defaults to gemini-2.0-flash for speed)

    Returns:
        Dict with:
            - results: Reranked top-k results
            - conflicts: List of detected conflicts [[idx1, idx2, reason], ...]
            - rerank_applied: True if reranking was applied
    """
    from json_repair import loads as json_repair_loads
    from litellm import acompletion

    # Use fast model for reranking by default
    if model is None:
        model = "gemini/gemini-2.0-flash"

    # If not enough results to rerank, return as-is
    if len(results) <= top_k:
        return {
            "results": results,
            "conflicts": [],
            "rerank_applied": False,
        }

    # Take top rerank_pool for LLM consideration
    pool = results[:rerank_pool]

    # Format memories for LLM - escape content to prevent injection
    formatted_memories = []
    for i, r in enumerate(pool):
        raw_content = r.get("content", "")[:RERANK_CONTENT_PREVIEW_LEN]
        content_preview = _escape_xml_chars(raw_content)
        mem_type = r.get("type", "unknown")
        formatted_memories.append(f"[{i}] ({mem_type}): {content_preview}")

    memories_text = "\n".join(formatted_memories)

    try:
        response = await acompletion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Rerank these memories by relevance to the query. Also identify any conflicting information.

<query>{query}</query>

<memories>
{memories_text}
</memories>

Return ONLY valid JSON with this exact structure:
{{
  "indices": [most_relevant_index, second_most_relevant, ...],
  "conflicts": [[index1, index2, "brief reason"], ...]
}}

Rules:
- "indices" must contain integers from 0 to {len(pool) - 1} ordered by relevance
- Include at least {min(top_k, len(pool))} indices
- "conflicts" lists pairs of memory indices that contain contradictory information
- If no conflicts found, use empty array: []""",
                }
            ],
            max_tokens=300,
            temperature=0.0,  # Deterministic for consistent ranking
        )

        # Safety check for LLM response
        if not response.choices or not response.choices[0].message.content:
            return {
                "results": results[:top_k],
                "conflicts": [],
                "rerank_applied": False,
                "error": "Empty LLM response",
            }

        content = response.choices[0].message.content
        parsed = json_repair_loads(content)

        if not isinstance(parsed, dict):
            return {
                "results": results[:top_k],
                "conflicts": [],
                "rerank_applied": False,
                "error": "Invalid JSON structure",
            }

        # Extract and validate indices
        indices = parsed.get("indices", [])
        if not isinstance(indices, list):
            indices = []

        # Filter valid indices and remove duplicates while preserving order
        seen = set()
        valid_indices = []
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(pool) and idx not in seen:
                valid_indices.append(idx)
                seen.add(idx)

        # If LLM didn't return enough indices, fill with remaining
        if len(valid_indices) < top_k:
            for i in range(len(pool)):
                if i not in seen:
                    valid_indices.append(i)
                    seen.add(i)
                if len(valid_indices) >= top_k:
                    break

        # Build reranked results
        reranked = [pool[i] for i in valid_indices[:top_k]]

        # Extract conflicts (validate format)
        conflicts_raw = parsed.get("conflicts", [])
        conflicts = []
        if isinstance(conflicts_raw, list):
            for c in conflicts_raw:
                if isinstance(c, list) and len(c) >= 2:
                    idx1, idx2 = c[0], c[1]
                    reason = c[2] if len(c) > 2 else "Conflicting information"
                    if (
                        isinstance(idx1, int)
                        and isinstance(idx2, int)
                        and 0 <= idx1 < len(pool)
                        and 0 <= idx2 < len(pool)
                    ):
                        conflicts.append([idx1, idx2, str(reason)])

        return {
            "results": reranked,
            "conflicts": conflicts,
            "rerank_applied": True,
        }

    except Exception as e:
        # On any error, return original results without reranking
        return {
            "results": results[:top_k],
            "conflicts": [],
            "rerank_applied": False,
            "error": str(e),
        }
