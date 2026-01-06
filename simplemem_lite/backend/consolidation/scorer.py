"""Phase 2: LLM scoring for consolidation candidates.

Uses batch prompting to classify candidates efficiently:
- Batches 5-10 comparisons per prompt
- Uses gemini-2.0-flash-lite for cheap, fast classification
- Structured JSON output with schema validation
- Async processing with asyncio.gather for parallelism

Cost: ~$0.01-0.05 per consolidation run (100-500 candidates)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from litellm import acompletion

from simplemem_lite.log_config import get_logger

from .candidates import EntityPair, MemoryPair, SupersessionPair

log = get_logger("consolidation.scorer")

# Batch size for prompt batching (5-10 comparisons per prompt)
BATCH_SIZE = 8

# LLM model for classification
CLASSIFIER_MODEL = "gemini/gemini-2.0-flash-lite"


@dataclass
class EntityDecision:
    """Decision for an entity pair."""

    pair: EntityPair
    same_entity: bool
    confidence: float
    canonical_name: str | None
    reason: str


@dataclass
class MemoryDecision:
    """Decision for a memory pair."""

    pair: MemoryPair
    should_merge: bool
    confidence: float
    merged_content: str | None
    reason: str


@dataclass
class SupersessionDecision:
    """Decision for a supersession pair."""

    pair: SupersessionPair
    supersedes: bool
    confidence: float
    supersession_type: str  # "full_replace", "partial_update", "none"
    reason: str


# ============================================================================
# Prompt Templates
# ============================================================================

ENTITY_DEDUP_PROMPT = """Compare these entity pairs and determine if they refer to the same thing.
For each pair, provide:
- same: true/false
- confidence: 0.0-1.0
- canonical: preferred name if same (keep the cleaner/shorter version)
- reason: brief explanation (max 20 words)

Entity pairs to compare:
{pairs}

Respond with a JSON array. ONLY output valid JSON, no other text:
[
  {{"pair": 1, "same": true, "confidence": 0.95, "canonical": "src/main.py", "reason": "Same file, different path format"}},
  ...
]"""

MEMORY_MERGE_PROMPT = """Determine if these memory pairs should be merged (contain essentially the same information).
For each pair, provide:
- merge: true/false
- confidence: 0.0-1.0
- merged: combined content if merging (preserve key details from both)
- reason: brief explanation (max 20 words)

Memory pairs to compare:
{pairs}

Respond with a JSON array. ONLY output valid JSON, no other text:
[
  {{"pair": 1, "merge": true, "confidence": 0.9, "merged": "Combined insight...", "reason": "Nearly identical content"}},
  ...
]"""

SUPERSESSION_PROMPT = """Determine if the newer memory supersedes (replaces/updates) the older one.
For each pair, provide:
- supersedes: true/false
- confidence: 0.0-1.0
- type: "full_replace" (completely replaces), "partial_update" (updates some info), or "none"
- reason: brief explanation (max 20 words)

Memory pairs to compare:
{pairs}

Respond with a JSON array. ONLY output valid JSON, no other text:
[
  {{"pair": 1, "supersedes": true, "confidence": 0.85, "type": "full_replace", "reason": "Newer provides updated solution"}},
  ...
]"""


# ============================================================================
# Batch Processing
# ============================================================================


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of specified size."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def format_entity_pairs(pairs: list[EntityPair]) -> str:
    """Format entity pairs for the prompt."""
    lines = []
    for i, p in enumerate(pairs, 1):
        lines.append(
            f"{i}. Entity A: \"{p.entity_a['name']}\" vs Entity B: \"{p.entity_b['name']}\" (type: {p.entity_type})"
        )
    return "\n".join(lines)


def format_memory_pairs(pairs: list[MemoryPair]) -> str:
    """Format memory pairs for the prompt."""
    lines = []
    for i, p in enumerate(pairs, 1):
        # Truncate content to avoid token overflow
        content_a = p.memory_a.get("content", "")[:200]
        content_b = p.memory_b.get("content", "")[:200]
        shared = ", ".join(p.shared_entities[:3]) if p.shared_entities else "none"
        lines.append(
            f'{i}. Memory A ({p.memory_a.get("type", "fact")}): "{content_a}..."\n'
            f'   Memory B ({p.memory_b.get("type", "fact")}): "{content_b}..."\n'
            f"   Shared entities: {shared}"
        )
    return "\n\n".join(lines)


def format_supersession_pairs(pairs: list[SupersessionPair]) -> str:
    """Format supersession pairs for the prompt."""
    lines = []
    for i, p in enumerate(pairs, 1):
        # Truncate content to avoid token overflow
        newer_content = p.newer.get("content", "")[:200]
        older_content = p.older.get("content", "")[:200]
        lines.append(
            f'{i}. NEWER ({p.time_delta_days} days ago): "{newer_content}..."\n'
            f'   OLDER: "{older_content}..."\n'
            f"   Shared entity: {p.entity}"
        )
    return "\n\n".join(lines)


async def call_llm_with_retry(
    prompt: str, max_retries: int = 3
) -> list[dict[str, Any]]:
    """Call LLM and parse JSON response with retry logic."""
    for attempt in range(max_retries):
        try:
            response = await acompletion(
                model=CLASSIFIER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1,  # Low temperature for consistent classification
            )

            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            # Parse JSON
            result = json.loads(content)
            if isinstance(result, list):
                return result
            else:
                log.warning(f"LLM returned non-list: {type(result)}")
                return []

        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                log.error(f"Failed to parse LLM response after {max_retries} attempts")
                return []
        except Exception as e:
            log.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                log.error(f"LLM call failed after {max_retries} attempts: {e}")
                return []

    return []


# ============================================================================
# Scoring Functions
# ============================================================================


async def score_entity_batch(pairs: list[EntityPair]) -> list[EntityDecision]:
    """Score a batch of entity pairs."""
    if not pairs:
        return []

    prompt = ENTITY_DEDUP_PROMPT.format(pairs=format_entity_pairs(pairs))
    results = await call_llm_with_retry(prompt)

    decisions = []
    for i, p in enumerate(pairs):
        # Find corresponding result
        result = next((r for r in results if r.get("pair") == i + 1), None)
        if result:
            decisions.append(
                EntityDecision(
                    pair=p,
                    same_entity=result.get("same", False),
                    confidence=float(result.get("confidence", 0.0)),
                    canonical_name=result.get("canonical"),
                    reason=result.get("reason", ""),
                )
            )
        else:
            # Default to not merging if result missing
            decisions.append(
                EntityDecision(
                    pair=p,
                    same_entity=False,
                    confidence=0.0,
                    canonical_name=None,
                    reason="No LLM response for this pair",
                )
            )

    return decisions


async def score_memory_batch(pairs: list[MemoryPair]) -> list[MemoryDecision]:
    """Score a batch of memory pairs."""
    if not pairs:
        return []

    prompt = MEMORY_MERGE_PROMPT.format(pairs=format_memory_pairs(pairs))
    results = await call_llm_with_retry(prompt)

    decisions = []
    for i, p in enumerate(pairs):
        result = next((r for r in results if r.get("pair") == i + 1), None)
        if result:
            decisions.append(
                MemoryDecision(
                    pair=p,
                    should_merge=result.get("merge", False),
                    confidence=float(result.get("confidence", 0.0)),
                    merged_content=result.get("merged"),
                    reason=result.get("reason", ""),
                )
            )
        else:
            decisions.append(
                MemoryDecision(
                    pair=p,
                    should_merge=False,
                    confidence=0.0,
                    merged_content=None,
                    reason="No LLM response for this pair",
                )
            )

    return decisions


async def score_supersession_batch(
    pairs: list[SupersessionPair],
) -> list[SupersessionDecision]:
    """Score a batch of supersession pairs."""
    if not pairs:
        return []

    prompt = SUPERSESSION_PROMPT.format(pairs=format_supersession_pairs(pairs))
    results = await call_llm_with_retry(prompt)

    decisions = []
    for i, p in enumerate(pairs):
        result = next((r for r in results if r.get("pair") == i + 1), None)
        if result:
            decisions.append(
                SupersessionDecision(
                    pair=p,
                    supersedes=result.get("supersedes", False),
                    confidence=float(result.get("confidence", 0.0)),
                    supersession_type=result.get("type", "none"),
                    reason=result.get("reason", ""),
                )
            )
        else:
            decisions.append(
                SupersessionDecision(
                    pair=p,
                    supersedes=False,
                    confidence=0.0,
                    supersession_type="none",
                    reason="No LLM response for this pair",
                )
            )

    return decisions


# ============================================================================
# Main Scoring Entry Points
# ============================================================================


async def score_entity_pairs(pairs: list[EntityPair]) -> list[EntityDecision]:
    """Score all entity pairs with batched LLM calls.

    Args:
        pairs: List of EntityPair candidates

    Returns:
        List of EntityDecision with classification results
    """
    if not pairs:
        return []

    log.info(f"Scoring {len(pairs)} entity pairs in batches of {BATCH_SIZE}")

    # Split into batches and process in parallel
    batches = chunk_list(pairs, BATCH_SIZE)
    tasks = [score_entity_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks)

    # Flatten results
    all_decisions = []
    for batch_decisions in results:
        all_decisions.extend(batch_decisions)

    log.info(f"Scored {len(all_decisions)} entity pairs")
    return all_decisions


async def score_memory_pairs(pairs: list[MemoryPair]) -> list[MemoryDecision]:
    """Score all memory pairs with batched LLM calls.

    Args:
        pairs: List of MemoryPair candidates

    Returns:
        List of MemoryDecision with classification results
    """
    if not pairs:
        return []

    log.info(f"Scoring {len(pairs)} memory pairs in batches of {BATCH_SIZE}")

    batches = chunk_list(pairs, BATCH_SIZE)
    tasks = [score_memory_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks)

    all_decisions = []
    for batch_decisions in results:
        all_decisions.extend(batch_decisions)

    log.info(f"Scored {len(all_decisions)} memory pairs")
    return all_decisions


async def score_supersession_pairs(
    pairs: list[SupersessionPair],
) -> list[SupersessionDecision]:
    """Score all supersession pairs with batched LLM calls.

    Args:
        pairs: List of SupersessionPair candidates

    Returns:
        List of SupersessionDecision with classification results
    """
    if not pairs:
        return []

    log.info(f"Scoring {len(pairs)} supersession pairs in batches of {BATCH_SIZE}")

    batches = chunk_list(pairs, BATCH_SIZE)
    tasks = [score_supersession_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks)

    all_decisions = []
    for batch_decisions in results:
        all_decisions.extend(batch_decisions)

    log.info(f"Scored {len(all_decisions)} supersession pairs")
    return all_decisions


__all__ = [
    "EntityDecision",
    "MemoryDecision",
    "SupersessionDecision",
    "score_entity_pairs",
    "score_memory_pairs",
    "score_supersession_pairs",
]
