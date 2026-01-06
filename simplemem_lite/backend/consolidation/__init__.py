"""Graph consolidation module.

Performs LLM-assisted graph maintenance:
- Entity deduplication (merge `main.py` ↔ `./main.py`)
- Memory merging (combine near-duplicate memories)
- Supersession detection (mark older memories as superseded)

Uses blocking + LSH strategy to achieve O(n) complexity instead of O(n²).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from simplemem_lite.log_config import get_logger

log = get_logger("consolidation")


class ConsolidationOperation(str, Enum):
    """Available consolidation operations."""

    ENTITY_DEDUP = "entity_dedup"
    MEMORY_MERGE = "memory_merge"
    SUPERSESSION = "supersession"


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation run."""

    operations: list[ConsolidationOperation] = field(
        default_factory=lambda: list(ConsolidationOperation)
    )
    confidence_threshold: float = 0.9  # Auto-merge threshold
    entity_similarity_threshold: float = 0.85  # LSH threshold for entities
    memory_similarity_threshold: float = 0.90  # LSH threshold for memories
    dry_run: bool = False  # Preview without executing
    max_candidates_per_type: int = 1000  # Limit candidates to prevent runaway


@dataclass
class ConsolidationReport:
    """Report from a consolidation run."""

    project_id: str
    operations_run: list[str]
    dry_run: bool

    # Entity deduplication results
    entity_candidates_found: int = 0
    entity_merges_executed: int = 0
    entity_merges_queued: int = 0

    # Memory merge results
    memory_candidates_found: int = 0
    memory_merges_executed: int = 0
    memory_merges_queued: int = 0

    # Supersession results
    supersession_candidates_found: int = 0
    supersessions_executed: int = 0
    supersessions_queued: int = 0

    # Errors and warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Details for review queue
    review_queue: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "project_id": self.project_id,
            "operations_run": self.operations_run,
            "dry_run": self.dry_run,
            "entity_dedup": {
                "candidates_found": self.entity_candidates_found,
                "merges_executed": self.entity_merges_executed,
                "merges_queued": self.entity_merges_queued,
            },
            "memory_merge": {
                "candidates_found": self.memory_candidates_found,
                "merges_executed": self.memory_merges_executed,
                "merges_queued": self.memory_merges_queued,
            },
            "supersession": {
                "candidates_found": self.supersession_candidates_found,
                "supersessions_executed": self.supersessions_executed,
                "supersessions_queued": self.supersessions_queued,
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "review_queue_count": len(self.review_queue),
        }


async def consolidate_project(
    project_id: str,
    operations: list[str] | None = None,
    dry_run: bool = False,
    confidence_threshold: float = 0.9,
) -> ConsolidationReport:
    """Run graph consolidation for a project.

    Main entry point for consolidation. Orchestrates the 3-phase pipeline:
    1. Candidate generation (no LLM, uses embeddings + LSH)
    2. LLM scoring & decision (gemini-flash classifier)
    3. Execution & reporting (auto-merge high confidence)

    Args:
        project_id: Project to consolidate
        operations: List of operations to run (default: all)
            - "entity_dedup": Merge duplicate entities
            - "memory_merge": Merge near-duplicate memories
            - "supersession": Detect memory supersession
        dry_run: If True, report candidates without executing
        confidence_threshold: Auto-merge threshold (default 0.9)

    Returns:
        ConsolidationReport with actions taken and review queue
    """
    from .candidates import (
        find_entity_candidates,
        find_memory_candidates,
        find_supersession_candidates,
    )
    from .executor import (
        execute_entity_merges,
        execute_memory_merges,
        execute_supersessions,
    )
    from .scorer import score_entity_pairs, score_memory_pairs, score_supersession_pairs

    log.info(f"Starting consolidation for project: {project_id}")

    # Parse operations
    if operations is None:
        ops = list(ConsolidationOperation)
    else:
        ops = [ConsolidationOperation(op) for op in operations]

    config = ConsolidationConfig(
        operations=ops,
        confidence_threshold=confidence_threshold,
        dry_run=dry_run,
    )

    report = ConsolidationReport(
        project_id=project_id,
        operations_run=[op.value for op in ops],
        dry_run=dry_run,
    )

    # Phase 1: Candidate Generation (no LLM)
    log.info("Phase 1: Candidate generation")

    if ConsolidationOperation.ENTITY_DEDUP in ops:
        try:
            entity_candidates = await find_entity_candidates(
                project_id, config.entity_similarity_threshold
            )
            report.entity_candidates_found = len(entity_candidates)
            log.info(f"Found {len(entity_candidates)} entity dedup candidates")
        except Exception as e:
            log.error(f"Entity candidate generation failed: {e}")
            report.errors.append(f"Entity candidate generation: {e}")
            entity_candidates = []
    else:
        entity_candidates = []

    if ConsolidationOperation.MEMORY_MERGE in ops:
        try:
            memory_candidates = await find_memory_candidates(
                project_id, config.memory_similarity_threshold
            )
            report.memory_candidates_found = len(memory_candidates)
            log.info(f"Found {len(memory_candidates)} memory merge candidates")
        except Exception as e:
            log.error(f"Memory candidate generation failed: {e}")
            report.errors.append(f"Memory candidate generation: {e}")
            memory_candidates = []
    else:
        memory_candidates = []

    if ConsolidationOperation.SUPERSESSION in ops:
        try:
            supersession_candidates = await find_supersession_candidates(project_id)
            report.supersession_candidates_found = len(supersession_candidates)
            log.info(f"Found {len(supersession_candidates)} supersession candidates")
        except Exception as e:
            log.error(f"Supersession candidate generation failed: {e}")
            report.errors.append(f"Supersession candidate generation: {e}")
            supersession_candidates = []
    else:
        supersession_candidates = []

    # Phase 2: LLM Scoring (gemini-flash)
    log.info("Phase 2: LLM scoring")

    if entity_candidates:
        try:
            entity_decisions = await score_entity_pairs(entity_candidates)
        except Exception as e:
            log.error(f"Entity scoring failed: {e}")
            report.errors.append(f"Entity scoring: {e}")
            entity_decisions = []
    else:
        entity_decisions = []

    if memory_candidates:
        try:
            memory_decisions = await score_memory_pairs(memory_candidates)
        except Exception as e:
            log.error(f"Memory scoring failed: {e}")
            report.errors.append(f"Memory scoring: {e}")
            memory_decisions = []
    else:
        memory_decisions = []

    if supersession_candidates:
        try:
            supersession_decisions = await score_supersession_pairs(supersession_candidates)
        except Exception as e:
            log.error(f"Supersession scoring failed: {e}")
            report.errors.append(f"Supersession scoring: {e}")
            supersession_decisions = []
    else:
        supersession_decisions = []

    # Phase 3: Execution
    if not dry_run:
        log.info("Phase 3: Execution")

        if entity_decisions:
            result = await execute_entity_merges(entity_decisions, config, project_id=project_id)
            report.entity_merges_executed = result["executed"]
            report.entity_merges_queued = result["queued"]
            report.review_queue.extend(result.get("review_items", []))

        if memory_decisions:
            result = await execute_memory_merges(memory_decisions, config, project_id=project_id)
            report.memory_merges_executed = result["executed"]
            report.memory_merges_queued = result["queued"]
            report.review_queue.extend(result.get("review_items", []))

        if supersession_decisions:
            result = await execute_supersessions(supersession_decisions, config, project_id=project_id)
            report.supersessions_executed = result["executed"]
            report.supersessions_queued = result["queued"]
            report.review_queue.extend(result.get("review_items", []))
    else:
        log.info("Dry run - skipping execution")
        # In dry run, all candidates go to review queue
        for d in entity_decisions:
            report.review_queue.append({"type": "entity_dedup", "decision": d})
        for d in memory_decisions:
            report.review_queue.append({"type": "memory_merge", "decision": d})
        for d in supersession_decisions:
            report.review_queue.append({"type": "supersession", "decision": d})

    log.info(f"Consolidation complete: {report.to_dict()}")
    return report


__all__ = [
    "ConsolidationOperation",
    "ConsolidationConfig",
    "ConsolidationReport",
    "consolidate_project",
]
