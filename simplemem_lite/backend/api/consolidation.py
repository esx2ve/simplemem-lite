"""Consolidation API endpoints for SimpleMem-Lite backend.

Graph consolidation operations:
- Entity deduplication
- Memory merging
- Supersession detection
"""

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from starlette import status

from simplemem_lite.backend.consolidation import (
    ConsolidationOperation,
    ConsolidationReport,
    consolidate_project,
)

log = logging.getLogger("simplemem_lite.backend.api.consolidation")

router = APIRouter()


class ConsolidateRequest(BaseModel):
    """Request body for consolidation endpoint."""

    project_id: str = Field(..., description="Project to consolidate")
    operations: list[str] | None = Field(
        default=None,
        description="Operations to run: entity_dedup, memory_merge, supersession (default: all)",
    )
    dry_run: bool = Field(
        default=False,
        description="Preview without executing",
    )
    confidence_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Auto-merge threshold (0.9 default)",
    )
    background: bool = Field(
        default=False,
        description="Run in background (returns job_id)",
    )


class ConsolidateResponse(BaseModel):
    """Response from consolidation endpoint."""

    project_id: str
    operations_run: list[str]
    dry_run: bool
    entity_dedup: dict[str, int]
    memory_merge: dict[str, int]
    supersession: dict[str, int]
    errors: list[str]
    warnings: list[str]
    review_queue_count: int


class JobResponse(BaseModel):
    """Response when consolidation is run in background."""

    job_id: str
    status: str
    message: str


# In-memory job status tracking (simple implementation)
# In production, would use Redis or database
_consolidation_jobs: dict[str, dict[str, Any]] = {}


@router.post("/run", response_model=ConsolidateResponse | JobResponse)
async def run_consolidation(
    request: ConsolidateRequest,
    background_tasks: BackgroundTasks,
) -> ConsolidateResponse | JobResponse:
    """Run graph consolidation for a project.

    Performs LLM-assisted graph maintenance:
    - Entity deduplication: Merge duplicate entities (main.py ↔ ./main.py)
    - Memory merging: Combine near-duplicate memories
    - Supersession detection: Mark older memories as superseded

    Args:
        request: Consolidation configuration

    Returns:
        If background=False: Full consolidation report
        If background=True: Job ID for status polling
    """
    log.info(f"Consolidation request received for project: {request.project_id}")

    # Validate operations if provided
    if request.operations:
        valid_ops = {op.value for op in ConsolidationOperation}
        for op in request.operations:
            if op not in valid_ops:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid operation: {op}. Valid: {valid_ops}",
                )

    if request.background:
        # Run in background
        import uuid

        job_id = str(uuid.uuid4())
        _consolidation_jobs[job_id] = {
            "status": "pending",
            "project_id": request.project_id,
            "result": None,
        }

        async def run_background():
            try:
                _consolidation_jobs[job_id]["status"] = "running"
                report = await consolidate_project(
                    project_id=request.project_id,
                    operations=request.operations,
                    dry_run=request.dry_run,
                    confidence_threshold=request.confidence_threshold,
                )
                _consolidation_jobs[job_id]["status"] = "completed"
                _consolidation_jobs[job_id]["result"] = report.to_dict()
            except Exception as e:
                log.error(f"Background consolidation failed: {e}", exc_info=True)
                _consolidation_jobs[job_id]["status"] = "failed"
                _consolidation_jobs[job_id]["error"] = str(e)

        background_tasks.add_task(run_background)

        return JobResponse(
            job_id=job_id,
            status="pending",
            message=f"Consolidation started for {request.project_id}. Poll /consolidate/status/{job_id} for results.",
        )

    # Run synchronously
    try:
        report = await consolidate_project(
            project_id=request.project_id,
            operations=request.operations,
            dry_run=request.dry_run,
            confidence_threshold=request.confidence_threshold,
        )
        result = report.to_dict()
        return ConsolidateResponse(**result)

    except Exception as e:
        log.error(f"Consolidation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/status/{job_id}")
async def get_consolidation_status(job_id: str) -> dict[str, Any]:
    """Get status of a background consolidation job.

    Args:
        job_id: Job ID from /consolidate/run response

    Returns:
        Job status with result if completed
    """
    if job_id not in _consolidation_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    job = _consolidation_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "project_id": job["project_id"],
        "result": job.get("result"),
        "error": job.get("error"),
    }


@router.get("/review-queue/{project_id}")
async def get_review_queue(
    project_id: str,
    status: str = "pending",
    type_filter: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Get pending merge candidates requiring manual review.

    Returns candidates with confidence 0.7-0.9 that weren't auto-merged.
    These require human review before approval.

    Args:
        project_id: Project to get review queue for
        status: Filter by status (pending, approved, rejected)
        type_filter: Filter by type (entity_dedup, memory_merge, supersession)
        limit: Maximum items to return

    Returns:
        List of review items with decision details
    """
    from simplemem_lite.backend.services import get_memory_store

    log.info(f"Review queue requested for project: {project_id}")

    store = get_memory_store()
    candidates = store.db.get_review_candidates(
        project_id=project_id,
        status=status,
        type_filter=type_filter,
        limit=limit,
    )

    return {
        "project_id": project_id,
        "items": candidates,
        "count": len(candidates),
    }


@router.post("/approve/{candidate_id}")
async def approve_candidate(candidate_id: str) -> dict[str, Any]:
    """Approve and execute a pending merge candidate.

    Executes the merge/supersession that was queued for review.
    Idempotent - approving twice returns success without re-executing.

    Args:
        candidate_id: Candidate ID from review queue

    Returns:
        Result of the approval action
    """
    import time

    from simplemem_lite.backend.services import get_memory_store

    log.info(f"Approving candidate: {candidate_id}")

    store = get_memory_store()
    candidate = store.db.get_review_candidate(candidate_id)

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate not found: {candidate_id}",
        )

    if candidate["status"] != "pending":
        return {
            "candidate_id": candidate_id,
            "status": "already_resolved",
            "previous_status": candidate["status"],
        }

    decision_data = candidate["decision_data"]

    # Execute the merge based on type
    try:
        if candidate["type"] == "entity_dedup":
            # Execute entity merge
            entity_a = decision_data.get("entity_a_name")
            entity_b = decision_data.get("entity_b_name")
            canonical = decision_data.get("canonical_name") or entity_a
            entity_type = decision_data.get("entity_type", "file")

            deprecated = entity_b if entity_a == canonical else entity_a

            log.info(f"Executing entity merge: {deprecated} -> {canonical}")

            # Redirect all edges from deprecated to canonical
            store.db.graph.query(
                """
                MATCH (m)-[r]->(e:Entity {name: $deprecated_name, type: $type})
                MATCH (canonical:Entity {name: $canonical_name, type: $type})
                CREATE (m)-[r2:REFERENCES]->(canonical)
                DELETE r
                """,
                {
                    "deprecated_name": deprecated,
                    "canonical_name": canonical,
                    "type": entity_type,
                },
            )

            # Delete deprecated entity if no remaining edges
            store.db.graph.query(
                """
                MATCH (e:Entity {name: $name, type: $type})
                WHERE NOT EXISTS((e)-[]-())
                DELETE e
                """,
                {"name": deprecated, "type": entity_type},
            )

        elif candidate["type"] == "memory_merge":
            # Execute memory merge
            mem_a_uuid = decision_data.get("memory_a_uuid")
            mem_b_uuid = decision_data.get("memory_b_uuid")
            merged_content = decision_data.get("merged_content")

            # Get timestamps to determine newer/older
            result = store.db.graph.query(
                """
                MATCH (a:Memory {uuid: $uuid_a})
                MATCH (b:Memory {uuid: $uuid_b})
                RETURN a.created_at, b.created_at
                """,
                {"uuid_a": mem_a_uuid, "uuid_b": mem_b_uuid},
            )

            if result.result_set:
                time_a, time_b = result.result_set[0]
                if (time_a or 0) >= (time_b or 0):
                    newer_uuid, older_uuid = mem_a_uuid, mem_b_uuid
                else:
                    newer_uuid, older_uuid = mem_b_uuid, mem_a_uuid
            else:
                newer_uuid, older_uuid = mem_a_uuid, mem_b_uuid

            log.info(f"Executing memory merge: {older_uuid[:8]}... -> {newer_uuid[:8]}...")

            # Update newer memory with merged content
            if merged_content:
                store.db.graph.query(
                    """
                    MATCH (m:Memory {uuid: $uuid})
                    SET m.content = $content,
                        m.merged_from = $older_uuid
                    """,
                    {
                        "uuid": newer_uuid,
                        "content": merged_content,
                        "older_uuid": older_uuid,
                    },
                )

            # Mark older as merged
            store.db.mark_merged(older_uuid, newer_uuid)

        elif candidate["type"] == "supersession":
            # Execute supersession
            newer_uuid = decision_data.get("newer_uuid")
            older_uuid = decision_data.get("older_uuid")
            supersession_type = decision_data.get("supersession_type", "full_replace")

            log.info(f"Executing supersession: {newer_uuid[:8]}... supersedes {older_uuid[:8]}...")

            store.db.add_supersession(
                newer_uuid=newer_uuid,
                older_uuid=older_uuid,
                confidence=candidate["confidence"],
                supersession_type=supersession_type,
            )

        # Update candidate status
        store.db.update_candidate_status(candidate_id, "approved", int(time.time()))

        return {
            "candidate_id": candidate_id,
            "status": "approved",
            "type": candidate["type"],
        }

    except Exception as e:
        log.error(f"Failed to execute approval: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute merge: {str(e)}",
        )


@router.post("/reject/{candidate_id}")
async def reject_candidate(
    candidate_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Reject a candidate and mark pair to skip in future runs.

    Creates a REJECTED_PAIR edge so future consolidation runs will
    automatically skip this pair.

    Args:
        candidate_id: Candidate ID from review queue
        reason: Optional reason for rejection

    Returns:
        Result of the rejection action
    """
    import time

    from simplemem_lite.backend.services import get_memory_store

    log.info(f"Rejecting candidate: {candidate_id} (reason: {reason})")

    store = get_memory_store()
    candidate = store.db.get_review_candidate(candidate_id)

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate not found: {candidate_id}",
        )

    if candidate["status"] != "pending":
        return {
            "candidate_id": candidate_id,
            "status": "already_resolved",
            "previous_status": candidate["status"],
        }

    # Add rejected pair edge for future skip
    store.db.add_rejected_pair(
        candidate["source_id"],
        candidate["target_id"],
        candidate_id,
    )

    # Update candidate status
    store.db.update_candidate_status(candidate_id, "rejected", int(time.time()))

    return {
        "candidate_id": candidate_id,
        "status": "rejected",
        "type": candidate["type"],
        "reason": reason,
    }


@router.delete("/review-queue/{project_id}")
async def clear_review_queue(
    project_id: str,
    status_filter: str = "pending",
) -> dict[str, Any]:
    """Clear all review candidates for a project.

    Use this to reset the queue after filter improvements or testing.

    Args:
        project_id: Project to clear queue for
        status_filter: Only clear items with this status (default: pending)

    Returns:
        Count of deleted items
    """
    from simplemem_lite.backend.services import get_memory_store

    log.info(f"Clearing review queue for project: {project_id} (status: {status_filter})")

    store = get_memory_store()

    # Delete matching review candidates
    result = store.db.graph.query(
        """
        MATCH (c:ReviewCandidate)
        WHERE c.project_id = $project_id AND c.status = $status
        DELETE c
        RETURN count(c) AS deleted
        """,
        {"project_id": project_id, "status": status_filter},
    )

    deleted = 0
    if result.result_set:
        deleted = result.result_set[0][0]

    log.info(f"Deleted {deleted} review candidates")

    return {
        "project_id": project_id,
        "status_filter": status_filter,
        "deleted": deleted,
    }
