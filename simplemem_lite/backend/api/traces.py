"""Trace processing API endpoints.

The MCP thin layer reads trace files locally, compresses them,
and sends the content here for processing.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from simplemem_lite.backend.services import get_hierarchical_indexer, get_job_manager
from simplemem_lite.backend.toon import toonify
from simplemem_lite.compression import decompress_payload
from simplemem_lite.log_config import get_logger

router = APIRouter()
log = get_logger("backend.api.traces")

# Simple in-memory job tracking (for MVP - use proper queue in production)
# Note: Jobs are lost on restart. Production should use Redis/Celery.
_jobs: dict[str, dict[str, Any]] = {}
MAX_JOBS = 1000  # Prevent unbounded memory growth


class ProcessTraceRequest(BaseModel):
    """Request model for processing a trace.

    The trace_content can be:
    - Raw JSON (dict) for small traces
    - Compressed base64 string for large traces (set compressed=True)
    """

    session_id: str = Field(..., description="Session UUID to index")
    trace_content: str | dict | list = Field(..., description="Trace content (raw or compressed)")
    compressed: bool = Field(default=False, description="Whether trace_content is gzip+base64")
    background: bool = Field(default=True, description="Run processing in background")
    project_id: str | None = Field(default=None, description="Project ID for memory isolation")


class ProcessTraceResponse(BaseModel):
    """Response model for trace processing."""

    job_id: str | None = Field(default=None, description="Background job ID if async")
    status: str = Field(default="submitted")
    message: str | None = None
    session_summary_id: str | None = Field(default=None, description="Summary ID (sync mode only)")
    chunk_count: int | None = Field(default=None, description="Chunk count (sync mode only)")
    message_count: int | None = Field(default=None, description="Message count (sync mode only)")


class TraceInput(BaseModel):
    """Input model for a single trace in batch processing."""

    session_id: str = Field(..., description="Session UUID")
    trace_content: str | dict | list = Field(..., description="Trace content (raw or compressed)")
    compressed: bool = Field(default=False, description="Whether trace_content is gzip+base64")
    project_id: str | None = Field(default=None, description="Project ID for this trace")


class ProcessTraceBatchRequest(BaseModel):
    """Request model for batch trace processing."""

    traces: list[TraceInput] = Field(..., description="List of trace objects to process")
    max_concurrent: int = Field(default=3, ge=1, le=10)
    project_id: str | None = Field(default=None, description="Project ID for memory isolation (applied to all traces)")


def _cleanup_old_jobs() -> None:
    """Remove old completed/failed jobs to prevent unbounded memory growth."""
    if len(_jobs) < MAX_JOBS:
        return

    # Sort jobs by created time, keep only the newest MAX_JOBS/2
    terminal_states = ("completed", "failed", "cancelled")
    jobs_by_time = sorted(
        _jobs.items(),
        key=lambda x: x[1]["timestamps"]["created"],
        reverse=True
    )

    # Keep all non-terminal jobs and newest MAX_JOBS/2 terminal jobs
    keep_count = 0
    for job_id, job in jobs_by_time:
        if job["status"] not in terminal_states:
            continue
        keep_count += 1
        if keep_count > MAX_JOBS // 2:
            del _jobs[job_id]


def _create_job(job_type: str, session_id: str) -> str:
    """Create a new job entry and return the job ID."""
    _cleanup_old_jobs()  # Cleanup before creating new job
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "id": job_id,
        "type": job_type,
        "session_id": session_id,
        "status": "pending",
        "progress": 0,
        "message": "",
        "result": None,
        "error": None,
        "timestamps": {
            "created": datetime.now(timezone.utc).isoformat(),
            "started": None,
            "completed": None,
        },
    }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    """Update job fields."""
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)


async def _process_trace_task(
    job_id: str,
    session_id: str,
    trace_data: str | dict | list,
    project_id: str | None = None,
) -> None:
    """Background task to process a trace."""
    try:
        _update_job(
            job_id,
            status="running",
            timestamps={**_jobs[job_id]["timestamps"], "started": datetime.now(timezone.utc).isoformat()},
        )

        indexer = get_hierarchical_indexer()

        def progress_callback(progress: int, message: str):
            _update_job(job_id, progress=progress, message=message)

        result = await indexer.index_session_content(
            session_id=session_id,
            trace_content=trace_data,
            progress_callback=progress_callback,
            project_id=project_id,
        )

        if result:
            _update_job(
                job_id,
                status="completed",
                progress=100,
                message="Processing complete",
                result={
                    "session_summary_id": result.session_summary_id,
                    "chunk_count": len(result.chunk_summary_ids),
                    "message_count": len(result.message_ids),
                    "goal_id": result.goal_id,
                },
                timestamps={**_jobs[job_id]["timestamps"], "completed": datetime.now(timezone.utc).isoformat()},
            )
        else:
            _update_job(
                job_id,
                status="failed",
                error="No messages found in trace",
                timestamps={**_jobs[job_id]["timestamps"], "completed": datetime.now(timezone.utc).isoformat()},
            )

    except Exception as e:
        log.error(f"Trace processing failed for job {job_id}: {e}")
        _update_job(
            job_id,
            status="failed",
            error=str(e),
            timestamps={**_jobs[job_id]["timestamps"], "completed": datetime.now(timezone.utc).isoformat()},
        )


@router.post("/process", response_model=ProcessTraceResponse)
async def process_trace(
    request: ProcessTraceRequest,
    background_tasks: BackgroundTasks,
) -> ProcessTraceResponse:
    """Process a Claude Code session trace.

    Creates hierarchical summaries:
    - session_summary (1) - Overall session summary
    - chunk_summary (5-15) - Summaries of activity chunks
    """
    # Decompress if needed
    if request.compressed and isinstance(request.trace_content, str):
        try:
            trace_data = decompress_payload(request.trace_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decompress trace: {e}")
    else:
        trace_data = request.trace_content

    if request.background:
        # Background processing
        job_id = _create_job("process_trace", request.session_id)
        background_tasks.add_task(
            _process_trace_task, job_id, request.session_id, trace_data, request.project_id
        )
        return ProcessTraceResponse(
            job_id=job_id,
            status="submitted",
            message=f"Processing session {request.session_id} in background",
        )
    else:
        # Synchronous processing
        try:
            indexer = get_hierarchical_indexer()
            result = await indexer.index_session_content(
                session_id=request.session_id,
                trace_content=trace_data,
                project_id=request.project_id,
            )

            if result:
                return ProcessTraceResponse(
                    status="completed",
                    message="Processing complete",
                    session_summary_id=result.session_summary_id,
                    chunk_count=len(result.chunk_summary_ids),
                    message_count=len(result.message_ids),
                )
            else:
                raise HTTPException(status_code=400, detail="No messages found in trace")

        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Trace processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-batch")
async def process_trace_batch(
    request: ProcessTraceBatchRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Process multiple session traces."""
    queued = []
    errors = []
    job_ids = {}

    for trace_input in request.traces:
        try:
            # Decompress if needed
            if trace_input.compressed and isinstance(trace_input.trace_content, str):
                try:
                    trace_data = decompress_payload(trace_input.trace_content)
                except Exception as e:
                    errors.append({
                        "session_id": trace_input.session_id,
                        "error": f"Decompression failed: {e}",
                    })
                    continue
            else:
                trace_data = trace_input.trace_content

            # Create job and queue background task
            # Per-trace project_id takes precedence, fallback to request-level
            effective_project_id = trace_input.project_id or request.project_id
            job_id = _create_job("process_trace", trace_input.session_id)
            background_tasks.add_task(_process_trace_task, job_id, trace_input.session_id, trace_data, effective_project_id)

            queued.append(trace_input.session_id)
            job_ids[trace_input.session_id] = job_id

        except Exception as e:
            errors.append({
                "session_id": trace_input.session_id,
                "error": str(e),
            })

    return {
        "queued": queued,
        "errors": errors,
        "job_ids": job_ids,
    }


@router.get("/job/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """Get status of a background processing job.

    Checks both trace jobs (in-memory) and code indexing jobs (JobManager).
    """
    # Check trace jobs first
    if job_id in _jobs:
        return _jobs[job_id]

    # Check services JobManager (used by code indexing)
    try:
        job_manager = get_job_manager()
        job_info = job_manager.get_status(job_id)
        if job_info:
            return job_info
    except Exception as e:
        log.debug(f"JobManager lookup failed: {e}")

    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


class ListJobsRequest(BaseModel):
    """Request model for listing jobs."""

    include_completed: bool = Field(default=True, description="Include completed/failed jobs")
    limit: int = Field(default=20, ge=1, le=100)
    output_format: str | None = Field(
        default=None,
        description="Response format: 'json' or 'toon'. Defaults to SIMPLEMEM_OUTPUT_FORMAT env var.",
    )


@router.post("/jobs/list")
@toonify(headers=["id", "type", "status", "progress", "message"], result_key="jobs")
async def list_jobs_post(request: ListJobsRequest) -> dict:
    """List all background jobs (POST version with TOON support).

    Includes both trace jobs (in-memory) and code indexing jobs (JobManager).
    """
    # Get trace jobs
    jobs = list(_jobs.values())

    # Get code indexing jobs from services JobManager
    try:
        job_manager = get_job_manager()
        code_jobs = job_manager.list_jobs(include_completed=request.include_completed, limit=request.limit)
        jobs.extend(code_jobs)
    except Exception as e:
        log.debug(f"JobManager list failed: {e}")

    if not request.include_completed:
        jobs = [j for j in jobs if j.get("status") in ("pending", "running")]

    # Sort by created timestamp (newest first)
    jobs.sort(key=lambda j: j.get("timestamps", {}).get("created", "") or j.get("created_at", ""), reverse=True)

    # Apply limit
    jobs = jobs[:request.limit]

    return {"jobs": jobs}


@router.get("/jobs")
async def list_jobs(include_completed: bool = True, limit: int = 20) -> dict:
    """List all background jobs (GET version, backwards compatible).

    Includes both trace jobs (in-memory) and code indexing jobs (JobManager).
    """
    # Get trace jobs
    jobs = list(_jobs.values())

    # Get code indexing jobs from services JobManager
    try:
        job_manager = get_job_manager()
        code_jobs = job_manager.list_jobs(include_completed=include_completed, limit=limit)
        jobs.extend(code_jobs)
    except Exception as e:
        log.debug(f"JobManager list failed: {e}")

    if not include_completed:
        jobs = [j for j in jobs if j.get("status") in ("pending", "running")]

    # Sort by created timestamp (newest first)
    jobs.sort(key=lambda j: j.get("timestamps", {}).get("created", "") or j.get("created_at", ""), reverse=True)

    # Apply limit
    jobs = jobs[:limit]

    return {"jobs": jobs}


@router.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    """Cancel a running background job.

    Note: This sets the status to cancelled but doesn't actually stop
    the running task. For proper cancellation, use a task queue like Celery.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]
    if job["status"] in ("completed", "failed", "cancelled"):
        return {
            "cancelled": False,
            "message": f"Job already in terminal state: {job['status']}",
        }

    _update_job(
        job_id,
        status="cancelled",
        message="Cancelled by user",
        timestamps={**job["timestamps"], "completed": datetime.now(timezone.utc).isoformat()},
    )

    return {
        "cancelled": True,
        "message": f"Job {job_id} marked as cancelled",
    }
