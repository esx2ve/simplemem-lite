"""Admin API endpoints for SimpleMem-Lite backend.

These endpoints are for administrative operations like data wipes.
Most are DEV MODE ONLY for safety.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette import status

from simplemem_lite.backend.config import get_config
from simplemem_lite.backend.services import get_database_manager

log = logging.getLogger("simplemem_lite.backend.api.admin")

router = APIRouter()


class WipeRequest(BaseModel):
    """Request body for wipe endpoint."""

    confirm: str
    include_logs: bool = False


class WipeResponse(BaseModel):
    """Response from wipe endpoint."""

    status: str
    message: str
    stats: dict


@router.post("/wipe", response_model=WipeResponse)
async def wipe_all_data(request: WipeRequest) -> WipeResponse:
    """Wipe ALL data from SimpleMem. DEV MODE ONLY.

    This is a destructive operation that clears:
    - All memories (graph nodes and vectors)
    - All code chunks
    - All relationships
    - Metadata files (projects.json, session_state.db, etc.)
    - Jobs directory
    - Optionally: logs directory

    After wipe, empty tables/schema are reinitialized so the system is ready to use.

    **DEV MODE ONLY** - Returns 403 Forbidden in PROD mode.

    Request body:
    - confirm: Must be exactly "WIPE_ALL_DATA" to proceed
    - include_logs: If true, also wipe logs directory (default: false)

    Returns:
        Detailed stats of what was deleted
    """
    config = get_config()

    # CRITICAL: Only allow in DEV mode
    if not config.is_dev_mode:
        log.warning("WIPE: Attempted wipe in PROD mode - DENIED")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Wipe endpoint is only available in DEV mode. "
            "Set SIMPLEMEM_MODE=dev to enable.",
        )

    # Require explicit confirmation
    if request.confirm != "WIPE_ALL_DATA":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Must confirm with {"confirm": "WIPE_ALL_DATA"} in request body',
        )

    log.warning("=" * 70)
    log.warning("  ADMIN: Wipe request received and authorized")
    log.warning(f"  include_logs: {request.include_logs}")
    log.warning("=" * 70)

    # Execute wipe
    db_manager = get_database_manager()
    stats = db_manager.wipe_all_data(include_logs=request.include_logs)

    return WipeResponse(
        status="wiped",
        message="All data has been wiped. System reinitialized with empty tables.",
        stats=stats,
    )


@router.get("/status")
async def admin_status() -> dict:
    """Get admin status including security mode."""
    config = get_config()
    return {
        "mode": config.mode.value,
        "is_dev_mode": config.is_dev_mode,
        "wipe_available": config.is_dev_mode,
        "require_auth": config.require_auth,
        "require_project_id": config.require_project_id,
    }


class ReindexRequest(BaseModel):
    """Request body for reindex endpoint."""

    background: bool = True


class ReindexResponse(BaseModel):
    """Response from reindex endpoint."""

    status: str
    project_id: str
    reindexed: int | None = None
    errors: int | None = None
    total: int | None = None
    job_id: str | None = None
    message: str | None = None


@router.post("/reindex/{project_id}", response_model=ReindexResponse)
async def reindex_project(
    project_id: str,
    request: ReindexRequest | None = None,
) -> ReindexResponse:
    """Re-generate embeddings for all memories in a project.

    This fixes embedding model mismatches where memories were embedded
    with one model but searches use a different model.

    Args:
        project_id: Project to reindex (URL path parameter)
        request.background: Run in background job (default: True)

    Returns:
        If background=True: {"job_id": "...", "status": "submitted"}
        If background=False: {"reindexed": N, "errors": 0, "project_id": "..."}
    """
    from simplemem_lite.backend.services import get_memory_store

    background = request.background if request else True

    log.info(f"ADMIN: reindex_project called for {project_id}, background={background}")

    store = get_memory_store()

    if background:
        # For background, we would need to submit to a job manager
        # For now, just run synchronously with a warning for large projects
        log.warning("Background reindex via REST API not yet implemented, running synchronously")

    try:
        result = store.reindex_memories(project_id)
        return ReindexResponse(
            status="completed",
            project_id=project_id,
            reindexed=result.get("reindexed", 0),
            errors=result.get("errors", 0),
            total=result.get("total", 0),
        )
    except Exception as e:
        log.error(f"Reindex failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
