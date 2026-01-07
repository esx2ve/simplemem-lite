"""Projects API endpoints for SimpleMem-Lite backend.

Provides endpoints for project status, bootstrap tracking, and metadata.
"""

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from simplemem_lite.backend.services import get_project_manager
from simplemem_lite.backend.toon import toonify
from simplemem_lite.log_config import get_logger

log = get_logger("backend.api.projects")

router = APIRouter()


class ProjectStatusResponse(BaseModel):
    """Response model for project status."""

    project_root: str
    project_name: str | None = None
    is_known: bool = False
    is_bootstrapped: bool = False
    never_ask: bool = False
    should_ask: bool = True
    is_watching: bool = False  # Placeholder - watchers are MCP-side
    error: str | None = None


class SetBootstrapRequest(BaseModel):
    """Request model for setting bootstrap status."""

    project_root: str
    is_bootstrapped: bool = True
    never_ask: bool = False


@router.get("/status")
async def get_project_status(project_root: str) -> ProjectStatusResponse:
    """Get bootstrap and metadata status for a project.

    Args:
        project_root: Absolute path to the project root directory

    Returns:
        ProjectStatusResponse with bootstrap state and metadata
    """
    log.info(f"get_project_status: {project_root}")
    fallback_name = Path(project_root).name

    try:
        manager = get_project_manager()
        state = manager.get_project_state(project_root)

        if state is None:
            # Unknown project
            log.debug(f"Project not found: {project_root}")
            return ProjectStatusResponse(
                project_root=project_root,
                project_name=fallback_name,
                is_known=False,
                is_bootstrapped=False,
                should_ask=True,
            )

        # Known project - return full state
        should_ask = manager.should_ask_bootstrap(project_root)
        log.info(f"Project status: bootstrapped={state.is_bootstrapped}, should_ask={should_ask}")

        return ProjectStatusResponse(
            project_root=state.project_root,
            project_name=state.project_name or fallback_name,
            is_known=True,
            is_bootstrapped=state.is_bootstrapped,
            never_ask=state.never_ask,
            should_ask=should_ask,
        )

    except Exception as e:
        log.error(f"get_project_status failed: {e}")
        return ProjectStatusResponse(
            project_root=project_root,
            error=str(e),
        )


@router.post("/bootstrap")
async def set_bootstrap_status(request: SetBootstrapRequest) -> dict:
    """Set the bootstrap status for a project.

    Args:
        request: SetBootstrapRequest with project_root and status

    Returns:
        Updated project state
    """
    log.info(f"set_bootstrap_status: {request.project_root} -> bootstrapped={request.is_bootstrapped}")

    try:
        manager = get_project_manager()
        state = manager.get_or_create_project(request.project_root)

        state.is_bootstrapped = request.is_bootstrapped
        if request.never_ask:
            state.never_ask = True

        manager.set_project_state(state)

        return {
            "success": True,
            "project_root": state.project_root,
            "is_bootstrapped": state.is_bootstrapped,
            "never_ask": state.never_ask,
        }

    except Exception as e:
        log.error(f"set_bootstrap_status failed: {e}")
        return {"success": False, "error": str(e)}


class ListProjectsRequest(BaseModel):
    """Request model for listing projects."""

    output_format: str | None = Field(
        default=None,
        description="Response format: 'toon' (default) or 'json'. Env var: SIMPLEMEM_OUTPUT_FORMAT.",
    )


@router.post("/list")
@toonify(headers=["project_root", "project_name", "is_bootstrapped", "never_ask"], result_key="projects")
async def list_projects_post(request: ListProjectsRequest) -> dict:
    """List all tracked projects (POST version with TOON support).

    Returns:
        List of project summaries with bootstrap status
    """
    log.debug("list_projects_post called")

    try:
        manager = get_project_manager()
        projects = manager.list_projects()
        return {"projects": projects, "count": len(projects)}

    except Exception as e:
        log.error(f"list_projects failed: {e}")
        return {"projects": [], "count": 0, "error": str(e)}


@router.get("/list")
async def list_projects() -> dict:
    """List all tracked projects (GET version, backwards compatible).

    Returns:
        List of project summaries with bootstrap status
    """
    log.debug("list_projects called")

    try:
        manager = get_project_manager()
        projects = manager.list_projects()
        return {"projects": projects, "count": len(projects)}

    except Exception as e:
        log.error(f"list_projects failed: {e}")
        return {"projects": [], "count": 0, "error": str(e)}
