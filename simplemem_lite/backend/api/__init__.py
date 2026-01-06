"""API routers for SimpleMem-Lite backend."""

from fastapi import APIRouter

from simplemem_lite.backend.api.memories import router as memories_router
from simplemem_lite.backend.api.traces import router as traces_router
from simplemem_lite.backend.api.code import router as code_router
from simplemem_lite.backend.api.graph import router as graph_router
from simplemem_lite.backend.api.projects import router as projects_router
from simplemem_lite.backend.api.admin import router as admin_router

# Main API router that aggregates all sub-routers
router = APIRouter()

router.include_router(memories_router, prefix="/memories", tags=["memories"])
router.include_router(traces_router, prefix="/traces", tags=["traces"])
router.include_router(code_router, prefix="/code", tags=["code"])
router.include_router(graph_router, prefix="/graph", tags=["graph"])
router.include_router(projects_router, prefix="/projects", tags=["projects"])
router.include_router(admin_router, prefix="/admin", tags=["admin"])

__all__ = ["router"]
