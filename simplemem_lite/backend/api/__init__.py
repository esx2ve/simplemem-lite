"""API routers for SimpleMem-Lite backend."""

from fastapi import APIRouter

from simplemem_lite.backend.api.memories import router as memories_router
from simplemem_lite.backend.api.traces import router as traces_router
from simplemem_lite.backend.api.code import router as code_router
from simplemem_lite.backend.api.graph import router as graph_router
from simplemem_lite.backend.api.projects import router as projects_router
from simplemem_lite.backend.api.admin import router as admin_router
from simplemem_lite.backend.api.consolidation import router as consolidation_router
from simplemem_lite.backend.api.scratchpad import router as scratchpad_router
from simplemem_lite.backend.api.unified import router as unified_router

# Main API router that aggregates all sub-routers
router = APIRouter()

# V2 Unified API (3 tools: remember, recall, index)
router.include_router(unified_router, prefix="/v2", tags=["unified"])

# Legacy routers (kept for backwards compatibility)
router.include_router(memories_router, prefix="/memories", tags=["memories"])
router.include_router(traces_router, prefix="/traces", tags=["traces"])
router.include_router(code_router, prefix="/code", tags=["code"])
router.include_router(graph_router, prefix="/graph", tags=["graph"])
router.include_router(projects_router, prefix="/projects", tags=["projects"])
router.include_router(admin_router, prefix="/admin", tags=["admin"])
router.include_router(consolidation_router, prefix="/consolidate", tags=["consolidation"])
router.include_router(scratchpad_router, prefix="/scratchpad", tags=["scratchpad"])

__all__ = ["router"]
