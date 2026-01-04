"""Code indexing and search API endpoints.

The MCP thin layer reads code files locally, compresses them,
and sends the content here for indexing.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simplemem_lite.backend.config import get_config
from simplemem_lite.backend.services import get_code_indexer, get_job_manager, get_memory_store
from simplemem_lite.compression import decompress_payload
from simplemem_lite.log_config import get_logger

router = APIRouter()
log = get_logger("backend.api.code")


def require_project_id(project_root: str | None, endpoint: str) -> None:
    """Validate that project_root/project_id is provided when required.

    For code endpoints, project_root serves as project_id.

    Args:
        project_root: The project_root from the request
        endpoint: Name of the endpoint for error message

    Raises:
        HTTPException: 400 if project_id is required but not provided
    """
    config = get_config()
    if config.require_project_id and not project_root:
        raise HTTPException(
            status_code=400,
            detail=f"project_root is required for {endpoint}. "
            "Cloud API requires project isolation for all operations.",
        )


class FileInput(BaseModel):
    """Input model for a file to be indexed."""

    path: str = Field(..., description="File path relative to project root")
    content: str | dict = Field(..., description="File content (string or dict if JSON)")
    compressed: bool = Field(default=False, description="Whether content is gzip+base64")


class IndexDirectoryRequest(BaseModel):
    """Request model for indexing code files.

    The MCP layer reads files locally and sends their content here.
    """

    project_root: str = Field(..., description="Project root path (for identification)")
    files: list[FileInput] = Field(..., description="List of files to index")
    clear_existing: bool = Field(default=True, description="Clear existing index")
    background: bool = Field(default=True, description="Run in background")


class FileUpdateInput(BaseModel):
    """Input model for a file update operation."""

    path: str = Field(..., description="File path relative to project root")
    action: str = Field(..., description="Action: 'add', 'modify', or 'delete'")
    content: str | None = Field(default=None, description="File content (required for add/modify)")
    compressed: bool = Field(default=False, description="Whether content is gzip+base64")


class UpdateCodeRequest(BaseModel):
    """Request model for incremental code index updates."""

    project_root: str = Field(..., description="Project root path")
    updates: list[FileUpdateInput] = Field(..., description="List of file updates")


class SearchCodeRequest(BaseModel):
    """Request model for semantic code search."""

    query: str = Field(..., description="Natural language search query")
    limit: int = Field(default=10, ge=1, le=100)
    project_root: str | None = Field(default=None, description="Filter to specific project")


class CodeRelatedMemoriesRequest(BaseModel):
    """Request model for finding memories related to code."""

    chunk_uuid: str = Field(..., description="Code chunk UUID from search results")
    limit: int = Field(default=10, ge=1, le=50)


class MemoryRelatedCodeRequest(BaseModel):
    """Request model for finding code related to a memory."""

    memory_uuid: str = Field(..., description="Memory UUID")
    limit: int = Field(default=10, ge=1, le=50)


class CodeIndexStatusRequest(BaseModel):
    """Request model for updating code index status."""

    status: str | None = Field(default=None, description="Status: idle, indexing, watching")
    watchers: int | None = Field(default=None, description="Number of active watchers")
    projects_watching: list[str] | None = Field(default=None, description="Projects being watched")
    indexing_in_progress: bool | None = Field(default=None, description="Whether indexing is running")
    files_done: int | None = Field(default=None, description="Files indexed so far")
    files_total: int | None = Field(default=None, description="Total files to index")
    current_file: str | None = Field(default=None, description="Currently indexing file")
    total_files: int | None = Field(default=None, description="Total indexed files (stats)")
    total_chunks: int | None = Field(default=None, description="Total code chunks (stats)")


@router.post("/index")
async def index_code(request: IndexDirectoryRequest) -> dict:
    """Index code files for semantic search.

    The MCP layer reads files locally and sends content here.
    Runs in background by default to return immediately.
    """
    require_project_id(request.project_root, "index_code")
    try:
        # Decompress files if needed
        processed_files = []
        for file_input in request.files:
            content = file_input.content
            if file_input.compressed and isinstance(content, str):
                try:
                    content = decompress_payload(content)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to decompress {file_input.path}: {e}"
                    )
            processed_files.append({
                "path": file_input.path,
                "content": content,
            })

        indexer = get_code_indexer()

        if request.background:
            # Submit background job and return immediately
            job_manager = get_job_manager()

            async def _index_job(
                proj_root: str, files: list[dict], clear: bool
            ) -> dict:
                """Background job wrapper for indexing."""
                log.info(f"Background job: index_code starting for {proj_root}")
                result = await indexer.index_files_content_async(
                    project_root=proj_root,
                    files=files,
                    clear_existing=clear,
                )
                log.info(
                    f"Background job: index_code complete: "
                    f"{result.get('files_indexed', 0)} files"
                )
                return result

            job_id = await job_manager.submit(
                "index_code",
                _index_job,
                request.project_root,
                processed_files,
                request.clear_existing,
            )
            log.info(f"index_code submitted as background job {job_id}")
            return {
                "job_id": job_id,
                "status": "submitted",
                "message": f"Use job_status('{job_id}') to check progress",
            }

        # Synchronous execution (not recommended for large codebases)
        log.debug("index_code running synchronously")
        result = await indexer.index_files_content_async(
            project_root=request.project_root,
            files=processed_files,
            clear_existing=request.clear_existing,
        )

        log.info(f"Indexed {result['files_indexed']} files, {result['chunks_created']} chunks")
        return result

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Code indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update")
async def update_code(request: UpdateCodeRequest) -> dict:
    """Incrementally update code index with file changes.

    Used by the file watcher for real-time updates.
    Supports add, modify, and delete operations.
    """
    require_project_id(request.project_root, "update_code")
    try:
        # Process updates, decompressing content if needed
        processed_updates = []
        for update in request.updates:
            content = update.content
            if update.compressed and content:
                try:
                    content = decompress_payload(content)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to decompress {update.path}: {e}"
                    )
            processed_updates.append({
                "path": update.path,
                "action": update.action,
                "content": content,
            })

        indexer = get_code_indexer()
        result = indexer.update_files_content(
            project_root=request.project_root,
            updates=processed_updates,
        )

        log.info(
            f"Updated {result['files_updated']} files: "
            f"+{result['chunks_created']}/-{result['chunks_deleted']} chunks"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Code update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_code(request: SearchCodeRequest) -> dict:
    """Semantic search over indexed code."""
    require_project_id(request.project_root, "search_code")
    try:
        indexer = get_code_indexer()
        results = indexer.search(
            query=request.query,
            limit=request.limit,
            project_root=request.project_root,
        )
        return {
            "results": results,
            "count": len(results),
        }

    except Exception as e:
        log.error(f"Code search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def code_stats(project_root: str | None = None) -> dict:
    """Get code index statistics."""
    require_project_id(project_root, "code_stats")
    try:
        indexer = get_code_indexer()
        stats = indexer.get_stats()
        return stats

    except Exception as e:
        log.error(f"Failed to get code stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/related-memories")
async def code_related_memories(request: CodeRelatedMemoriesRequest) -> dict:
    """Find memories related to a code chunk via shared entities."""
    try:
        store = get_memory_store()
        # Get related memories via entity graph
        related = store.db.get_code_related_memories(
            chunk_uuid=request.chunk_uuid,
            limit=request.limit,
        )
        return {
            "chunk_uuid": request.chunk_uuid,
            "related_memories": related,
            "count": len(related),
        }

    except Exception as e:
        log.error(f"Failed to get code-related memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/related-code")
async def memory_related_code(request: MemoryRelatedCodeRequest) -> dict:
    """Find code chunks related to a memory via shared entities."""
    try:
        store = get_memory_store()
        # Get related code via entity graph
        related = store.db.get_memory_related_code(
            memory_uuid=request.memory_uuid,
            limit=request.limit,
        )
        return {
            "memory_uuid": request.memory_uuid,
            "related_code": related,
            "count": len(related),
        }

    except Exception as e:
        log.error(f"Failed to get memory-related code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/status")
async def update_code_index_status(request: CodeIndexStatusRequest) -> dict:
    """Update code index status for statusline display.

    Called by the MCP watcher to report indexing/watching state.
    The status is included in the /stats endpoint response.
    """
    try:
        job_manager = get_job_manager()
        job_manager.update_code_index_status(
            status=request.status,
            watchers=request.watchers,
            projects_watching=request.projects_watching,
            indexing_in_progress=request.indexing_in_progress,
            files_done=request.files_done,
            files_total=request.files_total,
            current_file=request.current_file,
            total_files=request.total_files,
            total_chunks=request.total_chunks,
        )
        log.debug(f"Code index status updated: {request.status or 'inferred'}")
        return {
            "status": "updated",
            "code_index": job_manager.get_code_index_status(),
        }

    except Exception as e:
        log.error(f"Failed to update code index status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_code_index_status() -> dict:
    """Get current code index status for statusline display."""
    try:
        job_manager = get_job_manager()
        return {
            "code_index": job_manager.get_code_index_status(),
        }

    except Exception as e:
        log.error(f"Failed to get code index status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
