"""Code indexing and search API endpoints.

The MCP thin layer reads code files locally, compresses them,
and sends the content here for indexing.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simplemem_lite.backend.services import get_code_indexer, get_memory_store
from simplemem_lite.compression import decompress_payload
from simplemem_lite.log_config import get_logger

router = APIRouter()
log = get_logger("backend.api.code")


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


@router.post("/index")
async def index_code(request: IndexDirectoryRequest) -> dict:
    """Index code files for semantic search.

    The MCP layer reads files locally and sends content here.
    """
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
        result = indexer.index_files_content(
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


@router.post("/search")
async def search_code(request: SearchCodeRequest) -> dict:
    """Semantic search over indexed code."""
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
