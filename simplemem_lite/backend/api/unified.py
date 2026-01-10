"""Unified SimpleMem API - 3 tools to rule them all.

This module implements the consolidated API surface:
- remember: Save anything (memories, facts, decisions, lessons)
- recall: Find anything (search, get by ID, ask with synthesis)
- index: Make code/traces searchable

These tools internally route to existing backend functions, providing
a minimal surface area while preserving all functionality.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simplemem_lite.backend.config import get_config
from simplemem_lite.backend.scoring import rerank_results
from simplemem_lite.backend.services import (
    get_code_indexer,
    get_hierarchical_indexer,
    get_job_manager,
    get_memory_store,
)
from simplemem_lite.backend.toon import toonify
from simplemem_lite.compression import decompress_payload
from simplemem_lite.log_config import get_logger
from simplemem_lite.memory import MemoryItem

router = APIRouter()
log = get_logger("backend.api.unified")


def require_project_id(project_id: str | None, endpoint: str) -> None:
    """Validate that project_id is provided when required."""
    config = get_config()
    if config.require_project_id and not project_id:
        raise HTTPException(
            status_code=400,
            detail=f"project_id is required for {endpoint}. "
            "Cloud API requires project isolation for all operations.",
        )


# =============================================================================
# REMEMBER - Save anything to memory
# =============================================================================


class RememberRequest(BaseModel):
    """Request model for remembering/storing information."""

    content: str = Field(..., description="The content to store. Be specific and actionable.")
    project: str | None = Field(default=None, description="Project ID for isolation")
    type: Literal["fact", "lesson_learned", "decision", "pattern"] = Field(
        default="fact",
        description="Memory type: fact (default), lesson_learned, decision, pattern",
    )
    relations: list[str] | None = Field(
        default=None,
        description="List of memory UUIDs to relate this memory to",
    )


class RememberResponse(BaseModel):
    """Response model for remember operation."""

    uuid: str = Field(..., description="UUID of the stored memory")
    relations_created: int = Field(default=0, description="Number of relations created")


@router.post("/remember", response_model=RememberResponse)
async def remember(request: RememberRequest) -> RememberResponse:
    """Store a memory with optional relations.

    This is the unified entry point for all memory storage operations.
    It replaces: store_memory, relate_memories

    Examples:
        # Store a lesson learned
        remember(content="Fix: Check Docker when DB fails", type="lesson_learned")

        # Store a decision with relations
        remember(
            content="Decision: Use Redis for caching | Reason: Speed",
            type="decision",
            relations=["uuid-of-related-memory"]
        )
    """
    require_project_id(request.project, "remember")

    try:
        store = get_memory_store()

        # Build metadata
        metadata = {
            "type": request.type,
            "source": "user",
        }
        if request.project:
            metadata["project_id"] = request.project

        # Build relations
        relations = []
        if request.relations:
            relations = [{"target_id": r, "type": "relates"} for r in request.relations]

        item = MemoryItem(
            content=request.content,
            metadata=metadata,
            relations=relations,
        )

        uuid = store.store(item)
        relations_created = len(relations)

        log.info(f"Remembered: {uuid[:8]}... (type={request.type}, relations={relations_created})")
        return RememberResponse(uuid=uuid, relations_created=relations_created)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Remember failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RECALL - Find anything from memory
# =============================================================================


class RecallRequest(BaseModel):
    """Request model for recalling/finding information."""

    query: str | None = Field(default=None, description="Search query (required if no id)")
    id: str | None = Field(default=None, description="Exact memory UUID to fetch")
    project: str | None = Field(default=None, description="Project ID for isolation")
    mode: Literal["fast", "deep", "ask"] = Field(
        default="fast",
        description="Search mode: fast (vector search), deep (reranked), ask (LLM synthesis)",
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    output_format: str | None = Field(
        default=None,
        description="Response format: 'toon' (default) or 'json'",
    )


@router.post("/recall")
@toonify(headers=["uuid", "type", "score", "content"])
async def recall(request: RecallRequest) -> dict:
    """Find memories by query or exact ID.

    This is the unified entry point for all memory retrieval operations.
    It replaces: search_memories, search_memories_deep, ask_memories, get by ID

    Modes:
        - fast: Vector similarity search (default, quickest)
        - deep: LLM-reranked results with conflict detection
        - ask: LLM-synthesized answer with citations

    Examples:
        # Quick search
        recall(query="database timeout fix", project="myproject")

        # Get specific memory by ID
        recall(id="abc-123-uuid")

        # Deep search with reranking
        recall(query="authentication patterns", mode="deep")

        # Get synthesized answer
        recall(query="How did we fix the memory leak?", mode="ask")
    """
    require_project_id(request.project, "recall")

    # Validate: need either query or id
    if not request.query and not request.id:
        raise HTTPException(
            status_code=400,
            detail="Either 'query' or 'id' is required",
        )

    try:
        store = get_memory_store()

        # Exact fetch by ID - direct graph lookup
        if request.id:
            # Direct lookup by UUID (efficient graph query)
            memory = store.get(request.id)
            if not memory:
                raise HTTPException(
                    status_code=404,
                    detail=f"Memory {request.id} not found",
                )

            # Validate project ownership if project specified
            if request.project:
                valid_uuids = store.db.get_memories_in_project(
                    request.project, [request.id]
                )
                if request.id not in valid_uuids:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Memory {request.id} not found in project {request.project}",
                    )

            return {
                "results": [
                    {
                        "uuid": memory.uuid,
                        "content": memory.content,
                        "type": memory.type,
                        "session_id": memory.session_id,
                        "created_at": memory.created_at,
                    }
                ]
            }

        # Search by query based on mode
        # At this point, query is guaranteed to be non-None due to validation above
        if request.query is None:
            raise HTTPException(
                status_code=400,
                detail="Query is required for search operations",
            )
        query: str = request.query

        if request.mode == "fast":
            # Standard vector search with graph enhancement
            results = store.search(
                query=query,
                limit=request.limit,
                use_graph=True,
                project_id=request.project,
            )
            # Convert Memory objects to dicts
            result_dicts = [
                {
                    "uuid": m.uuid,
                    "content": m.content,
                    "type": m.type,
                    "score": m.score,
                }
                for m in results
            ]
            return {"results": result_dicts}

        elif request.mode == "deep":
            # LLM-reranked search with conflict detection
            rerank_pool = request.limit * 2
            results = store.search(
                query=query,
                limit=rerank_pool,
                use_graph=True,
                project_id=request.project,
            )
            # Convert to dicts for reranking
            result_dicts = [
                {
                    "uuid": m.uuid,
                    "content": m.content,
                    "type": m.type,
                    "score": m.score,
                    "created_at": m.created_at,
                }
                for m in results
            ]
            # Apply LLM reranking
            reranked = await rerank_results(
                query=query,
                results=result_dicts,
                top_k=request.limit,
                rerank_pool=rerank_pool,
            )
            return {
                "results": reranked.get("reranked", result_dicts[:request.limit]),
                "conflicts": reranked.get("conflicts", []),
                "rerank_applied": reranked.get("rerank_applied", False),
            }

        elif request.mode == "ask":
            # LLM-synthesized answer
            result = await store.ask_memories(
                query=query,
                max_memories=request.limit,
                max_hops=2,
                project_id=request.project,
            )
            # For ask mode, return the answer structure directly
            return {
                "answer": result.get("answer", ""),
                "memories_used": result.get("memories_used", 0),
                "cross_session_insights": result.get("cross_session_insights", []),
                "confidence": result.get("confidence", "low"),
                "sources": result.get("sources", []),
            }

        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Recall failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# INDEX - Make code or traces searchable
# =============================================================================


class FileInput(BaseModel):
    """Input model for a file to be indexed."""

    path: str = Field(..., description="File path relative to project root")
    content: str | dict = Field(..., description="File content")
    compressed: bool = Field(default=False, description="Whether content is gzip+base64")


class TraceInput(BaseModel):
    """Input model for a trace to be indexed."""

    session_id: str = Field(..., description="Session UUID")
    content: str | dict | list = Field(..., description="Trace content")
    compressed: bool = Field(default=False, description="Whether content is gzip+base64")


class IndexRequest(BaseModel):
    """Request model for indexing code or traces."""

    project: str = Field(..., description="Project ID for isolation")
    # Code indexing
    files: list[FileInput] | None = Field(default=None, description="Files to index")
    clear_existing: bool = Field(default=True, description="Clear existing code index")
    # Trace indexing
    traces: list[TraceInput] | None = Field(default=None, description="Traces to index")
    # Common
    wait: bool = Field(default=False, description="Wait for completion (default: background)")


class IndexResponse(BaseModel):
    """Response model for index operation."""

    job_id: str | None = Field(default=None, description="Background job ID")
    status: str = Field(default="submitted", description="Operation status")
    message: str | None = Field(default=None, description="Status message")
    # Sync results (when wait=True)
    files_indexed: int | None = Field(default=None)
    chunks_created: int | None = Field(default=None)
    traces_processed: int | None = Field(default=None)


@router.post("/index", response_model=IndexResponse)
async def index(request: IndexRequest) -> IndexResponse:
    """Index code files or session traces for semantic search.

    This is the unified entry point for all indexing operations.
    It replaces: index_directory, process_trace, process_trace_batch

    Provide either 'files' for code indexing or 'traces' for trace processing.

    Examples:
        # Index code files
        index(
            project="myproject",
            files=[{"path": "src/app.py", "content": "..."}],
        )

        # Index session traces
        index(
            project="myproject",
            traces=[{"session_id": "abc-123", "content": {...}}],
        )

        # Wait for completion
        index(project="myproject", files=[...], wait=True)
    """
    require_project_id(request.project, "index")

    # Validate: need either files or traces
    if not request.files and not request.traces:
        raise HTTPException(
            status_code=400,
            detail="Either 'files' or 'traces' is required",
        )

    if request.files and request.traces:
        raise HTTPException(
            status_code=400,
            detail="Cannot index both files and traces in same request",
        )

    try:
        # Code indexing
        if request.files:
            return await _index_code(request)

        # Trace indexing
        if request.traces:
            return await _index_traces(request)

        # Should never reach here due to validation above
        raise HTTPException(status_code=400, detail="Either 'files' or 'traces' is required")

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Index failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _index_code(request: IndexRequest) -> IndexResponse:
    """Handle code file indexing."""
    # Decompress files if needed
    files = request.files
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    processed_files = []
    for file_input in files:
        content = file_input.content
        if file_input.compressed and isinstance(content, str):
            try:
                content = decompress_payload(content)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decompress {file_input.path}: {e}",
                )
        processed_files.append({"path": file_input.path, "content": content})

    indexer = get_code_indexer()

    if not request.wait:
        # Background processing
        job_manager = get_job_manager()

        async def _index_job(project_id, files, clear, progress_callback=None):
            return await indexer.index_files_content_async(
                project_id=project_id,
                files=files,
                clear_existing=clear,
                progress_callback=progress_callback,
            )

        job_id = await job_manager.submit(
            "index_code",
            _index_job,
            request.project,
            processed_files,
            request.clear_existing,
        )
        return IndexResponse(
            job_id=job_id,
            status="submitted",
            message=f"Indexing {len(processed_files)} files in background",
        )

    # Synchronous processing
    result = await indexer.index_files_content_async(
        project_id=request.project,
        files=processed_files,
        clear_existing=request.clear_existing,
    )
    return IndexResponse(
        status="completed",
        files_indexed=result.get("files_indexed", 0),
        chunks_created=result.get("chunks_created", 0),
    )


async def _index_traces(request: IndexRequest) -> IndexResponse:
    """Handle trace indexing."""
    traces = request.traces
    if not traces:
        raise HTTPException(status_code=400, detail="No traces provided")

    indexer = get_hierarchical_indexer()
    job_manager = get_job_manager()

    traces_submitted = 0
    job_ids = []

    for trace_input in traces:
        # Decompress if needed
        content = trace_input.content
        if trace_input.compressed and isinstance(content, str):
            try:
                content = decompress_payload(content)
            except Exception as e:
                log.warning(f"Failed to decompress trace {trace_input.session_id}: {e}")
                continue

        if request.wait:
            # Synchronous processing
            result = await indexer.index_session_content(
                session_id=trace_input.session_id,
                trace_content=content,
                project_id=request.project,
            )
            if result:
                traces_submitted += 1
        else:
            # Background processing
            async def _trace_job(session_id, trace_content, project_id, progress_callback=None):
                return await indexer.index_session_content(
                    session_id=session_id,
                    trace_content=trace_content,
                    project_id=project_id,
                    progress_callback=progress_callback,
                )

            job_id = await job_manager.submit(
                "index_trace",
                _trace_job,
                trace_input.session_id,
                content,
                request.project,
            )
            job_ids.append(job_id)
            traces_submitted += 1

    if request.wait:
        return IndexResponse(
            status="completed",
            traces_processed=traces_submitted,
        )

    return IndexResponse(
        job_id=job_ids[0] if len(job_ids) == 1 else None,
        status="submitted",
        message=f"Processing {traces_submitted} traces in background"
        + (f" (job_ids: {job_ids})" if len(job_ids) > 1 else ""),
    )
