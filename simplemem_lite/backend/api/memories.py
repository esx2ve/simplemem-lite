"""Memory operations API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simplemem_lite.backend.config import get_config
from simplemem_lite.backend.scoring import ScoringWeights, apply_temporal_scoring
from simplemem_lite.backend.services import get_code_indexer, get_job_manager, get_memory_store
from simplemem_lite.log_config import get_logger
from simplemem_lite.memory import MemoryItem

router = APIRouter()
log = get_logger("backend.api.memories")


def require_project_id(project_id: str | None, endpoint: str) -> None:
    """Validate that project_id is provided when required by config.

    Args:
        project_id: The project_id from the request
        endpoint: Name of the endpoint for error message

    Raises:
        HTTPException: 400 if project_id is required but not provided
    """
    config = get_config()
    if config.require_project_id and not project_id:
        raise HTTPException(
            status_code=400,
            detail=f"project_id is required for {endpoint}. "
            "Cloud API requires project isolation for all operations.",
        )


class RelationInput(BaseModel):
    """Input model for a relation to another memory."""

    target_id: str = Field(..., description="Target memory UUID")
    type: str = Field(default="relates", description="Relation type")


class StoreMemoryRequest(BaseModel):
    """Request model for storing a memory."""

    text: str = Field(..., description="Memory content to store")
    type: str = Field(default="fact", description="Memory type")
    source: str = Field(default="user", description="Memory source")
    project_id: str | None = Field(default=None, description="Project identifier")
    relations: list[RelationInput] | None = Field(default=None, description="Relations to create")


class StoreMemoryResponse(BaseModel):
    """Response model for stored memory."""

    uuid: str
    message: str = "Memory stored successfully"


class SearchMemoriesRequest(BaseModel):
    """Request model for searching memories."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100)
    use_graph: bool = Field(default=True)
    type_filter: str | None = Field(default=None)
    project_id: str | None = Field(default=None)
    # Temporal scoring options
    use_temporal_scoring: bool = Field(
        default=True,
        description="Apply temporal decay and importance scoring to re-rank results",
    )
    scoring_details: bool = Field(
        default=False,
        description="Include detailed scoring breakdown in results",
    )
    vector_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity (default: 0.6)",
    )
    temporal_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for temporal decay (default: 0.25)",
    )
    importance_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for type importance (default: 0.15)",
    )


class AskMemoriesRequest(BaseModel):
    """Request model for ask_memories with LLM synthesis."""

    query: str = Field(..., description="Question to answer")
    max_memories: int = Field(default=8, ge=1, le=20)
    max_hops: int = Field(default=2, ge=1, le=3)
    project_id: str | None = Field(default=None, description="Project identifier")


class ReasonMemoriesRequest(BaseModel):
    """Request model for multi-hop reasoning."""

    query: str = Field(..., description="Query for reasoning")
    max_hops: int = Field(default=2, ge=1, le=3)
    min_score: float = Field(default=0.1, ge=0.0, le=1.0)
    project_id: str | None = Field(default=None, description="Project identifier")


@router.post("/store", response_model=StoreMemoryResponse)
async def store_memory(request: StoreMemoryRequest) -> StoreMemoryResponse:
    """Store a new memory with optional relations."""
    require_project_id(request.project_id, "store_memory")
    try:
        store = get_memory_store()

        # Build metadata
        metadata = {
            "type": request.type,
            "source": request.source,
        }
        if request.project_id:
            metadata["project_id"] = request.project_id

        # Build relations
        relations = []
        if request.relations:
            relations = [{"target_id": r.target_id, "type": r.type} for r in request.relations]

        item = MemoryItem(
            content=request.text,
            metadata=metadata,
            relations=relations,
        )

        uuid = store.store(item)
        log.info(f"Stored memory: {uuid[:8]}...")
        return StoreMemoryResponse(uuid=uuid)

    except Exception as e:
        log.error(f"Failed to store memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_memories(request: SearchMemoriesRequest) -> dict:
    """Search memories with hybrid vector + graph search.

    When use_temporal_scoring=True (default), results are re-ranked using
    multi-factor scoring that combines:
    - Vector similarity (normalized per-request)
    - Temporal decay (newer memories ranked higher)
    - Type importance (decisions > lessons > facts > summaries)

    Set use_temporal_scoring=False for pure vector similarity ranking.
    """
    require_project_id(request.project_id, "search_memories")
    try:
        store = get_memory_store()
        results = store.search(
            query=request.query,
            limit=request.limit,
            use_graph=request.use_graph,
            type_filter=request.type_filter,
            project_id=request.project_id,
        )

        # Convert Memory objects to dicts for scoring
        result_dicts = [
            {
                "uuid": m.uuid,
                "content": m.content,
                "type": m.type,
                "score": m.score,
                "session_id": m.session_id,
                "created_at": m.created_at,
            }
            for m in results
        ]

        # Apply temporal scoring if enabled
        if request.use_temporal_scoring and result_dicts:
            # Validate weights sum to ~1.0
            total_weight = request.vector_weight + request.temporal_weight + request.importance_weight
            if abs(total_weight - 1.0) > 0.01:
                raise HTTPException(
                    status_code=400,
                    detail=f"Scoring weights must sum to 1.0, got {total_weight:.2f}",
                )

            weights = ScoringWeights(
                vector=request.vector_weight,
                temporal=request.temporal_weight,
                importance=request.importance_weight,
            )
            result_dicts = apply_temporal_scoring(
                result_dicts,
                weights=weights,
                return_details=request.scoring_details,
            )

        return {
            "results": result_dicts,
            "count": len(result_dicts),
            "scoring": {
                "temporal_enabled": request.use_temporal_scoring,
                "weights": {
                    "vector": request.vector_weight,
                    "temporal": request.temporal_weight,
                    "importance": request.importance_weight,
                } if request.use_temporal_scoring else None,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def ask_memories(request: AskMemoriesRequest) -> dict:
    """Ask a question and get LLM-synthesized answer from memories."""
    require_project_id(request.project_id, "ask_memories")
    try:
        store = get_memory_store()
        result = await store.ask_memories(
            query=request.query,
            max_memories=request.max_memories,
            max_hops=request.max_hops,
            project_id=request.project_id,
        )
        return result

    except Exception as e:
        log.error(f"Ask memories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reason")
async def reason_memories(request: ReasonMemoriesRequest) -> dict:
    """Multi-hop reasoning over memory graph."""
    require_project_id(request.project_id, "reason_memories")
    try:
        store = get_memory_store()
        results = store.reason(
            query=request.query,
            max_hops=request.max_hops,
            min_score=request.min_score,
            project_id=request.project_id,
        )
        return {
            "conclusions": results,
            "count": len(results),
        }

    except Exception as e:
        log.error(f"Reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RelateMemoriesRequest(BaseModel):
    """Request model for creating a relationship between memories."""

    from_id: str = Field(..., description="Source memory UUID")
    to_id: str = Field(..., description="Target memory UUID")
    relation_type: str = Field(default="relates", description="Type of relationship")


@router.post("/relate")
async def relate_memories(request: RelateMemoriesRequest) -> dict:
    """Create a relationship between two memories."""
    try:
        store = get_memory_store()
        store.db.add_relationship(
            from_uuid=request.from_id,
            to_uuid=request.to_id,
            relation_type=request.relation_type,
        )
        return {
            "success": True,
            "from_id": request.from_id,
            "to_id": request.to_id,
            "relation_type": request.relation_type,
        }

    except Exception as e:
        log.error(f"Failed to create relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats() -> dict:
    """Get comprehensive statistics for statusline display.

    Returns memory stats, code index stats, and code_index status
    for Claude Code statusline integration.
    """
    try:
        store = get_memory_store()
        mem_stats = store.get_stats()

        # Add code index stats for statusline
        try:
            code_indexer = get_code_indexer()
            code_stats = code_indexer.get_stats()
            mem_stats["code_files"] = code_stats.get("unique_files", 0)
            mem_stats["code_chunks"] = code_stats.get("chunk_count", 0)
        except Exception:
            pass

        # Add code_index status for statusline
        try:
            job_manager = get_job_manager()
            code_index_status = job_manager.get_code_index_status()
            mem_stats["watchers"] = code_index_status.get("watchers", 0)
            mem_stats["code_index"] = code_index_status
        except Exception:
            pass

        return mem_stats

    except Exception as e:
        log.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset")
async def reset_all(confirm: bool = False) -> dict:
    """Reset all memories, vectors, and relationships.

    WARNING: This is a destructive operation that cannot be undone.

    Args:
        confirm: Must be True to proceed with reset
    """
    if not confirm:
        return {"error": "Must set confirm=True to reset all data"}

    try:
        store = get_memory_store()
        log.warning("API: reset_all called with confirm=True")
        result = store.reset_all()
        log.warning(f"API: reset_all complete: {result}")
        return result
    except Exception as e:
        log.error(f"Failed to reset: {e}")
        raise HTTPException(status_code=500, detail=str(e))
