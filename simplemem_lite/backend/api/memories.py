"""Memory operations API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simplemem_lite.backend.services import get_memory_store
from simplemem_lite.log_config import get_logger
from simplemem_lite.memory import MemoryItem

router = APIRouter()
log = get_logger("backend.api.memories")


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


class AskMemoriesRequest(BaseModel):
    """Request model for ask_memories with LLM synthesis."""

    query: str = Field(..., description="Question to answer")
    max_memories: int = Field(default=8, ge=1, le=20)
    max_hops: int = Field(default=2, ge=1, le=3)


class ReasonMemoriesRequest(BaseModel):
    """Request model for multi-hop reasoning."""

    query: str = Field(..., description="Query for reasoning")
    max_hops: int = Field(default=2, ge=1, le=3)
    min_score: float = Field(default=0.1, ge=0.0, le=1.0)


@router.post("/store", response_model=StoreMemoryResponse)
async def store_memory(request: StoreMemoryRequest) -> StoreMemoryResponse:
    """Store a new memory with optional relations."""
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
    """Search memories with hybrid vector + graph search."""
    try:
        store = get_memory_store()
        results = store.search(
            query=request.query,
            limit=request.limit,
            use_graph=request.use_graph,
            type_filter=request.type_filter,
            project_id=request.project_id,
        )

        return {
            "results": [
                {
                    "uuid": m.uuid,
                    "content": m.content,
                    "type": m.type,
                    "score": m.score,
                    "session_id": m.session_id,
                    "created_at": m.created_at,
                }
                for m in results
            ],
            "count": len(results),
        }

    except Exception as e:
        log.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def ask_memories(request: AskMemoriesRequest) -> dict:
    """Ask a question and get LLM-synthesized answer from memories."""
    try:
        store = get_memory_store()
        result = await store.ask_memories(
            query=request.query,
            max_memories=request.max_memories,
            max_hops=request.max_hops,
        )
        return result

    except Exception as e:
        log.error(f"Ask memories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reason")
async def reason_memories(request: ReasonMemoriesRequest) -> dict:
    """Multi-hop reasoning over memory graph."""
    try:
        store = get_memory_store()
        results = store.reason(
            query=request.query,
            max_hops=request.max_hops,
            min_score=request.min_score,
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
    """Get memory store statistics."""
    try:
        store = get_memory_store()
        return store.get_stats()

    except Exception as e:
        log.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
