"""Memory operations API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simplemem_lite.backend.config import get_config
from simplemem_lite.backend.scoring import ScoringWeights, apply_temporal_scoring, rerank_results
from simplemem_lite.backend.services import get_code_indexer, get_job_manager, get_memory_store
from simplemem_lite.backend.toon import toonify
from simplemem_lite.log_config import get_logger
from simplemem_lite.memory import MemoryItem, detect_contradictions

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
    output_format: str | None = Field(
        default=None,
        description="Response format: 'toon' (default) or 'json'. Env var: SIMPLEMEM_OUTPUT_FORMAT.",
    )
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
    output_format: str | None = Field(
        default=None,
        description="Response format: 'toon' (sources only) or 'json' (full response with answer). Env var: SIMPLEMEM_OUTPUT_FORMAT.",
    )


class ReasonMemoriesRequest(BaseModel):
    """Request model for multi-hop reasoning."""

    query: str = Field(..., description="Query for reasoning")
    max_hops: int = Field(default=2, ge=1, le=3)
    min_score: float = Field(default=0.1, ge=0.0, le=1.0)
    project_id: str | None = Field(default=None, description="Project identifier")
    output_format: str | None = Field(
        default=None,
        description="Response format: 'toon' (default) or 'json'. Env var: SIMPLEMEM_OUTPUT_FORMAT.",
    )


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
@toonify(headers=["uuid", "type", "score", "content"])
async def search_memories(request: SearchMemoriesRequest) -> dict:
    """Search memories with hybrid vector + graph search.

    When use_temporal_scoring=True (default), results are re-ranked using
    multi-factor scoring that combines:
    - Vector similarity (normalized per-request)
    - Temporal decay (newer memories ranked higher)
    - Type importance (decisions > lessons > facts > summaries)

    Set use_temporal_scoring=False for pure vector similarity ranking.

    When output_format="toon", returns tab-separated format for token efficiency.
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

        # Return results - decorator handles TOON conversion if requested
        return {"results": result_dicts}

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
@toonify(headers=["uuid", "type", "score", "hops", "cross_session"], result_key="sources")
async def ask_memories(request: AskMemoriesRequest) -> dict:
    """Ask a question and get LLM-synthesized answer from memories.

    Returns full structured response with answer + sources when output_format='json' (default).
    Returns only source memories as TOON when output_format='toon' (for token efficiency).
    """
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
@toonify(headers=["uuid", "type", "score", "hops", "cross_session"], result_key="sources")
async def reason_memories(request: ReasonMemoriesRequest) -> dict:
    """LLM-synthesized reasoning over memory graph.

    Returns full structured response with reasoning + conclusions when output_format='json' (default).
    Returns only conclusion memories as TOON when output_format='toon' (for token efficiency).
    """
    require_project_id(request.project_id, "reason_memories")
    try:
        store = get_memory_store()
        result = await store.reason_with_synthesis(
            query=request.query,
            max_hops=request.max_hops,
            min_score=request.min_score,
            project_id=request.project_id,
        )
        return result

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


# ============================================================================
# LLM-Enhanced Reasoning Endpoints (Phase 1-3)
# ============================================================================


class SearchDeepRequest(BaseModel):
    """Request model for LLM-reranked deep search."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    rerank_pool: int = Field(default=20, ge=1, le=50, description="Pool size for reranking")
    project_id: str | None = Field(default=None, description="Project identifier")
    output_format: str | None = Field(
        default=None,
        description="Response format: 'toon' (default) or 'json'. Env var: SIMPLEMEM_OUTPUT_FORMAT.",
    )


@router.post("/search-deep")
@toonify(headers=["uuid", "type", "score", "content"])
async def search_memories_deep(request: SearchDeepRequest) -> dict:
    """LLM-reranked semantic search with conflict detection.

    Performs vector search, then uses an LLM to rerank results by
    semantic relevance and detect potential conflicts between memories.
    Use this for higher precision when quality matters more than speed.
    """
    require_project_id(request.project_id, "search_memories_deep")
    try:
        store = get_memory_store()

        # Get expanded pool for reranking
        results = store.search(
            query=request.query,
            limit=request.rerank_pool,
            use_graph=True,
            project_id=request.project_id,
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
            query=request.query,
            results=result_dicts,
            top_k=request.limit,
            rerank_pool=request.rerank_pool,
        )

        # Map conflict indices to UUIDs
        conflicts_with_uuids = []
        for conflict in reranked.get("conflicts", []):
            if len(conflict) >= 3:
                idx1, idx2, reason = conflict[0], conflict[1], conflict[2]
                if idx1 < len(result_dicts) and idx2 < len(result_dicts):
                    conflicts_with_uuids.append([
                        result_dicts[idx1]["uuid"],
                        result_dicts[idx2]["uuid"],
                        reason,
                    ])

        return {
            "results": reranked["results"],
            "conflicts": conflicts_with_uuids,
            "rerank_applied": reranked.get("rerank_applied", False),
        }

    except Exception as e:
        log.error(f"Deep search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CheckContradictionsRequest(BaseModel):
    """Request model for contradiction detection."""

    content: str = Field(..., description="Content to check for contradictions")
    memory_uuid: str | None = Field(
        default=None,
        description="UUID of the new memory (required for apply_supersession)",
    )
    apply_supersession: bool = Field(
        default=False,
        description="Create SUPERSEDES edges from new memory to contradicted ones",
    )
    project_id: str | None = Field(default=None, description="Project identifier")
    output_format: str | None = Field(
        default=None,
        description="Response format: 'toon' (default) or 'json'. Env var: SIMPLEMEM_OUTPUT_FORMAT.",
    )


@router.post("/check-contradictions")
@toonify(headers=["uuid", "content", "reason", "confidence"], result_key="contradictions")
async def check_contradictions(request: CheckContradictionsRequest) -> dict:
    """Check if content contradicts existing memories.

    Searches for similar memories and uses LLM to detect contradictions.
    Optionally creates SUPERSEDES relationships to mark old memories
    as superseded by the new one.
    """
    require_project_id(request.project_id, "check_contradictions")
    try:
        store = get_memory_store()
        config = get_config()

        # Find similar memories
        similar = store.search(
            query=request.content,
            limit=10,
            use_graph=False,  # Pure vector for contradiction check
            project_id=request.project_id,
        )

        # Convert to dicts
        similar_dicts = [
            {
                "uuid": m.uuid,
                "content": m.content,
                "type": m.type,
                "score": m.score,
            }
            for m in similar
        ]

        # Detect contradictions via LLM
        contradictions = await detect_contradictions(
            new_content=request.content,
            similar_memories=similar_dicts,
            config=config,
        )

        supersessions_created = 0

        # Apply supersession if requested and we have a memory_uuid
        if request.apply_supersession and request.memory_uuid and contradictions:
            for c in contradictions:
                try:
                    # Map confidence to numeric value
                    confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                    confidence = confidence_map.get(c.get("confidence", "medium"), 0.7)

                    success = store.db.add_supersession(
                        newer_uuid=request.memory_uuid,
                        older_uuid=c["uuid"],
                        confidence=confidence,
                        supersession_type="contradiction",
                        reason=c.get("reason", ""),
                    )
                    if success:
                        supersessions_created += 1
                except Exception as e:
                    log.warning(f"Failed to create supersession: {e}")

        return {
            "contradictions": contradictions,
            "supersessions_created": supersessions_created,
            # Debug info
            "_debug": {
                "similar_memories_found": len(similar),
                "similar_memories_scores": [m.score for m in similar[:5]],
                "passed_to_llm": len([m for m in similar if m.score > 0.3]),
            },
        }

    except Exception as e:
        log.error(f"Contradiction check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync-health")
async def get_sync_health(project_id: str | None = None) -> dict:
    """Check synchronization health between graph and vector stores.

    Detects memories that exist in the graph but are missing from LanceDB.
    Use this to diagnose sync issues before running repair_sync.
    """
    try:
        store = get_memory_store()
        result = store.db.get_sync_health(project_id=project_id)
        return result

    except Exception as e:
        log.error(f"Sync health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RepairSyncRequest(BaseModel):
    """Request model for sync repair."""

    project_id: str | None = Field(default=None, description="Project identifier")
    dry_run: bool = Field(
        default=True,
        description="Preview changes without applying (default: True)",
    )


@router.post("/repair-sync")
async def repair_sync(request: RepairSyncRequest) -> dict:
    """Repair synchronization issues between graph and vector stores.

    Finds memories missing from LanceDB and regenerates their embeddings.
    Always run with dry_run=True first to preview changes.
    """
    try:
        store = get_memory_store()
        result = store.db.repair_sync(
            project_id=request.project_id,
            dry_run=request.dry_run,
        )
        return result

    except Exception as e:
        log.error(f"Sync repair failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
