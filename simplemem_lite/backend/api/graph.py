"""Graph database API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simplemem_lite.backend.services import get_memory_store
from simplemem_lite.log_config import get_logger

router = APIRouter()
log = get_logger("backend.api.graph")


class CypherQueryRequest(BaseModel):
    """Request model for executing Cypher queries."""

    query: str = Field(..., description="Cypher query to execute")
    params: dict | None = Field(default=None, description="Query parameters")
    max_results: int = Field(default=100, ge=1, le=1000)


@router.get("/schema")
async def get_graph_schema() -> dict:
    """Get the complete graph schema for query generation."""
    try:
        store = get_memory_store()
        schema = store.db.get_schema()
        return schema

    except Exception as e:
        log.error(f"Failed to get graph schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def run_cypher_query(request: CypherQueryRequest) -> dict:
    """Execute a validated Cypher query against the graph.

    Security: Read-only by default, LIMIT injection, result truncation.
    """
    try:
        store = get_memory_store()

        # Clamp max_results
        max_results = min(max(1, request.max_results), 1000)

        result = store.db.execute_validated_cypher(
            query=request.query,
            params=request.params,
            max_results=max_results,
            allow_mutations=False,  # Always read-only via API
        )
        return result

    except Exception as e:
        log.error(f"Cypher query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
