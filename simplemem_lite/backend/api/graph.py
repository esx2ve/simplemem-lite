"""Graph database API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simplemem_lite.backend.config import get_config
from simplemem_lite.backend.services import get_memory_store
from simplemem_lite.log_config import get_logger

router = APIRouter()
log = get_logger("backend.api.graph")


# Whitelisted query templates for PROD mode
# These are safe, parameterized queries that prevent injection
ALLOWED_QUERY_TEMPLATES = {
    "get_memory": "MATCH (m:Memory {uuid: $uuid}) RETURN m",
    "get_memory_by_type": "MATCH (m:Memory {type: $type}) RETURN m LIMIT $limit",
    "get_related": "MATCH (m:Memory {uuid: $uuid})-[r]-(n) RETURN m, r, n LIMIT $limit",
    "get_entities": "MATCH (e:Entity) RETURN e LIMIT $limit",
    "get_entity_by_name": "MATCH (e:Entity {name: $name}) RETURN e",
    "memory_entity_links": "MATCH (m:Memory)-[r:MENTIONS]->(e:Entity) WHERE m.uuid = $uuid RETURN e, r",
    "traverse_from_memory": "MATCH path = (m:Memory {uuid: $uuid})-[*1..2]-(n) RETURN path LIMIT $limit",
    "count_memories": "MATCH (m:Memory) RETURN count(m) as count",
    "count_by_type": "MATCH (m:Memory {type: $type}) RETURN count(m) as count",
    "recent_memories": "MATCH (m:Memory) RETURN m ORDER BY m.created_at DESC LIMIT $limit",
}


class CypherQueryRequest(BaseModel):
    """Request model for executing Cypher queries."""

    # For PROD mode: use query_name to select from whitelist
    query_name: str | None = Field(
        default=None,
        description="Name of whitelisted query template (required in PROD mode)",
    )

    # For DEV mode: raw query allowed
    query: str | None = Field(
        default=None,
        description="Raw Cypher query (only allowed in DEV mode)",
    )

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


@router.get("/templates")
async def get_query_templates() -> dict:
    """Get available query templates (for PROD mode)."""
    return {
        "templates": list(ALLOWED_QUERY_TEMPLATES.keys()),
        "details": {
            name: {"query": query}
            for name, query in ALLOWED_QUERY_TEMPLATES.items()
        },
    }


@router.post("/query")
async def run_cypher_query(request: CypherQueryRequest) -> dict:
    """Execute a validated Cypher query against the graph.

    In PROD mode: Only whitelisted query templates are allowed.
    In DEV mode: Arbitrary read-only queries are permitted.

    Security: Always read-only (mutations blocked), LIMIT injection, result truncation.
    """
    config = get_config()

    try:
        store = get_memory_store()

        # Clamp max_results
        max_results = min(max(1, request.max_results), 1000)

        # Determine query based on mode
        if config.allow_arbitrary_cypher:
            # DEV mode: allow raw queries
            if not request.query:
                raise HTTPException(
                    status_code=400,
                    detail="Query is required in DEV mode",
                )
            query = request.query
            log.debug(f"DEV mode: executing arbitrary Cypher query")
        else:
            # PROD mode: only allow whitelisted templates
            if not request.query_name:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "query_name is required in PROD mode. "
                        f"Available templates: {list(ALLOWED_QUERY_TEMPLATES.keys())}"
                    ),
                )

            if request.query_name not in ALLOWED_QUERY_TEMPLATES:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Query template '{request.query_name}' not found. "
                        f"Available: {list(ALLOWED_QUERY_TEMPLATES.keys())}"
                    ),
                )

            query = ALLOWED_QUERY_TEMPLATES[request.query_name]
            log.debug(f"PROD mode: using template '{request.query_name}'")

        result = store.db.execute_validated_cypher(
            query=query,
            params=request.params,
            max_results=max_results,
            allow_mutations=False,  # Always read-only via API
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Cypher query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
