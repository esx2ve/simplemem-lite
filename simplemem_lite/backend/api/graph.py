"""Graph database API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simplemem_lite.backend.config import get_config
from simplemem_lite.backend.services import get_memory_store
from simplemem_lite.backend.toon import toonify
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
    # Goal query templates
    "session_goals": """
        MATCH (s:Memory {type: 'session_summary'})-[:HAS_GOAL]->(g:Goal)
        WHERE ($session_id IS NULL OR s.session_id = $session_id)
          AND ($goal_id IS NULL OR g.id = $goal_id)
        RETURN g.id as goal_id, g.intent, g.status, s.session_id, s.uuid as session_uuid
        ORDER BY s.created_at DESC
        LIMIT $limit
    """,
    "all_goals": """
        MATCH (g:Goal)
        OPTIONAL MATCH (s:Memory)-[:HAS_GOAL]->(g)
        RETURN g.id as goal_id, g.intent, g.status, g.created_at, s.session_id
        ORDER BY g.created_at DESC
        LIMIT $limit
    """,
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
    output_format: str | None = Field(
        default=None,
        description="Response format: 'toon' (default) or 'json'. Env var: SIMPLEMEM_OUTPUT_FORMAT.",
    )


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
@toonify(headers=None, result_key="results")
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


class ComputePageRankRequest(BaseModel):
    """Request model for computing PageRank."""

    project_id: str | None = Field(default=None, description="Project filter (optional)")
    max_iterations: int = Field(default=100, ge=10, le=500, description="Max PageRank iterations")
    damping_factor: float = Field(default=0.85, ge=0.5, le=0.99, description="PageRank damping factor")


@router.post("/compute-pagerank")
async def compute_pagerank(request: ComputePageRankRequest) -> dict:
    """Compute and cache PageRank scores for all Memory nodes.

    Uses Memgraph MAGE's pagerank.get() algorithm and caches
    scores on Memory nodes for use in graph-enhanced search ranking.

    Should be called periodically (e.g., every 5 minutes) or after
    significant graph changes.
    """
    try:
        store = get_memory_store()

        # Try computing PageRank - this will raise if MAGE isn't available
        try:
            scores = store.db.compute_and_cache_pagerank(
                project_id=request.project_id,
                max_iterations=request.max_iterations,
                damping_factor=request.damping_factor,
            )
        except Exception as e:
            log.error(f"PageRank computation inner error: {e}")
            return {
                "success": False,
                "error": str(e),
                "nodes_scored": 0,
            }

        # Get stats for response
        if scores:
            max_score = max(scores.values())
            min_score = min(scores.values())
            avg_score = sum(scores.values()) / len(scores)
        else:
            max_score = min_score = avg_score = 0.0

        return {
            "success": True,
            "nodes_scored": len(scores),
            "max_pagerank": max_score,
            "min_pagerank": min_score,
            "avg_pagerank": avg_score,
        }

    except Exception as e:
        log.error(f"PageRank computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_graph_stats() -> dict:
    """Get graph statistics including normalization values for scoring."""
    try:
        store = get_memory_store()
        stats = store.db.get_graph_normalization_stats()
        return {
            "max_degree": stats.get("max_degree", 0),
            "avg_degree": stats.get("avg_degree", 0),
            "max_pagerank": stats.get("max_pagerank", 0),
            "node_count": stats.get("node_count", 0),
        }
    except Exception as e:
        log.error(f"Failed to get graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
