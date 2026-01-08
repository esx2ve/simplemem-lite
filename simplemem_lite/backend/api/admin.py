"""Admin API endpoints for SimpleMem-Lite backend.

These endpoints are for administrative operations like data wipes.
Most are DEV MODE ONLY for safety.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette import status

from simplemem_lite.backend.config import get_config
from simplemem_lite.backend.services import get_database_manager

log = logging.getLogger("simplemem_lite.backend.api.admin")

router = APIRouter()


class WipeRequest(BaseModel):
    """Request body for wipe endpoint."""

    confirm: str
    include_logs: bool = False


class WipeResponse(BaseModel):
    """Response from wipe endpoint."""

    status: str
    message: str
    stats: dict


@router.post("/wipe", response_model=WipeResponse)
async def wipe_all_data(request: WipeRequest) -> WipeResponse:
    """Wipe ALL data from SimpleMem. DEV MODE ONLY.

    This is a destructive operation that clears:
    - All memories (graph nodes and vectors)
    - All code chunks
    - All relationships
    - Metadata files (projects.json, session_state.db, etc.)
    - Jobs directory
    - Optionally: logs directory

    After wipe, empty tables/schema are reinitialized so the system is ready to use.

    **DEV MODE ONLY** - Returns 403 Forbidden in PROD mode.

    Request body:
    - confirm: Must be exactly "WIPE_ALL_DATA" to proceed
    - include_logs: If true, also wipe logs directory (default: false)

    Returns:
        Detailed stats of what was deleted
    """
    config = get_config()

    # CRITICAL: Only allow in DEV mode
    if not config.is_dev_mode:
        log.warning("WIPE: Attempted wipe in PROD mode - DENIED")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Wipe endpoint is only available in DEV mode. "
            "Set SIMPLEMEM_MODE=dev to enable.",
        )

    # Require explicit confirmation
    if request.confirm != "WIPE_ALL_DATA":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Must confirm with {"confirm": "WIPE_ALL_DATA"} in request body',
        )

    log.warning("=" * 70)
    log.warning("  ADMIN: Wipe request received and authorized")
    log.warning(f"  include_logs: {request.include_logs}")
    log.warning("=" * 70)

    # Execute wipe
    db_manager = get_database_manager()
    stats = db_manager.wipe_all_data(include_logs=request.include_logs)

    return WipeResponse(
        status="wiped",
        message="All data has been wiped. System reinitialized with empty tables.",
        stats=stats,
    )


@router.get("/status")
async def admin_status() -> dict:
    """Get admin status including security mode."""
    config = get_config()
    return {
        "mode": config.mode.value,
        "is_dev_mode": config.is_dev_mode,
        "wipe_available": config.is_dev_mode,
        "require_auth": config.require_auth,
        "require_project_id": config.require_project_id,
    }


class ReindexRequest(BaseModel):
    """Request body for reindex endpoint."""

    background: bool = True


class ReindexResponse(BaseModel):
    """Response from reindex endpoint."""

    status: str
    project_id: str
    reindexed: int | None = None
    errors: int | None = None
    total: int | None = None
    job_id: str | None = None
    message: str | None = None


@router.get("/vector-dimension-check")
async def vector_dimension_check() -> dict:
    """Diagnostic endpoint to check actual vector dimensions in LanceDB.

    Returns:
        {
            "config_dimension": int,
            "actual_dimension": int,
            "sample_count": int,
            "total_vectors": int,
            "dimension_mismatch": bool,
            "embedding_model": str
        }
    """
    from simplemem_lite.backend.services import get_database_manager
    from simplemem_lite.config import Config

    config = Config()
    db = get_database_manager()

    result = {
        "config_dimension": config.embedding_dim,
        "embedding_model": config.embedding_model,
        "actual_dimension": None,
        "sample_count": 0,
        "total_vectors": 0,
        "dimension_mismatch": False,
        "dimensions_found": {},
    }

    try:
        if db.lance_table is not None:
            # Get total count
            result["total_vectors"] = db.lance_table.count_rows()

            if result["total_vectors"] > 0:
                # LanceDB 0.1: only to_arrow() with no args works
                # Load all vectors (only ~4MB for 1368 vectors at 768D)
                table = db.lance_table.to_arrow()

                # Sample up to 100 vectors to check dimensions
                sample_limit = min(100, len(table))
                dims_count = {}

                for i in range(sample_limit):
                    vec = table.column("vector")[i].as_py()
                    dim = len(vec)
                    dims_count[dim] = dims_count.get(dim, 0) + 1

                result["dimensions_found"] = dims_count
                result["sample_count"] = sample_limit

                # Get most common dimension
                if dims_count:
                    result["actual_dimension"] = max(dims_count.keys(), key=dims_count.get)
                    result["dimension_mismatch"] = result["actual_dimension"] != config.embedding_dim

                # Check schema (for debugging)
                try:
                    schema_field = db.lance_table.schema.field("vector")
                    result["schema_vector_field"] = str(schema_field)
                except Exception as schema_err:
                    result["schema_error"] = str(schema_err)

                # Check indices (for debugging empty search)
                try:
                    indices = db.lance_table.list_indices()
                    result["indices"] = [{"name": idx.name, "type": str(idx.index_type)} for idx in indices] if indices else []
                except Exception as idx_err:
                    result["indices_error"] = str(idx_err)

    except Exception as e:
        log.error(f"Vector dimension check failed: {e}")
        result["error"] = str(e)

    return result


@router.get("/test-vector-search")
async def test_vector_search() -> dict:
    """Test raw vector search without filters (diagnostic for index issues).

    Returns:
        {
            "search_signature": str,
            "schema_columns": list,
            "indices": list,
            "test_search_results": int,
            "sample_results": list
        }
    """
    from simplemem_lite.backend.services import get_database_manager
    import inspect

    db = get_database_manager()
    result = {
        "search_signature": None,
        "schema_columns": [],
        "indices": [],
        "test_search_results": 0,
        "sample_results": [],
        "error": None,
    }

    try:
        if db.lance_table is None:
            result["error"] = "LanceDB table not initialized"
            return result

        # Get search method signature
        try:
            sig = inspect.signature(db.lance_table.search)
            result["search_signature"] = str(sig)
        except Exception as e:
            result["search_signature"] = f"Error: {e}"

        # Get schema columns
        try:
            result["schema_columns"] = db.lance_table.schema.names
        except Exception as e:
            result["schema_columns"] = [f"Error: {e}"]

        # Get indices
        try:
            indices = db.lance_table.list_indices()
            result["indices"] = [{"name": idx.name, "type": str(idx.index_type)} for idx in indices] if indices else []
        except Exception as e:
            result["indices"] = [f"Error: {e}"]

        # Test bare search with first vector (no filters)
        try:
            table = db.lance_table.to_arrow()
            if len(table) > 0:
                # Get first vector as query
                query_vec = table.column("vector")[0].as_py()

                # Bare search with correct LanceDB 0.1 pattern
                results_list = (
                    db.lance_table
                    .search(query_vec, vector_column_name="vector", query_type="auto")
                    .limit(5)
                    .to_list()
                )

                result["test_search_results"] = len(results_list)
                result["sample_results"] = results_list[:3]

        except Exception as e:
            result["error"] = f"Search test failed: {str(e)}"
            log.error(f"Search test error: {e}", exc_info=True)

    except Exception as e:
        log.error(f"Test vector search failed: {e}", exc_info=True)
        result["error"] = str(e)

    return result


@router.post("/build-vector-index")
async def build_vector_index() -> dict:
    """Build ANN vector index for LanceDB (required for search() on old versions).

    On LanceDB 0.1, search() returns empty without an index even if data exists.
    This endpoint builds the IVF-PQ index to enable vector search.

    Returns:
        {
            "status": "success" | "error",
            "message": str,
            "indices_before": list,
            "indices_after": list,
            "index_built": bool
        }
    """
    from simplemem_lite.backend.services import get_database_manager

    db = get_database_manager()
    result = {
        "status": "error",
        "message": "",
        "indices_before": [],
        "indices_after": [],
        "index_built": False,
    }

    try:
        if db.lance_table is None:
            result["message"] = "LanceDB table not initialized"
            return result

        # Check current indices
        try:
            indices_before = db.lance_table.list_indices()
            result["indices_before"] = [{"name": idx.name, "type": str(idx.index_type)} for idx in indices_before] if indices_before else []
        except Exception as e:
            result["indices_before"] = [f"Error: {e}"]

        # Build index (LanceDB 0.1 API)
        log.info("Building vector index for LanceDB table...")
        try:
            # Try cosine first (preferred for embeddings)
            db.lance_table.create_index(
                metric="cosine",
                num_partitions=64,
                num_sub_vectors=16,
                replace=True,
            )
            result["message"] = "Index built successfully with cosine metric"
        except Exception as e:
            # Fallback to L2 if cosine not supported
            log.warning(f"Cosine metric failed: {e}, trying L2")
            db.lance_table.create_index(
                metric="L2",
                num_partitions=64,
                num_sub_vectors=16,
                replace=True,
            )
            result["message"] = "Index built successfully with L2 metric (cosine not supported)"

        # Check indices after build
        try:
            indices_after = db.lance_table.list_indices()
            result["indices_after"] = [{"name": idx.name, "type": str(idx.index_type)} for idx in indices_after] if indices_after else []
            result["index_built"] = len(result["indices_after"]) > 0
        except Exception as e:
            result["indices_after"] = [f"Error: {e}"]

        result["status"] = "success"
        log.info(f"Vector index built: {result['index_built']}")

    except Exception as e:
        log.error(f"Failed to build vector index: {e}", exc_info=True)
        result["message"] = f"Error: {str(e)}"
        result["status"] = "error"

    return result


@router.post("/test-embedding")
async def test_embedding(query: str = "triton investigation") -> dict:
    """Test embedding generation and search with the current config.

    Generates an embedding for the query text and tests searching with it.
    Returns embedding info and search results for diagnostics.
    """
    from simplemem_lite.backend.services import get_database_manager, get_memory_store
    from simplemem_lite.config import Config
    from simplemem_lite.embeddings import embed

    config = Config()
    db = get_database_manager()

    result = {
        "query": query,
        "config_model": config.embedding_model,
        "config_dimension": config.embedding_dim,
        "generated_embedding_dim": None,
        "search_results_count": 0,
        "sample_results": [],
        "error": None,
    }

    try:
        # Generate embedding
        query_embedding = embed(query, config)
        result["generated_embedding_dim"] = len(query_embedding)

        # Test search with generated embedding
        search_results = (
            db.lance_table
            .search(query_embedding, vector_column_name="vector", query_type="auto")
            .limit(5)
            .to_list()
        )

        result["search_results_count"] = len(search_results)
        result["sample_results"] = [
            {
                "uuid": r.get("uuid"),
                "type": r.get("type"),
                "content_preview": r.get("content", "")[:100]
            }
            for r in search_results[:3]
        ]

    except Exception as e:
        log.error(f"Test embedding failed: {e}", exc_info=True)
        result["error"] = str(e)

    return result


@router.post("/reindex/{project_id}", response_model=ReindexResponse)
async def reindex_project(
    project_id: str,
    request: ReindexRequest | None = None,
) -> ReindexResponse:
    """Re-generate embeddings for all memories in a project.

    This fixes embedding model mismatches where memories were embedded
    with one model but searches use a different model.

    Args:
        project_id: Project to reindex (URL path parameter)
        request.background: Run in background job (default: True)

    Returns:
        If background=True: {"job_id": "...", "status": "submitted"}
        If background=False: {"reindexed": N, "errors": 0, "project_id": "..."}
    """
    from simplemem_lite.backend.services import get_memory_store

    background = request.background if request else True

    log.info(f"ADMIN: reindex_project called for {project_id}, background={background}")

    store = get_memory_store()

    if background:
        # For background, we would need to submit to a job manager
        # For now, just run synchronously with a warning for large projects
        log.warning("Background reindex via REST API not yet implemented, running synchronously")

    try:
        result = store.reindex_memories(project_id)
        return ReindexResponse(
            status="completed",
            project_id=project_id,
            reindexed=result.get("reindexed", 0),
            errors=result.get("errors", 0),
            total=result.get("total", 0),
        )
    except Exception as e:
        log.error(f"Reindex failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
