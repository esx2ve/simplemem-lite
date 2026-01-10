"""TOON (Token-Optimized Object Notation) decorator for FastAPI endpoints.

Provides a decorator that automatically converts list-of-dict responses to TOON format
when output_format="toon" is requested. This centralizes TOON conversion logic instead
of duplicating it in every endpoint.

Usage:
    @router.post("/search")
    @toonify(headers=["uuid", "type", "score", "content"], result_key="results")
    async def search_memories(request: SearchMemoriesRequest) -> dict:
        # ... business logic ...
        return {"results": result_dicts}

    # For hybrid mode (preserve reasoning text + TOON sources):
    @router.post("/reason")
    @toonify(headers=["uuid", "type", "score"], result_key="sources", hybrid=True)
    async def reason_memories(request: ReasonMemoriesRequest) -> dict:
        return {"reasoning": "...", "sources": [...], "confidence": "high"}
        # Returns: {"reasoning": "...", "sources": "uuid\\ttype\\tscore\\n...", "confidence": "high"}
"""

import functools
from typing import Any, Callable

from fastapi.responses import PlainTextResponse

from simplemem_lite.log_config import get_logger
from simplemem_lite.token_reduction import OUTPUT_FORMAT, to_toon

log = get_logger("backend.toon")


def toonify(
    headers: list[str],
    result_key: str | None = "results",
    format_param: str = "output_format",
    hybrid: bool = False,
):
    """Decorator to add TOON output support to an endpoint.

    Args:
        headers: Column headers for TOON output (order matters)
        result_key: Key in response dict containing the list to convert.
                    Use None if endpoint returns list directly.
        format_param: Name of the output_format field in request model.
        hybrid: If True, preserve all other fields and only convert result_key to TOON.
                Returns JSON with the result_key field containing TOON string.
                Use this for endpoints that return both synthesized text and source lists.

    The decorator looks for output_format in:
    1. The request Pydantic model (if it has output_format attribute)
    2. Falls back to SIMPLEMEM_OUTPUT_FORMAT env var

    Example (standard mode):
        @router.post("/search")
        @toonify(headers=["uuid", "type", "score", "content"])
        async def search_memories(request: SearchMemoriesRequest):
            return {"results": [...]}

        # When request.output_format="toon", returns:
        # uuid\ttype\tscore\tcontent
        # abc\tlesson\t0.9\tSome content...

    Example (hybrid mode):
        @router.post("/reason")
        @toonify(headers=["uuid", "type", "score"], result_key="sources", hybrid=True)
        async def reason_memories(request: ReasonMemoriesRequest):
            return {"reasoning": "...", "sources": [...], "confidence": "high"}

        # When request.output_format="toon", returns JSON:
        # {"reasoning": "...", "sources": "uuid\\ttype\\tscore\\n...", "confidence": "high"}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Execute the endpoint
            result = await func(*args, **kwargs)

            # Find output_format from request model in kwargs
            output_format = None
            for value in kwargs.values():
                if hasattr(value, format_param):
                    output_format = getattr(value, format_param)
                    break

            # Fall back to env var default
            fmt = output_format or OUTPUT_FORMAT

            if fmt != "toon":
                return result

            # Extract the list to convert
            if result_key and isinstance(result, dict):
                data = result.get(result_key, [])
            elif isinstance(result, list):
                data = result
            else:
                # Can't convert non-list to TOON
                log.warning(f"Cannot convert {type(result)} to TOON, returning as-is")
                return result

            # Auto-detect headers from first record if not provided
            header_list = headers
            if header_list is None:
                if data:
                    header_list = list(data[0].keys())
                else:
                    header_list = []

            # Convert to TOON
            if not data:
                toon_content = "\t".join(header_list)
            else:
                toon_content = to_toon(data, headers=header_list)
                log.debug(f"Converted {len(data)} records to TOON ({len(toon_content)} chars)")

            # Hybrid mode: preserve other fields, replace result_key with TOON string
            if hybrid and isinstance(result, dict):
                hybrid_result = {k: v for k, v in result.items() if k != result_key}
                hybrid_result[result_key] = toon_content
                return hybrid_result

            # Standard mode: return plain text TOON
            return PlainTextResponse(content=toon_content, media_type="text/plain")

        return wrapper

    return decorator
