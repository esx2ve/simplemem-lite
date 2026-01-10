"""Scratchpad API endpoints for SimpleMem-Lite backend.

Provides task state persistence with JSON+TOON hybrid format:
- save_scratchpad: Create/replace scratchpad
- load_scratchpad: Retrieve with optional memory expansion
- update_scratchpad: Partial field update
- attach_to_scratchpad: Link memory/session references
- render_scratchpad: Generate markdown view
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette import status

from simplemem_lite.backend.services import get_memory_store
from simplemem_lite.toon import (
    scratchpad_to_markdown,
    scratchpad_validate,
    scratchpad_expand_json,
    toon_table_parse,
    toon_table_render,
)

log = logging.getLogger("simplemem_lite.backend.api.scratchpad")

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class SaveScratchpadRequest(BaseModel):
    """Request body for save_scratchpad endpoint."""

    task_id: str = Field(..., description="Unique task identifier")
    scratchpad: dict[str, Any] = Field(..., description="Scratchpad data (JSON+TOON hybrid)")
    project_id: str = Field(..., description="Project identifier for isolation")


class UpdateScratchpadRequest(BaseModel):
    """Request body for update_scratchpad endpoint."""

    patch: dict[str, Any] = Field(..., description="Fields to update (partial)")
    project_id: str = Field(..., description="Project identifier for isolation")


class AttachRequest(BaseModel):
    """Request body for attach_to_scratchpad endpoint."""

    memory_ids: list[str] | None = Field(default=None, description="Memory UUIDs to attach")
    session_ids: list[str] | None = Field(default=None, description="Session IDs to attach")
    reasons: dict[str, str] | None = Field(default=None, description="Optional reasons per ID")
    project_id: str = Field(..., description="Project identifier for isolation")


class RenderRequest(BaseModel):
    """Query params for render endpoint."""

    format: str = Field(default="markdown", description="Output format: markdown or json")


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/{task_id}")
async def save_scratchpad(
    task_id: str,
    request: SaveScratchpadRequest,
) -> dict[str, Any]:
    """Create or replace a scratchpad for a task.

    Stores the scratchpad as a Memory node with type="scratchpad".
    Uses latest-wins semantics (overwrites existing).

    Args:
        task_id: Unique task identifier
        request: Scratchpad data and project_id

    Returns:
        {"success": True, "uuid": "...", "created": True/False}
    """
    log.info(f"Saving scratchpad for task: {task_id}")

    # Validate scratchpad structure
    scratchpad = request.scratchpad.copy()
    scratchpad["task_id"] = task_id  # Ensure task_id is set
    scratchpad["version"] = scratchpad.get("version", "1.1")
    scratchpad["updated_at"] = int(time.time())

    errors = scratchpad_validate(scratchpad)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scratchpad: {errors}",
        )

    store = get_memory_store()

    # Check if scratchpad already exists for this task
    existing = _find_scratchpad(task_id, request.project_id)

    if existing:
        # Update existing
        import json
        store.db.graph.query(
            """
            MATCH (m:Memory {uuid: $uuid})
            SET m.content = $content,
                m.updated_at = $updated_at
            """,
            {
                "uuid": existing["uuid"],
                "content": json.dumps(scratchpad),
                "updated_at": scratchpad["updated_at"],
            },
        )
        log.info(f"Updated existing scratchpad: {existing['uuid']}")
        return {"success": True, "uuid": existing["uuid"], "created": False}
    else:
        # Create new
        import json
        import uuid as uuid_mod

        new_uuid = str(uuid_mod.uuid4())
        store.db.graph.query(
            """
            CREATE (m:Memory {
                uuid: $uuid,
                content: $content,
                type: 'scratchpad',
                project_id: $project_id,
                task_id: $task_id,
                created_at: $created_at,
                updated_at: $updated_at
            })
            """,
            {
                "uuid": new_uuid,
                "content": json.dumps(scratchpad),
                "project_id": request.project_id,
                "task_id": task_id,
                "created_at": scratchpad["updated_at"],
                "updated_at": scratchpad["updated_at"],
            },
        )
        log.info(f"Created new scratchpad: {new_uuid}")
        return {"success": True, "uuid": new_uuid, "created": True}


@router.get("/{task_id}")
async def load_scratchpad(
    task_id: str,
    project_id: str,
    expand_memories: bool = False,
) -> dict[str, Any]:
    """Load a scratchpad for a task.

    Args:
        task_id: Unique task identifier
        project_id: Project identifier for isolation
        expand_memories: If True, fetch full content of attached memories

    Returns:
        {"scratchpad": {...}, "updated_at": ..., "expanded_memories": [...]}
    """
    log.info(f"Loading scratchpad for task: {task_id}")

    existing = _find_scratchpad(task_id, project_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scratchpad not found for task: {task_id}",
        )

    import json
    scratchpad = json.loads(existing["content"])

    result = {
        "scratchpad": scratchpad,
        "uuid": existing["uuid"],
        "updated_at": existing.get("updated_at"),
    }

    # Optionally expand attached memories
    if expand_memories and scratchpad.get("attached_memories"):
        attached = toon_table_parse(scratchpad["attached_memories"])
        memory_uuids = [row.get("uuid") for row in attached if row.get("uuid")]

        if memory_uuids:
            store = get_memory_store()
            expanded = []
            for mem_uuid in memory_uuids:
                mem_result = store.db.graph.query(
                    """
                    MATCH (m:Memory {uuid: $uuid})
                    RETURN m.uuid, m.content, m.type
                    """,
                    {"uuid": mem_uuid},
                )
                if mem_result.result_set:
                    row = mem_result.result_set[0]
                    expanded.append({
                        "uuid": row[0],
                        "content": row[1],
                        "type": row[2],
                    })
            result["expanded_memories"] = expanded

    return result


@router.patch("/{task_id}")
async def update_scratchpad(
    task_id: str,
    request: UpdateScratchpadRequest,
) -> dict[str, Any]:
    """Partially update a scratchpad.

    Only updates the fields provided in the patch.

    Args:
        task_id: Unique task identifier
        request: Patch data and project_id

    Returns:
        {"success": True, "updated_fields": [...]}
    """
    log.info(f"Updating scratchpad for task: {task_id}")

    existing = _find_scratchpad(task_id, request.project_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scratchpad not found for task: {task_id}",
        )

    import json
    scratchpad = json.loads(existing["content"])

    # Apply patch
    updated_fields = []
    for key, value in request.patch.items():
        if key not in ("task_id", "version", "uuid"):  # Protected fields
            scratchpad[key] = value
            updated_fields.append(key)

    scratchpad["updated_at"] = int(time.time())

    # Validate after patch
    errors = scratchpad_validate(scratchpad)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scratchpad after patch: {errors}",
        )

    # Save
    store = get_memory_store()
    store.db.graph.query(
        """
        MATCH (m:Memory {uuid: $uuid})
        SET m.content = $content,
            m.updated_at = $updated_at
        """,
        {
            "uuid": existing["uuid"],
            "content": json.dumps(scratchpad),
            "updated_at": scratchpad["updated_at"],
        },
    )

    return {"success": True, "updated_fields": updated_fields}


@router.post("/{task_id}/attach")
async def attach_to_scratchpad(
    task_id: str,
    request: AttachRequest,
) -> dict[str, Any]:
    """Attach memory and/or session references to a scratchpad.

    Creates graph edges for traversal and updates the scratchpad's
    attached_memories and attached_sessions fields.

    Args:
        task_id: Unique task identifier
        request: IDs to attach and optional reasons

    Returns:
        {"success": True, "attached": {"memories": N, "sessions": N}}
    """
    log.info(f"Attaching to scratchpad for task: {task_id}")

    existing = _find_scratchpad(task_id, request.project_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scratchpad not found for task: {task_id}",
        )

    import json
    scratchpad = json.loads(existing["content"])
    store = get_memory_store()
    reasons = request.reasons or {}

    attached_counts = {"memories": 0, "sessions": 0}

    # Attach memories
    if request.memory_ids:
        # Parse existing attached_memories
        existing_memories = toon_table_parse(scratchpad.get("attached_memories", ""))
        existing_uuids = {row.get("uuid") for row in existing_memories}

        new_rows = list(existing_memories)
        for mem_uuid in request.memory_ids:
            if mem_uuid not in existing_uuids:
                reason = reasons.get(mem_uuid, "")
                new_rows.append({"uuid": mem_uuid, "reason": reason})
                attached_counts["memories"] += 1

                # Create graph edge
                store.db.graph.query(
                    """
                    MATCH (s:Memory {uuid: $scratchpad_uuid})
                    MATCH (m:Memory {uuid: $memory_uuid})
                    MERGE (s)-[:REFERENCES {reason: $reason}]->(m)
                    """,
                    {
                        "scratchpad_uuid": existing["uuid"],
                        "memory_uuid": mem_uuid,
                        "reason": reason,
                    },
                )

        if new_rows:
            scratchpad["attached_memories"] = toon_table_render(
                new_rows, ["uuid", "reason"]
            )

    # Attach sessions
    if request.session_ids:
        existing_sessions = toon_table_parse(scratchpad.get("attached_sessions", ""))
        existing_sids = {row.get("session_id") for row in existing_sessions}

        new_rows = list(existing_sessions)
        for session_id in request.session_ids:
            if session_id not in existing_sids:
                description = reasons.get(session_id, "")
                new_rows.append({"session_id": session_id, "description": description})
                attached_counts["sessions"] += 1

                # Create graph edge (session may not exist as node yet)
                store.db.graph.query(
                    """
                    MATCH (s:Memory {uuid: $scratchpad_uuid})
                    MERGE (sess:Session {id: $session_id})
                    MERGE (s)-[:FROM_SESSION]->(sess)
                    """,
                    {
                        "scratchpad_uuid": existing["uuid"],
                        "session_id": session_id,
                    },
                )

        if new_rows:
            scratchpad["attached_sessions"] = toon_table_render(
                new_rows, ["session_id", "description"]
            )

    # Update scratchpad
    scratchpad["updated_at"] = int(time.time())
    store.db.graph.query(
        """
        MATCH (m:Memory {uuid: $uuid})
        SET m.content = $content,
            m.updated_at = $updated_at
        """,
        {
            "uuid": existing["uuid"],
            "content": json.dumps(scratchpad),
            "updated_at": scratchpad["updated_at"],
        },
    )

    return {"success": True, "attached": attached_counts}


@router.get("/{task_id}/render")
async def render_scratchpad(
    task_id: str,
    project_id: str,
    format: str = "markdown",
) -> dict[str, Any]:
    """Render a scratchpad in human-readable format.

    Args:
        task_id: Unique task identifier
        project_id: Project identifier for isolation
        format: Output format - "markdown" or "json" (expanded)

    Returns:
        {"rendered": "...", "format": "..."}
    """
    log.info(f"Rendering scratchpad for task: {task_id} (format: {format})")

    existing = _find_scratchpad(task_id, project_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scratchpad not found for task: {task_id}",
        )

    import json
    scratchpad = json.loads(existing["content"])

    if format == "markdown":
        rendered = scratchpad_to_markdown(scratchpad)
    elif format == "json":
        rendered = json.dumps(scratchpad_expand_json(scratchpad), indent=2)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown format: {format}. Use 'markdown' or 'json'.",
        )

    return {"rendered": rendered, "format": format}


@router.delete("/{task_id}")
async def delete_scratchpad(
    task_id: str,
    project_id: str,
) -> dict[str, Any]:
    """Delete a scratchpad.

    Also removes associated graph edges.

    Args:
        task_id: Unique task identifier
        project_id: Project identifier for isolation

    Returns:
        {"success": True, "deleted": True/False}
    """
    log.info(f"Deleting scratchpad for task: {task_id}")

    existing = _find_scratchpad(task_id, project_id)
    if not existing:
        return {"success": True, "deleted": False}

    store = get_memory_store()

    # Delete edges and node
    store.db.graph.query(
        """
        MATCH (m:Memory {uuid: $uuid})
        DETACH DELETE m
        """,
        {"uuid": existing["uuid"]},
    )

    return {"success": True, "deleted": True}


# =============================================================================
# Helper Functions
# =============================================================================


def _find_scratchpad(task_id: str, project_id: str) -> dict[str, Any] | None:
    """Find a scratchpad by task_id and project_id.

    Returns:
        Dict with uuid, content, updated_at or None if not found.
    """
    store = get_memory_store()
    result = store.db.graph.query(
        """
        MATCH (m:Memory {type: 'scratchpad', task_id: $task_id, project_id: $project_id})
        RETURN m.uuid, m.content, m.updated_at
        LIMIT 1
        """,
        {"task_id": task_id, "project_id": project_id},
    )

    if result.result_set:
        row = result.result_set[0]
        return {
            "uuid": row[0],
            "content": row[1],
            "updated_at": row[2],
        }
    return None
