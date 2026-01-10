"""Integration tests for scratchpad API endpoints.

Tests the full scratchpad CRUD lifecycle including:
- Save/load/update/delete operations
- Memory and session attachments with graph edges
- Markdown and JSON rendering
- TOON format validation
"""

import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from simplemem_lite.backend.app import create_app
from simplemem_lite.backend.services import clear_service_caches, get_memory_store
from simplemem_lite.toon import (
    toon_list_render,
    toon_table_render,
    toon_list_parse,
    toon_table_parse,
)


@pytest.fixture
def client(monkeypatch):
    """Create a test client for the API in dev mode (no auth required)."""
    # Reset config to ensure fresh state
    import simplemem_lite.backend.config as config_module
    config_module._config = None

    # Set dev mode to disable auth requirement
    monkeypatch.setenv("SIMPLEMEM_MODE", "dev")
    # Use KuzuDB to avoid Memgraph timeout during auto-detection
    monkeypatch.setenv("SIMPLEMEM_GRAPH_BACKEND", "kuzu")

    clear_service_caches()
    app = create_app()
    yield TestClient(app)

    # Reset config after test
    config_module._config = None
    clear_service_caches()


@pytest.fixture
def project_id():
    """Generate a unique project ID for test isolation."""
    return f"test-project-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def task_id():
    """Generate a unique task ID."""
    return f"task-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def minimal_scratchpad():
    """Create a minimal valid scratchpad."""
    return {
        "task_id": "test-task",
        "version": "1.1",
        "current_focus": "Implement feature X",
    }


@pytest.fixture
def full_scratchpad():
    """Create a fully populated scratchpad with TOON fields."""
    return {
        "task_id": "test-task",
        "version": "1.1",
        "current_focus": "Implement authentication system",
        "active_constraints": toon_list_render(["Must use OAuth2", "No plaintext passwords"]),
        "active_files": toon_list_render(["src/auth.py", "src/middleware.py"]),
        "pending_verification": toon_list_render(["Token expiry logic", "Rate limiting"]),
        "decisions": toon_table_render(
            [{"what": "Use JWT tokens", "why": "Stateless auth", "ts": "1704067200"}],
            ["what", "why", "ts"]
        ),
        "notes": "Need to review OAuth library options",
        "attached_memories": "",
        "attached_sessions": "",
    }


class TestSaveScratchpad:
    """Tests for POST /api/v1/scratchpad/{task_id} endpoint."""

    def test_save_minimal_scratchpad(self, client, task_id, project_id):
        """Save minimal scratchpad should succeed."""
        scratchpad = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Test focus",
        }

        response = client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={
                "task_id": task_id,
                "scratchpad": scratchpad,
                "project_id": project_id,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "uuid" in data
        assert data["created"] is True

    def test_save_full_scratchpad(self, client, task_id, project_id, full_scratchpad):
        """Save fully populated scratchpad should succeed."""
        full_scratchpad["task_id"] = task_id

        response = client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={
                "task_id": task_id,
                "scratchpad": full_scratchpad,
                "project_id": project_id,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["created"] is True

    def test_save_overwrites_existing(self, client, task_id, project_id):
        """Saving to same task_id should overwrite."""
        scratchpad1 = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Original focus",
        }
        scratchpad2 = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Updated focus",
        }

        # First save
        response1 = client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad1, "project_id": project_id}
        )
        assert response1.status_code == 200
        uuid1 = response1.json()["uuid"]

        # Second save (overwrite)
        response2 = client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad2, "project_id": project_id}
        )
        assert response2.status_code == 200
        assert response2.json()["created"] is False
        assert response2.json()["uuid"] == uuid1  # Same UUID

    def test_save_validates_scratchpad(self, client, task_id, project_id):
        """Save should reject invalid scratchpad."""
        invalid_scratchpad = {
            "task_id": task_id,
            # Missing required current_focus
        }

        response = client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": invalid_scratchpad, "project_id": project_id}
        )

        assert response.status_code == 400
        assert "Invalid scratchpad" in response.json()["detail"]


class TestLoadScratchpad:
    """Tests for GET /api/v1/scratchpad/{task_id} endpoint."""

    def test_load_existing_scratchpad(self, client, task_id, project_id):
        """Load should return saved scratchpad."""
        scratchpad = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Test focus for load",
        }

        # Save first
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad, "project_id": project_id}
        )

        # Load
        response = client.get(
            f"/api/v1/scratchpad/{task_id}",
            params={"project_id": project_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert "scratchpad" in data
        assert data["scratchpad"]["current_focus"] == "Test focus for load"
        assert "uuid" in data
        assert "updated_at" in data

    def test_load_nonexistent_returns_404(self, client, project_id):
        """Load nonexistent scratchpad should return 404."""
        response = client.get(
            "/api/v1/scratchpad/nonexistent-task-id",
            params={"project_id": project_id}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_load_with_expand_memories(self, client, task_id, project_id):
        """Load with expand_memories should include full memory content."""
        # First create a memory to attach
        mem_response = client.post(
            "/api/v1/memories/store",
            json={"text": "Test memory for expansion", "project_id": project_id}
        )
        memory_uuid = mem_response.json()["uuid"]

        # Create scratchpad with attached memory
        scratchpad = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Test expand",
            "attached_memories": toon_table_render(
                [{"uuid": memory_uuid, "reason": "test"}],
                ["uuid", "reason"]
            ),
        }

        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad, "project_id": project_id}
        )

        # Load with expansion
        response = client.get(
            f"/api/v1/scratchpad/{task_id}",
            params={"project_id": project_id, "expand_memories": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert "expanded_memories" in data
        assert len(data["expanded_memories"]) > 0
        assert data["expanded_memories"][0]["uuid"] == memory_uuid


class TestUpdateScratchpad:
    """Tests for PATCH /api/v1/scratchpad/{task_id} endpoint."""

    def test_update_single_field(self, client, task_id, project_id):
        """Update should modify only specified fields."""
        # Create initial
        scratchpad = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Original focus",
            "notes": "Original notes",
        }
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad, "project_id": project_id}
        )

        # Update focus only
        response = client.patch(
            f"/api/v1/scratchpad/{task_id}",
            json={
                "patch": {"current_focus": "Updated focus"},
                "project_id": project_id,
            }
        )

        assert response.status_code == 200
        assert response.json()["success"] is True
        assert "current_focus" in response.json()["updated_fields"]

        # Verify notes unchanged
        load_response = client.get(
            f"/api/v1/scratchpad/{task_id}",
            params={"project_id": project_id}
        )
        assert load_response.json()["scratchpad"]["notes"] == "Original notes"
        assert load_response.json()["scratchpad"]["current_focus"] == "Updated focus"

    def test_update_protected_fields_ignored(self, client, task_id, project_id):
        """Update should not modify protected fields (task_id, version)."""
        scratchpad = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Test focus",
        }
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad, "project_id": project_id}
        )

        # Try to update protected fields
        response = client.patch(
            f"/api/v1/scratchpad/{task_id}",
            json={
                "patch": {"task_id": "hacked", "version": "9.9"},
                "project_id": project_id,
            }
        )

        assert response.status_code == 200
        # Protected fields should not be in updated_fields
        assert "task_id" not in response.json()["updated_fields"]
        assert "version" not in response.json()["updated_fields"]

    def test_update_nonexistent_returns_404(self, client, project_id):
        """Update nonexistent scratchpad should return 404."""
        response = client.patch(
            "/api/v1/scratchpad/nonexistent-task",
            json={
                "patch": {"current_focus": "New focus"},
                "project_id": project_id,
            }
        )

        assert response.status_code == 404


class TestAttachToScratchpad:
    """Tests for POST /api/v1/scratchpad/{task_id}/attach endpoint."""

    def test_attach_memories(self, client, task_id, project_id):
        """Attach should add memory references to scratchpad."""
        # Create scratchpad
        scratchpad = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Test attach",
        }
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad, "project_id": project_id}
        )

        # Create memories to attach
        mem1 = client.post(
            "/api/v1/memories/store",
            json={"text": "Memory 1", "project_id": project_id}
        ).json()["uuid"]
        mem2 = client.post(
            "/api/v1/memories/store",
            json={"text": "Memory 2", "project_id": project_id}
        ).json()["uuid"]

        # Attach
        response = client.post(
            f"/api/v1/scratchpad/{task_id}/attach",
            json={
                "memory_ids": [mem1, mem2],
                "reasons": {mem1: "Relevant fix", mem2: "Related context"},
                "project_id": project_id,
            }
        )

        assert response.status_code == 200
        assert response.json()["success"] is True
        assert response.json()["attached"]["memories"] == 2

        # Verify in loaded scratchpad
        load_response = client.get(
            f"/api/v1/scratchpad/{task_id}",
            params={"project_id": project_id}
        )
        attached = toon_table_parse(
            load_response.json()["scratchpad"].get("attached_memories", "")
        )
        assert len(attached) == 2
        assert any(row.get("uuid") == mem1 for row in attached)

    def test_attach_sessions(self, client, task_id, project_id):
        """Attach should add session references to scratchpad."""
        # Create scratchpad
        scratchpad = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Test attach sessions",
        }
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad, "project_id": project_id}
        )

        session_id = f"session-{uuid.uuid4().hex[:8]}"

        response = client.post(
            f"/api/v1/scratchpad/{task_id}/attach",
            json={
                "session_ids": [session_id],
                "reasons": {session_id: "Related debugging session"},
                "project_id": project_id,
            }
        )

        assert response.status_code == 200
        assert response.json()["attached"]["sessions"] == 1

    def test_attach_idempotent(self, client, task_id, project_id):
        """Attaching same memory twice should not duplicate."""
        scratchpad = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Test idempotent attach",
        }
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad, "project_id": project_id}
        )

        mem = client.post(
            "/api/v1/memories/store",
            json={"text": "Test memory", "project_id": project_id}
        ).json()["uuid"]

        # Attach twice
        client.post(
            f"/api/v1/scratchpad/{task_id}/attach",
            json={"memory_ids": [mem], "project_id": project_id}
        )
        response = client.post(
            f"/api/v1/scratchpad/{task_id}/attach",
            json={"memory_ids": [mem], "project_id": project_id}
        )

        # Second attach should report 0 new
        assert response.json()["attached"]["memories"] == 0


class TestRenderScratchpad:
    """Tests for GET /api/v1/scratchpad/{task_id}/render endpoint."""

    def test_render_markdown(self, client, task_id, project_id, full_scratchpad):
        """Render markdown should produce human-readable output."""
        full_scratchpad["task_id"] = task_id
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": full_scratchpad, "project_id": project_id}
        )

        response = client.get(
            f"/api/v1/scratchpad/{task_id}/render",
            params={"project_id": project_id, "format": "markdown"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "markdown"
        rendered = data["rendered"]

        # Check markdown structure
        assert "# Scratchpad" in rendered
        assert "## Current Focus" in rendered
        assert "Implement authentication system" in rendered
        assert "## Active Constraints" in rendered
        assert "Must use OAuth2" in rendered

    def test_render_json(self, client, task_id, project_id, full_scratchpad):
        """Render JSON should produce expanded structure."""
        full_scratchpad["task_id"] = task_id
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": full_scratchpad, "project_id": project_id}
        )

        response = client.get(
            f"/api/v1/scratchpad/{task_id}/render",
            params={"project_id": project_id, "format": "json"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "json"

        # Parse and verify expanded structure
        expanded = json.loads(data["rendered"])
        assert isinstance(expanded["active_constraints"], list)
        assert "Must use OAuth2" in expanded["active_constraints"]
        assert isinstance(expanded["decisions"], list)

    def test_render_invalid_format(self, client, task_id, project_id):
        """Render with invalid format should return 400."""
        scratchpad = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Test",
        }
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad, "project_id": project_id}
        )

        response = client.get(
            f"/api/v1/scratchpad/{task_id}/render",
            params={"project_id": project_id, "format": "xml"}
        )

        assert response.status_code == 400
        assert "Unknown format" in response.json()["detail"]


class TestDeleteScratchpad:
    """Tests for DELETE /api/v1/scratchpad/{task_id} endpoint."""

    def test_delete_existing(self, client, task_id, project_id):
        """Delete should remove scratchpad."""
        scratchpad = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "To be deleted",
        }
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad, "project_id": project_id}
        )

        response = client.delete(
            f"/api/v1/scratchpad/{task_id}",
            params={"project_id": project_id}
        )

        assert response.status_code == 200
        assert response.json()["deleted"] is True

        # Verify gone
        load_response = client.get(
            f"/api/v1/scratchpad/{task_id}",
            params={"project_id": project_id}
        )
        assert load_response.status_code == 404

    def test_delete_nonexistent(self, client, project_id):
        """Delete nonexistent should succeed with deleted=False."""
        response = client.delete(
            "/api/v1/scratchpad/nonexistent-task",
            params={"project_id": project_id}
        )

        assert response.status_code == 200
        assert response.json()["deleted"] is False


class TestProjectIsolation:
    """Tests for project-level isolation."""

    def test_scratchpads_isolated_by_project(self, client, task_id):
        """Scratchpads should be isolated between projects."""
        project_a = f"project-a-{uuid.uuid4().hex[:8]}"
        project_b = f"project-b-{uuid.uuid4().hex[:8]}"

        scratchpad_a = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Focus for project A",
        }
        scratchpad_b = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Focus for project B",
        }

        # Save to project A
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad_a, "project_id": project_a}
        )

        # Save to project B (same task_id)
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad_b, "project_id": project_b}
        )

        # Load from each project
        response_a = client.get(
            f"/api/v1/scratchpad/{task_id}",
            params={"project_id": project_a}
        )
        response_b = client.get(
            f"/api/v1/scratchpad/{task_id}",
            params={"project_id": project_b}
        )

        assert response_a.json()["scratchpad"]["current_focus"] == "Focus for project A"
        assert response_b.json()["scratchpad"]["current_focus"] == "Focus for project B"


class TestGraphEdges:
    """Tests for graph edge creation on attachment."""

    def test_memory_attachment_creates_references_edge(self, client, task_id, project_id):
        """Attaching memory should create REFERENCES edge in graph."""
        # Create scratchpad
        scratchpad = {
            "task_id": task_id,
            "version": "1.1",
            "current_focus": "Test graph edge",
        }
        client.post(
            f"/api/v1/scratchpad/{task_id}",
            json={"task_id": task_id, "scratchpad": scratchpad, "project_id": project_id}
        )

        # Create and attach memory
        mem = client.post(
            "/api/v1/memories/store",
            json={"text": "Memory for edge test", "project_id": project_id}
        ).json()["uuid"]

        client.post(
            f"/api/v1/scratchpad/{task_id}/attach",
            json={
                "memory_ids": [mem],
                "reasons": {mem: "Test reason"},
                "project_id": project_id,
            }
        )

        # Query graph directly to verify edge
        store = get_memory_store()
        result = store.db.graph.query(
            """
            MATCH (s:Memory {type: 'scratchpad', task_id: $task_id})-[r:REFERENCES]->(m:Memory {uuid: $mem_uuid})
            RETURN r.reason
            """,
            {"task_id": task_id, "mem_uuid": mem}
        )

        assert result.result_set is not None
        assert len(result.result_set) == 1
        assert result.result_set[0][0] == "Test reason"
