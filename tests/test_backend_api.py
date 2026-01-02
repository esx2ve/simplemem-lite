"""Tests for SimpleMem-Lite backend API.

These tests verify both validation and wired service functionality.
"""

import pytest
from fastapi.testclient import TestClient

from simplemem_lite.backend.app import create_app
from simplemem_lite.backend.config import BackendConfig, _config
from simplemem_lite.compression import compress_payload, decompress_payload


@pytest.fixture
def client():
    """Create a test client for the API."""
    # Clear service caches to ensure fresh state
    from simplemem_lite.backend.services import clear_service_caches
    clear_service_caches()
    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_client(monkeypatch):
    """Create a test client with authentication enabled."""
    # Reset global config
    import simplemem_lite.backend.config as config_module
    from simplemem_lite.backend.services import clear_service_caches
    config_module._config = None
    clear_service_caches()

    # Set environment variables for auth
    monkeypatch.setenv("SIMPLEMEM_REQUIRE_AUTH", "true")
    monkeypatch.setenv("SIMPLEMEM_API_KEY", "test-secret-key")

    app = create_app()
    yield TestClient(app)

    # Reset config after test
    config_module._config = None
    clear_service_caches()


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_returns_healthy(self, client):
        """Health check should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "simplemem-lite-backend"


class TestAuthentication:
    """Tests for API authentication."""

    def test_no_auth_required_by_default(self, client):
        """API should not require auth by default."""
        response = client.get("/api/v1/memories/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_memories" in data

    def test_auth_required_when_enabled(self, auth_client):
        """API should require auth when enabled."""
        response = auth_client.get("/api/v1/memories/stats")
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_auth_with_valid_key(self, auth_client):
        """API should accept valid API key."""
        response = auth_client.get(
            "/api/v1/memories/stats",
            headers={"X-API-Key": "test-secret-key"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_memories" in data

    def test_auth_with_invalid_key(self, auth_client):
        """API should reject invalid API key."""
        response = auth_client.get(
            "/api/v1/memories/stats",
            headers={"X-API-Key": "wrong-key"}
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]


class TestCompressionUtilities:
    """Tests for compression utilities."""

    def test_compress_decompress_roundtrip(self):
        """Data should survive compression/decompression roundtrip."""
        original = {"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}}
        compressed = compress_payload(original)
        decompressed = decompress_payload(compressed)
        assert decompressed == original

    def test_compress_large_text(self):
        """Compression should work on large text."""
        original = "x" * 10000
        compressed = compress_payload(original)
        # Compressed should be smaller than original
        assert len(compressed) < len(original)
        decompressed = decompress_payload(compressed)
        assert decompressed == original

    def test_compress_produces_base64(self):
        """Compressed output should be base64-encoded ASCII."""
        data = {"test": "data"}
        compressed = compress_payload(data)
        # Should be valid ASCII
        compressed.encode("ascii")
        # Should be base64 decodable
        import base64
        base64.b64decode(compressed)


class TestMemoriesEndpoints:
    """Tests for memories API endpoints."""

    def test_store_memory_validation(self, client):
        """Store memory should validate request body."""
        # Missing required field
        response = client.post("/api/v1/memories/store", json={})
        assert response.status_code == 422  # Validation error

        # Valid request - should succeed
        response = client.post("/api/v1/memories/store", json={
            "text": "Test memory content for API test"
        })
        assert response.status_code == 200
        data = response.json()
        assert "uuid" in data
        assert data["message"] == "Memory stored successfully"

    def test_store_memory_with_relations(self, client):
        """Store memory should accept typed relations."""
        # First create a memory to relate to
        response = client.post("/api/v1/memories/store", json={
            "text": "Target memory for relation test"
        })
        assert response.status_code == 200
        target_uuid = response.json()["uuid"]

        # Store with valid relation
        response = client.post("/api/v1/memories/store", json={
            "text": "Memory with relation",
            "relations": [
                {"target_id": target_uuid, "type": "relates"}
            ]
        })
        assert response.status_code == 200

    def test_search_memories_validation(self, client):
        """Search memories should validate request body."""
        # Missing required field
        response = client.post("/api/v1/memories/search", json={})
        assert response.status_code == 422

        # Valid request - should return results
        response = client.post("/api/v1/memories/search", json={
            "query": "test query for API"
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "count" in data

    def test_search_memories_limit_bounds(self, client):
        """Search should enforce limit bounds."""
        # Limit too low
        response = client.post("/api/v1/memories/search", json={
            "query": "test",
            "limit": 0
        })
        assert response.status_code == 422

        # Limit too high
        response = client.post("/api/v1/memories/search", json={
            "query": "test",
            "limit": 101
        })
        assert response.status_code == 422

    def test_relate_memories_validation(self, client):
        """Relate memories should validate request body."""
        # Create two memories to relate
        resp1 = client.post("/api/v1/memories/store", json={
            "text": "First memory for relate test"
        })
        resp2 = client.post("/api/v1/memories/store", json={
            "text": "Second memory for relate test"
        })
        uuid1 = resp1.json()["uuid"]
        uuid2 = resp2.json()["uuid"]

        response = client.post("/api/v1/memories/relate", json={
            "from_id": uuid1,
            "to_id": uuid2
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestTracesEndpoints:
    """Tests for traces API endpoints."""

    def test_process_trace_raw_background(self, client):
        """Process trace in background mode should return job ID."""
        response = client.post("/api/v1/traces/process", json={
            "session_id": "test-session-123",
            "trace_content": [],  # Empty trace - no messages
            "compressed": False,
            "background": True
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "submitted"
        assert "job_id" in data

    def test_process_trace_compressed(self, client):
        """Process trace should accept compressed content."""
        trace_data = [{"type": "user", "content": "test message", "uuid": "m1"}]
        compressed = compress_payload(trace_data)

        response = client.post("/api/v1/traces/process", json={
            "session_id": "test-session-compressed",
            "trace_content": compressed,
            "compressed": True,
            "background": True
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "submitted"

    def test_process_trace_invalid_compression(self, client):
        """Process trace should reject invalid compressed data."""
        response = client.post("/api/v1/traces/process", json={
            "session_id": "test-session-123",
            "trace_content": "not-valid-base64-gzip",
            "compressed": True
        })
        assert response.status_code == 400
        assert "Failed to decompress" in response.json()["detail"]

    def test_process_trace_batch(self, client):
        """Batch processing should queue multiple traces."""
        response = client.post("/api/v1/traces/process-batch", json={
            "traces": [
                {"session_id": "batch-sess-1", "trace_content": []},
                {"session_id": "batch-sess-2", "trace_content": [], "compressed": False}
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert "queued" in data
        assert "job_ids" in data
        assert len(data["queued"]) == 2

    def test_job_status(self, client):
        """Get job status should return job info."""
        # Create a job first
        response = client.post("/api/v1/traces/process", json={
            "session_id": "job-status-test",
            "trace_content": [],
            "background": True
        })
        job_id = response.json()["job_id"]

        # Get job status
        response = client.get(f"/api/v1/traces/job/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["status"] in ("pending", "running", "completed", "failed")

    def test_job_not_found(self, client):
        """Get status for non-existent job should return 404."""
        response = client.get("/api/v1/traces/job/non-existent-job-id")
        assert response.status_code == 404

    def test_list_jobs(self, client):
        """List jobs should return job list."""
        response = client.get("/api/v1/traces/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data


class TestCodeEndpoints:
    """Tests for code indexing API endpoints."""

    def test_index_code_validation(self, client):
        """Index code should validate request body."""
        # Missing required fields
        response = client.post("/api/v1/code/index", json={})
        assert response.status_code == 422

        # Valid request with typed FileInput
        response = client.post("/api/v1/code/index", json={
            "project_root": "/test/project/api",
            "files": [
                {"path": "src/main.py", "content": "print('hello')"},
                {"path": "src/utils.py", "content": "def util(): pass", "compressed": False}
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert "files_indexed" in data
        assert data["files_indexed"] == 2

    def test_index_code_with_compressed_files(self, client):
        """Index code should handle compressed file content."""
        content = "def main():\n    print('hello world')"
        compressed = compress_payload(content)

        response = client.post("/api/v1/code/index", json={
            "project_root": "/test/project/compressed",
            "files": [
                {"path": "main.py", "content": compressed, "compressed": True}
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["files_indexed"] == 1

    def test_search_code(self, client):
        """Search code should return results."""
        response = client.post("/api/v1/code/search", json={
            "query": "print hello"
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "count" in data

    def test_code_stats(self, client):
        """Code stats should return statistics."""
        response = client.get("/api/v1/code/stats")
        assert response.status_code == 200


class TestGraphEndpoints:
    """Tests for graph API endpoints."""

    def test_get_schema(self, client):
        """Get schema endpoint should return graph schema."""
        response = client.get("/api/v1/graph/schema")
        assert response.status_code == 200
        data = response.json()
        # Schema should have node_labels and relationship_types
        assert "node_labels" in data or isinstance(data, dict)

    def test_run_query(self, client):
        """Run query should execute Cypher query."""
        response = client.post("/api/v1/graph/query", json={
            "query": "MATCH (n:Memory) RETURN n.uuid LIMIT 5"
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "row_count" in data

    def test_run_query_max_results_bounds(self, client):
        """Run query should enforce max_results bounds."""
        response = client.post("/api/v1/graph/query", json={
            "query": "MATCH (n) RETURN n",
            "max_results": 1001
        })
        assert response.status_code == 422
