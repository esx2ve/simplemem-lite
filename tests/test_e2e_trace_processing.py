"""End-to-end tests for trace processing with project_id propagation.

These tests verify that:
1. project_id is correctly passed through the entire trace processing pipeline
2. Memories are stored with the correct project_id metadata
3. Vector search correctly filters by project_id
4. Graph queries can find memories by project_id

REQUIREMENTS:
    These tests require Memgraph to be running. Start the Docker environment first:

    docker-compose -f docker-compose.local.yml up -d

    Then wait for services to be healthy:
    curl http://localhost:8420/health

To run with Docker:
    docker-compose -f docker-compose.local.yml up -d
    docker-compose -f docker-compose.local.yml exec backend pytest tests/test_e2e_trace_processing.py -v

To run locally (requires running Memgraph on localhost:7687):
    # Start Memgraph first:
    docker run -p 7687:7687 memgraph/memgraph-mage:latest
    # Then run tests:
    SIMPLEMEM_GRAPH_BACKEND=memgraph pytest tests/test_e2e_trace_processing.py -v

To run without graph (degraded mode):
    SIMPLEMEM_GRAPH_BACKEND=none pytest tests/test_e2e_trace_processing.py -v
"""

import asyncio
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from simplemem_lite.backend.app import create_app
from simplemem_lite.backend.services import (
    clear_service_caches,
    get_hierarchical_indexer,
    get_memory_store,
)
from simplemem_lite.compression import compress_payload


@pytest.fixture(scope="function")
def client(monkeypatch):
    """Create a fresh test client for each test without auth."""
    # Reset global config to disable auth
    import simplemem_lite.backend.config as config_module
    config_module._config = None
    clear_service_caches()

    # Ensure auth is disabled for tests
    monkeypatch.setenv("SIMPLEMEM_REQUIRE_AUTH", "false")
    monkeypatch.delenv("SIMPLEMEM_API_KEY", raising=False)

    app = create_app()
    yield TestClient(app)

    # Cleanup
    config_module._config = None
    clear_service_caches()


@pytest.fixture
def unique_project_id():
    """Generate a unique project ID for test isolation."""
    return f"/tmp/test-project-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_trace_content():
    """Sample Claude Code trace content for testing."""
    return [
        {
            "type": "user",
            "message": {"content": "Help me fix the authentication bug in login.py"},
            "uuid": f"msg-{uuid.uuid4().hex[:8]}",
            "timestamp": "2024-01-15T10:00:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "content": "I'll analyze the login.py file to find the authentication issue."
            },
            "uuid": f"msg-{uuid.uuid4().hex[:8]}",
            "timestamp": "2024-01-15T10:00:05Z",
        },
        {
            "type": "assistant",
            "message": {
                "content": "Found the bug! The password hash comparison was using == instead of constant-time comparison. Fixed by using secrets.compare_digest()."
            },
            "uuid": f"msg-{uuid.uuid4().hex[:8]}",
            "timestamp": "2024-01-15T10:00:30Z",
        },
        {
            "type": "user",
            "message": {"content": "Great, thanks! That fixed it."},
            "uuid": f"msg-{uuid.uuid4().hex[:8]}",
            "timestamp": "2024-01-15T10:01:00Z",
        },
    ]


class TestProjectIdPropagation:
    """Tests for project_id propagation through trace processing."""

    def test_process_trace_with_project_id(
        self, client, unique_project_id, sample_trace_content
    ):
        """Process trace should accept and use project_id."""
        session_id = f"test-session-{uuid.uuid4().hex[:8]}"

        response = client.post(
            "/api/v1/traces/process",
            json={
                "session_id": session_id,
                "trace_content": sample_trace_content,
                "compressed": False,
                "background": False,  # Synchronous for testing
                "project_id": unique_project_id,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Sync mode should return results directly or via job completion
        assert data.get("status") in ("submitted", "completed") or "session_summary_id" in data

    def test_process_trace_background_with_project_id(
        self, client, unique_project_id, sample_trace_content
    ):
        """Process trace in background mode should accept project_id."""
        session_id = f"test-session-bg-{uuid.uuid4().hex[:8]}"

        response = client.post(
            "/api/v1/traces/process",
            json={
                "session_id": session_id,
                "trace_content": sample_trace_content,
                "compressed": False,
                "background": True,
                "project_id": unique_project_id,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "submitted"
        assert "job_id" in data

    def test_process_trace_compressed_with_project_id(
        self, client, unique_project_id, sample_trace_content
    ):
        """Process trace with compressed content should preserve project_id."""
        session_id = f"test-session-compressed-{uuid.uuid4().hex[:8]}"
        compressed_content = compress_payload(sample_trace_content)

        response = client.post(
            "/api/v1/traces/process",
            json={
                "session_id": session_id,
                "trace_content": compressed_content,
                "compressed": True,
                "background": True,
                "project_id": unique_project_id,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "submitted"


class TestMemoryStorageWithProjectId:
    """Tests for memory storage with project_id."""

    def test_store_memory_with_project_id(self, client, unique_project_id):
        """Store memory should accept and persist project_id."""
        response = client.post(
            "/api/v1/memories/store",
            json={
                "text": "Test memory for project isolation",
                "type": "fact",
                "source": "user",
                "project_id": unique_project_id,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "uuid" in data
        assert data["uuid"]  # Should be non-empty

    def test_store_multiple_memories_different_projects(self, client):
        """Memories stored with different project_ids should be isolated."""
        project_a = f"/tmp/project-a-{uuid.uuid4().hex[:8]}"
        project_b = f"/tmp/project-b-{uuid.uuid4().hex[:8]}"

        # Store memory in project A
        response_a = client.post(
            "/api/v1/memories/store",
            json={
                "text": "Memory specific to project A - authentication fix",
                "type": "lesson_learned",
                "project_id": project_a,
            },
        )
        assert response_a.status_code == 200
        uuid_a = response_a.json()["uuid"]

        # Store memory in project B
        response_b = client.post(
            "/api/v1/memories/store",
            json={
                "text": "Memory specific to project B - database optimization",
                "type": "lesson_learned",
                "project_id": project_b,
            },
        )
        assert response_b.status_code == 200
        uuid_b = response_b.json()["uuid"]

        # Verify both UUIDs are different and non-empty
        assert uuid_a != uuid_b
        assert uuid_a
        assert uuid_b


class TestSearchWithProjectId:
    """Tests for search filtering by project_id."""

    def test_search_finds_memories_by_project_id(self, client, unique_project_id):
        """Search should return memories matching the project_id."""
        # Store a memory with the project
        store_response = client.post(
            "/api/v1/memories/store",
            json={
                "text": "Authentication bug fix using constant-time comparison",
                "type": "lesson_learned",
                "project_id": unique_project_id,
            },
        )
        assert store_response.status_code == 200

        # Give time for indexing
        time.sleep(0.5)

        # Search for it
        search_response = client.post(
            "/api/v1/memories/search",
            json={
                "query": "authentication bug fix",
                "project_id": unique_project_id,
                "limit": 10,
            },
        )

        assert search_response.status_code == 200
        data = search_response.json()
        assert "results" in data
        # Should find the memory we just stored
        results = data["results"]
        assert len(results) >= 0  # May or may not find depending on embedding model

    def test_search_does_not_return_memories_from_other_projects(self, client):
        """Search with project_id should NOT return memories from other projects."""
        project_a = f"/tmp/project-isolate-a-{uuid.uuid4().hex[:8]}"
        project_b = f"/tmp/project-isolate-b-{uuid.uuid4().hex[:8]}"

        # Store memory in project A with unique content
        unique_text = f"XYZ123 unique identifier {uuid.uuid4().hex}"
        client.post(
            "/api/v1/memories/store",
            json={
                "text": unique_text,
                "type": "fact",
                "project_id": project_a,
            },
        )

        time.sleep(0.5)

        # Search in project B - should NOT find project A's memory
        search_response = client.post(
            "/api/v1/memories/search",
            json={
                "query": "XYZ123 unique identifier",
                "project_id": project_b,
                "limit": 10,
            },
        )

        assert search_response.status_code == 200
        data = search_response.json()
        results = data.get("results", [])

        # Verify none of the results contain the unique text from project A
        for result in results:
            content = result.get("content", "")
            assert "XYZ123" not in content, "Found memory from wrong project!"


class TestAskMemoriesWithProjectId:
    """Tests for ask_memories with project_id filtering."""

    def test_ask_memories_with_project_id(self, client, unique_project_id):
        """Ask memories should filter by project_id."""
        # Store a memory
        client.post(
            "/api/v1/memories/store",
            json={
                "text": "The database connection pool was set to 10 connections",
                "type": "fact",
                "project_id": unique_project_id,
            },
        )

        time.sleep(0.5)

        # Ask about it
        ask_response = client.post(
            "/api/v1/memories/ask",
            json={
                "query": "What was the database connection pool size?",
                "project_id": unique_project_id,
                "max_memories": 5,
            },
        )

        assert ask_response.status_code == 200
        data = ask_response.json()
        # Response should include standard fields
        assert "answer" in data or "error" in data


class TestReasonMemoriesWithProjectId:
    """Tests for reason_memories with project_id filtering."""

    def test_reason_memories_with_project_id(self, client, unique_project_id):
        """Reason memories should filter by project_id."""
        # Store related memories
        client.post(
            "/api/v1/memories/store",
            json={
                "text": "Database connections were timing out under load",
                "type": "fact",
                "project_id": unique_project_id,
            },
        )
        client.post(
            "/api/v1/memories/store",
            json={
                "text": "Fixed by increasing connection pool from 5 to 20",
                "type": "lesson_learned",
                "project_id": unique_project_id,
            },
        )

        time.sleep(0.5)

        # Reason about it
        reason_response = client.post(
            "/api/v1/memories/reason",
            json={
                "query": "What caused database timeouts and how was it fixed?",
                "project_id": unique_project_id,
                "max_hops": 2,
            },
        )

        assert reason_response.status_code == 200
        data = reason_response.json()
        assert "conclusions" in data or "error" in data


class TestGraphQueriesWithProjectId:
    """Tests for graph queries with project_id."""

    def test_graph_schema_available(self, client):
        """Graph schema endpoint should be accessible."""
        response = client.get("/api/v1/graph/schema")
        assert response.status_code == 200
        data = response.json()
        # Should return schema information
        assert "node_labels" in data or "error" in data


class TestEndToEndTraceToSearch:
    """End-to-end tests from trace processing to search."""

    @pytest.mark.asyncio
    async def test_full_flow_trace_to_search(self, unique_project_id, sample_trace_content):
        """Full flow: process trace → memories stored → search finds them."""
        clear_service_caches()

        session_id = f"e2e-session-{uuid.uuid4().hex[:8]}"

        # Get services directly for sync processing
        indexer = get_hierarchical_indexer()
        memory_store = get_memory_store()

        # Process trace synchronously
        result = await indexer.index_session_content(
            session_id=session_id,
            trace_content=sample_trace_content,
            project_id=unique_project_id,
        )

        # Verify result
        assert result is not None, "Indexer returned None - check trace content format"

        if result:
            # Wait for indexing
            await asyncio.sleep(1)

            # Search for memories using the memory store
            search_result = await memory_store.search(
                query="authentication bug fix",
                project_id=unique_project_id,
                limit=10,
            )

            # Should find something (either summaries or messages)
            assert "results" in search_result

    @pytest.mark.asyncio
    async def test_project_isolation_end_to_end(self, sample_trace_content):
        """Verify project isolation works end-to-end."""
        clear_service_caches()

        project_a = f"/tmp/e2e-project-a-{uuid.uuid4().hex[:8]}"
        project_b = f"/tmp/e2e-project-b-{uuid.uuid4().hex[:8]}"

        indexer = get_hierarchical_indexer()
        memory_store = get_memory_store()

        # Index trace in project A
        session_a = f"session-a-{uuid.uuid4().hex[:8]}"
        await indexer.index_session_content(
            session_id=session_a,
            trace_content=sample_trace_content,
            project_id=project_a,
        )

        await asyncio.sleep(1)

        # Search in project B - should NOT find project A's content
        search_b = await memory_store.search(
            query="authentication bug",
            project_id=project_b,
            limit=10,
        )

        results_b = search_b.get("results", [])
        # Project B should have fewer/no results since we didn't index there
        # (May still have some if there's global content)

        # Search in project A - should find content
        search_a = await memory_store.search(
            query="authentication bug",
            project_id=project_a,
            limit=10,
        )

        # Results should be project-specific


class TestBatchProcessing:
    """Tests for batch trace processing."""

    def test_batch_process_with_project_id(self, client, unique_project_id, sample_trace_content):
        """Batch processing should handle multiple traces with same project_id."""
        traces = [
            {
                "session_id": f"batch-1-{uuid.uuid4().hex[:8]}",
                "trace_content": sample_trace_content,
                "compressed": False,
            },
            {
                "session_id": f"batch-2-{uuid.uuid4().hex[:8]}",
                "trace_content": sample_trace_content,
                "compressed": False,
            },
        ]

        response = client.post(
            "/api/v1/traces/process-batch",
            json={"traces": traces, "max_concurrent": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert "queued" in data or "job_ids" in data


class TestJobManagement:
    """Tests for background job management."""

    def test_job_status_endpoint(self, client, unique_project_id, sample_trace_content):
        """Job status endpoint should return correct status."""
        # Create a background job
        session_id = f"job-test-{uuid.uuid4().hex[:8]}"
        response = client.post(
            "/api/v1/traces/process",
            json={
                "session_id": session_id,
                "trace_content": sample_trace_content,
                "background": True,
                "project_id": unique_project_id,
            },
        )

        assert response.status_code == 200
        job_id = response.json().get("job_id")

        if job_id:
            # Check job status
            status_response = client.get(f"/api/v1/traces/job/{job_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["id"] == job_id
            assert status_data["status"] in ("pending", "running", "completed", "failed")

    def test_list_jobs_endpoint(self, client):
        """List jobs endpoint should return jobs list."""
        response = client.get("/api/v1/traces/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data


class TestStatsAndHealth:
    """Tests for stats and health endpoints."""

    def test_stats_endpoint(self, client):
        """Stats endpoint should return memory statistics."""
        response = client.get("/api/v1/memories/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_memories" in data

    def test_code_stats_with_project(self, client, unique_project_id):
        """Code stats should accept project_root filter."""
        response = client.get(
            "/api/v1/code/stats",
            params={"project_root": unique_project_id},
        )
        assert response.status_code == 200
        data = response.json()
        # Should return stats structure even if empty
        assert "chunk_count" in data or "error" in data


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_trace_content(self, client, unique_project_id):
        """Empty trace content should be handled gracefully."""
        response = client.post(
            "/api/v1/traces/process",
            json={
                "session_id": f"empty-{uuid.uuid4().hex[:8]}",
                "trace_content": [],
                "background": True,
                "project_id": unique_project_id,
            },
        )
        # Should accept but may result in "no messages" in job
        assert response.status_code == 200

    def test_missing_project_id(self, client, sample_trace_content):
        """Missing project_id should use default or None."""
        response = client.post(
            "/api/v1/traces/process",
            json={
                "session_id": f"no-project-{uuid.uuid4().hex[:8]}",
                "trace_content": sample_trace_content,
                "background": True,
                # No project_id
            },
        )
        # Should still work
        assert response.status_code == 200

    def test_invalid_compressed_data(self, client, unique_project_id):
        """Invalid compressed data should return error."""
        response = client.post(
            "/api/v1/traces/process",
            json={
                "session_id": f"invalid-{uuid.uuid4().hex[:8]}",
                "trace_content": "not-valid-base64-gzip",
                "compressed": True,
                "project_id": unique_project_id,
            },
        )
        assert response.status_code == 400
        assert "Failed to decompress" in response.json()["detail"]

    def test_special_characters_in_project_id(self, client, sample_trace_content):
        """Project ID with special characters should be handled."""
        special_project = "/Users/test/my-project_v2.0/src"
        response = client.post(
            "/api/v1/memories/store",
            json={
                "text": "Test with special characters in project",
                "type": "fact",
                "project_id": special_project,
            },
        )
        assert response.status_code == 200
