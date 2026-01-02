"""Tests for MCP thin layer client."""

import pytest
import respx
from httpx import Response

from simplemem_lite.compression import compress_payload
from simplemem_lite.mcp.client import BackendClient, BackendError


@pytest.fixture
def client():
    """Create a test backend client."""
    return BackendClient(base_url="http://test-backend:8420")


class TestBackendClient:
    """Tests for BackendClient."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_check(self, client):
        """Health check should return backend status."""
        respx.get("http://test-backend:8420/health").mock(
            return_value=Response(200, json={"status": "healthy"})
        )

        result = await client.health_check()
        assert result["status"] == "healthy"
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_healthy_returns_true(self, client):
        """is_healthy should return True when backend responds."""
        respx.get("http://test-backend:8420/health").mock(
            return_value=Response(200, json={"status": "healthy"})
        )

        assert await client.is_healthy() is True
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_healthy_returns_false_on_error(self, client):
        """is_healthy should return False when backend fails."""
        respx.get("http://test-backend:8420/health").mock(
            return_value=Response(500)
        )

        assert await client.is_healthy() is False
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_store_memory(self, client):
        """store_memory should POST to memories/store."""
        respx.post("http://test-backend:8420/api/v1/memories/store").mock(
            return_value=Response(200, json={
                "uuid": "test-uuid-123",
                "message": "Memory stored successfully"
            })
        )

        result = await client.store_memory(
            text="Test memory content",
            type="fact",
            project_id="test-project",
        )
        assert result["uuid"] == "test-uuid-123"
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_memories(self, client):
        """search_memories should POST to memories/search."""
        respx.post("http://test-backend:8420/api/v1/memories/search").mock(
            return_value=Response(200, json={
                "results": [{"uuid": "123", "content": "test"}],
                "count": 1
            })
        )

        result = await client.search_memories(
            query="test query",
            limit=5,
        )
        assert result["count"] == 1
        assert len(result["results"]) == 1
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_process_trace(self, client):
        """process_trace should POST compressed trace to traces/process."""
        route = respx.post("http://test-backend:8420/api/v1/traces/process").mock(
            return_value=Response(200, json={
                "job_id": "job-123",
                "status": "submitted"
            })
        )

        result = await client.process_trace(
            session_id="session-abc",
            trace_content=[{"type": "user", "content": "test"}],
            background=True,
        )
        assert result["job_id"] == "job-123"
        assert result["status"] == "submitted"

        # Verify the request was made
        assert route.called
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_process_trace_compresses_large_content(self, client):
        """process_trace should compress large trace content."""
        route = respx.post("http://test-backend:8420/api/v1/traces/process").mock(
            return_value=Response(200, json={"job_id": "job-123", "status": "submitted"})
        )

        # Create content larger than compression threshold (4KB)
        large_content = [{"type": "user", "content": "x" * 10000}]

        await client.process_trace(
            session_id="session-abc",
            trace_content=large_content,
            background=True,
        )

        # Verify request was made with compressed=True
        request = route.calls[0].request
        import json
        body = json.loads(request.content)
        assert body["compressed"] is True
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_index_code(self, client):
        """index_code should POST files to code/index."""
        respx.post("http://test-backend:8420/api/v1/code/index").mock(
            return_value=Response(200, json={
                "files_indexed": 2,
                "chunks_created": 10
            })
        )

        result = await client.index_code(
            project_root="/test/project",
            files=[
                {"path": "main.py", "content": "print('hello')"},
                {"path": "utils.py", "content": "def util(): pass"},
            ],
        )
        assert result["files_indexed"] == 2
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_code(self, client):
        """search_code should POST query to code/search."""
        respx.post("http://test-backend:8420/api/v1/code/search").mock(
            return_value=Response(200, json={
                "results": [{"chunk_id": "c1", "score": 0.95}],
                "count": 1
            })
        )

        result = await client.search_code(
            query="print hello",
            limit=5,
        )
        assert result["count"] == 1
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_job_status(self, client):
        """get_job_status should GET job info."""
        respx.get("http://test-backend:8420/api/v1/traces/job/job-123").mock(
            return_value=Response(200, json={
                "id": "job-123",
                "status": "completed",
                "progress": 100
            })
        )

        result = await client.get_job_status("job-123")
        assert result["status"] == "completed"
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_graph_schema(self, client):
        """get_graph_schema should GET schema."""
        respx.get("http://test-backend:8420/api/v1/graph/schema").mock(
            return_value=Response(200, json={
                "node_labels": ["Memory", "Entity"],
                "relationship_types": ["RELATES_TO"]
            })
        )

        result = await client.get_graph_schema()
        assert "node_labels" in result
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_error_handling(self, client):
        """Client should raise BackendError on HTTP errors."""
        respx.post("http://test-backend:8420/api/v1/memories/store").mock(
            return_value=Response(400, json={"detail": "Invalid request"})
        )

        with pytest.raises(BackendError) as exc_info:
            await client.store_memory(text="test")

        assert exc_info.value.status_code == 400
        assert "Invalid request" in exc_info.value.detail
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_key_header(self):
        """Client should send API key in header when provided."""
        route = respx.get("http://test-backend:8420/health").mock(
            return_value=Response(200, json={"status": "healthy"})
        )

        client = BackendClient(
            base_url="http://test-backend:8420",
            api_key="secret-key-123",
        )

        await client.health_check()

        # Verify header was sent
        request = route.calls[0].request
        assert request.headers.get("X-API-Key") == "secret-key-123"
        await client.close()


class TestBackendClientBatch:
    """Tests for batch operations."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_process_trace_batch(self, client):
        """process_trace_batch should POST multiple traces."""
        respx.post("http://test-backend:8420/api/v1/traces/process-batch").mock(
            return_value=Response(200, json={
                "queued": ["sess-1", "sess-2"],
                "errors": [],
                "job_ids": {"sess-1": "job-1", "sess-2": "job-2"}
            })
        )

        result = await client.process_trace_batch(
            traces=[
                {"session_id": "sess-1", "trace_content": []},
                {"session_id": "sess-2", "trace_content": []},
            ],
            max_concurrent=2,
        )
        assert len(result["queued"]) == 2
        await client.close()
