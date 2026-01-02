"""Tests for MCP thin layer server."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from simplemem_lite.mcp.client import BackendError


# Mock the BackendClient and LocalReader before importing server
@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state between tests."""
    import simplemem_lite.mcp.server as server_module
    server_module._client = None
    server_module._reader = None
    yield
    server_module._client = None
    server_module._reader = None


class TestMemoryTools:
    """Tests for memory tool proxies."""

    @pytest.mark.asyncio
    async def test_store_memory_success(self):
        """store_memory should proxy to backend and return uuid."""
        from simplemem_lite.mcp.server import store_memory

        mock_client = AsyncMock()
        mock_client.store_memory.return_value = {"uuid": "test-uuid-123"}

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await store_memory(
                text="Test memory",
                type="fact",
                project_id="test-project",
            )

        assert result == {"uuid": "test-uuid-123"}
        mock_client.store_memory.assert_called_once_with(
            text="Test memory",
            type="fact",
            source="user",
            relations=None,
            project_id="test-project",
        )

    @pytest.mark.asyncio
    async def test_store_memory_error(self):
        """store_memory should return error dict on failure."""
        from simplemem_lite.mcp.server import store_memory

        mock_client = AsyncMock()
        mock_client.store_memory.side_effect = BackendError(500, "Internal error")

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await store_memory(text="Test")

        assert "error" in result
        assert "Internal error" in result["error"]

    @pytest.mark.asyncio
    async def test_search_memories_success(self):
        """search_memories should proxy to backend and return results."""
        from simplemem_lite.mcp.server import search_memories

        mock_client = AsyncMock()
        mock_client.search_memories.return_value = {
            "results": [{"uuid": "123", "content": "test"}]
        }

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await search_memories(query="test query", limit=5)

        assert "results" in result
        assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_search_memories_error(self):
        """search_memories should return empty results on error."""
        from simplemem_lite.mcp.server import search_memories

        mock_client = AsyncMock()
        mock_client.search_memories.side_effect = BackendError(500, "Error")

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await search_memories(query="test")

        assert "error" in result
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_relate_memories_success(self):
        """relate_memories should proxy to backend and return success."""
        from simplemem_lite.mcp.server import relate_memories

        mock_client = AsyncMock()
        mock_client.relate_memories.return_value = {"success": True}

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await relate_memories(
                from_id="uuid-1234-5678",
                to_id="uuid-8765-4321",
                relation_type="supports",
            )

        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_relate_memories_error(self):
        """relate_memories should return false on error."""
        from simplemem_lite.mcp.server import relate_memories

        mock_client = AsyncMock()
        mock_client.relate_memories.side_effect = BackendError(400, "Invalid")

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await relate_memories(from_id="id1", to_id="id2")

        assert "error" in result
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_ask_memories_success(self):
        """ask_memories should proxy to backend."""
        from simplemem_lite.mcp.server import ask_memories

        mock_client = AsyncMock()
        mock_client.ask_memories.return_value = {
            "answer": "Test answer",
            "memories_used": 3,
        }

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await ask_memories(query="What is the solution?")

        assert result["answer"] == "Test answer"

    @pytest.mark.asyncio
    async def test_reason_memories_success(self):
        """reason_memories should proxy to backend."""
        from simplemem_lite.mcp.server import reason_memories

        mock_client = AsyncMock()
        mock_client.reason_memories.return_value = {
            "conclusions": [{"uuid": "123", "score": 0.9}]
        }

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await reason_memories(query="Find patterns")

        assert "conclusions" in result

    @pytest.mark.asyncio
    async def test_get_stats_success(self):
        """get_stats should proxy to backend."""
        from simplemem_lite.mcp.server import get_stats

        mock_client = AsyncMock()
        mock_client.get_stats.return_value = {
            "total_memories": 100,
            "total_relations": 50,
        }

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await get_stats()

        assert result["total_memories"] == 100


class TestTraceTools:
    """Tests for trace tool proxies."""

    @pytest.mark.asyncio
    async def test_process_trace_success(self):
        """process_trace should read local file and send to backend."""
        from simplemem_lite.mcp.server import process_trace

        mock_client = AsyncMock()
        mock_client.process_trace.return_value = {
            "job_id": "job-123",
            "status": "submitted",
        }

        mock_reader = MagicMock()
        mock_reader.read_trace_file.return_value = [{"type": "user", "content": "test"}]

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client), \
             patch("simplemem_lite.mcp.server._get_reader", return_value=mock_reader), \
             patch("asyncio.to_thread", side_effect=lambda f, *args: f(*args)):
            result = await process_trace(session_id="session-123")

        assert result["job_id"] == "job-123"

    @pytest.mark.asyncio
    async def test_process_trace_not_found(self):
        """process_trace should return error if session not found."""
        from simplemem_lite.mcp.server import process_trace

        mock_reader = MagicMock()
        mock_reader.read_trace_file.return_value = None

        with patch("simplemem_lite.mcp.server._get_reader", return_value=mock_reader), \
             patch("asyncio.to_thread", side_effect=lambda f, *args: f(*args)):
            result = await process_trace(session_id="nonexistent")

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_discover_sessions_success(self):
        """discover_sessions should return session metadata."""
        from simplemem_lite.mcp.server import discover_sessions

        mock_reader = MagicMock()
        mock_reader.discover_sessions.return_value = [
            {"session_id": "sess-1", "project": "proj-1"},
            {"session_id": "sess-2", "project": "proj-1"},
        ]

        with patch("simplemem_lite.mcp.server._get_reader", return_value=mock_reader), \
             patch("asyncio.to_thread", side_effect=lambda f, *args: f(*args)):
            result = await discover_sessions(days_back=7)

        assert result["total_count"] == 2
        assert len(result["sessions"]) == 2

    @pytest.mark.asyncio
    async def test_discover_sessions_grouped(self):
        """discover_sessions should group by project when requested."""
        from simplemem_lite.mcp.server import discover_sessions

        mock_reader = MagicMock()
        mock_reader.discover_sessions.return_value = [
            {"session_id": "sess-1", "project": "proj-1"},
            {"session_id": "sess-2", "project": "proj-2"},
        ]

        with patch("simplemem_lite.mcp.server._get_reader", return_value=mock_reader), \
             patch("asyncio.to_thread", side_effect=lambda f, *args: f(*args)):
            result = await discover_sessions(days_back=7, group_by="project")

        assert "proj-1" in result["sessions"]
        assert "proj-2" in result["sessions"]


class TestJobTools:
    """Tests for job management tools."""

    @pytest.mark.asyncio
    async def test_job_status(self):
        """job_status should proxy to backend."""
        from simplemem_lite.mcp.server import job_status

        mock_client = AsyncMock()
        mock_client.get_job_status.return_value = {
            "id": "job-123",
            "status": "completed",
        }

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await job_status(job_id="job-123")

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_jobs(self):
        """list_jobs should proxy to backend."""
        from simplemem_lite.mcp.server import list_jobs

        mock_client = AsyncMock()
        mock_client.list_jobs.return_value = {
            "jobs": [{"id": "job-1"}, {"id": "job-2"}]
        }

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await list_jobs(limit=10)

        assert len(result["jobs"]) == 2

    @pytest.mark.asyncio
    async def test_cancel_job(self):
        """cancel_job should proxy to backend."""
        from simplemem_lite.mcp.server import cancel_job

        mock_client = AsyncMock()
        mock_client.cancel_job.return_value = {"cancelled": True}

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await cancel_job(job_id="job-123")

        assert result["cancelled"] is True


class TestCodeTools:
    """Tests for code tool proxies."""

    @pytest.mark.asyncio
    async def test_search_code(self):
        """search_code should proxy to backend."""
        from simplemem_lite.mcp.server import search_code

        mock_client = AsyncMock()
        mock_client.search_code.return_value = {
            "results": [{"chunk_id": "c1", "score": 0.9}]
        }

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await search_code(query="find auth code")

        assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_index_directory_success(self, tmp_path):
        """index_directory should read local files and send to backend."""
        from simplemem_lite.mcp.server import index_directory

        # Create test files
        (tmp_path / "main.py").write_text("print('hello')")

        mock_client = AsyncMock()
        mock_client.index_code.return_value = {
            "files_indexed": 1,
            "chunks_created": 5,
        }

        mock_reader = MagicMock()
        mock_reader.read_code_files.return_value = [
            {"path": "main.py", "content": "print('hello')"}
        ]

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client), \
             patch("simplemem_lite.mcp.server._get_reader", return_value=mock_reader), \
             patch("asyncio.to_thread", side_effect=lambda f, *args, **kwargs: f(*args, **kwargs)):
            result = await index_directory(path=str(tmp_path))

        assert result["files_indexed"] == 1

    @pytest.mark.asyncio
    async def test_index_directory_not_found(self):
        """index_directory should return error if directory not found."""
        from simplemem_lite.mcp.server import index_directory

        result = await index_directory(path="/nonexistent/path")

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_code_stats(self):
        """code_stats should proxy to backend."""
        from simplemem_lite.mcp.server import code_stats

        mock_client = AsyncMock()
        mock_client.code_stats.return_value = {
            "total_chunks": 100,
            "unique_files": 20,
        }

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await code_stats(project_root="/test")

        assert result["total_chunks"] == 100


class TestGraphTools:
    """Tests for graph tool proxies."""

    @pytest.mark.asyncio
    async def test_get_graph_schema(self):
        """get_graph_schema should proxy to backend."""
        from simplemem_lite.mcp.server import get_graph_schema

        mock_client = AsyncMock()
        mock_client.get_graph_schema.return_value = {
            "node_labels": ["Memory", "Entity"],
            "relationship_types": ["RELATES_TO"],
        }

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await get_graph_schema()

        assert "node_labels" in result

    @pytest.mark.asyncio
    async def test_run_cypher_query(self):
        """run_cypher_query should proxy to backend."""
        from simplemem_lite.mcp.server import run_cypher_query

        mock_client = AsyncMock()
        mock_client.run_cypher_query.return_value = {
            "results": [{"uuid": "123"}],
            "row_count": 1,
        }

        with patch("simplemem_lite.mcp.server._get_client", return_value=mock_client):
            result = await run_cypher_query(
                query="MATCH (m:Memory) RETURN m.uuid LIMIT 10"
            )

        assert result["row_count"] == 1


class TestProjectTools:
    """Tests for project tool proxies."""

    @pytest.mark.asyncio
    async def test_get_project_status(self, tmp_path):
        """get_project_status should return directory info."""
        from simplemem_lite.mcp.server import get_project_status

        # Create .git dir
        (tmp_path / ".git").mkdir()
        (tmp_path / "main.py").write_text("print('hello')")

        mock_reader = MagicMock()
        mock_reader.get_directory_info.return_value = {
            "exists": True,
            "is_git": True,
            "file_count": 1,
        }

        with patch("simplemem_lite.mcp.server._get_reader", return_value=mock_reader), \
             patch("asyncio.to_thread", side_effect=lambda f, *args: f(*args)):
            result = await get_project_status(project_root=str(tmp_path))

        assert result["exists"] is True
        assert result["is_git"] is True


class TestThreadSafety:
    """Tests for thread-safe initialization."""

    @pytest.mark.asyncio
    async def test_get_client_creates_once(self):
        """_get_client should create client only once."""
        from simplemem_lite.mcp.server import _get_client
        import simplemem_lite.mcp.server as server_module

        # Reset state
        server_module._client = None

        with patch("simplemem_lite.mcp.server.BackendClient") as MockClient:
            MockClient.return_value = MagicMock()

            # Get client twice
            client1 = await _get_client()
            client2 = await _get_client()

            # Should be same instance
            assert client1 is client2
            # Constructor should only be called once
            assert MockClient.call_count == 1

    @pytest.mark.asyncio
    async def test_get_reader_creates_once(self):
        """_get_reader should create reader only once."""
        from simplemem_lite.mcp.server import _get_reader
        import simplemem_lite.mcp.server as server_module

        # Reset state
        server_module._reader = None

        with patch("simplemem_lite.mcp.server.LocalReader") as MockReader:
            MockReader.return_value = MagicMock()

            # Get reader twice
            reader1 = await _get_reader()
            reader2 = await _get_reader()

            # Should be same instance
            assert reader1 is reader2
            # Constructor should only be called once
            assert MockReader.call_count == 1
