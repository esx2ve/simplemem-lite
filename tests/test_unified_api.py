"""Tests for V2 Unified API endpoints.

Tests cover:
1. /v2/remember - Store memories
2. /v2/recall - Find memories (fast, deep, ask modes)
3. /v2/index - Index code files and traces
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_memory_store():
    """Mock memory store for testing."""
    store = MagicMock()
    store.store = MagicMock(return_value="test-uuid-123")
    store.search = MagicMock(return_value=[])
    store.ask_memories = AsyncMock(return_value={"answer": "Test answer", "sources": []})
    return store


@pytest.fixture
def mock_code_indexer():
    """Mock code indexer for testing."""
    indexer = MagicMock()
    indexer.index_files_content_async = AsyncMock(return_value={"files_indexed": 1, "chunks_created": 5})
    return indexer


@pytest.fixture
def mock_job_manager():
    """Mock job manager for testing."""
    manager = MagicMock()
    manager.submit = AsyncMock(return_value="job-uuid-123")
    return manager


# =============================================================================
# REMEMBER ENDPOINT TESTS
# =============================================================================


class TestRememberEndpoint:
    """Tests for /v2/remember endpoint."""

    @pytest.mark.asyncio
    async def test_remember_basic(self, mock_memory_store):
        """Test basic memory storage."""
        from simplemem_lite.backend.api.unified import remember, RememberRequest

        with patch("simplemem_lite.backend.api.unified.get_memory_store", return_value=mock_memory_store):
            with patch("simplemem_lite.backend.api.unified.get_config") as mock_config:
                mock_config.return_value.require_project_id = False

                request = RememberRequest(
                    content="Test content",
                    project="config:test",
                    type="fact",
                )
                result = await remember(request)

                assert result.uuid == "test-uuid-123"
                assert result.relations_created == 0

    @pytest.mark.asyncio
    async def test_remember_with_relations(self, mock_memory_store):
        """Test memory storage with relations."""
        from simplemem_lite.backend.api.unified import remember, RememberRequest

        with patch("simplemem_lite.backend.api.unified.get_memory_store", return_value=mock_memory_store):
            with patch("simplemem_lite.backend.api.unified.get_config") as mock_config:
                mock_config.return_value.require_project_id = False

                request = RememberRequest(
                    content="Test content",
                    project="config:test",
                    type="decision",
                    relations=["related-uuid-1", "related-uuid-2"],
                )
                result = await remember(request)

                assert result.uuid == "test-uuid-123"
                assert result.relations_created == 2

    @pytest.mark.asyncio
    async def test_remember_requires_project_id_when_configured(self):
        """Test that project_id is required when configured."""
        from simplemem_lite.backend.api.unified import remember, RememberRequest

        with patch("simplemem_lite.backend.api.unified.get_config") as mock_config:
            mock_config.return_value.require_project_id = True

            request = RememberRequest(
                content="Test content",
                type="fact",
            )

            with pytest.raises(HTTPException) as exc_info:
                await remember(request)

            assert exc_info.value.status_code == 400
            assert "project_id is required" in exc_info.value.detail


# =============================================================================
# RECALL ENDPOINT TESTS
# =============================================================================


class TestRecallEndpoint:
    """Tests for /v2/recall endpoint."""

    @pytest.mark.asyncio
    async def test_recall_requires_query_or_id(self):
        """Test that either query or id is required."""
        from simplemem_lite.backend.api.unified import recall, RecallRequest

        with patch("simplemem_lite.backend.api.unified.get_config") as mock_config:
            mock_config.return_value.require_project_id = False

            request = RecallRequest(project="config:test")

            with pytest.raises(HTTPException) as exc_info:
                await recall(request)

            assert exc_info.value.status_code == 400
            assert "Either 'query' or 'id' is required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_recall_fast_mode(self, mock_memory_store):
        """Test recall in fast mode (vector search)."""
        from simplemem_lite.backend.api.unified import recall, RecallRequest
        from simplemem_lite.memory import Memory

        mock_memory = MagicMock()
        mock_memory.uuid = "result-uuid"
        mock_memory.content = "Test content"
        mock_memory.type = "fact"
        mock_memory.score = 0.9
        mock_memory_store.search = MagicMock(return_value=[mock_memory])

        with patch("simplemem_lite.backend.api.unified.get_memory_store", return_value=mock_memory_store):
            with patch("simplemem_lite.backend.api.unified.get_config") as mock_config:
                mock_config.return_value.require_project_id = False

                request = RecallRequest(
                    query="test query",
                    project="config:test",
                    mode="fast",
                    limit=10,
                )
                result = await recall(request)

                assert "results" in result
                assert len(result["results"]) == 1
                assert result["results"][0]["uuid"] == "result-uuid"

    @pytest.mark.asyncio
    async def test_recall_ask_mode(self, mock_memory_store):
        """Test recall in ask mode (LLM synthesis)."""
        from simplemem_lite.backend.api.unified import recall, RecallRequest

        mock_memory_store.ask_memories = AsyncMock(return_value={
            "answer": "Synthesized answer",
            "memories_used": 3,
            "confidence": "high",
            "sources": [],
        })

        with patch("simplemem_lite.backend.api.unified.get_memory_store", return_value=mock_memory_store):
            with patch("simplemem_lite.backend.api.unified.get_config") as mock_config:
                mock_config.return_value.require_project_id = False

                request = RecallRequest(
                    query="How did we fix the bug?",
                    project="config:test",
                    mode="ask",
                )
                result = await recall(request)

                assert "answer" in result
                assert result["answer"] == "Synthesized answer"
                assert result["confidence"] == "high"


# =============================================================================
# INDEX ENDPOINT TESTS
# =============================================================================


class TestIndexEndpoint:
    """Tests for /v2/index endpoint."""

    @pytest.mark.asyncio
    async def test_index_requires_files_or_traces(self):
        """Test that either files or traces is required."""
        from simplemem_lite.backend.api.unified import index, IndexRequest

        with patch("simplemem_lite.backend.api.unified.get_config") as mock_config:
            mock_config.return_value.require_project_id = False

            request = IndexRequest(project="config:test")

            with pytest.raises(HTTPException) as exc_info:
                await index(request)

            assert exc_info.value.status_code == 400
            assert "Either 'files' or 'traces' is required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_index_cannot_mix_files_and_traces(self):
        """Test that files and traces cannot be mixed."""
        from simplemem_lite.backend.api.unified import index, IndexRequest, FileInput, TraceInput

        with patch("simplemem_lite.backend.api.unified.get_config") as mock_config:
            mock_config.return_value.require_project_id = False

            request = IndexRequest(
                project="config:test",
                files=[FileInput(path="test.py", content="print('hello')")],
                traces=[TraceInput(session_id="abc-123", content={"messages": []})],
            )

            with pytest.raises(HTTPException) as exc_info:
                await index(request)

            assert exc_info.value.status_code == 400
            assert "Cannot index both" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_index_files_background(self, mock_code_indexer, mock_job_manager):
        """Test code file indexing in background mode."""
        from simplemem_lite.backend.api.unified import index, IndexRequest, FileInput

        with patch("simplemem_lite.backend.api.unified.get_code_indexer", return_value=mock_code_indexer):
            with patch("simplemem_lite.backend.api.unified.get_job_manager", return_value=mock_job_manager):
                with patch("simplemem_lite.backend.api.unified.get_config") as mock_config:
                    mock_config.return_value.require_project_id = False

                    request = IndexRequest(
                        project="config:test",
                        files=[FileInput(path="test.py", content="print('hello')")],
                        wait=False,
                    )
                    result = await index(request)

                    assert result.status == "submitted"
                    assert result.job_id == "job-uuid-123"

    @pytest.mark.asyncio
    async def test_index_files_sync(self, mock_code_indexer):
        """Test code file indexing in sync mode."""
        from simplemem_lite.backend.api.unified import index, IndexRequest, FileInput

        with patch("simplemem_lite.backend.api.unified.get_code_indexer", return_value=mock_code_indexer):
            with patch("simplemem_lite.backend.api.unified.get_config") as mock_config:
                mock_config.return_value.require_project_id = False

                request = IndexRequest(
                    project="config:test",
                    files=[FileInput(path="test.py", content="print('hello')")],
                    wait=True,
                )
                result = await index(request)

                assert result.status == "completed"
                assert result.files_indexed == 1
                assert result.chunks_created == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
