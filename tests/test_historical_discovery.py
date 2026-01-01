"""Tests for Historical Session Discovery (P3).

Tests:
- discover_sessions filtering by days_back
- discover_sessions grouping by project/date
- discover_sessions include_indexed filter
- index_sessions_batch with session_ids
- index_sessions_batch with days_back
- index_sessions_batch respects max_sessions
"""

import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDiscoverSessions:
    """Test discover_sessions MCP tool."""

    @pytest.fixture
    def mock_deps(self):
        """Create mock dependencies."""
        deps = MagicMock()

        # Mock TraceParser.list_sessions()
        now = time.time()
        deps.indexer.parser.list_sessions.return_value = [
            {
                "session_id": "session-1",
                "project": "project-a",
                "path": "/path/to/session-1.jsonl",
                "size_kb": 100,
                "modified": now - (1 * 24 * 60 * 60),  # 1 day ago
            },
            {
                "session_id": "session-2",
                "project": "project-a",
                "path": "/path/to/session-2.jsonl",
                "size_kb": 200,
                "modified": now - (5 * 24 * 60 * 60),  # 5 days ago
            },
            {
                "session_id": "session-3",
                "project": "project-b",
                "path": "/path/to/session-3.jsonl",
                "size_kb": 150,
                "modified": now - (10 * 24 * 60 * 60),  # 10 days ago
            },
            {
                "session_id": "session-4",
                "project": "project-b",
                "path": "/path/to/session-4.jsonl",
                "size_kb": 50,
                "modified": now - (40 * 24 * 60 * 60),  # 40 days ago
            },
        ]

        # Mock SessionStateDB.list_sessions() for batch fetch
        def mock_list_sessions(status=None, limit=100):
            if status == "indexed":
                # Return list of mock SessionState objects
                indexed_state = MagicMock()
                indexed_state.session_id = "session-1"
                return [indexed_state]
            return []

        deps.session_state_db.list_sessions = mock_list_sessions

        return deps

    @pytest.mark.asyncio
    async def test_filters_by_days_back(self, mock_deps):
        """Sessions older than days_back should be excluded."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import discover_sessions

            result = await discover_sessions(days_back=7)

            # Only sessions 1 and 2 are within 7 days
            assert result["total_count"] == 2
            session_ids = [s["session_id"] for s in result["sessions"]]
            assert "session-1" in session_ids
            assert "session-2" in session_ids
            assert "session-3" not in session_ids
            assert "session-4" not in session_ids

    @pytest.mark.asyncio
    async def test_groups_by_project(self, mock_deps):
        """Sessions should be grouped by project when requested."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import discover_sessions

            result = await discover_sessions(days_back=30, group_by="project")

            sessions = result["sessions"]
            assert isinstance(sessions, dict)
            assert "project-a" in sessions
            assert "project-b" in sessions
            assert len(sessions["project-a"]) == 2
            assert len(sessions["project-b"]) == 1  # session-4 is > 30 days

    @pytest.mark.asyncio
    async def test_groups_by_date(self, mock_deps):
        """Sessions should be grouped by date when requested."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import discover_sessions

            result = await discover_sessions(days_back=30, group_by="date")

            sessions = result["sessions"]
            assert isinstance(sessions, dict)
            # Each session is on a different date
            assert len(sessions) >= 1

    @pytest.mark.asyncio
    async def test_excludes_indexed_when_requested(self, mock_deps):
        """Already-indexed sessions should be excluded when include_indexed=False."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import discover_sessions

            result = await discover_sessions(days_back=30, include_indexed=False)

            session_ids = [s["session_id"] for s in result["sessions"]]
            assert "session-1" not in session_ids  # This one is indexed
            assert "session-2" in session_ids
            assert "session-3" in session_ids

    @pytest.mark.asyncio
    async def test_includes_indexed_count(self, mock_deps):
        """Result should include indexed and unindexed counts."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import discover_sessions

            result = await discover_sessions(days_back=30)

            assert "indexed_count" in result
            assert "unindexed_count" in result
            assert result["indexed_count"] == 1  # session-1
            assert result["unindexed_count"] == 2  # session-2, session-3


class TestIndexSessionsBatch:
    """Test index_sessions_batch MCP tool."""

    @pytest.fixture
    def mock_deps(self):
        """Create mock dependencies."""
        deps = MagicMock()

        # Mock TraceParser.list_sessions()
        now = time.time()
        deps.indexer.parser.list_sessions.return_value = [
            {
                "session_id": "session-1",
                "project": "project-a",
                "path": "/path/to/session-1.jsonl",
                "size_kb": 100,
                "modified": now - (1 * 24 * 60 * 60),
            },
            {
                "session_id": "session-2",
                "project": "project-a",
                "path": "/path/to/session-2.jsonl",
                "size_kb": 200,
                "modified": now - (5 * 24 * 60 * 60),
            },
            {
                "session_id": "session-3",
                "project": "project-b",
                "path": "/path/to/session-3.jsonl",
                "size_kb": 150,
                "modified": now - (10 * 24 * 60 * 60),
            },
        ]

        # Mock SessionStateDB.list_sessions() for batch fetch
        def mock_list_sessions(status=None, limit=100):
            if status == "indexed":
                # Return list of mock SessionState objects
                indexed_state = MagicMock()
                indexed_state.session_id = "session-1"
                return [indexed_state]
            return []

        deps.session_state_db.list_sessions = mock_list_sessions

        # Mock project manager
        deps.project_manager.detect_project_root.return_value = "/detected/root"

        # Mock job manager
        deps.job_manager.submit_job = AsyncMock(return_value="job-123")

        # Mock indexer
        deps.indexer.index_session_delta = AsyncMock(return_value={"processed": 10})

        return deps

    @pytest.mark.asyncio
    async def test_requires_session_ids_or_days_back(self, mock_deps):
        """Should return error if neither session_ids nor days_back provided."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import index_sessions_batch

            result = await index_sessions_batch()

            assert "error" in result
            assert "Must provide either session_ids or days_back" in result["error"]

    @pytest.mark.asyncio
    async def test_queues_specific_sessions(self, mock_deps):
        """Should queue only the specified session IDs."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import index_sessions_batch

            result = await index_sessions_batch(session_ids=["session-2", "session-3"])

            assert len(result["queued"]) == 2
            assert "session-2" in result["queued"]
            assert "session-3" in result["queued"]

    @pytest.mark.asyncio
    async def test_skips_indexed_sessions(self, mock_deps):
        """Should skip already-indexed sessions."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import index_sessions_batch

            result = await index_sessions_batch(
                session_ids=["session-1", "session-2"],
                skip_indexed=True,
            )

            assert "session-1" in result["skipped"]  # Already indexed
            assert "session-2" in result["queued"]

    @pytest.mark.asyncio
    async def test_respects_max_sessions(self, mock_deps):
        """Should stop after queuing max_sessions."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import index_sessions_batch

            result = await index_sessions_batch(days_back=30, max_sessions=1)

            # Should queue at most 1 session (session-1 is indexed, so session-2 or session-3)
            assert len(result["queued"]) <= 1

    @pytest.mark.asyncio
    async def test_returns_job_ids(self, mock_deps):
        """Should return job IDs for queued sessions."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import index_sessions_batch

            result = await index_sessions_batch(session_ids=["session-2"])

            assert "job_ids" in result
            assert "session-2" in result["job_ids"]
            assert result["job_ids"]["session-2"] == "job-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
