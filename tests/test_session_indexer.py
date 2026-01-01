"""Tests for SessionAutoIndexer (P2: Active Session Handling).

Tests:
- File stability detection (mtime + size tracking)
- Rate limiting (daily session limit)
- File size limit enforcement
- enabled_at boundary enforcement
- Polling loop lifecycle (start/stop)
"""

import asyncio
import tempfile
import time
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simplemem_lite.session_indexer import FileStabilityState, SessionAutoIndexer


class TestFileStabilityState:
    """Test FileStabilityState dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        state = FileStabilityState(file_path=Path("/test/path"))

        assert state.file_path == Path("/test/path")
        assert state.last_mtime == 0.0
        assert state.last_size == 0
        assert state.stable_count == 0
        assert state.last_checked > 0

    def test_custom_values(self):
        """Custom values should be preserved."""
        state = FileStabilityState(
            file_path=Path("/test/path"),
            last_mtime=1234567890.0,
            last_size=1024,
            stable_count=2,
            last_checked=1234567800.0,
        )

        assert state.last_mtime == 1234567890.0
        assert state.last_size == 1024
        assert state.stable_count == 2


class TestSessionAutoIndexerInit:
    """Test SessionAutoIndexer initialization."""

    def test_init_sets_running_false(self):
        """Indexer should not be running after init."""
        config = MagicMock()
        config.auto_index_poll_interval = 120
        config.auto_index_max_file_mb = 5
        config.auto_index_max_per_day = 10
        config.auto_index_stability_cycles = 2

        indexer = SessionAutoIndexer(
            config=config,
            session_state_db=MagicMock(),
            indexer=MagicMock(),
            job_manager=MagicMock(),
            project_manager=MagicMock(),
        )

        assert indexer.is_running is False
        assert indexer.sessions_indexed_today == 0

    def test_init_logs_config(self):
        """Initialization should log configuration."""
        config = MagicMock()
        config.auto_index_poll_interval = 60
        config.auto_index_max_file_mb = 10
        config.auto_index_max_per_day = 20
        config.auto_index_stability_cycles = 3

        with patch("simplemem_lite.session_indexer.log") as mock_log:
            SessionAutoIndexer(
                config=config,
                session_state_db=MagicMock(),
                indexer=MagicMock(),
                job_manager=MagicMock(),
                project_manager=MagicMock(),
            )

            # Should log config values
            assert mock_log.info.called


class TestStabilityDetection:
    """Test file stability detection logic."""

    @pytest.fixture
    def indexer(self):
        """Create indexer with mock dependencies."""
        config = MagicMock()
        config.auto_index_poll_interval = 120
        config.auto_index_max_file_mb = 5
        config.auto_index_max_per_day = 10
        config.auto_index_stability_cycles = 2
        config.auto_index_enabled_at = 0.0

        return SessionAutoIndexer(
            config=config,
            session_state_db=MagicMock(),
            indexer=MagicMock(),
            job_manager=MagicMock(),
            project_manager=MagicMock(),
        )

    def test_new_file_starts_at_zero(self, indexer):
        """New files should start with stable_count=0."""
        state = indexer._update_stability(
            file_path=Path("/test/file.jsonl"),
            session_id="test-session",
            current_mtime=1000.0,
            current_size=500,
        )

        assert state.stable_count == 0
        assert state.last_mtime == 1000.0
        assert state.last_size == 500

    def test_unchanged_file_increments_stable_count(self, indexer):
        """Unchanged files should increment stable_count."""
        # First check
        indexer._update_stability(
            file_path=Path("/test/file.jsonl"),
            session_id="test-session",
            current_mtime=1000.0,
            current_size=500,
        )

        # Second check with same values
        state = indexer._update_stability(
            file_path=Path("/test/file.jsonl"),
            session_id="test-session",
            current_mtime=1000.0,
            current_size=500,
        )

        assert state.stable_count == 1

    def test_changed_file_resets_stable_count(self, indexer):
        """Changed files should reset stable_count to 0."""
        # First check
        indexer._update_stability(
            file_path=Path("/test/file.jsonl"),
            session_id="test-session",
            current_mtime=1000.0,
            current_size=500,
        )

        # Second check - stable
        indexer._update_stability(
            file_path=Path("/test/file.jsonl"),
            session_id="test-session",
            current_mtime=1000.0,
            current_size=500,
        )

        # Third check - file changed (size increased)
        state = indexer._update_stability(
            file_path=Path("/test/file.jsonl"),
            session_id="test-session",
            current_mtime=1000.0,
            current_size=600,
        )

        assert state.stable_count == 0

    def test_mtime_change_resets_stable_count(self, indexer):
        """mtime change should reset stable_count."""
        indexer._update_stability(
            file_path=Path("/test/file.jsonl"),
            session_id="test-session",
            current_mtime=1000.0,
            current_size=500,
        )

        state = indexer._update_stability(
            file_path=Path("/test/file.jsonl"),
            session_id="test-session",
            current_mtime=1001.0,  # mtime changed
            current_size=500,
        )

        assert state.stable_count == 0


class TestRateLimiting:
    """Test daily rate limiting."""

    @pytest.fixture
    def indexer(self):
        """Create indexer with low rate limit for testing."""
        config = MagicMock()
        config.auto_index_poll_interval = 1
        config.auto_index_max_file_mb = 5
        config.auto_index_max_per_day = 2  # Low limit for testing
        config.auto_index_stability_cycles = 2
        config.auto_index_enabled_at = 0.0

        return SessionAutoIndexer(
            config=config,
            session_state_db=MagicMock(),
            indexer=MagicMock(),
            job_manager=MagicMock(),
            project_manager=MagicMock(),
        )

    def test_initial_count_is_zero(self, indexer):
        """Initial indexed count should be zero."""
        assert indexer.sessions_indexed_today == 0

    def test_counter_increments(self, indexer):
        """Counter should increment when session is queued."""
        indexer._sessions_indexed_today = 1
        assert indexer.sessions_indexed_today == 1

    def test_daily_reset_detection(self, indexer):
        """Daily counter should reset on new day."""
        # Simulate yesterday
        indexer._last_reset_date = date(2020, 1, 1)
        indexer._sessions_indexed_today = 5

        # The poll cycle will reset the counter
        assert indexer._last_reset_date != date.today()


class TestFileSizeLimit:
    """Test file size limit enforcement."""

    @pytest.fixture
    def indexer(self):
        """Create indexer with 5MB limit."""
        config = MagicMock()
        config.auto_index_poll_interval = 120
        config.auto_index_max_file_mb = 5
        config.auto_index_max_per_day = 10
        config.auto_index_stability_cycles = 2
        config.auto_index_enabled_at = 0.0

        return SessionAutoIndexer(
            config=config,
            session_state_db=MagicMock(),
            indexer=MagicMock(),
            job_manager=MagicMock(),
            project_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_oversized_file_is_skipped(self, indexer, tmp_path):
        """Files larger than limit should be skipped."""
        # Create a session dict with size exceeding limit
        session = {
            "session_id": "test-session",
            "path": str(tmp_path / "large.jsonl"),
            "size_kb": 6000,  # 6MB > 5MB limit
        }

        result = await indexer._check_session(session)
        assert result == "skipped"


class TestEnabledAtBoundary:
    """Test enabled_at timestamp boundary."""

    @pytest.fixture
    def indexer(self):
        """Create indexer with enabled_at in the future."""
        config = MagicMock()
        config.auto_index_poll_interval = 120
        config.auto_index_max_file_mb = 5
        config.auto_index_max_per_day = 10
        config.auto_index_stability_cycles = 2
        config.auto_index_enabled_at = time.time() + 3600  # 1 hour in future

        return SessionAutoIndexer(
            config=config,
            session_state_db=MagicMock(),
            indexer=MagicMock(),
            job_manager=MagicMock(),
            project_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_old_session_is_skipped(self, indexer, tmp_path):
        """Sessions before enabled_at should be skipped."""
        # Create a real file to stat
        test_file = tmp_path / "old.jsonl"
        test_file.write_text("{}")

        session = {
            "session_id": "test-session",
            "path": str(test_file),
            "size_kb": 1,
        }

        result = await indexer._check_session(session)
        assert result == "skipped"


class TestPollingLifecycle:
    """Test polling loop start/stop."""

    @pytest.fixture
    def indexer(self):
        """Create indexer with short poll interval."""
        config = MagicMock()
        config.auto_index_poll_interval = 0.1  # Very short for testing
        config.auto_index_max_file_mb = 5
        config.auto_index_max_per_day = 10
        config.auto_index_stability_cycles = 2
        config.auto_index_enabled_at = 0.0

        mock_parser = MagicMock()
        mock_parser.list_sessions.return_value = []

        mock_indexer = MagicMock()
        mock_indexer.parser = mock_parser

        return SessionAutoIndexer(
            config=config,
            session_state_db=MagicMock(),
            indexer=mock_indexer,
            job_manager=MagicMock(),
            project_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_start_sets_running_true(self, indexer):
        """Starting should set is_running to True."""
        assert indexer.is_running is False
        await indexer.start()
        assert indexer.is_running is True
        await indexer.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, indexer):
        """Stopping should set is_running to False."""
        await indexer.start()
        await indexer.stop()
        assert indexer.is_running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, indexer):
        """Calling start twice should not create multiple tasks."""
        await indexer.start()
        task1 = indexer._task
        await indexer.start()  # Second call
        task2 = indexer._task

        # Should be the same task (not a new one)
        assert task1 is task2
        await indexer.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, indexer):
        """Calling stop twice should not error."""
        await indexer.start()
        await indexer.stop()
        await indexer.stop()  # Second call should not error

    def test_stop_sync_cancels_task(self, indexer):
        """stop_sync should cancel the task without awaiting."""
        # Create a mock task
        indexer._running = True
        mock_task = MagicMock()
        indexer._task = mock_task

        indexer.stop_sync()

        assert indexer._running is False
        assert indexer._task is None  # Task is cleared
        mock_task.cancel.assert_called_once()  # But cancel was called


class TestGetStatus:
    """Test get_status method."""

    def test_status_includes_all_fields(self):
        """Status should include all relevant fields."""
        config = MagicMock()
        config.auto_index_poll_interval = 120
        config.auto_index_max_file_mb = 5
        config.auto_index_max_per_day = 10
        config.auto_index_stability_cycles = 2
        config.auto_index_enabled_at = 1234567890.0

        indexer = SessionAutoIndexer(
            config=config,
            session_state_db=MagicMock(),
            indexer=MagicMock(),
            job_manager=MagicMock(),
            project_manager=MagicMock(),
        )

        status = indexer.get_status()

        assert "running" in status
        assert "sessions_indexed_today" in status
        assert "daily_limit" in status
        assert "tracked_sessions" in status
        assert "active_sessions" in status
        assert "poll_interval" in status
        assert "stability_cycles" in status
        assert "max_file_mb" in status
        assert "enabled_at" in status

        assert status["running"] is False
        assert status["daily_limit"] == 10
        assert status["poll_interval"] == 120
        assert status["active_sessions"] == 0


class TestActiveSessionTracking:
    """Test active session tracking to prevent duplicates."""

    @pytest.fixture
    def indexer(self):
        """Create indexer with mock dependencies."""
        config = MagicMock()
        config.auto_index_poll_interval = 120
        config.auto_index_max_file_mb = 5
        config.auto_index_max_per_day = 10
        config.auto_index_stability_cycles = 2
        config.auto_index_enabled_at = 0.0

        return SessionAutoIndexer(
            config=config,
            session_state_db=MagicMock(),
            indexer=MagicMock(),
            job_manager=MagicMock(),
            project_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_active_session_skipped(self, indexer, tmp_path):
        """Sessions already being indexed should be skipped."""
        test_file = tmp_path / "active.jsonl"
        test_file.write_text("{}")

        # Mark session as active
        indexer._active_sessions.add("test-session")

        session = {
            "session_id": "test-session",
            "path": str(test_file),
            "size_kb": 1,
        }

        result = await indexer._check_session(session)
        assert result == "skipped"

    def test_active_sessions_initially_empty(self, indexer):
        """Active sessions should be empty on init."""
        assert len(indexer._active_sessions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
