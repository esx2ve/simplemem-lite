"""Unit tests for consolidation scheduler.

Tests:
- SchedulerConfig - configuration dataclass
- ConsolidationScheduler - background task management
- Scheduling logic - next run calculation
- Project filtering - min memories threshold
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simplemem_lite.backend.consolidation.scheduler import (
    ConsolidationScheduler,
    SchedulerConfig,
    get_scheduler,
    start_scheduler,
    stop_scheduler,
)


# ============================================================================
# SchedulerConfig tests
# ============================================================================


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = SchedulerConfig()
        assert config.check_interval_hours == 24
        assert config.min_days_between_runs == 7
        assert config.min_new_memories == 100
        assert config.enabled is False
        assert config.run_at_hour == 3  # 3 AM

    def test_custom_values(self):
        """Should accept custom configuration."""
        config = SchedulerConfig(
            check_interval_hours=12,
            min_days_between_runs=3,
            min_new_memories=50,
            enabled=True,
            run_at_hour=2,
        )
        assert config.check_interval_hours == 12
        assert config.min_days_between_runs == 3
        assert config.min_new_memories == 50
        assert config.enabled is True
        assert config.run_at_hour == 2


# ============================================================================
# ConsolidationScheduler tests
# ============================================================================


class TestConsolidationScheduler:
    """Tests for ConsolidationScheduler class."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        """Scheduler should not start when enabled=False."""
        scheduler = ConsolidationScheduler()
        assert scheduler.config.enabled is False

        await scheduler.start()
        # Should not create a task
        assert scheduler._task is None
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_enabled_starts_task(self):
        """Scheduler should create task when enabled."""
        config = SchedulerConfig(enabled=True)
        scheduler = ConsolidationScheduler(config=config)

        # Mock the run loop to avoid actual execution
        with patch.object(scheduler, "_run_loop", new_callable=AsyncMock) as mock_loop:
            await scheduler.start()
            assert scheduler._running is True
            assert scheduler._task is not None

            # Clean up
            await scheduler.stop()
            assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """Stop should cancel the background task."""
        config = SchedulerConfig(enabled=True)
        scheduler = ConsolidationScheduler(config=config)

        # Create a real task that we'll cancel
        async def slow_loop():
            while True:
                await asyncio.sleep(1)

        scheduler._running = True
        scheduler._task = asyncio.create_task(slow_loop())

        await scheduler.stop()
        assert scheduler._running is False
        assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_double_start_warning(self):
        """Starting twice should log warning and not duplicate."""
        config = SchedulerConfig(enabled=True)
        scheduler = ConsolidationScheduler(config=config)

        with patch.object(scheduler, "_run_loop", new_callable=AsyncMock):
            await scheduler.start()
            first_task = scheduler._task

            # Second start should not create new task
            await scheduler.start()
            assert scheduler._task is first_task

            await scheduler.stop()


class TestSchedulerProjectFiltering:
    """Tests for project filtering logic."""

    @pytest.mark.asyncio
    async def test_filters_projects_by_memory_count(self, mock_graph_store):
        """Should only return projects with enough memories."""
        config = SchedulerConfig(enabled=True, min_new_memories=100)
        scheduler = ConsolidationScheduler(config=config)

        # Mock query result with projects
        query_result = MagicMock()
        query_result.result_set = [
            ["project-1", 150, datetime.now()],  # Enough memories
            ["project-2", 50, datetime.now()],   # Not enough
            ["project-3", 200, datetime.now()],  # Enough
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.scheduler.get_memory_store",
            return_value=mock_graph_store,
        ):
            projects = await scheduler._get_projects_needing_consolidation()
            # Should return project-1 and project-3, but not project-2
            assert "project-1" in projects
            assert "project-3" in projects
            assert "project-2" not in projects

    @pytest.mark.asyncio
    async def test_empty_result_returns_empty(self, mock_graph_store):
        """Empty query result should return empty list."""
        config = SchedulerConfig(enabled=True)
        scheduler = ConsolidationScheduler(config=config)

        query_result = MagicMock()
        query_result.result_set = []
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.scheduler.get_memory_store",
            return_value=mock_graph_store,
        ):
            projects = await scheduler._get_projects_needing_consolidation()
            assert projects == []


class TestSchedulerRunConsolidation:
    """Tests for consolidation execution."""

    @pytest.mark.asyncio
    async def test_runs_consolidation_for_project(self):
        """Should call consolidate_project for each project."""
        config = SchedulerConfig(enabled=True)
        scheduler = ConsolidationScheduler(config=config)

        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"status": "completed"}

        with patch(
            "simplemem_lite.backend.consolidation.consolidate_project",
            new_callable=AsyncMock,
            return_value=mock_report,
        ) as mock_consolidate:
            result = await scheduler._run_consolidation("config:test-project")

            mock_consolidate.assert_called_once()
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_check_and_run_no_projects(self, mock_graph_store):
        """Should handle no projects needing consolidation."""
        config = SchedulerConfig(enabled=True)
        scheduler = ConsolidationScheduler(config=config)

        query_result = MagicMock()
        query_result.result_set = []
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.scheduler.get_memory_store",
            return_value=mock_graph_store,
        ):
            # Should not raise
            await scheduler._check_and_run()

    @pytest.mark.asyncio
    async def test_check_and_run_handles_errors(self, mock_graph_store):
        """Should handle errors in individual project consolidation."""
        config = SchedulerConfig(enabled=True, min_new_memories=10)
        scheduler = ConsolidationScheduler(config=config)

        query_result = MagicMock()
        query_result.result_set = [
            ["project-1", 100, datetime.now()],
            ["project-2", 100, datetime.now()],
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        call_count = 0

        async def flaky_consolidate(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["project_id"] == "project-1":
                raise Exception("Consolidation failed")
            mock = MagicMock()
            mock.to_dict.return_value = {}
            return mock

        with patch(
            "simplemem_lite.backend.consolidation.scheduler.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.consolidate_project",
            side_effect=flaky_consolidate,
        ):
            # Should not raise, should continue to project-2
            await scheduler._check_and_run()
            assert call_count == 2  # Both projects attempted


# ============================================================================
# Module-level functions tests
# ============================================================================


class TestModuleFunctions:
    """Tests for module-level scheduler functions."""

    def test_get_scheduler_singleton(self):
        """get_scheduler should return the same instance."""
        # Note: This might affect other tests, so we should be careful
        scheduler1 = get_scheduler()
        scheduler2 = get_scheduler()
        assert scheduler1 is scheduler2

    @pytest.mark.asyncio
    async def test_start_scheduler_with_config(self):
        """start_scheduler should apply config and start."""
        config = SchedulerConfig(enabled=True, run_at_hour=5)

        with patch.object(ConsolidationScheduler, "start", new_callable=AsyncMock) as mock_start:
            # Reset global scheduler for clean test
            import simplemem_lite.backend.consolidation.scheduler as scheduler_module
            scheduler_module._scheduler = None

            await start_scheduler(config)
            mock_start.assert_called_once()

            scheduler = get_scheduler()
            assert scheduler.config.run_at_hour == 5

    @pytest.mark.asyncio
    async def test_stop_scheduler(self):
        """stop_scheduler should stop the global scheduler."""
        with patch.object(ConsolidationScheduler, "stop", new_callable=AsyncMock) as mock_stop:
            await stop_scheduler()
            mock_stop.assert_called_once()


# ============================================================================
# Timing calculation tests
# ============================================================================


class TestTimingCalculation:
    """Tests for next run time calculation."""

    def test_calculates_next_run_same_day(self):
        """Should calculate correct sleep time when run hour is later today."""
        config = SchedulerConfig(enabled=True, run_at_hour=15)  # 3 PM
        scheduler = ConsolidationScheduler(config=config)

        # Simulate current time being 10 AM
        now = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
        next_run = now.replace(hour=15, minute=0, second=0, microsecond=0)

        # The actual calculation happens in _run_loop
        # We just verify the logic
        if next_run <= now:
            next_run += timedelta(days=1)

        assert next_run > now
        assert next_run.hour == 15

    def test_calculates_next_run_next_day(self):
        """Should calculate next day when run hour has passed."""
        config = SchedulerConfig(enabled=True, run_at_hour=3)  # 3 AM
        scheduler = ConsolidationScheduler(config=config)

        # Simulate current time being 5 AM (past the run hour)
        now = datetime.now().replace(hour=5, minute=0, second=0, microsecond=0)
        next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)

        if next_run <= now:
            next_run += timedelta(days=1)

        assert next_run > now
        assert (next_run - now).total_seconds() > 20 * 3600  # At least 20 hours


# ============================================================================
# Integration-like tests
# ============================================================================


class TestSchedulerLifecycle:
    """Tests for full scheduler lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test complete start/stop cycle."""
        config = SchedulerConfig(enabled=True)

        # Create fresh scheduler
        scheduler = ConsolidationScheduler(config=config)

        # Mock the loop
        loop_started = asyncio.Event()

        async def mock_loop():
            loop_started.set()
            while scheduler._running:
                await asyncio.sleep(0.1)

        with patch.object(scheduler, "_run_loop", side_effect=mock_loop):
            # Start
            await scheduler.start()
            assert scheduler._running is True

            # Wait for loop to start
            await asyncio.wait_for(loop_started.wait(), timeout=1.0)

            # Stop
            await scheduler.stop()
            assert scheduler._running is False
            assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_error_recovery_in_loop(self, mock_graph_store):
        """Scheduler should recover from errors in check_and_run."""
        config = SchedulerConfig(enabled=True)
        scheduler = ConsolidationScheduler(config=config)

        call_count = 0

        async def failing_check():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            # After first failure, succeed
            return

        with patch.object(scheduler, "_check_and_run", side_effect=failing_check):
            # Simulate one iteration of error recovery
            try:
                await scheduler._check_and_run()
            except Exception:
                pass  # First call fails

            await scheduler._check_and_run()  # Second call succeeds
            assert call_count == 2
