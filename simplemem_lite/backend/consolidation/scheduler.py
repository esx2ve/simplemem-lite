"""Scheduler for periodic graph consolidation.

Runs consolidation automatically on a configurable schedule for projects
that have accumulated enough changes since their last consolidation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from simplemem_lite.backend.services import get_memory_store

log = logging.getLogger("consolidation.scheduler")


@dataclass
class SchedulerConfig:
    """Configuration for the consolidation scheduler."""

    # How often to check for projects needing consolidation (hours)
    check_interval_hours: int = 24

    # Minimum days since last consolidation before running again
    min_days_between_runs: int = 7

    # Minimum new memories since last consolidation to trigger
    min_new_memories: int = 100

    # Whether scheduler is enabled
    enabled: bool = False

    # Hour to run (0-23, default 3 AM)
    run_at_hour: int = 3


@dataclass
class ConsolidationScheduler:
    """Scheduler for automatic graph consolidation.

    This is a simple in-memory scheduler. For production, consider
    using APScheduler, Celery Beat, or cloud-native solutions.
    """

    config: SchedulerConfig = field(default_factory=SchedulerConfig)
    _task: asyncio.Task | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _last_check: datetime | None = field(default=None, init=False)

    async def start(self) -> None:
        """Start the scheduler background task."""
        if not self.config.enabled:
            log.info("Consolidation scheduler disabled")
            return

        if self._running:
            log.warning("Scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        log.info(
            f"Consolidation scheduler started (check every {self.config.check_interval_hours}h, "
            f"run at {self.config.run_at_hour}:00)"
        )

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("Consolidation scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Calculate sleep until next run hour
                now = datetime.now()
                next_run = now.replace(
                    hour=self.config.run_at_hour,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                if next_run <= now:
                    next_run += timedelta(days=1)

                sleep_seconds = (next_run - now).total_seconds()
                log.debug(f"Next consolidation check at {next_run} (in {sleep_seconds/3600:.1f}h)")

                await asyncio.sleep(sleep_seconds)

                if not self._running:
                    break

                await self._check_and_run()
                self._last_check = datetime.now()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Scheduler error: {e}", exc_info=True)
                # Sleep before retry on error
                await asyncio.sleep(3600)

    async def _check_and_run(self) -> None:
        """Check for projects needing consolidation and run them."""
        log.info("Checking for projects needing consolidation...")

        try:
            projects = await self._get_projects_needing_consolidation()

            if not projects:
                log.info("No projects need consolidation")
                return

            log.info(f"Found {len(projects)} projects needing consolidation")

            for project_id in projects:
                try:
                    await self._run_consolidation(project_id)
                except Exception as e:
                    log.error(f"Consolidation failed for {project_id}: {e}")

        except Exception as e:
            log.error(f"Failed to check projects: {e}", exc_info=True)

    async def _get_projects_needing_consolidation(self) -> list[str]:
        """Get list of project IDs that need consolidation.

        Criteria:
        - Has > min_new_memories since last consolidation
        - Last consolidation was > min_days_between_runs ago (or never)
        """
        store = get_memory_store()

        # Query projects with memory counts and last consolidation time
        result = store.db.graph.query(
            """
            MATCH (m:Memory)
            WHERE m.project_id IS NOT NULL
            RETURN m.project_id AS project_id,
                   count(m) AS memory_count,
                   max(m.created_at) AS last_memory_at
            """,
            {},
        )

        projects_needing_work = []
        cutoff_date = datetime.now() - timedelta(days=self.config.min_days_between_runs)

        for record in result.result_set or []:
            project_id, memory_count, last_memory_at = record

            if memory_count < self.config.min_new_memories:
                continue

            # Check last consolidation time (stored as metadata)
            # For now, assume all projects need consolidation if they meet the threshold
            # In production, track last consolidation time in a metadata store
            projects_needing_work.append(project_id)

        return projects_needing_work

    async def _run_consolidation(self, project_id: str) -> dict[str, Any]:
        """Run consolidation for a single project."""
        from simplemem_lite.backend.consolidation import consolidate_project

        log.info(f"Running scheduled consolidation for: {project_id}")

        report = await consolidate_project(
            project_id=project_id,
            operations=None,  # All operations
            dry_run=False,
            confidence_threshold=0.9,
        )

        result = report.to_dict()
        log.info(f"Scheduled consolidation complete for {project_id}: {result}")
        return result


# Global scheduler instance
_scheduler: ConsolidationScheduler | None = None


def get_scheduler() -> ConsolidationScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ConsolidationScheduler()
    return _scheduler


async def start_scheduler(config: SchedulerConfig | None = None) -> None:
    """Start the consolidation scheduler.

    Call this from app lifespan startup.
    """
    scheduler = get_scheduler()
    if config:
        scheduler.config = config
    await scheduler.start()


async def stop_scheduler() -> None:
    """Stop the consolidation scheduler.

    Call this from app lifespan shutdown.
    """
    scheduler = get_scheduler()
    await scheduler.stop()


__all__ = [
    "SchedulerConfig",
    "ConsolidationScheduler",
    "get_scheduler",
    "start_scheduler",
    "stop_scheduler",
]
