"""Background job manager for long-running SimpleMem operations.

Provides async background processing with:
- Job state tracking (pending -> running -> completed/failed/cancelled)
- File-based persistence for crash recovery
- Progress callbacks for real-time updates
- Status file for Claude Code statusline integration
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

# Throttle interval for progress persistence (seconds)
PROGRESS_PERSIST_INTERVAL = 0.5

from simplemem_lite.log_config import get_logger

log = get_logger("job_manager")


class JobStatus(str, Enum):
    """Job state machine states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a background job."""

    id: str
    job_type: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    progress_message: str = ""
    result: Any = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Create Job from dict."""
        data["status"] = JobStatus(data["status"])
        return cls(**data)


class JobManager:
    """Manages background jobs with persistence and status reporting."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize job manager.

        Args:
            data_dir: Directory for job persistence. Defaults to ~/.simplemem/
        """
        self.data_dir = data_dir or Path.home() / ".simplemem"
        self.jobs_dir = self.data_dir / "jobs"
        self.status_file = self.data_dir / "status.json"

        # Ensure directories exist
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        # In-memory job tracking
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}

        # Load persisted jobs on startup
        self._load_jobs()

        # Auto-cleanup old jobs to prevent accumulation
        cleaned = self.cleanup_old_jobs(max_age_hours=24)
        if cleaned > 0:
            log.info(f"Auto-cleaned {cleaned} old jobs")

        log.info(f"JobManager initialized with {len(self._jobs)} persisted jobs")

    def _load_jobs(self) -> None:
        """Load persisted jobs from disk."""
        log.debug(f"Loading persisted jobs from {self.jobs_dir}")
        loaded_count = 0
        interrupted_count = 0

        for job_file in self.jobs_dir.glob("*.json"):
            try:
                data = json.loads(job_file.read_text())
                job = Job.from_dict(data)
                log.debug(f"Loaded job {job.id} ({job.job_type}) status={job.status.value}")

                # Only load incomplete jobs (for potential resume)
                if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                    # Mark as failed since we can't resume
                    log.warning(f"Job {job.id} was interrupted (was {job.status.value}), marking as failed")
                    job.status = JobStatus.FAILED
                    job.error = "Job interrupted by server restart"
                    job.completed_at = datetime.now().isoformat()
                    self._persist_job(job)
                    interrupted_count += 1

                self._jobs[job.id] = job
                loaded_count += 1
            except Exception as e:
                log.warning(f"Failed to load job from {job_file}: {e}")

        log.info(f"Loaded {loaded_count} jobs ({interrupted_count} were interrupted)")

    def _write_atomic(self, path: Path, data: dict) -> None:
        """Write JSON data atomically using temp file + rename.

        This prevents corruption if the process crashes during write,
        or if external readers access the file mid-write.
        """
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(data, indent=2))
            os.replace(tmp_path, path)  # Atomic on POSIX
        except Exception as e:
            # Clean up temp file on failure
            tmp_path.unlink(missing_ok=True)
            raise e

    def _persist_job(self, job: Job) -> None:
        """Persist job state to disk atomically."""
        job_file = self.jobs_dir / f"{job.id}.json"
        try:
            self._write_atomic(job_file, job.to_dict())
            log.debug(f"Persisted job {job.id} status={job.status.value} progress={job.progress}%")
        except Exception as e:
            log.error(f"Failed to persist job {job.id}: {e}")

    def _update_status_file(self) -> None:
        """Update status file for Claude Code statusline (atomic write)."""
        active_jobs = [j for j in self._jobs.values() if j.status == JobStatus.RUNNING]

        if active_jobs:
            # Show the most recent running job
            current = max(active_jobs, key=lambda j: j.started_at or "")
            status = {
                "active_jobs": len(active_jobs),
                "current": {
                    "id": current.id,
                    "type": current.job_type,
                    "progress": current.progress,
                    "message": current.progress_message or f"{current.job_type}...",
                },
                "updated_at": datetime.now().isoformat(),
            }
        else:
            status = {
                "active_jobs": 0,
                "current": None,
                "updated_at": datetime.now().isoformat(),
            }

        try:
            self._write_atomic(self.status_file, status)
            log.debug(f"Updated status file: {status.get('active_jobs', 0)} active jobs")
        except Exception as e:
            log.error(f"Failed to update status file: {e}")

    async def submit(
        self,
        job_type: str,
        coro_factory: Callable[..., Coroutine],
        *args,
        progress_callback: Callable[[int, str], None] | None = None,
        **kwargs,
    ) -> str:
        """Submit a background job.

        Args:
            job_type: Type of job (e.g., "process_trace", "index_directory")
            coro_factory: Async function to run
            *args: Arguments for the coroutine
            progress_callback: Optional callback for progress updates
            **kwargs: Keyword arguments for the coroutine

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())[:8]
        job = Job(id=job_id, job_type=job_type)

        self._jobs[job_id] = job
        self._persist_job(job)
        self._update_status_file()

        log.info(f"Job submitted: id={job_id} type={job_type} args={args[:2] if args else 'none'}...")

        # Create wrapper that updates job state
        async def job_wrapper():
            log.info(f"Job {job_id} starting execution")
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now().isoformat()
            self._persist_job(job)
            self._update_status_file()

            # Track last persist time for throttling
            last_persist_time = time.time()

            try:
                # Create progress callback that updates job (throttled to prevent blocking)
                def update_progress(progress: int, message: str = ""):
                    nonlocal last_persist_time
                    log.debug(f"Job {job_id} progress: {progress}% - {message}")
                    job.progress = progress
                    job.progress_message = message

                    # Throttle persistence: only write if interval elapsed OR job complete
                    now = time.time()
                    if progress >= 100 or (now - last_persist_time) >= PROGRESS_PERSIST_INTERVAL:
                        self._persist_job(job)
                        self._update_status_file()
                        last_persist_time = now

                    if progress_callback:
                        progress_callback(progress, message)

                # Inject progress callback if the function accepts it
                if "progress_callback" in kwargs or _accepts_progress_callback(
                    coro_factory
                ):
                    kwargs["progress_callback"] = update_progress

                log.debug(f"Job {job_id} invoking coroutine factory")
                result = await coro_factory(*args, **kwargs)

                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.result = result
                job.completed_at = datetime.now().isoformat()
                elapsed = _elapsed_since(job.started_at)
                log.info(f"Job {job_id} completed successfully in {elapsed:.1f}s result_type={type(result).__name__}")

            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now().isoformat()
                elapsed = _elapsed_since(job.started_at)
                log.warning(f"Job {job_id} cancelled after {elapsed:.1f}s")
                raise

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now().isoformat()
                elapsed = _elapsed_since(job.started_at)
                log.error(f"Job {job_id} failed after {elapsed:.1f}s: {type(e).__name__}: {e}")

            finally:
                self._persist_job(job)
                self._update_status_file()
                # Clean up task reference
                self._tasks.pop(job_id, None)

        # Start the task
        task = asyncio.create_task(job_wrapper())
        self._tasks[job_id] = task

        return job_id

    def get_status(self, job_id: str) -> dict | None:
        """Get status of a job.

        Args:
            job_id: Job ID to check

        Returns:
            Job status dict or None if not found
        """
        job = self._jobs.get(job_id)
        if not job:
            return None

        return {
            "id": job.id,
            "type": job.job_type,
            "status": job.status.value,
            "progress": job.progress,
            "message": job.progress_message,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
        }

    def get_active_stats(self) -> dict:
        """Get summary stats for active jobs (for statusline).

        Returns:
            Dict with active job count and current job info
        """
        active = [j for j in self._jobs.values() if j.status == JobStatus.RUNNING]
        if active:
            current = max(active, key=lambda j: j.started_at or "")
            return {
                "active": len(active),
                "current": {
                    "type": current.job_type,
                    "progress": current.progress,
                },
            }
        return {"active": 0, "current": None}

    def list_jobs(self, include_completed: bool = True, limit: int = 20) -> list[dict]:
        """List all jobs.

        Args:
            include_completed: Include completed/failed/cancelled jobs
            limit: Maximum number of jobs to return

        Returns:
            List of job status dicts
        """
        jobs = list(self._jobs.values())

        if not include_completed:
            jobs = [j for j in jobs if j.status in (JobStatus.PENDING, JobStatus.RUNNING)]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return [
            {
                "id": j.id,
                "type": j.job_type,
                "status": j.status.value,
                "progress": j.progress,
                "message": j.progress_message,
            }
            for j in jobs[:limit]
        ]

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled, False if not found or not running
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status != JobStatus.RUNNING:
            return False

        task = self._tasks.get(job_id)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True

        return False

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove old completed jobs.

        Args:
            max_age_hours: Remove jobs older than this

        Returns:
            Number of jobs removed
        """
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        removed = 0

        for job_id, job in list(self._jobs.items()):
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                try:
                    completed = datetime.fromisoformat(job.completed_at or job.created_at)
                    if completed.timestamp() < cutoff:
                        del self._jobs[job_id]
                        job_file = self.jobs_dir / f"{job_id}.json"
                        job_file.unlink(missing_ok=True)
                        removed += 1
                except Exception:
                    pass

        return removed


def _accepts_progress_callback(func: Callable) -> bool:
    """Check if function accepts a progress_callback parameter."""
    import inspect

    try:
        sig = inspect.signature(func)
        return "progress_callback" in sig.parameters
    except (ValueError, TypeError):
        return False


def _elapsed_since(iso_timestamp: str | None) -> float:
    """Calculate seconds elapsed since an ISO timestamp."""
    if not iso_timestamp:
        return 0.0
    try:
        start = datetime.fromisoformat(iso_timestamp)
        return (datetime.now() - start).total_seconds()
    except Exception:
        return 0.0


# Global instance (initialized by server)
_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


def init_job_manager(data_dir: Path | None = None) -> JobManager:
    """Initialize the global job manager."""
    global _job_manager
    _job_manager = JobManager(data_dir)
    return _job_manager
