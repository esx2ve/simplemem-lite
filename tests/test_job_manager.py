"""Job manager tests for SimpleMem Lite.

Tests critical job management logic:
- Job dataclass serialization
- JobStatus enum
- Helper functions
- JobManager with temp directory
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest


class TestJobStatus:
    """Test JobStatus enum."""

    def test_all_statuses_exist(self):
        """All expected statuses should exist."""
        from simplemem_lite.job_manager import JobStatus

        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_status_is_str_subclass(self):
        """JobStatus should be str for JSON serialization."""
        from simplemem_lite.job_manager import JobStatus

        assert isinstance(JobStatus.PENDING, str)
        assert JobStatus.RUNNING.value == "running"


class TestJobDataclass:
    """Test Job dataclass operations."""

    def test_job_creation_defaults(self):
        """Job should have sensible defaults."""
        from simplemem_lite.job_manager import Job, JobStatus

        job = Job(id="test-123", job_type="process_trace")

        assert job.id == "test-123"
        assert job.job_type == "process_trace"
        assert job.status == JobStatus.PENDING
        assert job.progress == 0
        assert job.result is None
        assert job.error is None

    def test_job_to_dict(self):
        """to_dict should serialize job correctly."""
        from simplemem_lite.job_manager import Job, JobStatus

        job = Job(
            id="test-123",
            job_type="index_directory",
            status=JobStatus.RUNNING,
            progress=50,
            progress_message="Processing files...",
        )

        d = job.to_dict()

        assert d["id"] == "test-123"
        assert d["job_type"] == "index_directory"
        assert d["status"] == "running"  # String, not enum
        assert d["progress"] == 50
        assert d["progress_message"] == "Processing files..."

    def test_job_from_dict(self):
        """from_dict should deserialize job correctly."""
        from simplemem_lite.job_manager import Job, JobStatus

        data = {
            "id": "test-456",
            "job_type": "process_trace",
            "status": "completed",
            "progress": 100,
            "progress_message": "Done",
            "result": {"chunks": 5},
            "error": None,
            "created_at": "2024-01-01T00:00:00",
            "started_at": "2024-01-01T00:00:01",
            "completed_at": "2024-01-01T00:00:10",
        }

        job = Job.from_dict(data)

        assert job.id == "test-456"
        assert job.status == JobStatus.COMPLETED
        assert job.result == {"chunks": 5}

    def test_job_roundtrip(self):
        """Job should survive to_dict/from_dict roundtrip."""
        from simplemem_lite.job_manager import Job, JobStatus

        original = Job(
            id="roundtrip-test",
            job_type="test_job",
            status=JobStatus.FAILED,
            error="Something went wrong",
        )

        restored = Job.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.job_type == original.job_type
        assert restored.status == original.status
        assert restored.error == original.error


class TestHelperFunctions:
    """Test helper functions."""

    def test_accepts_progress_callback_true(self):
        """Should detect functions with progress_callback parameter."""
        from simplemem_lite.job_manager import _accepts_progress_callback

        def func_with_callback(x, progress_callback=None):
            pass

        async def async_func_with_callback(x, progress_callback=None):
            pass

        assert _accepts_progress_callback(func_with_callback) is True
        assert _accepts_progress_callback(async_func_with_callback) is True

    def test_accepts_progress_callback_false(self):
        """Should detect functions without progress_callback parameter."""
        from simplemem_lite.job_manager import _accepts_progress_callback

        def func_without_callback(x, y):
            pass

        assert _accepts_progress_callback(func_without_callback) is False

    def test_elapsed_since_valid_timestamp(self):
        """Should calculate elapsed time from ISO timestamp."""
        from simplemem_lite.job_manager import _elapsed_since

        # Create a timestamp 5 seconds ago
        past = datetime.now()
        past_iso = past.isoformat()

        elapsed = _elapsed_since(past_iso)

        # Should be very close to 0 (just created)
        assert elapsed >= 0
        assert elapsed < 1.0

    def test_elapsed_since_none(self):
        """Should return 0 for None timestamp."""
        from simplemem_lite.job_manager import _elapsed_since

        assert _elapsed_since(None) == 0.0

    def test_elapsed_since_invalid(self):
        """Should return 0 for invalid timestamp."""
        from simplemem_lite.job_manager import _elapsed_since

        assert _elapsed_since("not-a-timestamp") == 0.0


class TestJobManagerBasics:
    """Test JobManager basic operations."""

    def test_manager_creates_directories(self):
        """JobManager should create required directories."""
        from simplemem_lite.job_manager import JobManager

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "simplemem_test"
            manager = JobManager(data_dir=data_dir)

            assert data_dir.exists()
            assert (data_dir / "jobs").exists()

    def test_manager_starts_empty(self):
        """New manager should have no jobs."""
        from simplemem_lite.job_manager import JobManager

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(data_dir=Path(tmp))

            jobs = manager.list_jobs()
            assert jobs == []

    def test_get_status_unknown_job(self):
        """Getting status of unknown job should return None."""
        from simplemem_lite.job_manager import JobManager

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(data_dir=Path(tmp))

            result = manager.get_status("nonexistent-job-id")
            assert result is None

    def test_list_jobs_respects_limit(self):
        """list_jobs should respect limit parameter."""
        from simplemem_lite.job_manager import Job, JobManager, JobStatus

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(data_dir=Path(tmp))

            # Add 5 jobs directly to bypass async
            for i in range(5):
                job = Job(id=f"job-{i}", job_type="test", status=JobStatus.COMPLETED)
                manager._jobs[job.id] = job

            jobs = manager.list_jobs(limit=3)
            assert len(jobs) == 3


class TestJobPersistence:
    """Test job persistence to disk."""

    def test_job_persists_to_disk(self):
        """Jobs should be saved to disk."""
        from simplemem_lite.job_manager import Job, JobManager, JobStatus

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            manager = JobManager(data_dir=data_dir)

            # Create and persist a job
            job = Job(
                id="persist-test",
                job_type="test",
                status=JobStatus.COMPLETED,
            )
            manager._jobs[job.id] = job
            manager._persist_job(job)

            # Check file exists
            job_file = data_dir / "jobs" / "persist-test.json"
            assert job_file.exists()

    def test_jobs_survive_restart(self):
        """Jobs should be loaded on manager restart."""
        from simplemem_lite.job_manager import Job, JobManager, JobStatus

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            # First manager - create a completed job
            manager1 = JobManager(data_dir=data_dir)
            job = Job(
                id="survive-restart",
                job_type="test",
                status=JobStatus.COMPLETED,
                completed_at=datetime.now().isoformat(),
            )
            manager1._jobs[job.id] = job
            manager1._persist_job(job)

            # Second manager - should load the job
            manager2 = JobManager(data_dir=data_dir)
            loaded_job = manager2._jobs.get("survive-restart")

            assert loaded_job is not None
            assert loaded_job.status == JobStatus.COMPLETED

    def test_interrupted_jobs_marked_failed(self):
        """Jobs that were running when server stopped should be marked failed."""
        from simplemem_lite.job_manager import Job, JobManager, JobStatus

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            # First manager - create a running job (simulating crash)
            manager1 = JobManager(data_dir=data_dir)
            job = Job(
                id="interrupted-job",
                job_type="test",
                status=JobStatus.RUNNING,
                started_at=datetime.now().isoformat(),
            )
            manager1._jobs[job.id] = job
            manager1._persist_job(job)

            # Second manager - should mark it as failed
            manager2 = JobManager(data_dir=data_dir)
            loaded_job = manager2._jobs.get("interrupted-job")

            assert loaded_job is not None
            assert loaded_job.status == JobStatus.FAILED
            assert "interrupted" in loaded_job.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
