"""Tests for code index status tracking and statusline integration."""

import json
import tempfile
import time
from pathlib import Path

import pytest


class TestJobManagerCodeIndexStatus:
    """Test code index status tracking in JobManager."""

    @pytest.fixture
    def job_manager(self):
        """Create a JobManager with temporary directory."""
        from simplemem_lite.job_manager import JobManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            manager = JobManager(data_dir=tmppath)
            yield manager

    def test_initial_code_index_status(self, job_manager):
        """Code index status should be idle by default."""
        status = job_manager.get_code_index_status()
        assert status["status"] == "idle"
        assert status["watchers"] == 0
        assert status["projects_watching"] == []
        assert status["indexing"]["in_progress"] is False

    def test_update_watcher_count(self, job_manager):
        """Update watcher count should change status to watching."""
        job_manager.update_code_index_status(
            watchers=2,
            projects_watching=["/project/a", "/project/b"],
        )

        status = job_manager.get_code_index_status()
        assert status["status"] == "watching"
        assert status["watchers"] == 2
        assert status["projects_watching"] == ["/project/a", "/project/b"]

    def test_update_indexing_progress(self, job_manager):
        """Update indexing progress should change status to indexing."""
        job_manager.update_code_index_status(
            indexing_in_progress=True,
            files_done=5,
            files_total=100,
            current_file="src/main.py",
        )

        status = job_manager.get_code_index_status()
        assert status["status"] == "indexing"
        assert status["indexing"]["in_progress"] is True
        assert status["indexing"]["files_done"] == 5
        assert status["indexing"]["files_total"] == 100
        assert status["indexing"]["current_file"] == "src/main.py"

    def test_indexing_complete_reverts_to_watching(self, job_manager):
        """When indexing completes with active watchers, status should be watching."""
        # Set up watchers
        job_manager.update_code_index_status(
            watchers=1,
            projects_watching=["/project/a"],
        )

        # Start indexing
        job_manager.update_code_index_status(
            indexing_in_progress=True,
            files_total=10,
        )
        assert job_manager.get_code_index_status()["status"] == "indexing"

        # Complete indexing
        job_manager.update_code_index_status(
            indexing_in_progress=False,
        )
        assert job_manager.get_code_index_status()["status"] == "watching"

    def test_no_watchers_reverts_to_idle(self, job_manager):
        """When watchers are removed, status should be idle."""
        # Add watchers
        job_manager.update_code_index_status(
            watchers=2,
            projects_watching=["/project/a", "/project/b"],
        )
        assert job_manager.get_code_index_status()["status"] == "watching"

        # Remove all watchers
        job_manager.update_code_index_status(
            watchers=0,
            projects_watching=[],
        )
        assert job_manager.get_code_index_status()["status"] == "idle"

    def test_update_stats(self, job_manager):
        """Update stats should update the stats sub-dict."""
        job_manager.update_code_index_status(
            total_files=150,
            total_chunks=2340,
        )

        status = job_manager.get_code_index_status()
        assert status["stats"]["total_files"] == 150
        assert status["stats"]["total_chunks"] == 2340

    def test_explicit_status_override(self, job_manager):
        """Explicit status should override inferred status."""
        job_manager.update_code_index_status(
            watchers=1,
            status="idle",  # Explicitly set to idle despite having watchers
        )

        status = job_manager.get_code_index_status()
        assert status["status"] == "idle"
        assert status["watchers"] == 1

    def test_status_file_includes_code_index(self, job_manager):
        """Status file should include code_index data."""
        job_manager.update_code_index_status(
            watchers=1,
            projects_watching=["/test/project"],
        )

        # Read status file
        status_file = job_manager.status_file
        assert status_file.exists()

        with open(status_file) as f:
            data = json.load(f)

        assert "code_index" in data
        assert data["code_index"]["status"] == "watching"
        assert data["code_index"]["watchers"] == 1

    def test_partial_updates_preserve_other_fields(self, job_manager):
        """Partial updates should not reset other fields."""
        # Set initial state
        job_manager.update_code_index_status(
            watchers=2,
            projects_watching=["/a", "/b"],
            total_files=100,
            total_chunks=500,
        )

        # Update only indexing progress
        job_manager.update_code_index_status(
            indexing_in_progress=True,
            files_done=10,
        )

        status = job_manager.get_code_index_status()
        # Other fields should be preserved
        assert status["watchers"] == 2
        assert status["projects_watching"] == ["/a", "/b"]
        assert status["stats"]["total_files"] == 100
        assert status["stats"]["total_chunks"] == 500
        # New field should be set
        assert status["indexing"]["in_progress"] is True
        assert status["indexing"]["files_done"] == 10


class TestStatusEndpointIntegration:
    """Integration tests for the /stats endpoint code_index data."""

    @pytest.fixture
    def test_client(self):
        """Create a test client for the FastAPI backend with auth disabled."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from simplemem_lite.backend.app import create_app
        from simplemem_lite.backend.config import reset_config
        from simplemem_lite.backend.services import clear_service_caches

        # Clear caches before setting env vars
        clear_service_caches()
        reset_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            # Set dev mode to disable auth
            os.environ["SIMPLEMEM_DATA_DIR"] = tmpdir
            os.environ["SIMPLEMEM_MODE"] = "dev"  # Use SIMPLEMEM_MODE not DEV_MODE
            os.environ.pop("SIMPLEMEM_API_KEY", None)

            # Reset config again after setting env vars
            reset_config()

            app = create_app()
            client = TestClient(app)
            yield client

            # Clean up env vars and reset
            os.environ.pop("SIMPLEMEM_MODE", None)
            os.environ.pop("SIMPLEMEM_DATA_DIR", None)
            reset_config()
            clear_service_caches()

    def test_code_status_post_endpoint(self, test_client):
        """POST /api/v1/code/status should update code index status."""
        response = test_client.post(
            "/api/v1/code/status",
            json={
                "watchers": 1,
                "projects_watching": ["/test/project"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"
        assert data["code_index"]["watchers"] == 1

    def test_code_status_get_endpoint(self, test_client):
        """GET /api/v1/code/status should return current status."""
        # First set some status
        test_client.post(
            "/api/v1/code/status",
            json={"watchers": 2},
        )

        # Then get it
        response = test_client.get("/api/v1/code/status")
        assert response.status_code == 200
        data = response.json()
        assert "code_index" in data
        assert data["code_index"]["watchers"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
