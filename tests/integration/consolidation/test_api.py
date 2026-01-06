"""Integration tests for consolidation REST API endpoints.

Tests the FastAPI endpoints using TestClient with mocked dependencies.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from simplemem_lite.backend.app import create_app
from simplemem_lite.backend.config import reset_config
from simplemem_lite.backend.consolidation import ConsolidationReport


@pytest.fixture
def app():
    """Create test FastAPI application with DEV mode (no auth required)."""
    # Reset config and set DEV mode before creating the app
    reset_config()
    os.environ["SIMPLEMEM_MODE"] = "dev"
    app = create_app()
    yield app
    # Cleanup - reset config back to default
    if "SIMPLEMEM_MODE" in os.environ:
        del os.environ["SIMPLEMEM_MODE"]
    reset_config()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_consolidation_report():
    """Create a mock consolidation report."""
    return ConsolidationReport(
        project_id="config:test-project",
        operations_run=["entity_dedup", "memory_merge", "supersession"],
        dry_run=False,
        entity_candidates_found=5,
        entity_merges_executed=3,
        entity_merges_queued=2,
        memory_candidates_found=3,
        memory_merges_executed=2,
        memory_merges_queued=1,
        supersession_candidates_found=2,
        supersessions_executed=1,
        supersessions_queued=1,
        errors=[],
        warnings=[],
        review_queue=[],
    )


# ============================================================================
# POST /api/v1/consolidate/run tests
# ============================================================================


class TestConsolidateRunEndpoint:
    """Tests for the /consolidate/run endpoint."""

    def test_synchronous_consolidation(self, client, mock_consolidation_report):
        """Should run consolidation synchronously when background=False."""
        with patch(
            "simplemem_lite.backend.api.consolidation.consolidate_project",
            new_callable=AsyncMock,
            return_value=mock_consolidation_report,
        ):
            response = client.post(
                "/api/v1/consolidate/run",
                json={
                    "project_id": "config:test-project",
                    "background": False,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["project_id"] == "config:test-project"
            assert "entity_dedup" in data
            assert "memory_merge" in data
            assert "supersession" in data

    def test_background_consolidation_returns_job_id(self, client):
        """Should return job_id when background=True."""
        response = client.post(
            "/api/v1/consolidate/run",
            json={
                "project_id": "config:test-project",
                "background": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert "consolidate/status" in data["message"]

    def test_dry_run(self, client, mock_consolidation_report):
        """Should accept dry_run parameter."""
        mock_consolidation_report.dry_run = True

        with patch(
            "simplemem_lite.backend.api.consolidation.consolidate_project",
            new_callable=AsyncMock,
            return_value=mock_consolidation_report,
        ):
            response = client.post(
                "/api/v1/consolidate/run",
                json={
                    "project_id": "config:test-project",
                    "dry_run": True,
                    "background": False,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["dry_run"] is True

    def test_specific_operations(self, client, mock_consolidation_report):
        """Should accept specific operations list."""
        mock_consolidation_report.operations_run = ["entity_dedup"]

        with patch(
            "simplemem_lite.backend.api.consolidation.consolidate_project",
            new_callable=AsyncMock,
            return_value=mock_consolidation_report,
        ):
            response = client.post(
                "/api/v1/consolidate/run",
                json={
                    "project_id": "config:test-project",
                    "operations": ["entity_dedup"],
                    "background": False,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["operations_run"] == ["entity_dedup"]

    def test_invalid_operation_returns_400(self, client):
        """Should return 400 for invalid operation name."""
        response = client.post(
            "/api/v1/consolidate/run",
            json={
                "project_id": "config:test-project",
                "operations": ["invalid_operation"],
                "background": False,
            },
        )

        assert response.status_code == 400
        assert "Invalid operation" in response.json()["detail"]

    def test_custom_confidence_threshold(self, client, mock_consolidation_report):
        """Should accept custom confidence threshold."""
        with patch(
            "simplemem_lite.backend.api.consolidation.consolidate_project",
            new_callable=AsyncMock,
            return_value=mock_consolidation_report,
        ) as mock_consolidate:
            response = client.post(
                "/api/v1/consolidate/run",
                json={
                    "project_id": "config:test-project",
                    "confidence_threshold": 0.8,
                    "background": False,
                },
            )

            assert response.status_code == 200
            # Verify threshold was passed
            mock_consolidate.assert_called_once()
            call_kwargs = mock_consolidate.call_args[1]
            assert call_kwargs["confidence_threshold"] == 0.8

    def test_missing_project_id_returns_422(self, client):
        """Should return 422 for missing required project_id."""
        response = client.post(
            "/api/v1/consolidate/run",
            json={
                "background": False,
            },
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_consolidation_error_returns_500(self, client):
        """Should return 500 on consolidation failure."""
        with patch(
            "simplemem_lite.backend.api.consolidation.consolidate_project",
            new_callable=AsyncMock,
            side_effect=Exception("Database connection failed"),
        ):
            response = client.post(
                "/api/v1/consolidate/run",
                json={
                    "project_id": "config:test-project",
                    "background": False,
                },
            )

            assert response.status_code == 500
            assert "Database connection failed" in response.json()["detail"]


# ============================================================================
# GET /api/v1/consolidate/status/{job_id} tests
# ============================================================================


class TestConsolidateStatusEndpoint:
    """Tests for the /consolidate/status/{job_id} endpoint."""

    def test_pending_job_status(self, client):
        """Should return valid status for new job."""
        # First create a background job
        create_response = client.post(
            "/api/v1/consolidate/run",
            json={
                "project_id": "config:test-project",
                "background": True,
            },
        )
        job_id = create_response.json()["job_id"]

        # Check status - job may already be completed due to race condition
        response = client.get(f"/api/v1/consolidate/status/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        # Job may complete very fast, so all valid states are acceptable
        assert data["status"] in ["pending", "running", "completed"]
        assert data["project_id"] == "config:test-project"

    def test_nonexistent_job_returns_404(self, client):
        """Should return 404 for unknown job ID."""
        response = client.get("/api/v1/consolidate/status/nonexistent-job-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_completed_job_has_result(self, client, mock_consolidation_report):
        """Completed job should include result."""
        # We need to manually manipulate the job storage for this test
        from simplemem_lite.backend.api.consolidation import _consolidation_jobs

        job_id = "test-completed-job"
        _consolidation_jobs[job_id] = {
            "status": "completed",
            "project_id": "config:test-project",
            "result": mock_consolidation_report.to_dict(),
        }

        response = client.get(f"/api/v1/consolidate/status/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["result"]["project_id"] == "config:test-project"

        # Cleanup
        del _consolidation_jobs[job_id]

    def test_failed_job_has_error(self, client):
        """Failed job should include error message."""
        from simplemem_lite.backend.api.consolidation import _consolidation_jobs

        job_id = "test-failed-job"
        _consolidation_jobs[job_id] = {
            "status": "failed",
            "project_id": "config:test-project",
            "result": None,
            "error": "Connection timeout",
        }

        response = client.get(f"/api/v1/consolidate/status/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Connection timeout"

        # Cleanup
        del _consolidation_jobs[job_id]


# ============================================================================
# GET /api/v1/consolidate/review-queue/{project_id} tests
# ============================================================================


class TestReviewQueueEndpoint:
    """Tests for the /consolidate/review-queue endpoint."""

    def test_get_review_queue(self, client):
        """Should return review queue structure."""
        response = client.get("/api/v1/consolidate/review-queue/config:test-project")

        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == "config:test-project"
        assert "items" in data
        assert "count" in data
        # Currently returns empty (placeholder implementation)
        assert data["count"] == 0

    def test_review_queue_with_limit(self, client):
        """Should accept limit parameter."""
        response = client.get(
            "/api/v1/consolidate/review-queue/config:test-project",
            params={"limit": 10},
        )

        assert response.status_code == 200


# ============================================================================
# POST /api/v1/consolidate/approve/{candidate_id} tests
# ============================================================================


class TestApproveMergeEndpoint:
    """Tests for the /consolidate/approve and /consolidate/reject endpoints."""

    def test_approve_nonexistent_returns_404(self, client):
        """Approving nonexistent candidate should return 404."""
        with patch("simplemem_lite.backend.services.get_memory_store") as mock_store:
            mock_store.return_value.db.get_review_candidate.return_value = None

            response = client.post(
                "/api/v1/consolidate/approve/candidate-123",
            )

            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()

    def test_reject_nonexistent_returns_404(self, client):
        """Rejecting nonexistent candidate should return 404."""
        with patch("simplemem_lite.backend.services.get_memory_store") as mock_store:
            mock_store.return_value.db.get_review_candidate.return_value = None

            response = client.post(
                "/api/v1/consolidate/reject/candidate-456",
            )

            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()

    def test_approve_already_resolved_returns_resolved_status(self, client):
        """Approving already resolved candidate should indicate already resolved."""
        with patch("simplemem_lite.backend.services.get_memory_store") as mock_store:
            mock_store.return_value.db.get_review_candidate.return_value = {
                "uuid": "candidate-123",
                "status": "approved",
                "type": "entity_dedup",
            }

            response = client.post(
                "/api/v1/consolidate/approve/candidate-123",
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "already_resolved"
            assert data["previous_status"] == "approved"

    def test_reject_already_resolved_returns_resolved_status(self, client):
        """Rejecting already resolved candidate should indicate already resolved."""
        with patch("simplemem_lite.backend.services.get_memory_store") as mock_store:
            mock_store.return_value.db.get_review_candidate.return_value = {
                "uuid": "candidate-456",
                "status": "rejected",
                "type": "memory_merge",
            }

            response = client.post(
                "/api/v1/consolidate/reject/candidate-456",
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "already_resolved"
            assert data["previous_status"] == "rejected"


# ============================================================================
# Request validation tests
# ============================================================================


class TestRequestValidation:
    """Tests for request validation."""

    def test_confidence_threshold_range_min(self, client):
        """Should reject confidence < 0."""
        response = client.post(
            "/api/v1/consolidate/run",
            json={
                "project_id": "config:test",
                "confidence_threshold": -0.1,
                "background": False,
            },
        )

        assert response.status_code == 422

    def test_confidence_threshold_range_max(self, client):
        """Should reject confidence > 1."""
        response = client.post(
            "/api/v1/consolidate/run",
            json={
                "project_id": "config:test",
                "confidence_threshold": 1.5,
                "background": False,
            },
        )

        assert response.status_code == 422

    def test_empty_operations_list_allowed(self, client, mock_consolidation_report):
        """Empty operations list should be allowed (means all)."""
        with patch(
            "simplemem_lite.backend.api.consolidation.consolidate_project",
            new_callable=AsyncMock,
            return_value=mock_consolidation_report,
        ):
            response = client.post(
                "/api/v1/consolidate/run",
                json={
                    "project_id": "config:test",
                    "operations": [],
                    "background": False,
                },
            )

            # Empty list is valid input
            assert response.status_code == 200


# ============================================================================
# Health check (verifies app is working)
# ============================================================================


class TestHealthCheck:
    """Basic health check to ensure app is configured correctly."""

    def test_health_endpoint(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "simplemem-lite-backend"
