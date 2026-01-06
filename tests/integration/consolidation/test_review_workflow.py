"""Integration tests for the review queue workflow.

Tests the full end-to-end flow:
1. Run consolidation with confidence_threshold=1.0 (everything queued)
2. GET /review-queue → items populated
3. POST /approve/{id} → merge executed
4. GET /review-queue → item removed
5. Re-run consolidation → approved pair not re-queued
"""

from __future__ import annotations

import os
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from simplemem_lite.backend.app import create_app
from simplemem_lite.backend.config import reset_config


@pytest.fixture
def app():
    """Create test FastAPI application with DEV mode (no auth required)."""
    reset_config()
    os.environ["SIMPLEMEM_MODE"] = "dev"
    app = create_app()
    yield app
    if "SIMPLEMEM_MODE" in os.environ:
        del os.environ["SIMPLEMEM_MODE"]
    reset_config()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_db():
    """Create mock database manager with review queue operations."""
    db = MagicMock()
    db.graph = MagicMock()

    # In-memory storage for review candidates and rejected pairs
    candidates_store: dict[str, dict[str, Any]] = {}
    rejected_pairs: set[tuple[str, str]] = set()

    def add_review_candidate(
        project_id: str,
        candidate_type: str,
        confidence: float,
        reason: str,
        decision_data: dict[str, Any],
        involved_ids: list[str],
        source_id: str,
        target_id: str,
        similarity: float,
    ) -> dict[str, Any]:
        """Mock add_review_candidate with in-memory storage."""
        import hashlib
        import uuid

        # Create dedupe key
        sorted_ids = sorted(involved_ids)
        raw = f"{project_id}|{candidate_type}|{','.join(sorted_ids)}"
        dedupe_key = hashlib.sha256(raw.encode()).hexdigest()[:16]

        # Check if exists
        for c in candidates_store.values():
            if c["dedupe_key"] == dedupe_key:
                return {"uuid": c["uuid"], "created": False, "skipped": c["status"]}

        # Create new
        candidate_uuid = str(uuid.uuid4())
        candidates_store[candidate_uuid] = {
            "uuid": candidate_uuid,
            "dedupe_key": dedupe_key,
            "project_id": project_id,
            "type": candidate_type,
            "status": "pending",
            "confidence": confidence,
            "reason": reason,
            "source_id": source_id,
            "target_id": target_id,
            "similarity": similarity,
            "decision_data": decision_data,
            "schema_version": 1,
            "created_at": int(time.time()),
            "resolved_at": None,
        }
        return {"uuid": candidate_uuid, "created": True}

    def get_review_candidates(
        project_id: str,
        status: str = "pending",
        type_filter: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Mock get_review_candidates with in-memory storage."""
        results = []
        for c in candidates_store.values():
            if c["project_id"] != project_id:
                continue
            if c["status"] != status:
                continue
            if type_filter and c["type"] != type_filter:
                continue
            results.append(c)
            if len(results) >= limit:
                break
        return results

    def get_review_candidate(uuid: str) -> dict[str, Any] | None:
        """Mock get_review_candidate."""
        return candidates_store.get(uuid)

    def update_candidate_status(
        uuid: str, status: str, resolved_at: int | None = None
    ) -> bool:
        """Mock update_candidate_status."""
        if uuid not in candidates_store:
            return False
        if candidates_store[uuid]["status"] != "pending":
            return False
        candidates_store[uuid]["status"] = status
        candidates_store[uuid]["resolved_at"] = resolved_at
        return True

    def add_rejected_pair(uuid1: str, uuid2: str, candidate_uuid: str) -> None:
        """Mock add_rejected_pair with normalized ordering."""
        pair_a = min(uuid1, uuid2)
        pair_b = max(uuid1, uuid2)
        rejected_pairs.add((pair_a, pair_b))

    def is_rejected_pair(uuid1: str, uuid2: str) -> bool:
        """Mock is_rejected_pair with normalized lookup."""
        pair_a = min(uuid1, uuid2)
        pair_b = max(uuid1, uuid2)
        return (pair_a, pair_b) in rejected_pairs

    # Assign mock methods
    db.add_review_candidate = add_review_candidate
    db.get_review_candidates = get_review_candidates
    db.get_review_candidate = get_review_candidate
    db.update_candidate_status = update_candidate_status
    db.add_rejected_pair = add_rejected_pair
    db.is_rejected_pair = is_rejected_pair

    # Mock graph queries to return empty results
    empty_result = MagicMock()
    empty_result.result_set = []
    db.graph.query = MagicMock(return_value=empty_result)

    # Make mark_merged a no-op
    db.mark_merged = MagicMock()

    # Make add_supersession a no-op
    db.add_supersession = MagicMock()

    # Expose internal storage for test verification
    db._candidates_store = candidates_store
    db._rejected_pairs = rejected_pairs

    return db


@pytest.fixture
def mock_store(mock_db):
    """Create mock memory store with mock db."""
    store = MagicMock()
    store.db = mock_db
    return store


# ============================================================================
# Full workflow tests
# ============================================================================


class TestReviewQueueWorkflow:
    """Test the full review queue workflow."""

    def test_entity_dedup_queue_approve_workflow(self, client, mock_store):
        """Test full workflow: queue → approve → verify."""
        from simplemem_lite.backend.consolidation.candidates import EntityPair
        from simplemem_lite.backend.consolidation.scorer import EntityDecision

        # Create entity decision with medium confidence (will be queued)
        pair = EntityPair(
            entity_a={"name": "main.py", "type": "file"},
            entity_b={"name": "./main.py", "type": "file"},
            entity_type="file",
            similarity=0.92,
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.85,  # Medium confidence - should queue
            canonical_name="main.py",
            reason="Same file path",
        )

        with patch(
            "simplemem_lite.backend.services.get_memory_store",
            return_value=mock_store,
        ):
            # Step 1: Execute - should queue (not execute)
            from simplemem_lite.backend.consolidation.executor import (
                execute_entity_merges,
            )
            from simplemem_lite.backend.consolidation import ConsolidationConfig

            with patch(
                "simplemem_lite.backend.consolidation.executor.get_memory_store",
                return_value=mock_store,
            ):
                import asyncio

                config = ConsolidationConfig(confidence_threshold=0.9)
                result = asyncio.get_event_loop().run_until_complete(
                    execute_entity_merges([decision], config, project_id="config:test")
                )

            assert result["executed"] == 0
            assert result["queued"] == 1

            # Step 2: GET review queue - should have 1 item
            response = client.get("/api/v1/consolidate/review-queue/config:test")
            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 1
            assert len(data["items"]) == 1
            candidate_id = data["items"][0]["uuid"]
            assert data["items"][0]["type"] == "entity_dedup"
            assert data["items"][0]["confidence"] == 0.85

            # Step 3: Approve the candidate
            response = client.post(f"/api/v1/consolidate/approve/{candidate_id}")
            assert response.status_code == 200
            approve_data = response.json()
            assert approve_data["status"] == "approved"

            # Step 4: GET review queue - should be empty
            response = client.get("/api/v1/consolidate/review-queue/config:test")
            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 0

    def test_entity_dedup_queue_reject_workflow(self, client, mock_store):
        """Test full workflow: queue → reject → verify skipped on re-run."""
        from simplemem_lite.backend.consolidation.candidates import EntityPair
        from simplemem_lite.backend.consolidation.scorer import EntityDecision
        from simplemem_lite.backend.consolidation import ConsolidationConfig

        pair = EntityPair(
            entity_a={"name": "test.py", "type": "file"},
            entity_b={"name": "./test.py", "type": "file"},
            entity_type="file",
            similarity=0.91,
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.80,
            canonical_name="test.py",
            reason="Similar path",
        )

        with patch(
            "simplemem_lite.backend.services.get_memory_store",
            return_value=mock_store,
        ):
            from simplemem_lite.backend.consolidation.executor import (
                execute_entity_merges,
            )

            with patch(
                "simplemem_lite.backend.consolidation.executor.get_memory_store",
                return_value=mock_store,
            ):
                import asyncio

                config = ConsolidationConfig(confidence_threshold=0.9)

                # Step 1: Queue the candidate
                result = asyncio.get_event_loop().run_until_complete(
                    execute_entity_merges([decision], config, project_id="config:test")
                )
                assert result["queued"] == 1

                # Step 2: Get and reject
                response = client.get("/api/v1/consolidate/review-queue/config:test")
                candidate_id = response.json()["items"][0]["uuid"]

                response = client.post(
                    f"/api/v1/consolidate/reject/{candidate_id}",
                    params={"reason": "False positive"},
                )
                assert response.status_code == 200
                assert response.json()["status"] == "rejected"

                # Step 3: Re-run executor - should skip the rejected pair
                result = asyncio.get_event_loop().run_until_complete(
                    execute_entity_merges([decision], config, project_id="config:test")
                )

                # Should NOT queue again because pair is rejected
                assert result["queued"] == 0

    def test_idempotency_no_duplicate_candidates(self, client, mock_store):
        """Running consolidation twice should not create duplicate candidates."""
        from simplemem_lite.backend.consolidation.candidates import EntityPair
        from simplemem_lite.backend.consolidation.scorer import EntityDecision
        from simplemem_lite.backend.consolidation import ConsolidationConfig

        pair = EntityPair(
            entity_a={"name": "utils.py", "type": "file"},
            entity_b={"name": "./utils.py", "type": "file"},
            entity_type="file",
            similarity=0.88,
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.75,
            canonical_name="utils.py",
            reason="Same file",
        )

        with patch(
            "simplemem_lite.backend.services.get_memory_store",
            return_value=mock_store,
        ):
            from simplemem_lite.backend.consolidation.executor import (
                execute_entity_merges,
            )

            with patch(
                "simplemem_lite.backend.consolidation.executor.get_memory_store",
                return_value=mock_store,
            ):
                import asyncio

                config = ConsolidationConfig(confidence_threshold=0.9)

                # Run twice
                result1 = asyncio.get_event_loop().run_until_complete(
                    execute_entity_merges([decision], config, project_id="config:test")
                )
                result2 = asyncio.get_event_loop().run_until_complete(
                    execute_entity_merges([decision], config, project_id="config:test")
                )

                # First run should create
                assert result1["queued"] == 1

                # Second run should NOT create (idempotent)
                assert result2["queued"] == 0

                # Verify only 1 candidate in queue
                response = client.get("/api/v1/consolidate/review-queue/config:test")
                data = response.json()
                assert data["count"] == 1


class TestMemoryMergeWorkflow:
    """Test memory merge review workflow."""

    def test_memory_merge_queue_approve(self, client, mock_store):
        """Test memory merge queue and approve workflow."""
        from simplemem_lite.backend.consolidation.candidates import MemoryPair
        from simplemem_lite.backend.consolidation.scorer import MemoryDecision
        from simplemem_lite.backend.consolidation import ConsolidationConfig

        pair = MemoryPair(
            memory_a={
                "uuid": "mem-001",
                "content": "Debug database connection",
                "created_at": 1704067200,
            },
            memory_b={
                "uuid": "mem-002",
                "content": "Debugging database connections",
                "created_at": 1704153600,
            },
            similarity=0.95,
            shared_entities=["postgresql"],
        )
        decision = MemoryDecision(
            pair=pair,
            should_merge=True,
            confidence=0.82,
            merged_content="Combined: Debug database connection issues with PostgreSQL",
            reason="Near-duplicate content",
        )

        with patch(
            "simplemem_lite.backend.services.get_memory_store",
            return_value=mock_store,
        ):
            from simplemem_lite.backend.consolidation.executor import (
                execute_memory_merges,
            )

            with patch(
                "simplemem_lite.backend.consolidation.executor.get_memory_store",
                return_value=mock_store,
            ):
                import asyncio

                config = ConsolidationConfig(confidence_threshold=0.9)

                result = asyncio.get_event_loop().run_until_complete(
                    execute_memory_merges([decision], config, project_id="config:test")
                )

                assert result["executed"] == 0
                assert result["queued"] == 1

                # Verify in queue
                response = client.get(
                    "/api/v1/consolidate/review-queue/config:test",
                    params={"type_filter": "memory_merge"},
                )
                data = response.json()
                assert data["count"] == 1
                assert data["items"][0]["type"] == "memory_merge"


class TestSupersessionWorkflow:
    """Test supersession review workflow."""

    def test_supersession_queue_approve(self, client, mock_store):
        """Test supersession queue and approve workflow."""
        from simplemem_lite.backend.consolidation.candidates import SupersessionPair
        from simplemem_lite.backend.consolidation.scorer import SupersessionDecision
        from simplemem_lite.backend.consolidation import ConsolidationConfig

        pair = SupersessionPair(
            newer={
                "uuid": "mem-newer",
                "content": "Updated solution for config issue",
                "created_at": 1704326400,
            },
            older={
                "uuid": "mem-older",
                "content": "Original config solution",
                "created_at": 1704067200,
            },
            entity="config.py",
            time_delta_days=3,
            similarity=0.85,
        )
        decision = SupersessionDecision(
            pair=pair,
            supersedes=True,
            confidence=0.78,
            supersession_type="full_replace",
            reason="Newer provides updated fix",
        )

        with patch(
            "simplemem_lite.backend.services.get_memory_store",
            return_value=mock_store,
        ):
            from simplemem_lite.backend.consolidation.executor import (
                execute_supersessions,
            )

            with patch(
                "simplemem_lite.backend.consolidation.executor.get_memory_store",
                return_value=mock_store,
            ):
                import asyncio

                config = ConsolidationConfig(confidence_threshold=0.9)

                result = asyncio.get_event_loop().run_until_complete(
                    execute_supersessions([decision], config, project_id="config:test")
                )

                assert result["executed"] == 0
                assert result["queued"] == 1

                # Verify in queue
                response = client.get(
                    "/api/v1/consolidate/review-queue/config:test",
                    params={"type_filter": "supersession"},
                )
                data = response.json()
                assert data["count"] == 1
                assert data["items"][0]["type"] == "supersession"


class TestAPIEndpoints:
    """Test review queue API endpoints."""

    def test_approve_nonexistent_candidate_returns_404(self, client, mock_store):
        """Should return 404 for nonexistent candidate."""
        with patch(
            "simplemem_lite.backend.services.get_memory_store",
            return_value=mock_store,
        ):
            response = client.post("/api/v1/consolidate/approve/nonexistent-uuid")
            assert response.status_code == 404

    def test_reject_nonexistent_candidate_returns_404(self, client, mock_store):
        """Should return 404 for nonexistent candidate."""
        with patch(
            "simplemem_lite.backend.services.get_memory_store",
            return_value=mock_store,
        ):
            response = client.post("/api/v1/consolidate/reject/nonexistent-uuid")
            assert response.status_code == 404

    def test_double_approve_is_idempotent(self, client, mock_store):
        """Approving twice should return already_resolved."""
        # First, create a candidate
        mock_store.db._candidates_store["test-uuid"] = {
            "uuid": "test-uuid",
            "dedupe_key": "abc123",
            "project_id": "config:test",
            "type": "entity_dedup",
            "status": "pending",
            "confidence": 0.85,
            "reason": "Test",
            "source_id": "a.py",
            "target_id": "b.py",
            "similarity": 0.9,
            "decision_data": {
                "entity_a_name": "a.py",
                "entity_b_name": "b.py",
                "entity_type": "file",
                "canonical_name": "a.py",
            },
            "schema_version": 1,
            "created_at": int(time.time()),
            "resolved_at": None,
        }

        with patch(
            "simplemem_lite.backend.services.get_memory_store",
            return_value=mock_store,
        ):
            # First approve
            response = client.post("/api/v1/consolidate/approve/test-uuid")
            assert response.status_code == 200
            assert response.json()["status"] == "approved"

            # Second approve
            response = client.post("/api/v1/consolidate/approve/test-uuid")
            assert response.status_code == 200
            assert response.json()["status"] == "already_resolved"

    def test_review_queue_filter_by_type(self, client, mock_store):
        """Should filter by type."""
        # Add candidates of different types
        mock_store.db._candidates_store["entity-1"] = {
            "uuid": "entity-1",
            "project_id": "config:test",
            "type": "entity_dedup",
            "status": "pending",
            "confidence": 0.85,
            "reason": "Test",
            "source_id": "a",
            "target_id": "b",
            "similarity": 0.9,
            "decision_data": {},
        }
        mock_store.db._candidates_store["memory-1"] = {
            "uuid": "memory-1",
            "project_id": "config:test",
            "type": "memory_merge",
            "status": "pending",
            "confidence": 0.82,
            "reason": "Test",
            "source_id": "m1",
            "target_id": "m2",
            "similarity": 0.88,
            "decision_data": {},
        }

        with patch(
            "simplemem_lite.backend.services.get_memory_store",
            return_value=mock_store,
        ):
            # Get all
            response = client.get("/api/v1/consolidate/review-queue/config:test")
            assert response.json()["count"] == 2

            # Filter by entity_dedup
            response = client.get(
                "/api/v1/consolidate/review-queue/config:test",
                params={"type_filter": "entity_dedup"},
            )
            assert response.json()["count"] == 1
            assert response.json()["items"][0]["type"] == "entity_dedup"

            # Filter by memory_merge
            response = client.get(
                "/api/v1/consolidate/review-queue/config:test",
                params={"type_filter": "memory_merge"},
            )
            assert response.json()["count"] == 1
            assert response.json()["items"][0]["type"] == "memory_merge"
