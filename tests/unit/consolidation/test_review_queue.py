"""Unit tests for review queue persistence.

Tests:
- ReviewCandidate CRUD operations
- Idempotency via dedupe_key
- Rejected pair lookup normalization
- Status transitions
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest


# ============================================================================
# make_dedupe_key tests
# ============================================================================


class TestMakeDedupeKey:
    """Tests for dedupe_key generation."""

    def test_deterministic(self, mock_graph_store):
        """Same inputs should always produce same key."""
        from simplemem_lite.db.manager import DatabaseManager

        db = DatabaseManager.__new__(DatabaseManager)

        key1 = db.make_dedupe_key("project-1", "entity_dedup", ["a", "b"])
        key2 = db.make_dedupe_key("project-1", "entity_dedup", ["a", "b"])

        assert key1 == key2

    def test_order_independent(self, mock_graph_store):
        """Order of involved_ids should not affect key."""
        from simplemem_lite.db.manager import DatabaseManager

        db = DatabaseManager.__new__(DatabaseManager)

        key1 = db.make_dedupe_key("project-1", "entity_dedup", ["a", "b"])
        key2 = db.make_dedupe_key("project-1", "entity_dedup", ["b", "a"])

        assert key1 == key2

    def test_different_project_different_key(self, mock_graph_store):
        """Different projects should produce different keys."""
        from simplemem_lite.db.manager import DatabaseManager

        db = DatabaseManager.__new__(DatabaseManager)

        key1 = db.make_dedupe_key("project-1", "entity_dedup", ["a", "b"])
        key2 = db.make_dedupe_key("project-2", "entity_dedup", ["a", "b"])

        assert key1 != key2

    def test_different_type_different_key(self, mock_graph_store):
        """Different candidate types should produce different keys."""
        from simplemem_lite.db.manager import DatabaseManager

        db = DatabaseManager.__new__(DatabaseManager)

        key1 = db.make_dedupe_key("project-1", "entity_dedup", ["a", "b"])
        key2 = db.make_dedupe_key("project-1", "memory_merge", ["a", "b"])

        assert key1 != key2

    def test_key_length(self, mock_graph_store):
        """Key should be 16 characters (hex digest truncation)."""
        from simplemem_lite.db.manager import DatabaseManager

        db = DatabaseManager.__new__(DatabaseManager)

        key = db.make_dedupe_key("project-1", "entity_dedup", ["a", "b", "c"])

        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)


# ============================================================================
# add_review_candidate tests
# ============================================================================


class TestAddReviewCandidate:
    """Tests for adding review candidates with idempotency."""

    @pytest.fixture
    def db_with_mock_graph(self):
        """Create DatabaseManager with mocked graph."""
        from simplemem_lite.db.manager import DatabaseManager

        db = DatabaseManager.__new__(DatabaseManager)
        db.graph = MagicMock()
        return db

    def test_creates_new_candidate(self, db_with_mock_graph):
        """Should create candidate when none exists."""
        db = db_with_mock_graph

        # Mock no existing candidate
        empty_result = MagicMock()
        empty_result.result_set = []
        db.graph.query = MagicMock(return_value=empty_result)

        result = db.add_review_candidate(
            project_id="config:test",
            candidate_type="entity_dedup",
            confidence=0.85,
            reason="Same file path",
            decision_data={"entity_a_name": "main.py", "entity_b_name": "./main.py"},
            involved_ids=["main.py", "./main.py"],
            source_id="main.py",
            target_id="./main.py",
            similarity=0.92,
        )

        assert result["created"] is True
        assert "uuid" in result
        assert db.graph.query.call_count == 2  # Check + Create

    def test_returns_existing_pending(self, db_with_mock_graph):
        """Should return existing pending candidate without creating."""
        db = db_with_mock_graph

        # Mock existing pending candidate
        existing_result = MagicMock()
        existing_result.result_set = [["existing-uuid", "pending"]]
        db.graph.query = MagicMock(return_value=existing_result)

        result = db.add_review_candidate(
            project_id="config:test",
            candidate_type="entity_dedup",
            confidence=0.85,
            reason="Same file",
            decision_data={},
            involved_ids=["a", "b"],
            source_id="a",
            target_id="b",
            similarity=0.9,
        )

        assert result["created"] is False
        assert result["uuid"] == "existing-uuid"
        assert db.graph.query.call_count == 1  # Only check, no create

    def test_skips_rejected_candidate(self, db_with_mock_graph):
        """Should not recreate rejected candidates."""
        db = db_with_mock_graph

        # Mock existing rejected candidate
        rejected_result = MagicMock()
        rejected_result.result_set = [["rejected-uuid", "rejected"]]
        db.graph.query = MagicMock(return_value=rejected_result)

        result = db.add_review_candidate(
            project_id="config:test",
            candidate_type="entity_dedup",
            confidence=0.85,
            reason="Same file",
            decision_data={},
            involved_ids=["a", "b"],
            source_id="a",
            target_id="b",
            similarity=0.9,
        )

        assert result["created"] is False
        assert result.get("skipped") == "rejected"
        assert result["uuid"] == "rejected-uuid"


# ============================================================================
# get_review_candidates tests
# ============================================================================


class TestGetReviewCandidates:
    """Tests for fetching review candidates."""

    @pytest.fixture
    def db_with_mock_graph(self):
        """Create DatabaseManager with mocked graph."""
        from simplemem_lite.db.manager import DatabaseManager

        db = DatabaseManager.__new__(DatabaseManager)
        db.graph = MagicMock()
        return db

    def test_returns_candidates(self, db_with_mock_graph):
        """Should return list of candidates."""
        db = db_with_mock_graph

        now = int(time.time())
        result = MagicMock()
        result.result_set = [
            [
                "uuid-1",
                "dedupe-key-1",
                "config:test",
                "entity_dedup",
                "pending",
                0.85,
                "Same file",
                "main.py",
                "./main.py",
                0.92,
                '{"entity_a_name": "main.py"}',
                1,
                now,
                None,
            ]
        ]
        db.graph.query = MagicMock(return_value=result)

        candidates = db.get_review_candidates("config:test")

        assert len(candidates) == 1
        assert candidates[0]["uuid"] == "uuid-1"
        assert candidates[0]["type"] == "entity_dedup"
        assert candidates[0]["confidence"] == 0.85
        assert candidates[0]["decision_data"]["entity_a_name"] == "main.py"

    def test_filters_by_status(self, db_with_mock_graph):
        """Should filter by status in query."""
        db = db_with_mock_graph

        empty_result = MagicMock()
        empty_result.result_set = []
        db.graph.query = MagicMock(return_value=empty_result)

        db.get_review_candidates("config:test", status="approved")

        # Check that status was passed in query params
        call_args = db.graph.query.call_args
        assert call_args[0][1]["status"] == "approved"

    def test_filters_by_type(self, db_with_mock_graph):
        """Should filter by type when specified."""
        db = db_with_mock_graph

        empty_result = MagicMock()
        empty_result.result_set = []
        db.graph.query = MagicMock(return_value=empty_result)

        db.get_review_candidates("config:test", type_filter="memory_merge")

        call_args = db.graph.query.call_args
        assert call_args[0][1]["type_filter"] == "memory_merge"


# ============================================================================
# update_candidate_status tests
# ============================================================================


class TestUpdateCandidateStatus:
    """Tests for status updates."""

    @pytest.fixture
    def db_with_mock_graph(self):
        """Create DatabaseManager with mocked graph."""
        from simplemem_lite.db.manager import DatabaseManager

        db = DatabaseManager.__new__(DatabaseManager)
        db.graph = MagicMock()
        return db

    def test_updates_pending_to_approved(self, db_with_mock_graph):
        """Should update pending candidate to approved."""
        db = db_with_mock_graph

        result = MagicMock()
        result.result_set = [["uuid-1"]]  # Indicates update succeeded
        db.graph.query = MagicMock(return_value=result)

        updated = db.update_candidate_status("uuid-1", "approved")

        assert updated is True

    def test_returns_false_for_already_resolved(self, db_with_mock_graph):
        """Should return False if candidate not pending."""
        db = db_with_mock_graph

        result = MagicMock()
        result.result_set = []  # No rows updated
        db.graph.query = MagicMock(return_value=result)

        updated = db.update_candidate_status("uuid-1", "approved")

        assert updated is False


# ============================================================================
# Rejected pair tests
# ============================================================================


class TestRejectedPair:
    """Tests for rejected pair tracking."""

    @pytest.fixture
    def db_with_mock_graph(self):
        """Create DatabaseManager with mocked graph."""
        from simplemem_lite.db.manager import DatabaseManager

        db = DatabaseManager.__new__(DatabaseManager)
        db.graph = MagicMock()
        return db

    def test_add_rejected_pair_normalized(self, db_with_mock_graph):
        """Should normalize pair order (min, max)."""
        db = db_with_mock_graph

        empty_result = MagicMock()
        empty_result.result_set = []
        db.graph.query = MagicMock(return_value=empty_result)

        db.add_rejected_pair("z_second", "a_first", "candidate-uuid")

        call_args = db.graph.query.call_args
        params = call_args[0][1]
        assert params["pair_a"] == "a_first"  # min
        assert params["pair_b"] == "z_second"  # max

    def test_is_rejected_pair_normalized_lookup(self, db_with_mock_graph):
        """Lookup should work regardless of argument order."""
        db = db_with_mock_graph

        found_result = MagicMock()
        found_result.result_set = [["a_first"]]
        db.graph.query = MagicMock(return_value=found_result)

        # Test both orders
        result1 = db.is_rejected_pair("a_first", "z_second")
        result2 = db.is_rejected_pair("z_second", "a_first")

        assert result1 is True
        assert result2 is True

        # Both calls should use normalized order
        for call in db.graph.query.call_args_list:
            params = call[0][1]
            assert params["pair_a"] == "a_first"
            assert params["pair_b"] == "z_second"

    def test_is_rejected_pair_not_found(self, db_with_mock_graph):
        """Should return False when pair not rejected."""
        db = db_with_mock_graph

        empty_result = MagicMock()
        empty_result.result_set = []
        db.graph.query = MagicMock(return_value=empty_result)

        result = db.is_rejected_pair("a", "b")

        assert result is False


# ============================================================================
# Executor integration tests (with mocked db)
# ============================================================================


class TestExecutorPersistence:
    """Tests for executor persisting to review queue."""

    @pytest.mark.asyncio
    async def test_entity_executor_persists_medium_confidence(self, mock_graph_store):
        """Should persist medium-confidence decisions to review queue."""
        from unittest.mock import patch

        from simplemem_lite.backend.consolidation.executor import execute_entity_merges
        from simplemem_lite.backend.consolidation import ConsolidationConfig
        from simplemem_lite.backend.consolidation.scorer import EntityDecision
        from simplemem_lite.backend.consolidation.candidates import EntityPair

        # Add is_rejected_pair and add_review_candidate mocks
        mock_graph_store.db.is_rejected_pair = MagicMock(return_value=False)
        mock_graph_store.db.add_review_candidate = MagicMock(return_value={"uuid": "test-uuid", "created": True})

        config = ConsolidationConfig(confidence_threshold=0.9)

        # Create decision with 0.85 confidence (should queue)
        pair = EntityPair(
            entity_a={"name": "main.py", "type": "file"},
            entity_b={"name": "./main.py", "type": "file"},
            entity_type="file",
            similarity=0.92,
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.85,  # Medium confidence
            canonical_name="main.py",
            reason="Same file",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_graph_store,
        ):
            result = await execute_entity_merges([decision], config, project_id="config:test")

        assert result["queued"] == 1
        assert result["executed"] == 0
        mock_graph_store.db.add_review_candidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_executor_skips_rejected_pair(self, mock_graph_store):
        """Should skip pairs that were previously rejected."""
        from unittest.mock import patch

        from simplemem_lite.backend.consolidation.executor import execute_entity_merges
        from simplemem_lite.backend.consolidation import ConsolidationConfig
        from simplemem_lite.backend.consolidation.scorer import EntityDecision
        from simplemem_lite.backend.consolidation.candidates import EntityPair

        # Mock pair as rejected
        mock_graph_store.db.is_rejected_pair = MagicMock(return_value=True)
        mock_graph_store.db.add_review_candidate = MagicMock()

        config = ConsolidationConfig(confidence_threshold=0.9)

        pair = EntityPair(
            entity_a={"name": "main.py", "type": "file"},
            entity_b={"name": "./main.py", "type": "file"},
            entity_type="file",
            similarity=0.92,
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.85,
            canonical_name="main.py",
            reason="Same file",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_graph_store,
        ):
            result = await execute_entity_merges([decision], config, project_id="config:test")

        assert result["queued"] == 0  # Should be skipped
        mock_graph_store.db.add_review_candidate.assert_not_called()
