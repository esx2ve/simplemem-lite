"""Unit tests for consolidation executor.

Tests:
- execute_entity_merges() - confidence thresholds and merge execution
- execute_memory_merges() - keeps newer, marks older as merged
- execute_supersessions() - creates SUPERSEDES relationships
- Error handling - database errors, review queue population
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simplemem_lite.backend.consolidation import ConsolidationConfig
from simplemem_lite.backend.consolidation.candidates import (
    EntityPair,
    MemoryPair,
    SupersessionPair,
)
from simplemem_lite.backend.consolidation.executor import (
    execute_entity_merges,
    execute_memory_merges,
    execute_supersessions,
)
from simplemem_lite.backend.consolidation.scorer import (
    EntityDecision,
    MemoryDecision,
    SupersessionDecision,
)


@pytest.fixture
def default_config() -> ConsolidationConfig:
    """Default consolidation config with 0.9 threshold."""
    return ConsolidationConfig(confidence_threshold=0.9)


@pytest.fixture
def low_threshold_config() -> ConsolidationConfig:
    """Config with lower threshold for testing."""
    return ConsolidationConfig(confidence_threshold=0.7)


@pytest.fixture
def project_id() -> str:
    """Standard test project ID."""
    return "config:test-project"


# ============================================================================
# execute_entity_merges() tests
# ============================================================================


class TestExecuteEntityMerges:
    """Tests for entity merge execution."""

    @pytest.fixture
    def mock_store_with_review(self, mock_graph_store):
        """Add review queue methods to mock store."""
        mock_graph_store.db.is_rejected_pair = MagicMock(return_value=False)
        mock_graph_store.db.add_review_candidate = MagicMock(
            return_value={"uuid": "test-uuid", "created": True}
        )
        return mock_graph_store

    @pytest.mark.asyncio
    async def test_empty_decisions_returns_empty_stats(
        self, mock_store_with_review, default_config, project_id
    ):
        """Empty decision list should return zeros."""
        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_entity_merges([], default_config, project_id=project_id)
            assert result["executed"] == 0
            assert result["queued"] == 0
            assert result["review_items"] == []

    @pytest.mark.asyncio
    async def test_high_confidence_auto_executes(self, mock_store_with_review, default_config, project_id):
        """Decisions with confidence >= 0.9 should auto-execute."""
        pair = EntityPair(
            entity_a={"name": "main.py", "type": "file"},
            entity_b={"name": "./main.py", "type": "file"},
            similarity=0.95,
            entity_type="file",
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.95,  # >= 0.9
            canonical_name="main.py",
            reason="Same file",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_entity_merges([decision], default_config, project_id=project_id)
            assert result["executed"] == 1
            assert result["queued"] == 0
            # Verify database was called
            assert mock_store_with_review.db.graph.query.call_count >= 1

    @pytest.mark.asyncio
    async def test_medium_confidence_queued_for_review(
        self, mock_store_with_review, default_config, project_id
    ):
        """Decisions with 0.7 <= confidence < 0.9 should be queued."""
        pair = EntityPair(
            entity_a={"name": "main.py", "type": "file"},
            entity_b={"name": "Main.py", "type": "file"},
            similarity=0.85,
            entity_type="file",
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.8,  # 0.7 <= x < 0.9
            canonical_name="main.py",
            reason="Possibly same file",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_entity_merges([decision], default_config, project_id=project_id)
            assert result["executed"] == 0
            assert result["queued"] == 1
            assert len(result["review_items"]) == 1
            assert result["review_items"][0]["action"] == "review_required"

    @pytest.mark.asyncio
    async def test_low_confidence_skipped(self, mock_store_with_review, default_config, project_id):
        """Decisions with confidence < 0.7 should be skipped."""
        pair = EntityPair(
            entity_a={"name": "main.py", "type": "file"},
            entity_b={"name": "test.py", "type": "file"},
            similarity=0.5,
            entity_type="file",
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.5,  # < 0.7
            canonical_name="main.py",
            reason="Unsure",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_entity_merges([decision], default_config, project_id=project_id)
            assert result["executed"] == 0
            assert result["queued"] == 0

    @pytest.mark.asyncio
    async def test_not_same_entity_skipped(self, mock_store_with_review, default_config, project_id):
        """Decisions with same_entity=False should be skipped."""
        pair = EntityPair(
            entity_a={"name": "main.py", "type": "file"},
            entity_b={"name": "test.py", "type": "file"},
            similarity=0.5,
            entity_type="file",
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=False,  # Not a match
            confidence=0.95,
            canonical_name=None,
            reason="Different files",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_entity_merges([decision], default_config, project_id=project_id)
            assert result["executed"] == 0
            assert result["queued"] == 0

    @pytest.mark.asyncio
    async def test_canonical_name_selection(self, mock_store_with_review, default_config, project_id):
        """Should use canonical_name to determine which entity to keep."""
        pair = EntityPair(
            entity_a={"name": "./main.py", "type": "file"},
            entity_b={"name": "main.py", "type": "file"},
            similarity=0.95,
            entity_type="file",
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.95,
            canonical_name="main.py",  # Keep this one
            reason="Same file",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_entity_merges([decision], default_config, project_id=project_id)
            # Check that the deprecated_name used was ./main.py
            calls = mock_store_with_review.db.graph.query.call_args_list
            if calls:
                # First call should redirect edges from ./main.py to main.py
                assert any("deprecated_name" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_db_error_adds_to_review(self, mock_store_with_review, default_config, project_id):
        """Database errors should add decision to review queue."""
        pair = EntityPair(
            entity_a={"name": "main.py", "type": "file"},
            entity_b={"name": "./main.py", "type": "file"},
            similarity=0.95,
            entity_type="file",
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.95,
            canonical_name="main.py",
            reason="Same file",
        )

        mock_store_with_review.db.graph.query.side_effect = Exception("Database error")

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_entity_merges([decision], default_config, project_id=project_id)
            assert result["executed"] == 0
            assert len(result["review_items"]) == 1
            assert "error" in result["review_items"][0]


# ============================================================================
# execute_memory_merges() tests
# ============================================================================


class TestExecuteMemoryMerges:
    """Tests for memory merge execution."""

    @pytest.fixture
    def mock_store_with_review(self, mock_graph_store):
        """Add review queue methods to mock store."""
        mock_graph_store.db.is_rejected_pair = MagicMock(return_value=False)
        mock_graph_store.db.add_review_candidate = MagicMock(
            return_value={"uuid": "test-uuid", "created": True}
        )
        mock_graph_store.db.mark_merged = MagicMock()
        return mock_graph_store

    @pytest.mark.asyncio
    async def test_empty_decisions_returns_empty_stats(
        self, mock_store_with_review, default_config, project_id
    ):
        """Empty decision list should return zeros."""
        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_memory_merges([], default_config, project_id=project_id)
            assert result["executed"] == 0
            assert result["queued"] == 0
            assert result["review_items"] == []

    @pytest.mark.asyncio
    async def test_keeps_newer_memory(self, mock_store_with_review, default_config, project_id):
        """Should keep the newer memory and mark older as merged."""
        pair = MemoryPair(
            memory_a={"uuid": "mem-old", "content": "old content", "type": "fact", "created_at": 1000},
            memory_b={"uuid": "mem-new", "content": "new content", "type": "fact", "created_at": 2000},
            similarity=0.95,
            shared_entities=["db.py"],
        )
        decision = MemoryDecision(
            pair=pair,
            should_merge=True,
            confidence=0.92,
            merged_content="Combined content from both",
            reason="Nearly identical",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_memory_merges([decision], default_config, project_id=project_id)
            assert result["executed"] == 1
            # Should have called mark_merged with older->newer
            mock_store_with_review.db.mark_merged.assert_called_once_with("mem-old", "mem-new")

    @pytest.mark.asyncio
    async def test_updates_content_when_provided(self, mock_store_with_review, default_config, project_id):
        """Should update newer memory with merged content."""
        pair = MemoryPair(
            memory_a={"uuid": "mem-old", "content": "old", "type": "fact", "created_at": 1000},
            memory_b={"uuid": "mem-new", "content": "new", "type": "fact", "created_at": 2000},
            similarity=0.9,
            shared_entities=[],
        )
        decision = MemoryDecision(
            pair=pair,
            should_merge=True,
            confidence=0.95,
            merged_content="Combined insight",
            reason="Duplicate",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_memory_merges([decision], default_config, project_id=project_id)
            # Check that content was updated via query
            calls = mock_store_with_review.db.graph.query.call_args_list
            content_updated = any("SET m.content" in str(c) for c in calls)
            assert content_updated or result["executed"] == 1

    @pytest.mark.asyncio
    async def test_medium_confidence_queued(self, mock_store_with_review, default_config, project_id):
        """Medium confidence should queue for review."""
        pair = MemoryPair(
            memory_a={"uuid": "m1", "content": "c1", "type": "fact", "created_at": 1000},
            memory_b={"uuid": "m2", "content": "c2", "type": "fact", "created_at": 2000},
            similarity=0.85,
            shared_entities=[],
        )
        decision = MemoryDecision(
            pair=pair,
            should_merge=True,
            confidence=0.75,  # 0.7 <= x < 0.9
            merged_content="merged",
            reason="Maybe same",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_memory_merges([decision], default_config, project_id=project_id)
            assert result["executed"] == 0
            assert result["queued"] == 1

    @pytest.mark.asyncio
    async def test_should_not_merge_skipped(self, mock_store_with_review, default_config, project_id):
        """Decisions with should_merge=False should be skipped."""
        pair = MemoryPair(
            memory_a={"uuid": "m1", "content": "c1", "type": "fact", "created_at": 1000},
            memory_b={"uuid": "m2", "content": "c2", "type": "fact", "created_at": 2000},
            similarity=0.5,
            shared_entities=[],
        )
        decision = MemoryDecision(
            pair=pair,
            should_merge=False,
            confidence=0.95,
            merged_content=None,
            reason="Different topics",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_memory_merges([decision], default_config, project_id=project_id)
            assert result["executed"] == 0
            assert result["queued"] == 0


# ============================================================================
# execute_supersessions() tests
# ============================================================================


class TestExecuteSupersessions:
    """Tests for supersession execution."""

    @pytest.fixture
    def mock_store_with_review(self, mock_graph_store):
        """Add review queue methods to mock store."""
        mock_graph_store.db.is_rejected_pair = MagicMock(return_value=False)
        mock_graph_store.db.add_review_candidate = MagicMock(
            return_value={"uuid": "test-uuid", "created": True}
        )
        mock_graph_store.db.add_supersession = MagicMock()
        return mock_graph_store

    @pytest.mark.asyncio
    async def test_empty_decisions_returns_empty_stats(
        self, mock_store_with_review, default_config, project_id
    ):
        """Empty decision list should return zeros."""
        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_supersessions([], default_config, project_id=project_id)
            assert result["executed"] == 0
            assert result["queued"] == 0
            assert result["review_items"] == []

    @pytest.mark.asyncio
    async def test_creates_supersedes_relationship(
        self, mock_store_with_review, default_config, project_id
    ):
        """Should create SUPERSEDES relationship for high confidence."""
        pair = SupersessionPair(
            newer={"uuid": "mem-new", "content": "updated solution"},
            older={"uuid": "mem-old", "content": "original approach"},
            entity="db.py",
            similarity=0.75,
            time_delta_days=5,
        )
        decision = SupersessionDecision(
            pair=pair,
            supersedes=True,
            confidence=0.92,
            supersession_type="full_replace",
            reason="Newer replaces older",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_supersessions([decision], default_config, project_id=project_id)
            assert result["executed"] == 1
            mock_store_with_review.db.add_supersession.assert_called_once()
            # Verify correct parameters
            call_kwargs = mock_store_with_review.db.add_supersession.call_args[1]
            assert call_kwargs["newer_uuid"] == "mem-new"
            assert call_kwargs["older_uuid"] == "mem-old"
            assert call_kwargs["supersession_type"] == "full_replace"

    @pytest.mark.asyncio
    async def test_stores_metadata(self, mock_store_with_review, default_config, project_id):
        """Should store confidence and type metadata."""
        pair = SupersessionPair(
            newer={"uuid": "new", "content": "updated"},
            older={"uuid": "old", "content": "original"},
            entity="file.py",
            similarity=0.8,
            time_delta_days=3,
        )
        decision = SupersessionDecision(
            pair=pair,
            supersedes=True,
            confidence=0.95,
            supersession_type="partial_update",
            reason="Adds context",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_supersessions([decision], default_config, project_id=project_id)
            call_kwargs = mock_store_with_review.db.add_supersession.call_args[1]
            assert call_kwargs["confidence"] == 0.95
            assert call_kwargs["supersession_type"] == "partial_update"

    @pytest.mark.asyncio
    async def test_skips_none_supersession_type(self, mock_store_with_review, default_config, project_id):
        """Should skip decisions with type='none'."""
        pair = SupersessionPair(
            newer={"uuid": "new", "content": "different"},
            older={"uuid": "old", "content": "original"},
            entity="file.py",
            similarity=0.65,
            time_delta_days=10,
        )
        decision = SupersessionDecision(
            pair=pair,
            supersedes=True,  # Even if true
            confidence=0.95,
            supersession_type="none",  # But type is none
            reason="Different aspects",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_supersessions([decision], default_config, project_id=project_id)
            assert result["executed"] == 0

    @pytest.mark.asyncio
    async def test_not_supersedes_skipped(self, mock_store_with_review, default_config, project_id):
        """Decisions with supersedes=False should be skipped."""
        pair = SupersessionPair(
            newer={"uuid": "new", "content": "content"},
            older={"uuid": "old", "content": "other"},
            entity="file.py",
            similarity=0.7,
            time_delta_days=5,
        )
        decision = SupersessionDecision(
            pair=pair,
            supersedes=False,
            confidence=0.95,
            supersession_type="full_replace",
            reason="Not related",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_supersessions([decision], default_config, project_id=project_id)
            assert result["executed"] == 0

    @pytest.mark.asyncio
    async def test_medium_confidence_queued(self, mock_store_with_review, default_config, project_id):
        """Medium confidence should queue for review."""
        pair = SupersessionPair(
            newer={"uuid": "new", "content": "updated"},
            older={"uuid": "old", "content": "original"},
            entity="file.py",
            similarity=0.75,
            time_delta_days=5,
        )
        decision = SupersessionDecision(
            pair=pair,
            supersedes=True,
            confidence=0.8,  # 0.7 <= x < 0.9
            supersession_type="partial_update",
            reason="Maybe supersedes",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_supersessions([decision], default_config, project_id=project_id)
            assert result["executed"] == 0
            assert result["queued"] == 1
            assert result["review_items"][0]["type"] == "supersession"


# ============================================================================
# Confidence threshold boundary tests
# ============================================================================


class TestConfidenceThresholds:
    """Test confidence threshold boundaries."""

    @pytest.fixture
    def mock_store_with_review(self, mock_graph_store):
        """Add review queue methods to mock store."""
        mock_graph_store.db.is_rejected_pair = MagicMock(return_value=False)
        mock_graph_store.db.add_review_candidate = MagicMock(
            return_value={"uuid": "test-uuid", "created": True}
        )
        return mock_graph_store

    @pytest.mark.asyncio
    async def test_exactly_at_threshold_executes(self, mock_store_with_review, project_id):
        """Confidence exactly at threshold (0.9) should execute."""
        config = ConsolidationConfig(confidence_threshold=0.9)
        pair = EntityPair(
            entity_a={"name": "a", "type": "file"},
            entity_b={"name": "b", "type": "file"},
            similarity=0.9,
            entity_type="file",
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.9,  # Exactly at threshold
            canonical_name="a",
            reason="Same",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_entity_merges([decision], config, project_id=project_id)
            assert result["executed"] == 1

    @pytest.mark.asyncio
    async def test_just_below_threshold_queued(self, mock_store_with_review, project_id):
        """Confidence just below threshold should be queued."""
        config = ConsolidationConfig(confidence_threshold=0.9)
        pair = EntityPair(
            entity_a={"name": "a", "type": "file"},
            entity_b={"name": "b", "type": "file"},
            similarity=0.89,
            entity_type="file",
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.89,  # Just below 0.9
            canonical_name="a",
            reason="Same",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_entity_merges([decision], config, project_id=project_id)
            assert result["executed"] == 0
            assert result["queued"] == 1

    @pytest.mark.asyncio
    async def test_custom_threshold(self, mock_store_with_review, project_id):
        """Should respect custom confidence threshold."""
        config = ConsolidationConfig(confidence_threshold=0.8)
        pair = EntityPair(
            entity_a={"name": "a", "type": "file"},
            entity_b={"name": "b", "type": "file"},
            similarity=0.85,
            entity_type="file",
        )
        decision = EntityDecision(
            pair=pair,
            same_entity=True,
            confidence=0.85,  # Above custom threshold of 0.8
            canonical_name="a",
            reason="Same",
        )

        with patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_review,
        ):
            result = await execute_entity_merges([decision], config, project_id=project_id)
            assert result["executed"] == 1
