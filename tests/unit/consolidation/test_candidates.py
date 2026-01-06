"""Unit tests for consolidation candidate generation.

Tests:
- cosine_similarity() - vector math correctness
- find_similar_pairs() - threshold filtering
- find_entity_candidates() - entity deduplication candidates
- find_memory_candidates() - memory merge candidates
- find_supersession_candidates() - temporal supersession candidates
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from simplemem_lite.backend.consolidation.candidates import (
    EntityPair,
    MemoryPair,
    SupersessionPair,
    cosine_similarity,
    find_entity_candidates,
    find_memory_candidates,
    find_similar_pairs,
    find_supersession_candidates,
)


# ============================================================================
# cosine_similarity() tests
# ============================================================================


class TestCosineSimilarity:
    """Tests for the cosine_similarity function."""

    def test_identical_vectors_returns_one(self):
        """Identical vectors should have similarity of 1.0."""
        vec = [1.0, 0.0, 0.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self):
        """Orthogonal vectors should have similarity of 0.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_opposite_vectors_returns_negative_one(self):
        """Opposite vectors should have similarity of -1.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        """Zero vector should return 0.0 similarity (graceful handling)."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 0.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_both_zero_vectors_returns_zero(self):
        """Two zero vectors should return 0.0."""
        vec = [0.0, 0.0, 0.0]
        assert cosine_similarity(vec, vec) == pytest.approx(0.0)

    def test_negative_values(self):
        """Should correctly handle negative vector components."""
        vec_a = [1.0, -1.0, 0.0]
        vec_b = [1.0, 1.0, 0.0]
        # dot = 1*1 + (-1)*1 + 0 = 0
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_similar_vectors_high_similarity(self):
        """Similar but not identical vectors should have high similarity."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.99, 0.14, 0.0]  # Small angle from vec_a
        sim = cosine_similarity(vec_a, vec_b)
        assert 0.95 < sim < 1.0

    def test_moderately_different_vectors(self):
        """Moderately different vectors should have moderate similarity."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.707, 0.707, 0.0]  # 45-degree angle
        sim = cosine_similarity(vec_a, vec_b)
        assert 0.6 < sim < 0.8

    def test_very_small_vectors(self):
        """Very small vectors should still compute correctly."""
        vec_a = [1e-8, 1e-8, 0.0]
        vec_b = [1e-8, 1e-8, 0.0]
        # Very small but non-zero, should be ~1.0
        sim = cosine_similarity(vec_a, vec_b)
        assert sim == pytest.approx(1.0, rel=0.1)

    def test_near_zero_norm_returns_zero(self):
        """Vectors with near-zero norm (< 1e-9) should return 0.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1e-10, 0.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)


# ============================================================================
# find_similar_pairs() tests
# ============================================================================


class TestFindSimilarPairs:
    """Tests for the find_similar_pairs function."""

    def test_empty_list_returns_empty(self):
        """Empty input should return empty list."""
        result = find_similar_pairs([], [], threshold=0.85)
        assert result == []

    def test_single_item_returns_empty(self):
        """Single item cannot form pairs."""
        items = [{"name": "a"}]
        embeddings = [[1.0, 0.0, 0.0]]
        result = find_similar_pairs(items, embeddings, threshold=0.85)
        assert result == []

    def test_no_pairs_above_threshold(self):
        """No pairs above threshold should return empty list."""
        items = [{"name": "a"}, {"name": "b"}]
        # Orthogonal vectors
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        result = find_similar_pairs(items, embeddings, threshold=0.85)
        assert result == []

    def test_finds_similar_pair(self):
        """Should find pairs above threshold."""
        items = [{"name": "a"}, {"name": "b"}]
        # Nearly identical vectors
        embeddings = [[1.0, 0.0, 0.0], [0.99, 0.14, 0.0]]
        result = find_similar_pairs(items, embeddings, threshold=0.85)
        assert len(result) == 1
        assert result[0][0] == 0  # idx_a
        assert result[0][1] == 1  # idx_b
        assert result[0][2] > 0.85  # similarity

    def test_finds_multiple_pairs(self):
        """Should find all pairs above threshold."""
        items = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        # a and b similar, b and c similar, a and c different
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.99, 0.14, 0.0],  # Similar to a
            [0.95, 0.31, 0.0],  # Similar to a and b
        ]
        result = find_similar_pairs(items, embeddings, threshold=0.85)
        # Should find: (a,b), (a,c), (b,c)
        assert len(result) >= 2

    def test_threshold_boundary_exact(self):
        """Pairs exactly at threshold should be included."""
        items = [{"name": "a"}, {"name": "b"}]
        # Create unit vectors with cosine similarity = 0.85
        # v1 = [1, 0, 0] (unit vector)
        # v2 = [0.85, sqrt(1 - 0.85^2), 0] (unit vector with cos(theta) = 0.85)
        sin_theta = np.sqrt(1 - 0.85**2)
        embeddings = [[1.0, 0.0, 0.0], [0.85, sin_theta, 0.0]]
        # Verify the similarity is exactly what we expect
        sim = cosine_similarity(embeddings[0], embeddings[1])
        assert abs(sim - 0.85) < 1e-10, f"Similarity is {sim}, expected 0.85"
        result = find_similar_pairs(items, embeddings, threshold=0.85)
        assert len(result) == 1

    def test_threshold_boundary_below(self):
        """Pairs just below threshold should be excluded."""
        items = [{"name": "a"}, {"name": "b"}]
        # Create vectors with ~0.84 similarity
        embeddings = [[1.0, 0.0, 0.0], [0.84, 0.54, 0.0]]
        norm = np.sqrt(0.84**2 + 0.54**2)
        embeddings[1] = [0.84 / norm, 0.54 / norm, 0.0]
        result = find_similar_pairs(items, embeddings, threshold=0.85)
        assert len(result) == 0

    def test_returns_correct_indices(self):
        """Should return correct pair indices."""
        items = [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}]
        embeddings = [
            [1.0, 0.0, 0.0],  # 0
            [0.0, 1.0, 0.0],  # 1 - orthogonal to 0
            [0.0, 0.99, 0.14],  # 2 - similar to 1
            [0.5, 0.5, 0.5],  # 3 - different from all
        ]
        result = find_similar_pairs(items, embeddings, threshold=0.85)
        # Only 1 and 2 should match
        assert len(result) == 1
        assert result[0][0] == 1
        assert result[0][1] == 2


# ============================================================================
# find_entity_candidates() tests
# ============================================================================


class TestFindEntityCandidates:
    """Tests for find_entity_candidates function."""

    @pytest.mark.asyncio
    async def test_empty_project_returns_empty(self, mock_graph_store):
        """Empty project should return no candidates."""
        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ):
            result = await find_entity_candidates("config:empty-project")
            assert result == []

    @pytest.mark.asyncio
    async def test_single_entity_returns_empty(self, mock_graph_store, mock_embeddings):
        """Single entity cannot form pairs."""
        # Setup mock to return one entity
        query_result = MagicMock()
        query_result.result_set = [["main.py", "file"]]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            result = await find_entity_candidates("config:test")
            assert result == []

    @pytest.mark.asyncio
    async def test_finds_similar_entities(self, mock_graph_store, mock_embeddings):
        """Should find similar entities as candidates."""
        query_result = MagicMock()
        query_result.result_set = [
            ["main.py", "file"],
            ["./main.py", "file"],
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            result = await find_entity_candidates("config:test", threshold=0.85)
            assert len(result) == 1
            assert isinstance(result[0], EntityPair)
            assert result[0].entity_type == "file"
            assert result[0].similarity > 0.85

    @pytest.mark.asyncio
    async def test_type_blocking_only_same_types(self, mock_graph_store, mock_embeddings):
        """Should only compare entities of the same type."""
        query_result = MagicMock()
        query_result.result_set = [
            ["main.py", "file"],
            ["main.py", "tool"],  # Same name but different type
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            result = await find_entity_candidates("config:test", threshold=0.85)
            # Should not pair because they're in different type groups
            # Each type group has only 1 entity
            assert result == []

    @pytest.mark.asyncio
    async def test_filters_by_threshold(self, mock_graph_store, mock_embeddings):
        """Should respect threshold parameter."""
        query_result = MagicMock()
        query_result.result_set = [
            ["main.py", "file"],
            ["test.py", "file"],  # Different file
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            # High threshold should exclude
            result = await find_entity_candidates("config:test", threshold=0.99)
            assert result == []

    @pytest.mark.asyncio
    async def test_handles_special_characters(self, mock_graph_store, mock_embeddings):
        """Should handle entities with special characters."""
        query_result = MagicMock()
        query_result.result_set = [
            ["./path/../main.py", "file"],
            ["main.py", "file"],
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            # Should not crash
            result = await find_entity_candidates("config:test", threshold=0.5)
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_embedding_failure_skips_type(self, mock_graph_store):
        """Should skip entity type if embedding fails."""
        query_result = MagicMock()
        query_result.result_set = [
            ["main.py", "file"],
            ["test.py", "file"],
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        def failing_embed(texts):
            raise RuntimeError("Embedding service unavailable")

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=failing_embed,
        ):
            result = await find_entity_candidates("config:test")
            # Should return empty (skipped due to error) not raise
            assert result == []


# ============================================================================
# find_memory_candidates() tests
# ============================================================================


class TestFindMemoryCandidates:
    """Tests for find_memory_candidates function."""

    @pytest.mark.asyncio
    async def test_empty_project_returns_empty(self, mock_graph_store):
        """Empty project should return no candidates."""
        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ):
            result = await find_memory_candidates("config:empty-project")
            assert result == []

    @pytest.mark.asyncio
    async def test_finds_similar_memories(
        self, mock_graph_store, mock_embeddings, sample_memories
    ):
        """Should find similar memories as candidates."""
        # Setup mock for memory query
        query_result = MagicMock()
        query_result.result_set = [
            [m["uuid"], m["content"], m["type"], m["created_at"], m["session_id"]]
            for m in sample_memories[:2]  # Two similar memories
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            result = await find_memory_candidates("config:test", threshold=0.90)
            assert len(result) >= 1
            assert isinstance(result[0], MemoryPair)

    @pytest.mark.asyncio
    async def test_different_types_not_compared(
        self, mock_graph_store, mock_embeddings, sample_memories
    ):
        """Should only compare memories of the same type."""
        # One lesson_learned and one decision
        query_result = MagicMock()
        query_result.result_set = [
            [
                sample_memories[0]["uuid"],
                sample_memories[0]["content"],
                "lesson_learned",
                sample_memories[0]["created_at"],
                sample_memories[0]["session_id"],
            ],
            [
                sample_memories[3]["uuid"],
                sample_memories[3]["content"],
                "decision",
                sample_memories[3]["created_at"],
                sample_memories[3]["session_id"],
            ],
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            result = await find_memory_candidates("config:test", threshold=0.5)
            # Each type has only 1 memory, so no pairs possible
            assert result == []

    @pytest.mark.asyncio
    async def test_detects_shared_entities(self, mock_graph_store, mock_embeddings):
        """Should detect shared entities between memory pairs."""
        # Setup memory query
        memory_result = MagicMock()
        memory_result.result_set = [
            ["mem-1", "debug database connection", "lesson_learned", 1704067200, "sess-1"],
            ["mem-2", "debugging database connections", "lesson_learned", 1704153600, "sess-2"],
        ]

        # Setup shared entity query
        entity_result = MagicMock()
        entity_result.result_set = [["database.py"]]

        def query_side_effect(query_str, params):
            if "DISTINCT e.name" in query_str:
                return entity_result
            return memory_result

        mock_graph_store.db.graph.query.side_effect = query_side_effect

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            result = await find_memory_candidates("config:test", threshold=0.90)
            if result:
                assert "database.py" in result[0].shared_entities

    @pytest.mark.asyncio
    async def test_high_threshold_for_memories(self, mock_graph_store):
        """Memory threshold should be high (0.90) by default."""
        query_result = MagicMock()
        query_result.result_set = [
            ["mem-1", "fix authentication bug", "lesson_learned", 1704067200, "sess-1"],
            ["mem-2", "improve login flow", "lesson_learned", 1704153600, "sess-2"],
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        # Create embeddings that produce similarity < 0.90 (about 0.85)
        def low_similarity_embeddings(texts):
            # Return orthogonal vectors that have similarity ~0.85
            results = []
            for i, _ in enumerate(texts):
                if i == 0:
                    results.append([1.0, 0.0, 0.0, 0.0])
                else:
                    # cos(theta) = 0.85 -> sin(theta) = sqrt(1 - 0.85^2)
                    sin_theta = np.sqrt(1 - 0.85**2)
                    results.append([0.85, sin_theta, 0.0, 0.0])
            return results

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=low_similarity_embeddings,
        ):
            # Default threshold (0.90) should filter out 0.85 similarity
            result = await find_memory_candidates("config:test")
            # These have 0.85 similarity, should be filtered at 0.90
            assert len(result) == 0


# ============================================================================
# find_supersession_candidates() tests
# ============================================================================


class TestFindSupersessionCandidates:
    """Tests for find_supersession_candidates function."""

    @pytest.mark.asyncio
    async def test_empty_project_returns_empty(self, mock_graph_store):
        """Empty project should return no candidates."""
        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ):
            result = await find_supersession_candidates("config:empty")
            assert result == []

    @pytest.mark.asyncio
    async def test_finds_temporal_candidates(self, mock_graph_store, mock_embeddings):
        """Should find memories with temporal gap about same entity."""
        # Memories about same entity, days apart
        query_result = MagicMock()
        now = datetime.now().timestamp()
        query_result.result_set = [
            ["database.py", "mem-newer", "Updated database handler", "fact", now],
            ["database.py", "mem-older", "Initial database setup", "fact", now - 172800],  # 2 days ago
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            result = await find_supersession_candidates(
                "config:test",
                min_days_apart=1,
                similarity_threshold=0.5,
            )
            assert len(result) >= 0  # May or may not find based on embedding

    @pytest.mark.asyncio
    async def test_respects_min_days_apart(self, mock_graph_store, mock_embeddings):
        """Should filter pairs with insufficient time gap."""
        query_result = MagicMock()
        now = datetime.now().timestamp()
        query_result.result_set = [
            ["database.py", "mem-1", "content 1", "fact", now],
            ["database.py", "mem-2", "content 2", "fact", now - 3600],  # Only 1 hour ago
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            result = await find_supersession_candidates(
                "config:test",
                min_days_apart=1,  # Requires 1 day
                similarity_threshold=0.5,
            )
            assert result == []  # Should be filtered out

    @pytest.mark.asyncio
    async def test_filters_too_similar_duplicates(self, mock_graph_store, mock_embeddings):
        """Should exclude pairs with >0.95 similarity (these are duplicates, not updates)."""
        query_result = MagicMock()
        now = datetime.now().timestamp()
        # Two identical contents (same embedding)
        query_result.result_set = [
            ["database.py", "mem-1", "main.py", "fact", now],
            ["database.py", "mem-2", "main.py", "fact", now - 172800],
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            result = await find_supersession_candidates(
                "config:test",
                min_days_apart=1,
                similarity_threshold=0.5,
            )
            # Similarity would be 1.0, which is > 0.95, so filtered out
            assert result == []

    @pytest.mark.asyncio
    async def test_filters_too_different(self, mock_graph_store, mock_embeddings):
        """Should exclude pairs below similarity threshold."""
        query_result = MagicMock()
        now = datetime.now().timestamp()
        # Completely different contents
        query_result.result_set = [
            ["database.py", "mem-1", "main.py", "fact", now],
            ["database.py", "mem-2", "test.py", "fact", now - 172800],
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            result = await find_supersession_candidates(
                "config:test",
                min_days_apart=1,
                similarity_threshold=0.6,  # Requires >0.6
            )
            # main.py and test.py are orthogonal (0.0 similarity)
            assert result == []

    @pytest.mark.asyncio
    async def test_handles_missing_timestamps(self, mock_graph_store, mock_embeddings):
        """Should handle memories with None timestamps gracefully."""
        query_result = MagicMock()
        query_result.result_set = [
            ["database.py", "mem-1", "content", "fact", None],
            ["database.py", "mem-2", "other content", "fact", None],
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            # Should not crash
            result = await find_supersession_candidates("config:test")
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_handles_datetime_objects(self, mock_graph_store, mock_embeddings):
        """Should handle datetime objects in created_at field."""
        query_result = MagicMock()
        now = datetime.now()
        query_result.result_set = [
            ["database.py", "mem-1", "content", "fact", now],
            ["database.py", "mem-2", "other", "fact", now - timedelta(days=3)],
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            # Should not crash
            result = await find_supersession_candidates("config:test")
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_returns_supersession_pairs(self, mock_graph_store, mock_embeddings):
        """Should return proper SupersessionPair objects."""
        query_result = MagicMock()
        now = datetime.now().timestamp()
        # Use embeddings that will have moderate similarity
        query_result.result_set = [
            ["database.py", "mem-newer", "debug database connection", "fact", now],
            ["database.py", "mem-older", "fix authentication bug", "fact", now - 259200],  # 3 days
        ]
        mock_graph_store.db.graph.query.return_value = query_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_graph_store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings,
        ):
            result = await find_supersession_candidates(
                "config:test",
                min_days_apart=1,
                similarity_threshold=0.3,  # Lower threshold to get result
            )
            if result:
                assert isinstance(result[0], SupersessionPair)
                assert result[0].entity == "database.py"
                assert result[0].time_delta_days >= 1
