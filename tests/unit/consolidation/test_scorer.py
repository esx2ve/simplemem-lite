"""Unit tests for consolidation LLM scorer.

Tests:
- chunk_list() - batch splitting
- format_*_pairs() - prompt formatting
- score_*_batch() - single batch scoring
- score_*_pairs() - full scoring with batching
- Error handling - malformed JSON, network errors, retries
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simplemem_lite.backend.consolidation.candidates import (
    EntityPair,
    MemoryPair,
    SupersessionPair,
)
from simplemem_lite.backend.consolidation.scorer import (
    BATCH_SIZE,
    EntityDecision,
    MemoryDecision,
    SupersessionDecision,
    call_llm_with_retry,
    chunk_list,
    format_entity_pairs,
    format_memory_pairs,
    format_supersession_pairs,
    score_entity_batch,
    score_entity_pairs,
    score_memory_batch,
    score_memory_pairs,
    score_supersession_batch,
    score_supersession_pairs,
)


# ============================================================================
# chunk_list() tests
# ============================================================================


class TestChunkList:
    """Tests for the chunk_list utility function."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        result = chunk_list([], 5)
        assert result == []

    def test_exact_division(self):
        """List that divides evenly into chunks."""
        result = chunk_list([1, 2, 3, 4, 5, 6], 3)
        assert result == [[1, 2, 3], [4, 5, 6]]

    def test_remainder(self):
        """List with leftover items."""
        result = chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_smaller_than_batch(self):
        """List smaller than batch size."""
        result = chunk_list([1, 2], 10)
        assert result == [[1, 2]]

    def test_single_item(self):
        """Single item list."""
        result = chunk_list([1], 5)
        assert result == [[1]]

    def test_batch_size_one(self):
        """Batch size of 1."""
        result = chunk_list([1, 2, 3], 1)
        assert result == [[1], [2], [3]]


# ============================================================================
# format_*_pairs() tests
# ============================================================================


class TestFormatEntityPairs:
    """Tests for entity pair formatting."""

    def test_formats_single_pair(self):
        """Should format a single entity pair correctly."""
        pair = EntityPair(
            entity_a={"name": "main.py", "type": "file"},
            entity_b={"name": "./main.py", "type": "file"},
            similarity=0.95,
            entity_type="file",
        )
        result = format_entity_pairs([pair])
        assert '1. Entity A: "main.py"' in result
        assert 'Entity B: "./main.py"' in result
        assert "(type: file)" in result

    def test_formats_multiple_pairs(self):
        """Should number multiple pairs correctly."""
        pairs = [
            EntityPair(
                entity_a={"name": "a.py", "type": "file"},
                entity_b={"name": "b.py", "type": "file"},
                similarity=0.9,
                entity_type="file",
            ),
            EntityPair(
                entity_a={"name": "c.py", "type": "file"},
                entity_b={"name": "d.py", "type": "file"},
                similarity=0.85,
                entity_type="file",
            ),
        ]
        result = format_entity_pairs(pairs)
        assert "1." in result
        assert "2." in result


class TestFormatMemoryPairs:
    """Tests for memory pair formatting."""

    def test_formats_with_shared_entities(self):
        """Should include shared entities in formatting."""
        pair = MemoryPair(
            memory_a={"uuid": "m1", "content": "debug db", "type": "lesson_learned"},
            memory_b={"uuid": "m2", "content": "fix db bug", "type": "lesson_learned"},
            similarity=0.9,
            shared_entities=["database.py", "config.py"],
        )
        result = format_memory_pairs([pair])
        assert "database.py" in result
        assert "lesson_learned" in result

    def test_truncates_long_content(self):
        """Should truncate content to avoid token overflow."""
        long_content = "x" * 500
        pair = MemoryPair(
            memory_a={"uuid": "m1", "content": long_content, "type": "fact"},
            memory_b={"uuid": "m2", "content": "short", "type": "fact"},
            similarity=0.5,
            shared_entities=[],
        )
        result = format_memory_pairs([pair])
        # Content should be truncated (200 chars max + "...")
        assert len(result) < len(long_content)

    def test_handles_empty_shared_entities(self):
        """Should handle case with no shared entities."""
        pair = MemoryPair(
            memory_a={"uuid": "m1", "content": "content", "type": "fact"},
            memory_b={"uuid": "m2", "content": "other", "type": "fact"},
            similarity=0.5,
            shared_entities=[],
        )
        result = format_memory_pairs([pair])
        assert "Shared entities: none" in result


class TestFormatSupersessionPairs:
    """Tests for supersession pair formatting."""

    def test_includes_time_delta(self):
        """Should include days apart in formatting."""
        pair = SupersessionPair(
            newer={"uuid": "new", "content": "updated solution"},
            older={"uuid": "old", "content": "original approach"},
            entity="database.py",
            similarity=0.7,
            time_delta_days=5,
        )
        result = format_supersession_pairs([pair])
        assert "5 days ago" in result
        assert "NEWER" in result
        assert "OLDER" in result
        assert "database.py" in result


# ============================================================================
# call_llm_with_retry() tests
# ============================================================================


class TestCallLlmWithRetry:
    """Tests for LLM call with retry logic."""

    @pytest.mark.asyncio
    async def test_valid_json_response(self, mock_llm_response):
        """Should parse valid JSON response."""
        response = mock_llm_response('[{"pair": 1, "same": true, "confidence": 0.95}]')

        with patch("simplemem_lite.backend.consolidation.scorer.acompletion") as mock:
            mock.return_value = response
            result = await call_llm_with_retry("test prompt")
            assert result == [{"pair": 1, "same": True, "confidence": 0.95}]

    @pytest.mark.asyncio
    async def test_markdown_code_block_stripping(self, mock_llm_response):
        """Should strip markdown code blocks from response."""
        response = mock_llm_response('```json\n[{"pair": 1, "same": true}]\n```')

        with patch("simplemem_lite.backend.consolidation.scorer.acompletion") as mock:
            mock.return_value = response
            result = await call_llm_with_retry("test prompt")
            assert result == [{"pair": 1, "same": True}]

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(self, mock_llm_response):
        """Should return empty list on malformed JSON after retries."""
        response = mock_llm_response("not valid json")

        with patch("simplemem_lite.backend.consolidation.scorer.acompletion") as mock:
            mock.return_value = response
            result = await call_llm_with_retry("test prompt", max_retries=2)
            assert result == []

    @pytest.mark.asyncio
    async def test_retry_on_network_error(self, mock_llm_response):
        """Should retry on network errors."""
        good_response = mock_llm_response('[{"pair": 1}]')

        call_count = 0

        async def flaky_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network unavailable")
            return good_response

        with patch(
            "simplemem_lite.backend.consolidation.scorer.acompletion",
            side_effect=flaky_call,
        ):
            result = await call_llm_with_retry("test prompt", max_retries=3)
            assert result == [{"pair": 1}]
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, mock_llm_response):
        """Should give up after max retries."""
        with patch(
            "simplemem_lite.backend.consolidation.scorer.acompletion",
            side_effect=ConnectionError("Network unavailable"),
        ):
            result = await call_llm_with_retry("test prompt", max_retries=2)
            assert result == []

    @pytest.mark.asyncio
    async def test_non_list_response_returns_empty(self, mock_llm_response):
        """Should return empty if LLM returns non-list JSON."""
        response = mock_llm_response('{"pair": 1, "same": true}')  # Object, not array

        with patch("simplemem_lite.backend.consolidation.scorer.acompletion") as mock:
            mock.return_value = response
            result = await call_llm_with_retry("test prompt")
            assert result == []


# ============================================================================
# score_entity_batch() tests
# ============================================================================


class TestScoreEntityBatch:
    """Tests for single-batch entity scoring."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        """Empty input should return empty list."""
        result = await score_entity_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_parses_llm_response(self, mock_llm_response):
        """Should correctly parse LLM classification response."""
        response = mock_llm_response(
            '[{"pair": 1, "same": true, "confidence": 0.95, "canonical": "main.py", "reason": "Same file"}]'
        )

        pair = EntityPair(
            entity_a={"name": "main.py", "type": "file"},
            entity_b={"name": "./main.py", "type": "file"},
            similarity=0.95,
            entity_type="file",
        )

        with patch("simplemem_lite.backend.consolidation.scorer.acompletion") as mock:
            mock.return_value = response
            result = await score_entity_batch([pair])

            assert len(result) == 1
            assert isinstance(result[0], EntityDecision)
            assert result[0].same_entity is True
            assert result[0].confidence == 0.95
            assert result[0].canonical_name == "main.py"

    @pytest.mark.asyncio
    async def test_missing_pair_in_response(self, mock_llm_response):
        """Should handle missing pair numbers with default values."""
        # Response missing pair 2
        response = mock_llm_response(
            '[{"pair": 1, "same": true, "confidence": 0.9, "canonical": "a", "reason": "test"}]'
        )

        pairs = [
            EntityPair(
                entity_a={"name": "a", "type": "file"},
                entity_b={"name": "b", "type": "file"},
                similarity=0.9,
                entity_type="file",
            ),
            EntityPair(
                entity_a={"name": "c", "type": "file"},
                entity_b={"name": "d", "type": "file"},
                similarity=0.85,
                entity_type="file",
            ),
        ]

        with patch("simplemem_lite.backend.consolidation.scorer.acompletion") as mock:
            mock.return_value = response
            result = await score_entity_batch(pairs)

            assert len(result) == 2
            # First pair has response
            assert result[0].same_entity is True
            # Second pair missing from response - should default to not merging
            assert result[1].same_entity is False
            assert result[1].confidence == 0.0
            assert "No LLM response" in result[1].reason


# ============================================================================
# score_memory_batch() tests
# ============================================================================


class TestScoreMemoryBatch:
    """Tests for single-batch memory scoring."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        """Empty input should return empty list."""
        result = await score_memory_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_parses_merge_decision(self, mock_llm_response):
        """Should correctly parse merge decision."""
        response = mock_llm_response(
            '[{"pair": 1, "merge": true, "confidence": 0.92, "merged": "Combined content", "reason": "Similar"}]'
        )

        pair = MemoryPair(
            memory_a={"uuid": "m1", "content": "debug db", "type": "lesson"},
            memory_b={"uuid": "m2", "content": "fix db", "type": "lesson"},
            similarity=0.92,
            shared_entities=["db.py"],
        )

        with patch("simplemem_lite.backend.consolidation.scorer.acompletion") as mock:
            mock.return_value = response
            result = await score_memory_batch([pair])

            assert len(result) == 1
            assert isinstance(result[0], MemoryDecision)
            assert result[0].should_merge is True
            assert result[0].merged_content == "Combined content"


# ============================================================================
# score_supersession_batch() tests
# ============================================================================


class TestScoreSupersessionBatch:
    """Tests for single-batch supersession scoring."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        """Empty input should return empty list."""
        result = await score_supersession_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_parses_supersession_types(self, mock_llm_response):
        """Should correctly parse supersession types."""
        response = mock_llm_response(
            '[{"pair": 1, "supersedes": true, "confidence": 0.88, "type": "full_replace", "reason": "Newer replaces"}]'
        )

        pair = SupersessionPair(
            newer={"uuid": "new", "content": "updated"},
            older={"uuid": "old", "content": "original"},
            entity="db.py",
            similarity=0.7,
            time_delta_days=5,
        )

        with patch("simplemem_lite.backend.consolidation.scorer.acompletion") as mock:
            mock.return_value = response
            result = await score_supersession_batch([pair])

            assert len(result) == 1
            assert isinstance(result[0], SupersessionDecision)
            assert result[0].supersedes is True
            assert result[0].supersession_type == "full_replace"

    @pytest.mark.asyncio
    async def test_handles_partial_update_type(self, mock_llm_response):
        """Should handle partial_update supersession type."""
        response = mock_llm_response(
            '[{"pair": 1, "supersedes": true, "confidence": 0.75, "type": "partial_update", "reason": "Adds info"}]'
        )

        pair = SupersessionPair(
            newer={"uuid": "new", "content": "extended"},
            older={"uuid": "old", "content": "original"},
            entity="db.py",
            similarity=0.8,
            time_delta_days=2,
        )

        with patch("simplemem_lite.backend.consolidation.scorer.acompletion") as mock:
            mock.return_value = response
            result = await score_supersession_batch([pair])

            assert result[0].supersession_type == "partial_update"

    @pytest.mark.asyncio
    async def test_handles_none_type(self, mock_llm_response):
        """Should handle 'none' supersession type."""
        response = mock_llm_response(
            '[{"pair": 1, "supersedes": false, "confidence": 0.1, "type": "none", "reason": "Different topics"}]'
        )

        pair = SupersessionPair(
            newer={"uuid": "new", "content": "something"},
            older={"uuid": "old", "content": "other"},
            entity="db.py",
            similarity=0.65,
            time_delta_days=10,
        )

        with patch("simplemem_lite.backend.consolidation.scorer.acompletion") as mock:
            mock.return_value = response
            result = await score_supersession_batch([pair])

            assert result[0].supersedes is False
            assert result[0].supersession_type == "none"


# ============================================================================
# score_*_pairs() main entry point tests
# ============================================================================


class TestScoreEntityPairs:
    """Tests for the main score_entity_pairs function."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        """Empty input should return empty list."""
        result = await score_entity_pairs([])
        assert result == []

    @pytest.mark.asyncio
    async def test_batches_correctly(self, mock_llm_response):
        """Should split into correct batch sizes."""
        # Create more pairs than BATCH_SIZE
        pairs = [
            EntityPair(
                entity_a={"name": f"a{i}.py", "type": "file"},
                entity_b={"name": f"b{i}.py", "type": "file"},
                similarity=0.9,
                entity_type="file",
            )
            for i in range(BATCH_SIZE + 2)  # e.g., 10 pairs
        ]

        # Track number of LLM calls
        call_count = 0

        async def mock_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return response for each batch
            return mock_llm_response('[{"pair": 1, "same": true, "confidence": 0.9, "canonical": "a", "reason": "test"}]')

        with patch(
            "simplemem_lite.backend.consolidation.scorer.acompletion",
            side_effect=mock_completion,
        ):
            result = await score_entity_pairs(pairs)
            # Should have made 2 LLM calls (one for each batch)
            assert call_count == 2
            assert len(result) == len(pairs)


class TestScoreMemoryPairs:
    """Tests for the main score_memory_pairs function."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        """Empty input should return empty list."""
        result = await score_memory_pairs([])
        assert result == []


class TestScoreSupersessionPairs:
    """Tests for the main score_supersession_pairs function."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        """Empty input should return empty list."""
        result = await score_supersession_pairs([])
        assert result == []

    @pytest.mark.asyncio
    async def test_processes_all_pairs(self, mock_llm_response):
        """Should return decisions for all input pairs."""
        pairs = [
            SupersessionPair(
                newer={"uuid": f"new{i}", "content": "updated"},
                older={"uuid": f"old{i}", "content": "original"},
                entity=f"file{i}.py",
                similarity=0.7,
                time_delta_days=i + 1,
            )
            for i in range(3)
        ]

        response = mock_llm_response(
            '[{"pair": 1, "supersedes": true, "confidence": 0.8, "type": "full_replace", "reason": "test"},'
            '{"pair": 2, "supersedes": false, "confidence": 0.2, "type": "none", "reason": "test"},'
            '{"pair": 3, "supersedes": true, "confidence": 0.9, "type": "partial_update", "reason": "test"}]'
        )

        with patch("simplemem_lite.backend.consolidation.scorer.acompletion") as mock:
            mock.return_value = response
            result = await score_supersession_pairs(pairs)

            assert len(result) == 3
            assert all(isinstance(d, SupersessionDecision) for d in result)
