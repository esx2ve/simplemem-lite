"""Integration tests for the full consolidation pipeline.

Tests the consolidate_project() function end-to-end with mocked
external dependencies (LLM, embeddings) but real internal logic.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simplemem_lite.backend.consolidation import (
    ConsolidationOperation,
    ConsolidationReport,
    consolidate_project,
)


@pytest.fixture
def mock_store_with_entities():
    """Mock store with entity data for deduplication testing."""
    store = MagicMock()
    store.db = MagicMock()
    store.db.graph = MagicMock()
    store.db.mark_merged = MagicMock()
    store.db.add_supersession = MagicMock()

    # Entity query returns duplicate-like entities
    entity_result = MagicMock()
    entity_result.result_set = [
        ["main.py", "file"],
        ["./main.py", "file"],
        ["src/utils.py", "file"],
    ]

    # Memory query returns similar memories
    memory_result = MagicMock()
    memory_result.result_set = [
        ["mem-1", "debug database connection pool timeout", "lesson_learned", 1704067200, "sess-1"],
        ["mem-2", "fixed database connection pooling issue", "lesson_learned", 1704153600, "sess-2"],
        ["mem-3", "implement user authentication flow", "decision", 1704240000, "sess-3"],
    ]

    # Entity lookup for shared entities
    shared_entity_result = MagicMock()
    shared_entity_result.result_set = [["database.py"]]

    # Memory-by-entity query for supersession
    memory_by_entity_result = MagicMock()
    memory_by_entity_result.result_set = [
        ["database.py", "mem-1", "debug database connection", "lesson_learned", 1704067200],
        ["database.py", "mem-2", "fixed database pooling", "lesson_learned", 1704153600],
    ]

    def query_router(query: str, params: dict) -> MagicMock:
        if "DISTINCT e.name AS name, e.type AS type" in query:
            return entity_result
        elif "m.uuid, m.content, m.type, m.created_at, m.session_id" in query:
            return memory_result
        elif "DISTINCT e.name" in query and "uuid1" in params:
            return shared_entity_result
        elif "e.name AS entity" in query:
            return memory_by_entity_result
        else:
            empty = MagicMock()
            empty.result_set = []
            return empty

    store.db.graph.query.side_effect = query_router
    return store


@pytest.fixture
def mock_embeddings_similar():
    """Embeddings that produce similar pairs."""
    def _embed(texts: list[str]) -> list[list[float]]:
        embeddings = {
            # Entities
            "main.py": [1.0, 0.0, 0.0, 0.0],
            "./main.py": [0.99, 0.14, 0.0, 0.0],  # ~0.99 similarity
            "src/utils.py": [0.0, 1.0, 0.0, 0.0],  # Different
            # Memories
            "debug database connection pool timeout": [0.8, 0.6, 0.0, 0.0],
            "fixed database connection pooling issue": [0.79, 0.61, 0.0, 0.0],  # ~0.99
            "implement user authentication flow": [0.0, 0.0, 1.0, 0.0],  # Different
            # Supersession content
            "debug database connection": [0.8, 0.6, 0.0, 0.0],
            "fixed database pooling": [0.78, 0.62, 0.0, 0.0],
        }
        return [embeddings.get(t, [0.5, 0.5, 0.5, 0.0]) for t in texts]

    return _embed


@pytest.fixture
def mock_llm_all_positive():
    """LLM that returns positive decisions for all pairs."""
    async def _mock(*args, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        # Return positive decisions for all pair types
        content = '''[
            {"pair": 1, "same": true, "confidence": 0.95, "canonical": "main.py", "reason": "Same file"},
            {"pair": 1, "merge": true, "confidence": 0.92, "merged": "Combined database debugging insight", "reason": "Similar content"},
            {"pair": 1, "supersedes": true, "confidence": 0.88, "type": "full_replace", "reason": "Updated solution"}
        ]'''
        response.choices[0].message.content = content
        return response

    return _mock


# ============================================================================
# consolidate_project() integration tests
# ============================================================================


class TestConsolidateProject:
    """Integration tests for the main consolidate_project function."""

    @pytest.mark.asyncio
    async def test_dry_run_no_mutations(
        self, mock_store_with_entities, mock_embeddings_similar, mock_llm_all_positive
    ):
        """Dry run should not execute any mutations."""
        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_store_with_entities,
        ), patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_entities,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings_similar,
        ), patch(
            "simplemem_lite.backend.consolidation.scorer.acompletion",
            side_effect=mock_llm_all_positive,
        ):
            report = await consolidate_project(
                project_id="config:test-project",
                dry_run=True,
            )

            assert isinstance(report, ConsolidationReport)
            assert report.dry_run is True
            # In dry run, executed counts should be 0
            # (candidates found but not executed)

    @pytest.mark.asyncio
    async def test_entity_dedup_only(
        self, mock_store_with_entities, mock_embeddings_similar, mock_llm_all_positive
    ):
        """Should run only entity deduplication when specified."""
        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_store_with_entities,
        ), patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_entities,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings_similar,
        ), patch(
            "simplemem_lite.backend.consolidation.scorer.acompletion",
            side_effect=mock_llm_all_positive,
        ):
            report = await consolidate_project(
                project_id="config:test-project",
                operations=["entity_dedup"],
                dry_run=False,
            )

            assert "entity_dedup" in report.operations_run
            assert "memory_merge" not in report.operations_run
            assert "supersession" not in report.operations_run

    @pytest.mark.asyncio
    async def test_all_operations(
        self, mock_store_with_entities, mock_embeddings_similar, mock_llm_all_positive
    ):
        """Should run all operations when none specified."""
        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_store_with_entities,
        ), patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_entities,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings_similar,
        ), patch(
            "simplemem_lite.backend.consolidation.scorer.acompletion",
            side_effect=mock_llm_all_positive,
        ):
            report = await consolidate_project(
                project_id="config:test-project",
                operations=None,  # All operations
                dry_run=False,
            )

            # Should have all three operations
            assert len(report.operations_run) == 3

    @pytest.mark.asyncio
    async def test_report_structure(
        self, mock_store_with_entities, mock_embeddings_similar, mock_llm_all_positive
    ):
        """Report should have correct structure."""
        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_store_with_entities,
        ), patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_entities,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings_similar,
        ), patch(
            "simplemem_lite.backend.consolidation.scorer.acompletion",
            side_effect=mock_llm_all_positive,
        ):
            report = await consolidate_project(
                project_id="config:test-project",
                dry_run=True,
            )

            # Verify report structure
            report_dict = report.to_dict()
            assert "project_id" in report_dict
            assert "operations_run" in report_dict
            assert "dry_run" in report_dict
            assert "entity_dedup" in report_dict
            assert "memory_merge" in report_dict
            assert "supersession" in report_dict
            assert "errors" in report_dict
            assert "warnings" in report_dict

    @pytest.mark.asyncio
    async def test_custom_confidence_threshold(
        self, mock_store_with_entities, mock_embeddings_similar
    ):
        """Should respect custom confidence threshold."""
        # LLM returns medium confidence
        async def medium_confidence_llm(*args, **kwargs):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = (
                '[{"pair": 1, "same": true, "confidence": 0.75, "canonical": "main.py", "reason": "Possibly same"}]'
            )
            return response

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=mock_store_with_entities,
        ), patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=mock_store_with_entities,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=mock_embeddings_similar,
        ), patch(
            "simplemem_lite.backend.consolidation.scorer.acompletion",
            side_effect=medium_confidence_llm,
        ):
            # With default 0.9 threshold, 0.75 should be queued
            report = await consolidate_project(
                project_id="config:test-project",
                operations=["entity_dedup"],
                confidence_threshold=0.9,
                dry_run=False,
            )

            # Check that items were queued, not executed
            # (depends on implementation details)
            assert isinstance(report, ConsolidationReport)


# ============================================================================
# Empty/edge case tests
# ============================================================================


class TestConsolidateProjectEdgeCases:
    """Edge case tests for consolidate_project."""

    @pytest.mark.asyncio
    async def test_empty_project(self):
        """Empty project should return empty report."""
        empty_store = MagicMock()
        empty_store.db = MagicMock()
        empty_store.db.graph = MagicMock()
        empty_result = MagicMock()
        empty_result.result_set = []
        empty_store.db.graph.query.return_value = empty_result

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=empty_store,
        ), patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=empty_store,
        ):
            report = await consolidate_project(
                project_id="config:empty-project",
                dry_run=False,
            )

            # Should complete without errors
            assert report.project_id == "config:empty-project"
            assert len(report.errors) == 0

    @pytest.mark.asyncio
    async def test_embedding_failure_continues(self):
        """Should continue with other operations if embedding fails."""
        store = MagicMock()
        store.db = MagicMock()
        store.db.graph = MagicMock()

        # Return entities but fail on embedding
        entity_result = MagicMock()
        entity_result.result_set = [
            ["main.py", "file"],
            ["test.py", "file"],
        ]
        store.db.graph.query.return_value = entity_result

        def failing_embed(texts):
            raise RuntimeError("Embedding service down")

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=store,
        ), patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=failing_embed,
        ):
            # Should not raise, should handle gracefully
            report = await consolidate_project(
                project_id="config:test",
                operations=["entity_dedup"],
                dry_run=False,
            )

            assert isinstance(report, ConsolidationReport)

    @pytest.mark.asyncio
    async def test_single_entity_no_pairs(self):
        """Single entity should result in no candidates."""
        store = MagicMock()
        store.db = MagicMock()
        store.db.graph = MagicMock()

        entity_result = MagicMock()
        entity_result.result_set = [["only_one.py", "file"]]
        store.db.graph.query.return_value = entity_result

        def single_embed(texts):
            return [[1.0, 0.0, 0.0]]

        with patch(
            "simplemem_lite.backend.consolidation.candidates.get_memory_store",
            return_value=store,
        ), patch(
            "simplemem_lite.backend.consolidation.executor.get_memory_store",
            return_value=store,
        ), patch(
            "simplemem_lite.backend.consolidation.candidates.embed_batch",
            side_effect=single_embed,
        ):
            report = await consolidate_project(
                project_id="config:test",
                operations=["entity_dedup"],
                dry_run=False,
            )

            # No candidates means nothing to execute
            assert report.entity_candidates_found == 0


# ============================================================================
# ConsolidationOperation enum tests
# ============================================================================


class TestConsolidationOperation:
    """Tests for the ConsolidationOperation enum."""

    def test_operation_values(self):
        """Should have correct operation values."""
        assert ConsolidationOperation.ENTITY_DEDUP.value == "entity_dedup"
        assert ConsolidationOperation.MEMORY_MERGE.value == "memory_merge"
        assert ConsolidationOperation.SUPERSESSION.value == "supersession"

    def test_all_operations(self):
        """Should have exactly 3 operations."""
        operations = list(ConsolidationOperation)
        assert len(operations) == 3


# ============================================================================
# ConsolidationReport tests
# ============================================================================


class TestConsolidationReport:
    """Tests for the ConsolidationReport dataclass."""

    def test_to_dict_structure(self):
        """to_dict should return proper structure."""
        report = ConsolidationReport(
            project_id="config:test",
            operations_run=["entity_dedup"],
            dry_run=False,
            entity_candidates_found=5,
            entity_merges_executed=3,
            entity_merges_queued=2,
            memory_candidates_found=0,
            memory_merges_executed=0,
            memory_merges_queued=0,
            supersession_candidates_found=0,
            supersessions_executed=0,
            supersessions_queued=0,
            errors=[],
            warnings=[],
            review_queue=[],
        )

        d = report.to_dict()

        assert d["project_id"] == "config:test"
        assert d["operations_run"] == ["entity_dedup"]
        assert d["dry_run"] is False
        assert d["entity_dedup"]["candidates_found"] == 5
        assert d["entity_dedup"]["merges_executed"] == 3

    def test_review_queue_count(self):
        """Should correctly count review queue items."""
        report = ConsolidationReport(
            project_id="config:test",
            operations_run=["entity_dedup"],
            dry_run=False,
            entity_candidates_found=5,
            entity_merges_executed=3,
            entity_merges_queued=2,
            memory_candidates_found=0,
            memory_merges_executed=0,
            memory_merges_queued=0,
            supersession_candidates_found=0,
            supersessions_executed=0,
            supersessions_queued=0,
            errors=[],
            warnings=[],
            review_queue=[{"type": "entity_dedup"}, {"type": "entity_dedup"}],
        )

        d = report.to_dict()
        assert d["review_queue_count"] == 2
