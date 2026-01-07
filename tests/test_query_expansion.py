"""Tests for query expansion functionality.

Tests the LLM-based query expansion that improves recall by generating
semantic variations of search queries.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestExpandQuery:
    """Test expand_query() function."""

    @pytest.mark.asyncio
    async def test_returns_valid_json_array(self):
        """expand_query should return a list of strings."""
        from simplemem_lite.memory import expand_query
        from simplemem_lite.config import Config

        config = Config()

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='["database connection", "db connection", "database connectivity", "sql connection"]'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await expand_query("database connection", config)

        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(v, str) for v in result)

    @pytest.mark.asyncio
    async def test_original_query_always_first(self):
        """Original query should always be first in results."""
        from simplemem_lite.memory import expand_query
        from simplemem_lite.config import Config

        config = Config()
        original_query = "authentication error"

        # Mock LLM response that doesn't include original first
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='["auth error", "login failure", "authentication error"]'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await expand_query(original_query, config)

        assert result[0] == original_query

    @pytest.mark.asyncio
    async def test_deduplicates_variations(self):
        """Duplicate variations should be removed."""
        from simplemem_lite.memory import expand_query
        from simplemem_lite.config import Config

        config = Config()

        # Mock LLM response with duplicates
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='["query", "variation", "query", "variation", "unique"]'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await expand_query("query", config)

        # Should only have unique entries
        assert len(result) == len(set(result))
        assert "query" in result
        assert "variation" in result
        assert "unique" in result

    @pytest.mark.asyncio
    async def test_caps_at_five_variations(self):
        """Result should be capped at 5 variations."""
        from simplemem_lite.memory import expand_query
        from simplemem_lite.config import Config

        config = Config()

        # Mock LLM response with many variations
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"]'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await expand_query("original", config)

        assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        """Should return original query on invalid JSON."""
        from simplemem_lite.memory import expand_query
        from simplemem_lite.config import Config

        config = Config()
        original_query = "test query"

        # Mock LLM response with invalid JSON
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="not valid json at all"))
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await expand_query(original_query, config)

        # json_repair may try to fix it, but worst case we get original
        assert original_query in result

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        """Should return original query on empty LLM response."""
        from simplemem_lite.memory import expand_query
        from simplemem_lite.config import Config

        config = Config()
        original_query = "test query"

        # Mock empty response
        mock_response = MagicMock()
        mock_response.choices = []

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await expand_query(original_query, config)

        assert result == [original_query]

    @pytest.mark.asyncio
    async def test_handles_llm_exception(self):
        """Should return original query on LLM exception."""
        from simplemem_lite.memory import expand_query
        from simplemem_lite.config import Config

        config = Config()
        original_query = "test query"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM API error")
            result = await expand_query(original_query, config)

        assert result == [original_query]

    @pytest.mark.asyncio
    async def test_handles_empty_list_response(self):
        """Should return original query when LLM returns empty list."""
        from simplemem_lite.memory import expand_query
        from simplemem_lite.config import Config

        config = Config()
        original_query = "test query"

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="[]"))
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await expand_query(original_query, config)

        assert result == [original_query]

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_variations(self):
        """Variations should have whitespace stripped."""
        from simplemem_lite.memory import expand_query
        from simplemem_lite.config import Config

        config = Config()

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='["  padded  ", "  also padded  "]'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await expand_query("original", config)

        # Check stripped values exist (original first, then stripped)
        assert "original" in result
        assert "padded" in result
        assert "also padded" in result


class TestSearchMultiQuery:
    """Test search_multi_query() method."""

    def test_deduplicates_by_uuid(self):
        """Should deduplicate results by UUID, keeping highest score."""
        from simplemem_lite.memory import MemoryStore, Memory

        with patch.object(MemoryStore, "__init__", lambda x, y=None: None):
            store = MemoryStore()
            store.config = MagicMock()

            # Mock search to return overlapping results with different scores
            call_count = [0]

            def mock_search(query, limit, use_graph, type_filter, project_id):
                call_count[0] += 1
                if call_count[0] == 1:
                    return [
                        Memory(uuid="uuid-1", content="A", type="fact", created_at=0, score=0.8),
                        Memory(uuid="uuid-2", content="B", type="fact", created_at=0, score=0.7),
                    ]
                else:
                    return [
                        Memory(uuid="uuid-1", content="A", type="fact", created_at=0, score=0.9),  # Higher
                        Memory(uuid="uuid-3", content="C", type="fact", created_at=0, score=0.6),
                    ]

            store.search = mock_search

            results = store.search_multi_query(
                queries=["query1", "query2"],
                limit=10,
                use_graph=True,
                type_filter=None,
                project_id=None,
            )

        # Should have 3 unique UUIDs
        uuids = [r.uuid for r in results]
        assert len(uuids) == 3
        assert set(uuids) == {"uuid-1", "uuid-2", "uuid-3"}

        # uuid-1 should have higher score (0.9)
        uuid1_result = next(r for r in results if r.uuid == "uuid-1")
        assert uuid1_result.score == 0.9

    def test_returns_empty_for_empty_queries(self):
        """Should return empty list for empty query list."""
        from simplemem_lite.memory import MemoryStore

        with patch.object(MemoryStore, "__init__", lambda x, y=None: None):
            store = MemoryStore()
            results = store.search_multi_query(
                queries=[],
                limit=10,
                use_graph=True,
                type_filter=None,
                project_id=None,
            )

        assert results == []

    def test_respects_limit(self):
        """Should respect the limit parameter."""
        from simplemem_lite.memory import MemoryStore, Memory

        with patch.object(MemoryStore, "__init__", lambda x, y=None: None):
            store = MemoryStore()
            store.config = MagicMock()

            # Mock search to return many results
            def mock_search(query, limit, use_graph, type_filter, project_id):
                return [
                    Memory(uuid=f"uuid-{i}", content=f"Content {i}", type="fact", created_at=0, score=1.0 - i * 0.1)
                    for i in range(10)
                ]

            store.search = mock_search

            results = store.search_multi_query(
                queries=["query1"],
                limit=3,
                use_graph=True,
                type_filter=None,
                project_id=None,
            )

        assert len(results) == 3

    def test_sorts_by_score_descending(self):
        """Results should be sorted by score in descending order."""
        from simplemem_lite.memory import MemoryStore, Memory

        with patch.object(MemoryStore, "__init__", lambda x, y=None: None):
            store = MemoryStore()
            store.config = MagicMock()

            def mock_search(query, limit, use_graph, type_filter, project_id):
                return [
                    Memory(uuid="uuid-low", content="Low", type="fact", created_at=0, score=0.3),
                    Memory(uuid="uuid-high", content="High", type="fact", created_at=0, score=0.9),
                    Memory(uuid="uuid-mid", content="Mid", type="fact", created_at=0, score=0.6),
                ]

            store.search = mock_search

            results = store.search_multi_query(
                queries=["query1"],
                limit=10,
                use_graph=True,
                type_filter=None,
                project_id=None,
            )

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestSearchMemoriesWithExpansion:
    """Test search_memories MCP tool with expand_query parameter."""

    @pytest.mark.asyncio
    async def test_expand_query_false_uses_normal_search(self):
        """With expand_query=False, should use normal search."""
        from simplemem_lite.server import search_memories
        from simplemem_lite.memory import Memory
        import simplemem_lite.server as server_module

        mock_store = MagicMock()
        mock_store.search.return_value = [
            Memory(uuid="uuid-1", content="Test", type="fact", created_at=0, score=0.8)
        ]

        # Patch the _deps object's internal attributes
        with patch.object(server_module._deps, "_store", mock_store):
            with patch.object(server_module._deps, "_initialized", True):
                results = await search_memories(
                    query="test query",
                    limit=10,
                    expand_query=False,
                )

        mock_store.search.assert_called_once()
        mock_store.search_multi_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_expand_query_true_uses_multi_query_search(self):
        """With expand_query=True, should expand and use multi-query search."""
        from simplemem_lite.server import search_memories
        from simplemem_lite.memory import Memory
        import simplemem_lite.server as server_module

        mock_store = MagicMock()
        mock_store.search_multi_query.return_value = [
            Memory(uuid="uuid-1", content="Test", type="fact", created_at=0, score=0.8)
        ]

        mock_config = MagicMock()

        # Mock expand_query to return variations
        async def mock_expand(query, config):
            return [query, "variation1", "variation2"]

        with patch.object(server_module._deps, "_store", mock_store):
            with patch.object(server_module._deps, "_config", mock_config):
                with patch.object(server_module._deps, "_initialized", True):
                    with patch("simplemem_lite.memory.expand_query", mock_expand):
                        results = await search_memories(
                            query="test query",
                            limit=10,
                            expand_query=True,
                        )

        mock_store.search_multi_query.assert_called_once()
        mock_store.search.assert_not_called()
