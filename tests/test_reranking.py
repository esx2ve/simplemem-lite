"""Tests for LLM reranking functionality.

Tests the LLM-based reranking that improves precision by having
an LLM reorder vector search results by semantic relevance.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestEscapeXmlChars:
    """Test _escape_xml_chars() helper function."""

    def test_escapes_less_than(self):
        """< should be escaped to &lt;."""
        from simplemem_lite.backend.scoring import _escape_xml_chars

        assert _escape_xml_chars("<tag>") == "&lt;tag&gt;"

    def test_escapes_greater_than(self):
        """> should be escaped to &gt;."""
        from simplemem_lite.backend.scoring import _escape_xml_chars

        assert _escape_xml_chars("x > y") == "x &gt; y"

    def test_escapes_ampersand(self):
        """& should be escaped to &amp;."""
        from simplemem_lite.backend.scoring import _escape_xml_chars

        assert _escape_xml_chars("A & B") == "A &amp; B"

    def test_escapes_all_together(self):
        """All XML chars should be escaped together."""
        from simplemem_lite.backend.scoring import _escape_xml_chars

        result = _escape_xml_chars("<script>alert('&XSS')</script>")
        assert result == "&lt;script&gt;alert('&amp;XSS')&lt;/script&gt;"

    def test_returns_unchanged_for_safe_text(self):
        """Safe text without XML chars should be unchanged."""
        from simplemem_lite.backend.scoring import _escape_xml_chars

        safe_text = "This is normal text without special chars."
        assert _escape_xml_chars(safe_text) == safe_text

    def test_handles_empty_string(self):
        """Empty string should return empty string."""
        from simplemem_lite.backend.scoring import _escape_xml_chars

        assert _escape_xml_chars("") == ""


class TestRerankResults:
    """Test rerank_results() async function."""

    @pytest.mark.asyncio
    async def test_returns_results_unchanged_when_under_top_k(self):
        """When results <= top_k, should return as-is without reranking."""
        from simplemem_lite.backend.scoring import rerank_results

        results = [
            {"uuid": "1", "content": "Test 1", "type": "fact"},
            {"uuid": "2", "content": "Test 2", "type": "fact"},
        ]

        result = await rerank_results("test query", results, top_k=10)

        assert result["results"] == results
        assert result["conflicts"] == []
        assert result["rerank_applied"] is False

    @pytest.mark.asyncio
    async def test_reranks_results_using_llm(self):
        """Should reorder results based on LLM response."""
        from simplemem_lite.backend.scoring import rerank_results

        results = [
            {"uuid": "0", "content": "Content 0", "type": "fact"},
            {"uuid": "1", "content": "Content 1", "type": "fact"},
            {"uuid": "2", "content": "Content 2", "type": "fact"},
        ]

        # Mock LLM to return reversed order
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"indices": [2, 1, 0], "conflicts": []}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rerank_results("test query", results, top_k=2)

        assert result["rerank_applied"] is True
        assert len(result["results"]) == 2
        assert result["results"][0]["uuid"] == "2"  # First in reranked
        assert result["results"][1]["uuid"] == "1"  # Second in reranked

    @pytest.mark.asyncio
    async def test_detects_conflicts(self):
        """Should return conflicts detected by LLM."""
        from simplemem_lite.backend.scoring import rerank_results

        results = [
            {"uuid": "0", "content": "A is true", "type": "fact"},
            {"uuid": "1", "content": "A is false", "type": "fact"},
            {"uuid": "2", "content": "B is true", "type": "fact"},
        ]

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"indices": [0, 1, 2], "conflicts": [[0, 1, "Contradictory statements about A"]]}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rerank_results("test query", results, top_k=2)

        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0] == [0, 1, "Contradictory statements about A"]

    @pytest.mark.asyncio
    async def test_handles_empty_results(self):
        """Empty results should return empty without reranking."""
        from simplemem_lite.backend.scoring import rerank_results

        result = await rerank_results("test query", [], top_k=10)

        assert result["results"] == []
        assert result["rerank_applied"] is False

    @pytest.mark.asyncio
    async def test_handles_llm_exception(self):
        """LLM exception should fallback to original results."""
        from simplemem_lite.backend.scoring import rerank_results

        results = [
            {"uuid": str(i), "content": f"Content {i}", "type": "fact"}
            for i in range(15)
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM API error")
            result = await rerank_results("test query", results, top_k=5)

        assert result["rerank_applied"] is False
        assert len(result["results"]) == 5
        assert "error" in result
        # Results should be first 5 in original order
        assert result["results"][0]["uuid"] == "0"

    @pytest.mark.asyncio
    async def test_handles_empty_llm_response(self):
        """Empty LLM response should fallback to original results."""
        from simplemem_lite.backend.scoring import rerank_results

        results = [
            {"uuid": str(i), "content": f"Content {i}", "type": "fact"}
            for i in range(15)
        ]

        mock_response = MagicMock()
        mock_response.choices = []

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rerank_results("test query", results, top_k=5)

        assert result["rerank_applied"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        """Invalid JSON response should fallback to original results."""
        from simplemem_lite.backend.scoring import rerank_results

        results = [
            {"uuid": str(i), "content": f"Content {i}", "type": "fact"}
            for i in range(15)
        ]

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="not json at all"))
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rerank_results("test query", results, top_k=5)

        # json_repair may fix it or return original
        # Either way should not crash
        assert "results" in result

    @pytest.mark.asyncio
    async def test_validates_index_bounds(self):
        """Out-of-bounds indices should be filtered out."""
        from simplemem_lite.backend.scoring import rerank_results

        results = [
            {"uuid": "0", "content": "Content 0", "type": "fact"},
            {"uuid": "1", "content": "Content 1", "type": "fact"},
            {"uuid": "2", "content": "Content 2", "type": "fact"},
        ]

        # LLM returns out-of-bounds indices
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"indices": [0, 99, 1, -1, 2], "conflicts": []}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rerank_results("test query", results, top_k=2)

        # Should only include valid indices [0, 1, 2]
        assert result["rerank_applied"] is True
        assert len(result["results"]) == 2
        assert result["results"][0]["uuid"] == "0"
        assert result["results"][1]["uuid"] == "1"

    @pytest.mark.asyncio
    async def test_deduplicates_indices(self):
        """Duplicate indices should be removed."""
        from simplemem_lite.backend.scoring import rerank_results

        results = [
            {"uuid": "0", "content": "Content 0", "type": "fact"},
            {"uuid": "1", "content": "Content 1", "type": "fact"},
            {"uuid": "2", "content": "Content 2", "type": "fact"},
        ]

        # LLM returns duplicate indices
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"indices": [0, 0, 1, 1, 2], "conflicts": []}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rerank_results("test query", results, top_k=2)

        assert result["rerank_applied"] is True
        assert len(result["results"]) == 2
        # First occurrence of each should be kept
        assert result["results"][0]["uuid"] == "0"
        assert result["results"][1]["uuid"] == "1"

    @pytest.mark.asyncio
    async def test_fills_missing_indices_from_pool(self):
        """If LLM returns too few indices, fill from remaining pool."""
        from simplemem_lite.backend.scoring import rerank_results

        results = [
            {"uuid": str(i), "content": f"Content {i}", "type": "fact"}
            for i in range(15)
        ]

        # LLM returns only 2 indices when we need 5
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"indices": [5, 10], "conflicts": []}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rerank_results("test query", results, top_k=5)

        assert len(result["results"]) == 5
        # First two should be from LLM
        assert result["results"][0]["uuid"] == "5"
        assert result["results"][1]["uuid"] == "10"
        # Remaining should be filled from pool

    @pytest.mark.asyncio
    async def test_respects_rerank_pool_limit(self):
        """Should only consider rerank_pool results for LLM."""
        from simplemem_lite.backend.scoring import rerank_results

        results = [
            {"uuid": str(i), "content": f"Content {i}", "type": "fact"}
            for i in range(50)
        ]

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"indices": [0, 1, 2, 3, 4], "conflicts": []}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rerank_results(
                "test query", results, top_k=5, rerank_pool=10
            )

        # Should work correctly with limited pool
        assert len(result["results"]) == 5


class TestSearchMemoriesDeep:
    """Test search_memories_deep MCP tool."""

    @pytest.mark.asyncio
    async def test_calls_rerank_on_results(self):
        """search_memories_deep should call rerank_results."""
        from simplemem_lite.server import search_memories_deep
        from simplemem_lite.memory import Memory
        import simplemem_lite.server as server_module

        mock_store = MagicMock()
        # Return enough results to trigger reranking
        mock_store.search_multi_query.return_value = [
            Memory(uuid=f"uuid-{i}", content=f"Test {i}", type="fact", created_at=0, score=0.8 - i * 0.1)
            for i in range(15)
        ]

        mock_config = MagicMock()

        async def mock_expand(query, config):
            return [query]

        mock_rerank_result = {
            "results": [{"uuid": f"uuid-{i}", "content": f"Test {i}", "type": "fact"} for i in range(5)],
            "conflicts": [],
            "rerank_applied": True,
        }

        with patch.object(server_module._deps, "_store", mock_store):
            with patch.object(server_module._deps, "_config", mock_config):
                with patch.object(server_module._deps, "_initialized", True):
                    with patch("simplemem_lite.memory.expand_query", mock_expand):
                        with patch(
                            "simplemem_lite.backend.scoring.rerank_results",
                            new_callable=AsyncMock,
                        ) as mock_rerank:
                            mock_rerank.return_value = mock_rerank_result
                            result = await search_memories_deep(
                                query="test query",
                                limit=5,
                            )

        mock_rerank.assert_called_once()
        assert result["rerank_applied"] is True

    @pytest.mark.asyncio
    async def test_maps_conflict_indices_to_uuids(self):
        """Conflicts should be mapped from indices to UUIDs."""
        from simplemem_lite.server import search_memories_deep
        from simplemem_lite.memory import Memory
        import simplemem_lite.server as server_module

        mock_store = MagicMock()
        mock_store.search_multi_query.return_value = [
            Memory(uuid=f"uuid-{i}", content=f"Test {i}", type="fact", created_at=0, score=0.8)
            for i in range(15)
        ]

        mock_config = MagicMock()

        async def mock_expand(query, config):
            return [query]

        mock_rerank_result = {
            "results": [{"uuid": f"uuid-{i}", "content": f"Test {i}", "type": "fact"} for i in range(5)],
            "conflicts": [[0, 1, "Conflicting info"]],  # Indices
            "rerank_applied": True,
        }

        with patch.object(server_module._deps, "_store", mock_store):
            with patch.object(server_module._deps, "_config", mock_config):
                with patch.object(server_module._deps, "_initialized", True):
                    with patch("simplemem_lite.memory.expand_query", mock_expand):
                        with patch(
                            "simplemem_lite.backend.scoring.rerank_results",
                            new_callable=AsyncMock,
                        ) as mock_rerank:
                            mock_rerank.return_value = mock_rerank_result
                            result = await search_memories_deep(
                                query="test query",
                                limit=5,
                            )

        # Conflicts should have UUIDs now
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0][0] == "uuid-0"
        assert result["conflicts"][0][1] == "uuid-1"
        assert result["conflicts"][0][2] == "Conflicting info"
