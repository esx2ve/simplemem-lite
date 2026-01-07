"""Tests for contradiction detection functionality.

Tests the LLM-based contradiction detection that identifies memories
which may be superseded by new content, and the DAG enforcement
that prevents cycles in the supersession graph.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestEscapeXmlCharsMemory:
    """Test _escape_xml_chars() helper function in memory.py."""

    def test_escapes_less_than(self):
        """< should be escaped to &lt;."""
        from simplemem_lite.memory import _escape_xml_chars

        assert _escape_xml_chars("<tag>") == "&lt;tag&gt;"

    def test_escapes_greater_than(self):
        """> should be escaped to &gt;."""
        from simplemem_lite.memory import _escape_xml_chars

        assert _escape_xml_chars("x > y") == "x &gt; y"

    def test_escapes_ampersand(self):
        """& should be escaped to &amp;."""
        from simplemem_lite.memory import _escape_xml_chars

        assert _escape_xml_chars("A & B") == "A &amp; B"

    def test_escapes_all_together(self):
        """All XML chars should be escaped together."""
        from simplemem_lite.memory import _escape_xml_chars

        result = _escape_xml_chars("<script>alert('&XSS')</script>")
        assert result == "&lt;script&gt;alert('&amp;XSS')&lt;/script&gt;"

    def test_returns_unchanged_for_safe_text(self):
        """Safe text without XML chars should be unchanged."""
        from simplemem_lite.memory import _escape_xml_chars

        safe_text = "This is normal text without special chars."
        assert _escape_xml_chars(safe_text) == safe_text

    def test_handles_empty_string(self):
        """Empty string should return empty string."""
        from simplemem_lite.memory import _escape_xml_chars

        assert _escape_xml_chars("") == ""


class TestDetectContradictions:
    """Test detect_contradictions() async function."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_similar_memories(self):
        """Should return empty list when no similar memories provided."""
        from simplemem_lite.memory import detect_contradictions

        config = MagicMock()
        result = await detect_contradictions(
            new_content="New decision: use PostgreSQL",
            similar_memories=[],
            config=config,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_filters_low_score_memories(self):
        """Should filter out memories with score <= 0.3."""
        from simplemem_lite.memory import detect_contradictions

        config = MagicMock()
        similar_memories = [
            {"uuid": "1", "content": "Old content", "type": "fact", "score": 0.2},
            {"uuid": "2", "content": "Another old content", "type": "fact", "score": 0.1},
        ]

        # No LLM call should happen since all memories are filtered
        result = await detect_contradictions(
            new_content="New decision",
            similar_memories=similar_memories,
            config=config,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_detects_contradictions_from_llm(self):
        """Should detect contradictions based on LLM response."""
        from simplemem_lite.memory import detect_contradictions

        config = MagicMock()
        config.summary_model = "gemini/gemini-2.0-flash"

        similar_memories = [
            {"uuid": "uuid-0", "content": "Decision: use MySQL", "type": "decision", "score": 0.8},
            {"uuid": "uuid-1", "content": "Setup guide for MySQL", "type": "fact", "score": 0.6},
            {"uuid": "uuid-2", "content": "Unrelated content", "type": "fact", "score": 0.5},
        ]

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"contradictions": [{"index": 0, "reason": "New decision uses PostgreSQL, replacing MySQL decision", "confidence": "high"}]}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detect_contradictions(
                new_content="Decision: use PostgreSQL for database",
                similar_memories=similar_memories,
                config=config,
            )

        assert len(result) == 1
        assert result[0]["uuid"] == "uuid-0"
        assert result[0]["confidence"] == "high"
        assert "PostgreSQL" in result[0]["reason"] or "MySQL" in result[0]["reason"]

    @pytest.mark.asyncio
    async def test_handles_empty_llm_response(self):
        """Should return empty list on empty LLM response."""
        from simplemem_lite.memory import detect_contradictions

        config = MagicMock()
        config.summary_model = "gemini/gemini-2.0-flash"

        similar_memories = [
            {"uuid": "uuid-0", "content": "Some content", "type": "fact", "score": 0.8},
        ]

        mock_response = MagicMock()
        mock_response.choices = []

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detect_contradictions(
                new_content="New content",
                similar_memories=similar_memories,
                config=config,
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_llm_exception(self):
        """Should return empty list on LLM exception."""
        from simplemem_lite.memory import detect_contradictions

        config = MagicMock()
        config.summary_model = "gemini/gemini-2.0-flash"

        similar_memories = [
            {"uuid": "uuid-0", "content": "Some content", "type": "fact", "score": 0.8},
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM API error")
            result = await detect_contradictions(
                new_content="New content",
                similar_memories=similar_memories,
                config=config,
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_invalid_json_response(self):
        """Should return empty list on invalid JSON response."""
        from simplemem_lite.memory import detect_contradictions

        config = MagicMock()
        config.summary_model = "gemini/gemini-2.0-flash"

        similar_memories = [
            {"uuid": "uuid-0", "content": "Some content", "type": "fact", "score": 0.8},
        ]

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="not valid json at all"))
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            # json_repair may try to fix it, but should not crash
            result = await detect_contradictions(
                new_content="New content",
                similar_memories=similar_memories,
                config=config,
            )

        # Should return empty or parsed result, not crash
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_validates_index_bounds(self):
        """Should filter out invalid indices from LLM response."""
        from simplemem_lite.memory import detect_contradictions

        config = MagicMock()
        config.summary_model = "gemini/gemini-2.0-flash"

        similar_memories = [
            {"uuid": "uuid-0", "content": "Content 0", "type": "fact", "score": 0.8},
            {"uuid": "uuid-1", "content": "Content 1", "type": "fact", "score": 0.7},
        ]

        # LLM returns out-of-bounds indices
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"contradictions": [{"index": 0, "reason": "Valid", "confidence": "high"}, {"index": 99, "reason": "Invalid index", "confidence": "high"}, {"index": -1, "reason": "Negative index", "confidence": "high"}]}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detect_contradictions(
                new_content="New content",
                similar_memories=similar_memories,
                config=config,
            )

        # Only valid index should be included
        assert len(result) == 1
        assert result[0]["uuid"] == "uuid-0"

    @pytest.mark.asyncio
    async def test_validates_confidence_values(self):
        """Should normalize invalid confidence values to 'medium'."""
        from simplemem_lite.memory import detect_contradictions

        config = MagicMock()
        config.summary_model = "gemini/gemini-2.0-flash"

        similar_memories = [
            {"uuid": "uuid-0", "content": "Content", "type": "fact", "score": 0.8},
        ]

        # LLM returns invalid confidence
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"contradictions": [{"index": 0, "reason": "Test", "confidence": "super_high"}]}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detect_contradictions(
                new_content="New content",
                similar_memories=similar_memories,
                config=config,
            )

        assert len(result) == 1
        assert result[0]["confidence"] == "medium"  # Normalized to medium

    @pytest.mark.asyncio
    async def test_truncates_long_reasons(self):
        """Should truncate reasons longer than 200 chars."""
        from simplemem_lite.memory import detect_contradictions

        config = MagicMock()
        config.summary_model = "gemini/gemini-2.0-flash"

        similar_memories = [
            {"uuid": "uuid-0", "content": "Content", "type": "fact", "score": 0.8},
        ]

        long_reason = "A" * 500  # 500 character reason
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=f'{{"contradictions": [{{"index": 0, "reason": "{long_reason}", "confidence": "high"}}]}}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detect_contradictions(
                new_content="New content",
                similar_memories=similar_memories,
                config=config,
            )

        assert len(result) == 1
        assert len(result[0]["reason"]) <= 200

    @pytest.mark.asyncio
    async def test_handles_no_contradictions_found(self):
        """Should return empty list when LLM finds no contradictions."""
        from simplemem_lite.memory import detect_contradictions

        config = MagicMock()
        config.summary_model = "gemini/gemini-2.0-flash"

        similar_memories = [
            {"uuid": "uuid-0", "content": "Content", "type": "fact", "score": 0.8},
        ]

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content='{"contradictions": []}')
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detect_contradictions(
                new_content="New content",
                similar_memories=similar_memories,
                config=config,
            )

        assert result == []


class TestDAGEnforcement:
    """Test DAG enforcement in supersession graph."""

    def test_would_create_supersession_cycle_detects_direct_cycle(self):
        """Should detect when creating edge would complete a direct cycle."""
        from simplemem_lite.db.manager import DatabaseManager

        mock_graph = MagicMock()
        # Simulate A -> B exists, and we're trying to add B -> A (would create cycle)
        mock_graph.query.return_value = MagicMock(result_set=[[True]])

        manager = MagicMock(spec=DatabaseManager)
        manager.graph = mock_graph

        # Call the actual method
        from simplemem_lite.db.manager import DatabaseManager
        result = DatabaseManager.would_create_supersession_cycle(
            manager, newer_uuid="uuid-B", older_uuid="uuid-A"
        )

        assert result is True

    def test_would_create_supersession_cycle_allows_new_edge(self):
        """Should allow edge when no cycle would be created."""
        from simplemem_lite.db.manager import DatabaseManager

        mock_graph = MagicMock()
        # No existing path from older to newer
        mock_graph.query.return_value = MagicMock(result_set=[[False]])

        manager = MagicMock(spec=DatabaseManager)
        manager.graph = mock_graph

        result = DatabaseManager.would_create_supersession_cycle(
            manager, newer_uuid="uuid-A", older_uuid="uuid-B"
        )

        assert result is False

    def test_would_create_supersession_cycle_handles_empty_result(self):
        """Should return False when query returns empty result."""
        from simplemem_lite.db.manager import DatabaseManager

        mock_graph = MagicMock()
        mock_graph.query.return_value = MagicMock(result_set=[])

        manager = MagicMock(spec=DatabaseManager)
        manager.graph = mock_graph

        result = DatabaseManager.would_create_supersession_cycle(
            manager, newer_uuid="uuid-A", older_uuid="uuid-B"
        )

        assert result is False

    def test_add_supersession_blocks_self_supersession(self):
        """Should not allow a memory to supersede itself."""
        from simplemem_lite.db.manager import DatabaseManager

        manager = MagicMock(spec=DatabaseManager)
        manager.graph = MagicMock()

        result = DatabaseManager.add_supersession(
            manager,
            newer_uuid="uuid-A",
            older_uuid="uuid-A",  # Same UUID
            confidence=0.9,
        )

        assert result is False
        # Should not even call graph.query for cycle check
        manager.graph.query.assert_not_called()

    def test_add_supersession_blocks_cycle(self):
        """Should block supersession that would create a cycle."""
        from simplemem_lite.db.manager import DatabaseManager

        mock_graph = MagicMock()
        # Simulate cycle detection returns True
        mock_graph.query.return_value = MagicMock(result_set=[[True]])

        manager = MagicMock(spec=DatabaseManager)
        manager.graph = mock_graph
        manager.would_create_supersession_cycle = MagicMock(return_value=True)

        result = DatabaseManager.add_supersession(
            manager,
            newer_uuid="uuid-B",
            older_uuid="uuid-A",
            confidence=0.9,
        )

        assert result is False

    def test_add_supersession_creates_edge_when_no_cycle(self):
        """Should create SUPERSEDES edge when no cycle would be created."""
        from simplemem_lite.db.manager import DatabaseManager

        mock_graph = MagicMock()
        mock_graph.query.return_value = MagicMock(result_set=[])

        manager = MagicMock(spec=DatabaseManager)
        manager.graph = mock_graph
        manager.would_create_supersession_cycle = MagicMock(return_value=False)

        result = DatabaseManager.add_supersession(
            manager,
            newer_uuid="uuid-A",
            older_uuid="uuid-B",
            confidence=0.9,
            supersession_type="contradiction",
            reason="Test reason",
        )

        assert result is True
        # Should have called graph.query to create the edge
        assert mock_graph.query.called


class TestCheckContradictionsTool:
    """Test check_contradictions MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_similar_memories(self):
        """Should return empty when no similar memories found."""
        from simplemem_lite.server import check_contradictions
        import simplemem_lite.server as server_module

        mock_store = MagicMock()
        mock_store.search.return_value = []

        mock_config = MagicMock()

        with patch.object(server_module._deps, "_store", mock_store):
            with patch.object(server_module._deps, "_config", mock_config):
                with patch.object(server_module._deps, "_initialized", True):
                    result = await check_contradictions(
                        content="Test content",
                        project_id="test-project",
                    )

        assert result["contradictions"] == []
        assert result["supersessions_created"] == 0
        assert result["similar_count"] == 0

    @pytest.mark.asyncio
    async def test_returns_contradictions_found(self):
        """Should return contradictions detected by LLM."""
        from simplemem_lite.server import check_contradictions
        from simplemem_lite.memory import Memory
        import simplemem_lite.server as server_module

        mock_store = MagicMock()
        mock_store.search.return_value = [
            Memory(uuid="uuid-0", content="Old decision", type="decision", created_at=0, score=0.8),
            Memory(uuid="uuid-1", content="Another memory", type="fact", created_at=0, score=0.6),
        ]

        mock_config = MagicMock()
        mock_config.summary_model = "gemini/gemini-2.0-flash"

        mock_contradictions = [
            {"uuid": "uuid-0", "reason": "Contradicting decision", "confidence": "high"}
        ]

        with patch.object(server_module._deps, "_store", mock_store):
            with patch.object(server_module._deps, "_config", mock_config):
                with patch.object(server_module._deps, "_initialized", True):
                    with patch(
                        "simplemem_lite.memory.detect_contradictions",
                        new_callable=AsyncMock,
                    ) as mock_detect:
                        mock_detect.return_value = mock_contradictions
                        result = await check_contradictions(
                            content="New decision",
                            project_id="test-project",
                        )

        assert len(result["contradictions"]) == 1
        assert result["contradictions"][0]["uuid"] == "uuid-0"
        assert result["similar_count"] == 2

    @pytest.mark.asyncio
    async def test_applies_supersession_when_requested(self):
        """Should create SUPERSEDES edges when apply_supersession=True."""
        from simplemem_lite.server import check_contradictions
        from simplemem_lite.memory import Memory
        import simplemem_lite.server as server_module

        mock_db = MagicMock()
        mock_db.add_supersession.return_value = True

        mock_store = MagicMock()
        mock_store.db = mock_db
        mock_store.search.return_value = [
            Memory(uuid="uuid-old", content="Old decision", type="decision", created_at=0, score=0.8),
        ]

        mock_config = MagicMock()
        mock_config.summary_model = "gemini/gemini-2.0-flash"

        mock_contradictions = [
            {"uuid": "uuid-old", "reason": "Outdated", "confidence": "high"}
        ]

        with patch.object(server_module._deps, "_store", mock_store):
            with patch.object(server_module._deps, "_config", mock_config):
                with patch.object(server_module._deps, "_initialized", True):
                    with patch(
                        "simplemem_lite.memory.detect_contradictions",
                        new_callable=AsyncMock,
                    ) as mock_detect:
                        mock_detect.return_value = mock_contradictions
                        result = await check_contradictions(
                            content="New decision",
                            memory_uuid="uuid-new",
                            apply_supersession=True,
                        )

        assert result["supersessions_created"] == 1
        mock_db.add_supersession.assert_called_once_with(
            newer_uuid="uuid-new",
            older_uuid="uuid-old",
            confidence=0.9,  # high -> 0.9
            supersession_type="contradiction",
            reason="Outdated",
        )

    @pytest.mark.asyncio
    async def test_does_not_apply_supersession_without_memory_uuid(self):
        """Should not apply supersession when memory_uuid is not provided."""
        from simplemem_lite.server import check_contradictions
        from simplemem_lite.memory import Memory
        import simplemem_lite.server as server_module

        mock_db = MagicMock()
        mock_store = MagicMock()
        mock_store.db = mock_db
        mock_store.search.return_value = [
            Memory(uuid="uuid-old", content="Old", type="fact", created_at=0, score=0.8),
        ]

        mock_config = MagicMock()

        mock_contradictions = [
            {"uuid": "uuid-old", "reason": "Outdated", "confidence": "high"}
        ]

        with patch.object(server_module._deps, "_store", mock_store):
            with patch.object(server_module._deps, "_config", mock_config):
                with patch.object(server_module._deps, "_initialized", True):
                    with patch(
                        "simplemem_lite.memory.detect_contradictions",
                        new_callable=AsyncMock,
                    ) as mock_detect:
                        mock_detect.return_value = mock_contradictions
                        result = await check_contradictions(
                            content="New content",
                            apply_supersession=True,  # But no memory_uuid!
                        )

        # Should not create supersession without memory_uuid
        assert result["supersessions_created"] == 0
        mock_db.add_supersession.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_supersession_failure(self):
        """Should count only successful supersessions."""
        from simplemem_lite.server import check_contradictions
        from simplemem_lite.memory import Memory
        import simplemem_lite.server as server_module

        mock_db = MagicMock()
        # First succeeds, second fails (cycle detected)
        mock_db.add_supersession.side_effect = [True, False]

        mock_store = MagicMock()
        mock_store.db = mock_db
        mock_store.search.return_value = [
            Memory(uuid="uuid-0", content="Old 0", type="fact", created_at=0, score=0.8),
            Memory(uuid="uuid-1", content="Old 1", type="fact", created_at=0, score=0.7),
        ]

        mock_config = MagicMock()

        mock_contradictions = [
            {"uuid": "uuid-0", "reason": "Outdated", "confidence": "high"},
            {"uuid": "uuid-1", "reason": "Also outdated", "confidence": "medium"},
        ]

        with patch.object(server_module._deps, "_store", mock_store):
            with patch.object(server_module._deps, "_config", mock_config):
                with patch.object(server_module._deps, "_initialized", True):
                    with patch(
                        "simplemem_lite.memory.detect_contradictions",
                        new_callable=AsyncMock,
                    ) as mock_detect:
                        mock_detect.return_value = mock_contradictions
                        result = await check_contradictions(
                            content="New content",
                            memory_uuid="uuid-new",
                            apply_supersession=True,
                        )

        # Only 1 succeeded
        assert result["supersessions_created"] == 1
