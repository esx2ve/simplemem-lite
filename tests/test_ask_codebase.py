"""Tests for ask_codebase LLM-powered code question answering.

Tests the integration of:
- Semantic code search
- LLM synthesis with citations
- Metadata formatting
"""

import pytest
from unittest.mock import patch, MagicMock

from simplemem_lite.code_index import CodeIndexer
from simplemem_lite.config import Config

# Enable pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


class TestAskCodebase:
    """Tests for ask_codebase method."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database manager."""
        db = MagicMock()
        db.get_code_embedding_dimension.return_value = 768
        return db

    @pytest.fixture
    def config(self):
        """Create a test config."""
        cfg = Config()
        cfg.code_embedding_provider = "local"
        cfg.summary_model = "gpt-4o-mini"
        return cfg

    @pytest.fixture
    def indexer(self, mock_db, config):
        """Create a CodeIndexer with mocked dependencies."""
        return CodeIndexer(mock_db, config)

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_ask_codebase_synthesizes_answer(self, mock_llm, indexer):
        """Verify ask_codebase calls LLM and returns synthesized answer."""
        # Setup mock search results
        with patch.object(indexer, "search") as mock_search:
            mock_search.return_value = [
                {
                    "uuid": "abc-123",
                    "filepath": "src/auth.py",
                    "content": "def login(user, password): pass",
                    "start_line": 10,
                    "end_line": 15,
                    "score": 0.85,
                    "function_name": "login",
                    "class_name": None,
                    "language": "python",
                    "node_type": "function",
                }
            ]

            # Setup mock LLM response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "The login function [1] handles authentication."
            mock_llm.return_value = mock_response

            result = await indexer.ask_codebase("How does authentication work?")

            # Verify LLM was called
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args

            # Verify prompt contains the code
            prompt = call_args.kwargs["messages"][0]["content"]
            assert "def login" in prompt
            assert "src/auth.py" in prompt

            # Verify result structure
            assert result["answer"] == "The login function [1] handles authentication."
            assert result["chunks_used"] == 1
            assert result["confidence"] == "high"  # score 0.85 >= 0.5
            assert len(result["sources"]) == 1
            assert result["sources"][0]["filepath"] == "src/auth.py"
            assert result["sources"][0]["function_name"] == "login"

    @pytest.mark.asyncio
    async def test_ask_codebase_no_results(self, indexer):
        """Verify ask_codebase handles no search results gracefully."""
        with patch.object(indexer, "search") as mock_search:
            mock_search.return_value = []

            result = await indexer.ask_codebase("What is the meaning of life?")

            assert result["chunks_used"] == 0
            assert result["confidence"] == "none"
            assert "couldn't find" in result["answer"].lower() or "no relevant" in result["answer"].lower()
            assert result["sources"] == []

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_ask_codebase_confidence_levels(self, mock_llm, indexer):
        """Verify confidence is calculated correctly from scores."""
        # Setup mock LLM
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer"
        mock_llm.return_value = mock_response

        with patch.object(indexer, "search") as mock_search:
            # High confidence (avg >= 0.5)
            mock_search.return_value = [{"score": 0.8}, {"score": 0.6}]
            result = await indexer.ask_codebase("test")
            assert result["confidence"] == "high"

            # Medium confidence (0.25 <= avg < 0.5)
            mock_search.return_value = [{"score": 0.3}, {"score": 0.35}]
            result = await indexer.ask_codebase("test")
            assert result["confidence"] == "medium"

            # Low confidence (avg < 0.25)
            mock_search.return_value = [{"score": 0.1}, {"score": 0.15}]
            result = await indexer.ask_codebase("test")
            assert result["confidence"] == "low"

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_ask_codebase_llm_error_handling(self, mock_llm, indexer):
        """Verify ask_codebase handles LLM errors gracefully."""
        with patch.object(indexer, "search") as mock_search:
            mock_search.return_value = [{"score": 0.8, "content": "code"}]

            # Simulate LLM error
            mock_llm.side_effect = Exception("API rate limit exceeded")

            result = await indexer.ask_codebase("test")

            assert "Error generating answer" in result["answer"]
            assert "API rate limit exceeded" in result["answer"]
            assert result["confidence"] == "error"


class TestFormatChunksForLlm:
    """Tests for _format_chunks_for_llm helper."""

    @pytest.fixture
    def indexer(self):
        """Create a CodeIndexer with minimal mocks."""
        mock_db = MagicMock()
        config = Config()
        return CodeIndexer(mock_db, config)

    def test_format_includes_metadata(self, indexer):
        """Verify format includes all metadata."""
        chunks = [
            {
                "filepath": "src/api/handler.py",
                "start_line": 45,
                "end_line": 78,
                "function_name": "handle_request",
                "class_name": "APIHandler",
                "language": "python",
                "node_type": "function",
                "score": 0.89,
                "content": "def handle_request(self):\n    pass",
            }
        ]

        result = indexer._format_chunks_for_llm(chunks)

        assert "[1]" in result
        assert "src/api/handler.py" in result
        assert "Lines: 45-78" in result
        assert "Function: handle_request" in result
        assert "Class: APIHandler" in result
        assert "Language: python" in result
        assert "Relevance: 0.89" in result
        assert "def handle_request" in result
        assert "```python" in result

    def test_format_multiple_chunks(self, indexer):
        """Verify multiple chunks are numbered correctly."""
        chunks = [
            {"filepath": "a.py", "start_line": 1, "end_line": 5, "score": 0.9, "content": "# a"},
            {"filepath": "b.py", "start_line": 1, "end_line": 5, "score": 0.8, "content": "# b"},
            {"filepath": "c.py", "start_line": 1, "end_line": 5, "score": 0.7, "content": "# c"},
        ]

        result = indexer._format_chunks_for_llm(chunks)

        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        assert "a.py" in result
        assert "b.py" in result
        assert "c.py" in result

    def test_format_infers_language_from_extension(self, indexer):
        """Verify language is inferred from file extension when not provided."""
        chunks = [
            {"filepath": "script.js", "start_line": 1, "end_line": 5, "score": 0.9, "content": "// js"},
        ]

        result = indexer._format_chunks_for_llm(chunks)

        assert "```javascript" in result


class TestInferLanguage:
    """Tests for _infer_language helper."""

    @pytest.fixture
    def indexer(self):
        """Create a CodeIndexer with minimal mocks."""
        mock_db = MagicMock()
        config = Config()
        return CodeIndexer(mock_db, config)

    def test_python_extension(self, indexer):
        assert indexer._infer_language("test.py") == "python"

    def test_javascript_extension(self, indexer):
        assert indexer._infer_language("app.js") == "javascript"

    def test_typescript_extension(self, indexer):
        assert indexer._infer_language("types.ts") == "typescript"

    def test_tsx_extension(self, indexer):
        assert indexer._infer_language("Component.tsx") == "tsx"

    def test_rust_extension(self, indexer):
        assert indexer._infer_language("main.rs") == "rust"

    def test_go_extension(self, indexer):
        assert indexer._infer_language("server.go") == "go"

    def test_unknown_extension(self, indexer):
        assert indexer._infer_language("readme.txt") == ""

    def test_no_extension(self, indexer):
        assert indexer._infer_language("Makefile") == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
