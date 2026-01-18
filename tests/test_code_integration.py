"""Integration tests for code indexing with AST chunking.

Tests the integration of:
- AST chunker with code_index.py
- embed_code_batch with proper metadata
- Database schema with new fields
- Search returning rich metadata
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from simplemem_lite.code_index import CodeIndexer
from simplemem_lite.config import Config
from simplemem_lite.ast_chunker import CodeChunk
from simplemem_lite.embeddings import EmbeddingResult


class TestCodeIndexAST:
    """Tests for AST-aware code indexing."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database manager."""
        db = MagicMock()
        db.get_code_embedding_dimension.return_value = None  # No existing chunks
        db.add_code_chunks.return_value = None
        db.add_code_chunk_node.return_value = None
        db.link_code_to_entity.return_value = None
        return db

    @pytest.fixture
    def config(self):
        """Create a test config."""
        cfg = Config()
        cfg.code_embedding_provider = "local"
        return cfg

    @pytest.fixture
    def indexer(self, mock_db, config):
        """Create a CodeIndexer with mocked dependencies."""
        return CodeIndexer(mock_db, config)

    @patch("simplemem_lite.code_index.chunk_file")
    @patch("simplemem_lite.code_index.embed_code_batch")
    def test_index_content_uses_ast_chunker(
        self, mock_embed, mock_chunk, indexer, mock_db
    ):
        """Verify _index_content uses AST chunker instead of line-based splitting."""
        # Setup mock chunks with AST metadata
        mock_chunk.return_value = [
            CodeChunk(
                content="def foo(): pass",
                start_line=1,
                end_line=1,
                node_type="function",
                function_name="foo",
                class_name=None,
                language="python",
                filepath="test.py",
            )
        ]

        # Setup mock embeddings
        mock_embed.return_value = EmbeddingResult(
            embeddings=[[0.1] * 768],
            provider="local",
            model="jina-embeddings-v2-base-code",
            dimension=768,
            elapsed_ms=50.0,
        )

        # Index content
        result = indexer._index_content("def foo(): pass", "test.py", "test-project")

        # Verify AST chunker was called
        mock_chunk.assert_called_once_with("def foo(): pass", "test.py")

        # Verify embed_code_batch was called
        mock_embed.assert_called_once()

        # Verify correct number of chunks created
        assert result == 1

    @patch("simplemem_lite.code_index.chunk_file")
    @patch("simplemem_lite.code_index.embed_code_batch")
    def test_index_content_stores_ast_metadata(
        self, mock_embed, mock_chunk, indexer, mock_db
    ):
        """Verify that AST metadata is stored in the database records."""
        # Setup mock chunk with full metadata
        mock_chunk.return_value = [
            CodeChunk(
                content="class MyClass:\n    def method(self): pass",
                start_line=1,
                end_line=2,
                node_type="class",
                function_name=None,
                class_name="MyClass",
                language="python",
                filepath="test.py",
            )
        ]

        mock_embed.return_value = EmbeddingResult(
            embeddings=[[0.1] * 768],
            provider="local",
            model="jina-embeddings-v2-base-code",
            dimension=768,
            elapsed_ms=50.0,
        )

        # Index content
        indexer._index_content("class MyClass:\n    def method(self): pass", "test.py", "test-project")

        # Verify add_code_chunks was called with correct metadata
        mock_db.add_code_chunks.assert_called_once()
        records = mock_db.add_code_chunks.call_args[0][0]

        assert len(records) == 1
        record = records[0]

        # Verify AST metadata fields
        assert record["function_name"] is None
        assert record["class_name"] == "MyClass"
        assert record["language"] == "python"
        assert record["node_type"] == "class"

        # Verify timestamp is present
        assert "indexed_at" in record
        assert record["indexed_at"] is not None

        # Verify embedding provenance
        assert record["embedding_model"] == "jina-embeddings-v2-base-code"
        assert record["embedding_dim"] == 768

    @patch("simplemem_lite.code_index.chunk_file")
    @patch("simplemem_lite.code_index.embed_code_batch")
    def test_index_content_passes_expected_dim(
        self, mock_embed, mock_chunk, indexer, mock_db
    ):
        """Verify expected dimension is passed to embed_code_batch."""
        mock_db.get_code_embedding_dimension.return_value = 1024  # Existing chunks have this dim

        mock_chunk.return_value = [
            CodeChunk(
                content="def foo(): pass",
                start_line=1,
                end_line=1,
                node_type="function",
                function_name="foo",
                class_name=None,
                language="python",
                filepath="test.py",
            )
        ]

        mock_embed.return_value = EmbeddingResult(
            embeddings=[[0.1] * 1024],
            provider="voyage",
            model="voyage-code-3",
            dimension=1024,
            elapsed_ms=50.0,
        )

        # Index content
        indexer._index_content("def foo(): pass", "test.py", "test-project")

        # Verify expected_dim was passed
        call_kwargs = mock_embed.call_args
        assert call_kwargs[1]["expected_dim"] == 1024


class TestCodeSearch:
    """Tests for code search with rich metadata."""

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
        return cfg

    @pytest.fixture
    def indexer(self, mock_db, config):
        """Create a CodeIndexer with mocked dependencies."""
        return CodeIndexer(mock_db, config)

    @patch("simplemem_lite.code_index.embed_code_batch")
    def test_search_uses_code_embeddings(self, mock_embed, indexer, mock_db):
        """Verify search uses code embeddings, not memory embeddings."""
        mock_embed.return_value = EmbeddingResult(
            embeddings=[[0.1] * 768],
            provider="local",
            model="jina-embeddings-v2-base-code",
            dimension=768,
            elapsed_ms=50.0,
        )

        mock_db.search_code.return_value = []

        # Perform search
        indexer.search("def foo")

        # Verify embed_code_batch was called
        mock_embed.assert_called_once()
        call_args = mock_embed.call_args[0]
        assert call_args[0] == ["def foo"]  # Query as list

    @patch("simplemem_lite.code_index.embed_code_batch")
    def test_search_returns_ast_metadata(self, mock_embed, indexer, mock_db):
        """Verify search results include AST metadata."""
        mock_embed.return_value = EmbeddingResult(
            embeddings=[[0.1] * 768],
            provider="local",
            model="jina-embeddings-v2-base-code",
            dimension=768,
            elapsed_ms=50.0,
        )

        mock_db.search_code.return_value = [
            {
                "uuid": "abc-123",
                "filepath": "test.py",
                "content": "def foo(): pass",
                "start_line": 1,
                "end_line": 1,
                "project_id": "test-project",
                "_distance": 0.1,
                "function_name": "foo",
                "class_name": None,
                "language": "python",
                "node_type": "function",
                "indexed_at": "2024-01-01T00:00:00Z",
                "embedding_model": "jina-embeddings-v2-base-code",
            }
        ]

        # Perform search
        results = indexer.search("def foo")

        # Verify result includes metadata
        assert len(results) == 1
        r = results[0]
        assert r["function_name"] == "foo"
        assert r["class_name"] is None
        assert r["language"] == "python"
        assert r["node_type"] == "function"
        assert r["indexed_at"] == "2024-01-01T00:00:00Z"
        assert r["embedding_model"] == "jina-embeddings-v2-base-code"


class TestGetChunk:
    """Tests for get_chunk with rich metadata."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database manager."""
        return MagicMock()

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return Config()

    @pytest.fixture
    def indexer(self, mock_db, config):
        """Create a CodeIndexer with mocked dependencies."""
        return CodeIndexer(mock_db, config)

    def test_get_chunk_returns_ast_metadata(self, indexer, mock_db):
        """Verify get_chunk returns AST metadata."""
        mock_db.get_code_chunk_by_uuid.return_value = {
            "uuid": "abc-123",
            "filepath": "test.py",
            "content": "class MyClass:\n    pass",
            "start_line": 1,
            "end_line": 2,
            "project_id": "test-project",
            "function_name": None,
            "class_name": "MyClass",
            "language": "python",
            "node_type": "class",
            "indexed_at": "2024-01-01T00:00:00Z",
            "embedding_model": "voyage-code-3",
        }

        result = indexer.get_chunk("abc-123")

        assert result is not None
        assert result["function_name"] is None
        assert result["class_name"] == "MyClass"
        assert result["language"] == "python"
        assert result["node_type"] == "class"
        assert result["indexed_at"] == "2024-01-01T00:00:00Z"
        assert result["embedding_model"] == "voyage-code-3"


class TestDatabaseDimensionValidation:
    """Tests for database-level dimension validation."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database manager."""
        db = MagicMock()
        return db

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return Config()

    @pytest.fixture
    def indexer(self, mock_db, config):
        """Create a CodeIndexer with mocked dependencies."""
        return CodeIndexer(mock_db, config)

    @patch("simplemem_lite.code_index.chunk_file")
    @patch("simplemem_lite.code_index.embed_code_batch")
    def test_new_project_no_expected_dim(self, mock_embed, mock_chunk, indexer, mock_db):
        """Verify new projects have no expected dimension constraint."""
        mock_db.get_code_embedding_dimension.return_value = None  # New project

        mock_chunk.return_value = [
            CodeChunk(
                content="def foo(): pass",
                start_line=1,
                end_line=1,
                node_type="function",
                function_name="foo",
                class_name=None,
                language="python",
                filepath="test.py",
            )
        ]

        mock_embed.return_value = EmbeddingResult(
            embeddings=[[0.1] * 768],
            provider="local",
            model="jina-embeddings-v2-base-code",
            dimension=768,
            elapsed_ms=50.0,
        )

        # Index content
        indexer._index_content("def foo(): pass", "test.py", "test-project")

        # Verify expected_dim was None (no constraint)
        call_kwargs = mock_embed.call_args
        assert call_kwargs[1]["expected_dim"] is None

    @patch("simplemem_lite.code_index.chunk_file")
    @patch("simplemem_lite.code_index.embed_code_batch")
    def test_existing_project_uses_expected_dim(
        self, mock_embed, mock_chunk, indexer, mock_db
    ):
        """Verify existing projects use stored dimension for validation."""
        mock_db.get_code_embedding_dimension.return_value = 3072  # OpenRouter dim

        mock_chunk.return_value = [
            CodeChunk(
                content="def foo(): pass",
                start_line=1,
                end_line=1,
                node_type="function",
                function_name="foo",
                class_name=None,
                language="python",
                filepath="test.py",
            )
        ]

        mock_embed.return_value = EmbeddingResult(
            embeddings=[[0.1] * 3072],
            provider="openrouter",
            model="openai/text-embedding-3-large",
            dimension=3072,
            elapsed_ms=100.0,
        )

        # Index content
        indexer._index_content("def foo(): pass", "test.py", "test-project")

        # Verify expected_dim was 3072 (from existing chunks)
        call_kwargs = mock_embed.call_args
        assert call_kwargs[1]["expected_dim"] == 3072


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
