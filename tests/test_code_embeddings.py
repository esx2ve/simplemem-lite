"""Tests for code embedding providers.

Tests the dual embedding pipeline with provider fallback:
Voyage → OpenRouter → Local

Uses mocking to test fallback behavior without requiring API keys.
"""

import pytest
from unittest.mock import patch, MagicMock

from simplemem_lite.config import Config
from simplemem_lite.embeddings import (
    embed_code_batch,
    EmbeddingResult,
    _embed_voyage,
    _embed_code_local,
    _embed_openrouter,
)


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_embedding_result_attributes(self):
        """Verify EmbeddingResult has all required attributes."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            provider="voyage",
            model="voyage-code-3",
            dimension=3,
            elapsed_ms=100.5,
        )
        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.provider == "voyage"
        assert result.model == "voyage-code-3"
        assert result.dimension == 3
        assert result.elapsed_ms == 100.5

    def test_empty_result(self):
        """Verify empty result for empty input."""
        result = embed_code_batch([])
        assert result.embeddings == []
        assert result.provider == "none"
        assert result.dimension == 0


class TestVoyageEmbeddings:
    """Tests for Voyage AI embeddings."""

    @patch("simplemem_lite.embeddings._get_voyage_client")
    def test_voyage_returns_correct_dimensions(self, mock_get_client):
        """Verify Voyage embeddings have correct dimensions."""
        # Mock the Voyage client
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]  # Voyage code-3 returns 1024 dims
        mock_client.embed.return_value = mock_result
        mock_get_client.return_value = mock_client

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = "test-key"

        result = embed_code_batch(["def foo(): pass"], config)

        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert result.provider == "voyage"
        assert result.model == config.voyage_code_model

    @patch("simplemem_lite.embeddings._get_voyage_client")
    def test_voyage_batch_processing(self, mock_get_client):
        """Verify Voyage handles multiple texts correctly."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        mock_client.embed.return_value = mock_result
        mock_get_client.return_value = mock_client

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = "test-key"

        texts = ["def foo(): pass", "def bar(): pass", "class Baz: pass"]
        result = embed_code_batch(texts, config)

        assert len(result.embeddings) == 3
        assert result.provider == "voyage"

    @patch("simplemem_lite.embeddings._get_voyage_client")
    def test_voyage_uses_document_input_type(self, mock_get_client):
        """Verify Voyage is called with input_type='document' for indexing."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result
        mock_get_client.return_value = mock_client

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = "test-key"

        _embed_voyage(["def foo(): pass"], config.voyage_code_model, config.voyage_api_key)

        mock_client.embed.assert_called_once()
        call_kwargs = mock_client.embed.call_args
        assert call_kwargs.kwargs.get("input_type") == "document"


class TestOpenRouterEmbeddings:
    """Tests for OpenRouter embeddings."""

    @patch("litellm.embedding")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_openrouter_returns_correct_dimensions(self, mock_embedding):
        """Verify OpenRouter embeddings have correct dimensions."""
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * 3072}]  # text-embedding-3-large returns 3072
        mock_embedding.return_value = mock_response

        embeddings = _embed_openrouter(["def foo(): pass"], "openai/text-embedding-3-large")

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 3072

    @patch.dict("os.environ", {}, clear=True)
    def test_openrouter_raises_without_api_key(self):
        """Verify OpenRouter raises error without API key."""
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            _embed_openrouter(["def foo(): pass"], "openai/text-embedding-3-large")


class TestLocalEmbeddings:
    """Tests for local code embeddings."""

    @patch("simplemem_lite.embeddings._get_local_code_model")
    def test_local_returns_correct_dimensions(self, mock_get_model):
        """Verify local embeddings have correct dimensions."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 768])  # jina returns 768
        mock_get_model.return_value = mock_model

        embeddings = _embed_code_local(["def foo(): pass"], "jinaai/jina-embeddings-v2-base-code")

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768

    @patch("simplemem_lite.embeddings._get_local_code_model")
    def test_local_batch_processing(self, mock_get_model):
        """Verify local model handles batches correctly."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768,
        ])
        mock_get_model.return_value = mock_model

        texts = ["def foo(): pass", "def bar(): pass", "class Baz: pass"]
        embeddings = _embed_code_local(texts, "jinaai/jina-embeddings-v2-base-code")

        assert len(embeddings) == 3
        mock_model.encode.assert_called_once()


class TestFallbackChain:
    """Tests for the automatic fallback chain."""

    @patch("simplemem_lite.embeddings._embed_voyage")
    @patch("simplemem_lite.embeddings._embed_openrouter")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_fallback_voyage_to_openrouter(self, mock_openrouter, mock_voyage):
        """Verify fallback from Voyage to OpenRouter on error."""
        mock_voyage.side_effect = Exception("API error")
        mock_openrouter.return_value = [[0.1] * 3072]

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = "test-key"

        result = embed_code_batch(["def foo(): pass"], config)

        assert mock_voyage.called
        assert mock_openrouter.called
        assert result.provider == "openrouter"
        assert len(result.embeddings[0]) == 3072

    @patch("simplemem_lite.embeddings._embed_voyage")
    @patch("simplemem_lite.embeddings._embed_openrouter")
    @patch("simplemem_lite.embeddings._embed_code_local")
    def test_fallback_to_local_when_both_fail(self, mock_local, mock_openrouter, mock_voyage):
        """Verify fallback all the way to local when both APIs fail."""
        mock_voyage.side_effect = Exception("Voyage API error")
        mock_openrouter.side_effect = Exception("OpenRouter API error")
        mock_local.return_value = [[0.1] * 768]

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = "test-key"

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = embed_code_batch(["def foo(): pass"], config)

        assert mock_voyage.called
        assert mock_openrouter.called
        assert mock_local.called
        assert result.provider == "local"
        assert len(result.embeddings[0]) == 768

    @patch("simplemem_lite.embeddings._embed_code_local")
    def test_fallback_to_local_when_no_api_keys(self, mock_local):
        """Verify fallback to local when no API keys configured."""
        mock_local.return_value = [[0.1] * 768]

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = None  # No Voyage key

        with patch.dict("os.environ", {}, clear=True):
            result = embed_code_batch(["def foo(): pass"], config)

        assert mock_local.called
        assert result.provider == "local"

    @patch("simplemem_lite.embeddings._embed_code_local")
    def test_direct_local_provider(self, mock_local):
        """Verify local provider is used directly when configured."""
        mock_local.return_value = [[0.1] * 768]

        config = Config()
        config.code_embedding_provider = "local"

        result = embed_code_batch(["def foo(): pass"], config)

        assert mock_local.called
        assert result.provider == "local"


class TestEmbeddingResultMetadata:
    """Tests for embedding result metadata."""

    @patch("simplemem_lite.embeddings._get_voyage_client")
    def test_result_includes_timing(self, mock_get_client):
        """Verify EmbeddingResult includes elapsed time."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result
        mock_get_client.return_value = mock_client

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = "test-key"

        result = embed_code_batch(["def foo(): pass"], config)

        assert result.elapsed_ms >= 0

    @patch("simplemem_lite.embeddings._get_voyage_client")
    def test_result_includes_dimension(self, mock_get_client):
        """Verify EmbeddingResult includes correct dimension."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result
        mock_get_client.return_value = mock_client

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = "test-key"

        result = embed_code_batch(["def foo(): pass"], config)

        assert result.dimension == 1024


class TestBatchSizeHandling:
    """Tests for batch size handling."""

    @patch("simplemem_lite.embeddings._get_voyage_client")
    def test_large_batch_is_chunked(self, mock_get_client):
        """Verify large batches are split according to batch_size."""
        mock_client = MagicMock()

        # Return different embeddings for each batch call
        call_count = [0]

        def mock_embed(*args, **kwargs):
            call_count[0] += 1
            batch_size = len(args[0])
            mock_result = MagicMock()
            mock_result.embeddings = [[0.1] * 1024] * batch_size
            return mock_result

        mock_client.embed.side_effect = mock_embed
        mock_get_client.return_value = mock_client

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = "test-key"

        # Create 300 texts (more than default batch_size of 128)
        texts = [f"def foo_{i}(): pass" for i in range(300)]
        result = embed_code_batch(texts, config, batch_size=128)

        # Should make 3 calls: 128 + 128 + 44
        assert call_count[0] == 3
        assert len(result.embeddings) == 300


class TestDimensionValidation:
    """Tests for dimension compatibility validation during fallback."""

    @patch("simplemem_lite.embeddings._embed_voyage")
    def test_dimension_validation_skips_incompatible_provider(self, mock_voyage):
        """Verify incompatible providers are skipped when expected_dim is set."""
        mock_voyage.return_value = [[0.1] * 1024]

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = "test-key"

        # Expect 1024 (Voyage dimension) - should work
        result = embed_code_batch(["def foo(): pass"], config, expected_dim=1024)
        assert result.provider == "voyage"
        assert len(result.embeddings[0]) == 1024

    @patch("simplemem_lite.embeddings._embed_code_local")
    def test_dimension_mismatch_falls_to_compatible_local(self, mock_local):
        """Verify fallback to local when it matches expected_dim."""
        mock_local.return_value = [[0.1] * 768]

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = None  # No Voyage key

        with patch.dict("os.environ", {}, clear=True):
            # Expect 768 (local dimension) - should fall through to local
            result = embed_code_batch(["def foo(): pass"], config, expected_dim=768)

        assert result.provider == "local"
        assert len(result.embeddings[0]) == 768

    def test_dimension_mismatch_raises_when_no_compatible_provider(self):
        """Verify ValueError when no provider can produce expected dimension."""
        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = None  # No Voyage key

        with patch.dict("os.environ", {}, clear=True):
            # Expect 2048 - no provider offers this dimension
            with pytest.raises(ValueError, match="No embedding provider available"):
                embed_code_batch(["def foo(): pass"], config, expected_dim=2048)

    @patch("simplemem_lite.embeddings._embed_voyage")
    @patch("simplemem_lite.embeddings._embed_openrouter")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_dimension_skip_voyage_use_openrouter(self, mock_openrouter, mock_voyage):
        """Verify Voyage is skipped and OpenRouter used when dimensions match."""
        mock_voyage.return_value = [[0.1] * 1024]  # Would return 1024
        mock_openrouter.return_value = [[0.1] * 3072]  # Returns 3072

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = "test-key"

        # Expect 3072 (OpenRouter dimension) - should skip Voyage
        result = embed_code_batch(["def foo(): pass"], config, expected_dim=3072)

        assert not mock_voyage.called  # Voyage should be skipped
        assert mock_openrouter.called
        assert result.provider == "openrouter"
        assert len(result.embeddings[0]) == 3072

    @patch("simplemem_lite.embeddings._embed_voyage")
    def test_no_expected_dim_allows_any_provider(self, mock_voyage):
        """Verify any provider is used when expected_dim is None."""
        mock_voyage.return_value = [[0.1] * 1024]

        config = Config()
        config.code_embedding_provider = "voyage"
        config.voyage_api_key = "test-key"

        # No expected_dim - should use primary provider
        result = embed_code_batch(["def foo(): pass"], config, expected_dim=None)

        assert result.provider == "voyage"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
