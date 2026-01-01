"""Config tests for SimpleMem Lite.

Tests critical configuration pathways:
- Database limit constants exist and have sensible defaults
- Environment variable override mechanism works
"""

import os
from unittest.mock import patch

import pytest


class TestConfigDefaults:
    """Test that config has expected default values."""

    def test_embedding_cache_size_default(self):
        """Embedding cache size should default to 1000."""
        from simplemem_lite.config import Config

        config = Config()
        assert config.embedding_cache_size == 1000

    def test_memory_content_max_size_default(self):
        """Memory content max size should default to 5000."""
        from simplemem_lite.config import Config

        config = Config()
        assert config.memory_content_max_size == 5000

    def test_summary_max_size_default(self):
        """Summary max size should default to 500."""
        from simplemem_lite.config import Config

        config = Config()
        assert config.summary_max_size == 500

    def test_max_graph_hops_default(self):
        """Max graph hops should default to 3."""
        from simplemem_lite.config import Config

        config = Config()
        assert config.max_graph_hops == 3

    def test_graph_path_limit_default(self):
        """Graph path limit should default to 100."""
        from simplemem_lite.config import Config

        config = Config()
        assert config.graph_path_limit == 100

    def test_cross_session_limit_default(self):
        """Cross session limit should default to 50."""
        from simplemem_lite.config import Config

        config = Config()
        assert config.cross_session_limit == 50


class TestConfigEnvironmentOverrides:
    """Test environment variable overrides for config."""

    def test_memory_content_max_size_env_override(self):
        """Memory content max size should be overridable via env var."""
        with patch.dict(os.environ, {"SIMPLEMEM_LITE_MEMORY_CONTENT_MAX_SIZE": "10000"}):
            # Need to reimport to pick up env change
            from importlib import reload

            import simplemem_lite.config as config_module

            reload(config_module)
            config = config_module.Config()
            assert config.memory_content_max_size == 10000

    def test_max_graph_hops_env_override(self):
        """Max graph hops should be overridable via env var."""
        with patch.dict(os.environ, {"SIMPLEMEM_LITE_MAX_GRAPH_HOPS": "5"}):
            from importlib import reload

            import simplemem_lite.config as config_module

            reload(config_module)
            config = config_module.Config()
            assert config.max_graph_hops == 5


class TestConfigLimitsArePositive:
    """Test that config limits are positive integers."""

    def test_all_limits_are_positive(self):
        """All limit config values should be positive."""
        from simplemem_lite.config import Config

        config = Config()
        assert config.embedding_cache_size > 0
        assert config.memory_content_max_size > 0
        assert config.summary_max_size > 0
        assert config.max_graph_hops > 0
        assert config.graph_path_limit > 0
        assert config.cross_session_limit > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
