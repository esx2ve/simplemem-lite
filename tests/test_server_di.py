"""Dependency injection tests for SimpleMem Lite server.

Tests the Dependencies container pattern for testability.
"""

import pytest


class TestDependenciesContainer:
    """Test the Dependencies container class."""

    def test_dependencies_class_exists(self):
        """Dependencies class should be importable."""
        from simplemem_lite.server import Dependencies

        assert Dependencies is not None

    def test_get_dependencies_returns_container(self):
        """get_dependencies should return the global container."""
        from simplemem_lite.server import Dependencies, get_dependencies

        deps = get_dependencies()
        assert isinstance(deps, Dependencies)

    def test_dependencies_has_configure_for_testing(self):
        """Dependencies should have configure_for_testing method."""
        from simplemem_lite.server import Dependencies

        deps = Dependencies()
        assert hasattr(deps, "configure_for_testing")
        assert callable(deps.configure_for_testing)

    def test_dependencies_has_all_properties(self):
        """Dependencies should expose all required properties."""
        from simplemem_lite.server import Dependencies

        expected_properties = [
            "config",
            "store",
            "parser",
            "indexer",
            "code_indexer",
            "watcher_manager",
            "project_manager",
            "bootstrap",
            "job_manager",
        ]

        for prop in expected_properties:
            assert hasattr(Dependencies, prop), f"Missing property: {prop}"


class TestDependencyInjectionPattern:
    """Test that DI pattern enables testability."""

    def test_configure_for_testing_accepts_mock(self):
        """configure_for_testing should accept mock objects."""
        from unittest.mock import Mock

        from simplemem_lite.server import Dependencies

        # Create a fresh container (not the global one)
        deps = Dependencies()

        # Create mocks
        mock_config = Mock()
        mock_store = Mock()

        # Configure with mocks (should not raise)
        deps.configure_for_testing(
            config=mock_config,
            store=mock_store,
        )

        # Note: We can't easily verify the mocks are used because
        # _ensure_initialized fills in None values. This test verifies
        # the API exists and accepts mocks.

    def test_fresh_container_not_initialized(self):
        """Fresh Dependencies container should not be initialized."""
        from simplemem_lite.server import Dependencies

        deps = Dependencies()
        assert deps._initialized is False

    def test_global_deps_is_singleton(self):
        """Global _deps should be same instance across imports."""
        from simplemem_lite.server import _deps as deps1
        from simplemem_lite.server import _deps as deps2

        assert deps1 is deps2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
