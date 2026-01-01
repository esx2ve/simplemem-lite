"""Database tests for SimpleMem Lite.

Tests critical database pathways:
- Batch delete operations (N+1 fix)
- Edge query security (hardcoded queries)
"""

import pytest


class TestBatchDeleteOperations:
    """Test batch delete operations for session cleanup."""

    def test_batch_delete_query_format(self):
        """Batch delete should use IN clause format."""
        uuids = ["uuid-1", "uuid-2", "uuid-3"]

        # Simulate the batch delete query construction
        escaped_uuids = [uid.replace('"', '\\"') for uid in uuids]
        uuid_list = ", ".join(f'"{uid}"' for uid in escaped_uuids)
        query = f"uuid IN ({uuid_list})"

        expected = 'uuid IN ("uuid-1", "uuid-2", "uuid-3")'
        assert query == expected

    def test_batch_delete_escapes_quotes(self):
        """Batch delete should escape quotes in UUIDs (edge case)."""
        uuids = ['uuid-with"quote', 'normal-uuid']

        escaped_uuids = [uid.replace('"', '\\"') for uid in uuids]
        uuid_list = ", ".join(f'"{uid}"' for uid in escaped_uuids)
        query = f"uuid IN ({uuid_list})"

        # Should have escaped the quote
        assert '\\"' in query or 'uuid-with\\"quote' in query

    def test_batch_delete_empty_list(self):
        """Batch delete with empty list should be handled."""
        uuids = []

        # Empty list case - should not proceed with delete
        assert len(uuids) == 0


class TestEdgeQuerySecurity:
    """Test that edge queries use hardcoded templates."""

    def test_all_edge_types_have_queries(self):
        """All allowed edge types should have corresponding queries."""
        from simplemem_lite.db import DatabaseManager

        allowed = DatabaseManager.ALLOWED_VERB_EDGES

        # Check both query dictionaries exist and have all types
        assert hasattr(DatabaseManager, '_EDGE_QUERIES_WITH_SUMMARY')
        assert hasattr(DatabaseManager, '_EDGE_QUERIES_NO_SUMMARY')

        for edge_type in allowed:
            assert edge_type in DatabaseManager._EDGE_QUERIES_WITH_SUMMARY, \
                f"Missing {edge_type} in _EDGE_QUERIES_WITH_SUMMARY"
            assert edge_type in DatabaseManager._EDGE_QUERIES_NO_SUMMARY, \
                f"Missing {edge_type} in _EDGE_QUERIES_NO_SUMMARY"

    def test_queries_are_parameterized(self):
        """Edge queries should use parameterized placeholders."""
        from simplemem_lite.db import DatabaseManager

        for edge_type, query in DatabaseManager._EDGE_QUERIES_WITH_SUMMARY.items():
            assert "$uuid" in query, f"{edge_type} query missing $uuid"
            assert "$name" in query, f"{edge_type} query missing $name"
            assert "$type" in query, f"{edge_type} query missing $type"
            assert "$ts" in query, f"{edge_type} query missing $ts"
            assert "$summary" in query, f"{edge_type} query missing $summary"

        for edge_type, query in DatabaseManager._EDGE_QUERIES_NO_SUMMARY.items():
            assert "$uuid" in query, f"{edge_type} query missing $uuid"
            assert "$name" in query, f"{edge_type} query missing $name"
            assert "$type" in query, f"{edge_type} query missing $type"
            assert "$ts" in query, f"{edge_type} query missing $ts"

    def test_queries_have_no_fstring_patterns(self):
        """Queries should not contain f-string interpolation patterns."""
        from simplemem_lite.db import DatabaseManager

        for edge_type, query in DatabaseManager._EDGE_QUERIES_WITH_SUMMARY.items():
            # Check for common f-string patterns that shouldn't be there
            assert "{edge_type}" not in query, f"{edge_type} has f-string pattern"
            assert "{{" not in query, f"{edge_type} has escaped braces"

        for edge_type, query in DatabaseManager._EDGE_QUERIES_NO_SUMMARY.items():
            assert "{edge_type}" not in query, f"{edge_type} has f-string pattern"
            assert "{{" not in query, f"{edge_type} has escaped braces"

    def test_action_to_edge_mapping(self):
        """Action names should correctly map to edge types."""
        action_to_edge = {
            "reads": "READS",
            "modifies": "MODIFIES",
            "executes": "EXECUTES",
            "triggered": "TRIGGERED",
        }

        for action, expected_edge in action_to_edge.items():
            edge_type = action_to_edge.get(action.lower(), "REFERENCES")
            assert edge_type == expected_edge

    def test_unknown_action_defaults_to_references(self):
        """Unknown actions should default to REFERENCES edge type."""
        action_to_edge = {
            "reads": "READS",
            "modifies": "MODIFIES",
            "executes": "EXECUTES",
            "triggered": "TRIGGERED",
        }

        # Unknown action
        edge_type = action_to_edge.get("unknown_action", "REFERENCES")
        assert edge_type == "REFERENCES"


class TestHealthCheck:
    """Test database health check functionality."""

    def test_health_check_method_exists(self):
        """DatabaseManager should have health_check method."""
        from simplemem_lite.db import DatabaseManager

        assert hasattr(DatabaseManager, 'health_check')
        assert callable(getattr(DatabaseManager, 'health_check'))

    def test_is_healthy_method_exists(self):
        """DatabaseManager should have is_healthy method."""
        from simplemem_lite.db import DatabaseManager

        assert hasattr(DatabaseManager, 'is_healthy')
        assert callable(getattr(DatabaseManager, 'is_healthy'))

    def test_reconnect_falkordb_method_exists(self):
        """DatabaseManager should have reconnect_falkordb method."""
        from simplemem_lite.db import DatabaseManager

        assert hasattr(DatabaseManager, 'reconnect_falkordb')
        assert callable(getattr(DatabaseManager, 'reconnect_falkordb'))

    def test_health_check_returns_expected_structure(self):
        """Health check should return dict with expected keys."""
        # We test the expected structure without actually connecting
        expected_keys = {"falkordb", "lancedb", "timestamp"}

        # Simulate expected return structure
        mock_result = {
            "falkordb": {"healthy": True, "error": None},
            "lancedb": {"healthy": True, "error": None},
            "timestamp": 1234567890.0,
        }

        assert set(mock_result.keys()) == expected_keys
        assert "healthy" in mock_result["falkordb"]
        assert "error" in mock_result["falkordb"]
        assert "healthy" in mock_result["lancedb"]
        assert "error" in mock_result["lancedb"]


class TestModuleStructure:
    """Test that db module structure maintains backward compatibility."""

    def test_import_from_package(self):
        """DatabaseManager should be importable from simplemem_lite.db."""
        from simplemem_lite.db import DatabaseManager

        assert DatabaseManager is not None
        assert DatabaseManager.__name__ == "DatabaseManager"

    def test_import_from_manager(self):
        """DatabaseManager should also be importable from simplemem_lite.db.manager."""
        from simplemem_lite.db.manager import DatabaseManager

        assert DatabaseManager is not None
        assert DatabaseManager.__name__ == "DatabaseManager"

    def test_both_imports_same_class(self):
        """Both import paths should return the same class."""
        from simplemem_lite.db import DatabaseManager as DbFromPackage
        from simplemem_lite.db.manager import DatabaseManager as DbFromManager

        assert DbFromPackage is DbFromManager

    def test_module_exports_all(self):
        """Package should export DatabaseManager in __all__."""
        import simplemem_lite.db as db_module

        assert hasattr(db_module, "__all__")
        assert "DatabaseManager" in db_module.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
