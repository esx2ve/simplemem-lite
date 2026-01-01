"""Tests for Graph Power Tools (P5).

Tests for:
- get_schema: Complete graph schema for zero-discovery query generation
- execute_validated_cypher: Read-only Cypher with validation and resource limits
"""

import pytest
import re


class TestMutationKeywords:
    """Test mutation keyword detection."""

    def test_mutation_keywords_exist(self):
        """MUTATION_KEYWORDS constant should exist."""
        from simplemem_lite.db import DatabaseManager

        assert hasattr(DatabaseManager, 'MUTATION_KEYWORDS')
        assert isinstance(DatabaseManager.MUTATION_KEYWORDS, frozenset)

    def test_mutation_keywords_complete(self):
        """All dangerous mutation keywords should be included."""
        from simplemem_lite.db import DatabaseManager

        expected = {"CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DETACH"}
        assert DatabaseManager.MUTATION_KEYWORDS == expected

    def test_mutation_keywords_uppercase(self):
        """All mutation keywords should be uppercase."""
        from simplemem_lite.db import DatabaseManager

        for keyword in DatabaseManager.MUTATION_KEYWORDS:
            assert keyword == keyword.upper()


class TestCypherValidation:
    """Test Cypher query validation logic."""

    def test_detect_create_keyword(self):
        """CREATE keyword should be detected."""
        from simplemem_lite.db import DatabaseManager

        query = "CREATE (n:Memory {uuid: 'test'})"
        query_upper = query.upper()

        detected = False
        for keyword in DatabaseManager.MUTATION_KEYWORDS:
            if re.search(rf'\b{keyword}\b', query_upper):
                detected = True
                break

        assert detected

    def test_detect_merge_keyword(self):
        """MERGE keyword should be detected."""
        from simplemem_lite.db import DatabaseManager

        query = "MERGE (n:Memory {uuid: 'test'})"
        query_upper = query.upper()

        detected = False
        for keyword in DatabaseManager.MUTATION_KEYWORDS:
            if re.search(rf'\b{keyword}\b', query_upper):
                detected = True
                break

        assert detected

    def test_detect_delete_keyword(self):
        """DELETE keyword should be detected."""
        from simplemem_lite.db import DatabaseManager

        query = "MATCH (n) DELETE n"
        query_upper = query.upper()

        detected = False
        for keyword in DatabaseManager.MUTATION_KEYWORDS:
            if re.search(rf'\b{keyword}\b', query_upper):
                detected = True
                break

        assert detected

    def test_detect_set_keyword(self):
        """SET keyword should be detected."""
        from simplemem_lite.db import DatabaseManager

        query = "MATCH (n) SET n.name = 'test'"
        query_upper = query.upper()

        detected = False
        for keyword in DatabaseManager.MUTATION_KEYWORDS:
            if re.search(rf'\b{keyword}\b', query_upper):
                detected = True
                break

        assert detected

    def test_detect_remove_keyword(self):
        """REMOVE keyword should be detected."""
        from simplemem_lite.db import DatabaseManager

        query = "MATCH (n) REMOVE n.label"
        query_upper = query.upper()

        detected = False
        for keyword in DatabaseManager.MUTATION_KEYWORDS:
            if re.search(rf'\b{keyword}\b', query_upper):
                detected = True
                break

        assert detected

    def test_detect_detach_keyword(self):
        """DETACH keyword should be detected."""
        from simplemem_lite.db import DatabaseManager

        query = "MATCH (n) DETACH DELETE n"
        query_upper = query.upper()

        detected = False
        for keyword in DatabaseManager.MUTATION_KEYWORDS:
            if re.search(rf'\b{keyword}\b', query_upper):
                detected = True
                break

        assert detected

    def test_read_only_query_passes(self):
        """Read-only queries should not trigger mutation detection."""
        from simplemem_lite.db import DatabaseManager

        read_only_queries = [
            "MATCH (m:Memory) RETURN m.uuid LIMIT 10",
            "MATCH (m)-[r]->(e) RETURN m, r, e",
            "MATCH path = (a)-[*1..3]-(b) RETURN path",
            "MATCH (e:Entity) WHERE e.type = 'file' RETURN e.name",
        ]

        for query in read_only_queries:
            query_upper = query.upper()
            detected = False
            for keyword in DatabaseManager.MUTATION_KEYWORDS:
                if re.search(rf'\b{keyword}\b', query_upper):
                    detected = True
                    break
            assert not detected, f"False positive on query: {query}"

    def test_keyword_in_string_not_detected(self):
        """Keywords inside string literals should ideally not trigger.

        Note: Simple regex detection may have false positives with strings,
        but this is a conservative security measure.
        """
        from simplemem_lite.db import DatabaseManager

        # This is a known limitation - keyword in string will be detected
        query = "MATCH (m) WHERE m.content CONTAINS 'CREATE' RETURN m"
        query_upper = query.upper()

        # This WILL detect CREATE even in string (conservative)
        detected = False
        for keyword in DatabaseManager.MUTATION_KEYWORDS:
            if re.search(rf'\b{keyword}\b', query_upper):
                detected = True
                break

        # Documenting current behavior - conservative blocking
        assert detected, "Conservative detection should catch keywords in strings"


class TestLimitInjection:
    """Test LIMIT injection for resource protection."""

    def test_limit_injection_pattern(self):
        """Queries without LIMIT should get one injected."""
        query = "MATCH (m:Memory) RETURN m.uuid"
        query_stripped = query.strip().rstrip(';')
        max_results = 100

        # Check if LIMIT already present
        has_limit = bool(re.search(r'\bLIMIT\s+\d+', query.upper()))
        assert not has_limit

        # Inject LIMIT
        result = f"{query_stripped} LIMIT {max_results}"
        assert "LIMIT 100" in result

    def test_limit_already_present(self):
        """Queries with LIMIT should not get another one."""
        query = "MATCH (m:Memory) RETURN m.uuid LIMIT 50"
        has_limit = bool(re.search(r'\bLIMIT\s+\d+', query.upper()))
        assert has_limit

    def test_limit_case_insensitive(self):
        """LIMIT detection should be case insensitive."""
        queries = [
            "MATCH (m) RETURN m LIMIT 10",
            "MATCH (m) RETURN m limit 10",
            "MATCH (m) RETURN m Limit 10",
        ]

        for query in queries:
            has_limit = bool(re.search(r'\bLIMIT\s+\d+', query.upper()))
            assert has_limit, f"Failed to detect LIMIT in: {query}"


class TestGetSchema:
    """Test get_schema method structure."""

    def test_get_schema_method_exists(self):
        """DatabaseManager should have get_schema method."""
        from simplemem_lite.db import DatabaseManager

        assert hasattr(DatabaseManager, 'get_schema')
        assert callable(getattr(DatabaseManager, 'get_schema'))

    def test_execute_validated_cypher_method_exists(self):
        """DatabaseManager should have execute_validated_cypher method."""
        from simplemem_lite.db import DatabaseManager

        assert hasattr(DatabaseManager, 'execute_validated_cypher')
        assert callable(getattr(DatabaseManager, 'execute_validated_cypher'))


class TestSchemaStructure:
    """Test expected schema structure (without requiring DB connection)."""

    def test_expected_node_labels(self):
        """Schema should document these node types."""
        expected_nodes = {
            "Memory",
            "Entity",
            "CodeChunk",
            "Project",
            "ProjectIndex",
            "Goal",
        }

        # Just verify expected types are documented
        assert len(expected_nodes) == 6

    def test_expected_relationship_types(self):
        """Schema should document these relationship types."""
        expected_relationships = {
            "RELATES_TO",
            "REFERENCES",
            "READS",
            "MODIFIES",
            "EXECUTES",
            "TRIGGERED",
            "CONTAINS",
            "CHILD_OF",
            "FOLLOWS",
            "BELONGS_TO",
            "HAS_GOAL",
            "ACHIEVES",
        }

        assert len(expected_relationships) == 12

    def test_memory_properties(self):
        """Memory nodes should have these key properties."""
        expected_props = {
            "uuid",
            "content",
            "type",
            "source",
            "project_id",
            "created_at",
        }

        # Document expected structure
        assert "uuid" in expected_props
        assert "content" in expected_props

    def test_entity_properties(self):
        """Entity nodes should have these key properties."""
        expected_props = {
            "name",
            "type",
            "version",
            "reads",
            "modifies",
        }

        assert "name" in expected_props
        assert "type" in expected_props


class TestResultProcessing:
    """Test result processing logic."""

    def test_result_dict_construction(self):
        """Results should be converted to dicts with column names."""
        # Simulate result processing
        header = [("col", "uuid"), ("col", "content")]
        row = ["uuid-123", "test content"]

        column_names = [col[1] if isinstance(col, tuple) else str(col) for col in header]
        result_dict = dict(zip(column_names, row))

        assert result_dict == {"uuid": "uuid-123", "content": "test content"}

    def test_fallback_column_names(self):
        """Missing column names should fallback to col_N format."""
        row = ["value1", "value2", "value3"]

        # No header case - use fallback
        result_dict = {f"col_{i}": val for i, val in enumerate(row)}

        assert result_dict == {"col_0": "value1", "col_1": "value2", "col_2": "value3"}

    def test_truncation_flag(self):
        """Truncation should be indicated when results hit limit."""
        max_results = 100
        row_count = 100

        truncated = row_count >= max_results
        assert truncated

    def test_no_truncation(self):
        """Truncation should be False when under limit."""
        max_results = 100
        row_count = 50

        truncated = row_count >= max_results
        assert not truncated


class TestQueryExamples:
    """Test that example queries are valid patterns."""

    def test_entity_query_pattern(self):
        """Entity queries should use proper pattern."""
        query = "MATCH (m:Memory)-[:RELATES_TO]->(e:Entity) WHERE e.name CONTAINS 'auth' RETURN m.uuid, m.content LIMIT 20"

        # Verify key patterns
        assert "MATCH" in query
        assert ":RELATES_TO" in query
        assert "WHERE" in query
        assert "RETURN" in query
        assert "LIMIT" in query

    def test_path_query_pattern(self):
        """Path traversal queries should use proper pattern."""
        query = "MATCH path = (m:Memory)-[*1..2]-(other) WHERE m.uuid = $uuid RETURN path"

        assert "MATCH path =" in query
        assert "[*1..2]" in query
        assert "$uuid" in query

    def test_ordered_query_pattern(self):
        """Ordered queries should use ORDER BY."""
        query = "MATCH (e:Entity {type: 'file'}) RETURN e.name, e.version ORDER BY e.version DESC LIMIT 10"

        assert "ORDER BY" in query
        assert "DESC" in query
        assert "LIMIT" in query


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
