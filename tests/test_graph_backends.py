"""Tests for graph backend abstraction layer.

Tests FalkorDB, KuzuDB, and Memgraph backends through the common GraphBackend protocol.
"""

import tempfile
from pathlib import Path

import pytest

from simplemem_lite.db.graph_protocol import GraphBackend, QueryResult, BaseGraphBackend
from simplemem_lite.db.graph_factory import get_backend_info


class TestQueryResult:
    """Test QueryResult dataclass."""

    def test_empty_result(self):
        """Empty result should be falsy and have len 0."""
        result = QueryResult(result_set=[])
        assert not result
        assert len(result) == 0

    def test_non_empty_result(self):
        """Non-empty result should be truthy."""
        result = QueryResult(result_set=[["a", 1], ["b", 2]])
        assert result
        assert len(result) == 2

    def test_iteration(self):
        """Should iterate over result set."""
        result = QueryResult(result_set=[["a"], ["b"], ["c"]])
        values = [row[0] for row in result]
        assert values == ["a", "b", "c"]

    def test_with_header(self):
        """Should store header."""
        result = QueryResult(
            result_set=[["alice", 30]],
            header=["name", "age"],
        )
        assert result.header == ["name", "age"]

    def test_with_stats(self):
        """Should store stats."""
        result = QueryResult(
            result_set=[],
            stats={"nodes_created": 1},
        )
        assert result.stats["nodes_created"] == 1


class TestKuzuDBBackend:
    """Test KuzuDB backend implementation."""

    @pytest.fixture
    def kuzu_backend(self):
        """Create a temporary KuzuDB backend for testing."""
        try:
            from simplemem_lite.db.kuzu_backend import KuzuDBBackend
        except ImportError:
            pytest.skip("KuzuDB not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = KuzuDBBackend(Path(tmpdir) / "test_kuzu")
            backend.init_schema()
            yield backend
            backend.close()

    def test_backend_name(self, kuzu_backend):
        """Should return 'kuzu'."""
        assert kuzu_backend.backend_name == "kuzu"

    def test_health_check(self, kuzu_backend):
        """Health check should pass."""
        assert kuzu_backend.health_check() is True

    def test_simple_query(self, kuzu_backend):
        """Should execute simple Cypher."""
        result = kuzu_backend.query("RETURN 1 AS num")
        assert len(result) == 1
        assert result.result_set[0][0] == 1

    def test_create_and_match_memory(self, kuzu_backend):
        """Should create and retrieve a Memory node."""
        import time

        ts = int(time.time())

        # Create
        kuzu_backend.merge_memory(
            uuid="test-uuid-123",
            content="Test memory content",
            mem_type="fact",
            source="test",
            session_id=None,
            created_at=ts,
        )

        # Match
        result = kuzu_backend.query(
            "MATCH (m:Memory {uuid: $uuid}) RETURN m.content",
            {"uuid": "test-uuid-123"},
        )
        assert len(result) == 1
        assert result.result_set[0][0] == "Test memory content"

    def test_entity_composite_id(self, kuzu_backend):
        """Should generate correct composite ID for entities."""
        entity_id = kuzu_backend.get_entity_id("test.py", "file")
        assert entity_id == "file:test.py"

    def test_create_and_match_entity(self, kuzu_backend):
        """Should create and retrieve an Entity node."""
        import time

        ts = int(time.time())
        entity_id = kuzu_backend.merge_entity("main.py", "file", ts)

        result = kuzu_backend.query(
            "MATCH (e:Entity {id: $id}) RETURN e.name, e.type",
            {"id": entity_id},
        )
        assert len(result) == 1
        assert result.result_set[0][0] == "main.py"
        assert result.result_set[0][1] == "file"

    def test_protocol_compliance(self, kuzu_backend):
        """KuzuDBBackend should implement GraphBackend protocol."""
        assert isinstance(kuzu_backend, GraphBackend)
        assert hasattr(kuzu_backend, "backend_name")
        assert hasattr(kuzu_backend, "query")
        assert hasattr(kuzu_backend, "health_check")
        assert hasattr(kuzu_backend, "close")
        assert hasattr(kuzu_backend, "init_schema")


class TestFalkorDBBackend:
    """Test FalkorDB backend implementation."""

    @pytest.fixture
    def falkor_backend(self):
        """Create FalkorDB backend if available."""
        try:
            from simplemem_lite.db.falkor_backend import FalkorDBBackend, is_falkordb_available
        except ImportError:
            pytest.skip("FalkorDB package not installed")

        if not is_falkordb_available():
            pytest.skip("FalkorDB server not running")

        backend = FalkorDBBackend()
        backend.init_schema()
        yield backend
        backend.close()

    def test_backend_name(self, falkor_backend):
        """Should return 'falkordb'."""
        assert falkor_backend.backend_name == "falkordb"

    def test_health_check(self, falkor_backend):
        """Health check should pass."""
        assert falkor_backend.health_check() is True

    def test_simple_query(self, falkor_backend):
        """Should execute simple Cypher."""
        result = falkor_backend.query("RETURN 1 AS num")
        assert len(result) == 1
        assert result.result_set[0][0] == 1

    def test_protocol_compliance(self, falkor_backend):
        """FalkorDBBackend should implement GraphBackend protocol."""
        assert isinstance(falkor_backend, GraphBackend)


class TestMemgraphBackend:
    """Test Memgraph backend implementation."""

    @pytest.fixture
    def memgraph_backend(self):
        """Create Memgraph backend if available."""
        try:
            from simplemem_lite.db.memgraph_backend import MemgraphBackend, is_memgraph_available
        except ImportError:
            pytest.skip("neo4j package not installed")

        if not is_memgraph_available():
            pytest.skip("Memgraph server not running")

        backend = MemgraphBackend()
        backend.init_schema()
        yield backend
        backend.close()

    def test_backend_name(self, memgraph_backend):
        """Should return 'memgraph'."""
        assert memgraph_backend.backend_name == "memgraph"

    def test_health_check(self, memgraph_backend):
        """Health check should pass."""
        assert memgraph_backend.health_check() is True

    def test_simple_query(self, memgraph_backend):
        """Should execute simple Cypher."""
        result = memgraph_backend.query("RETURN 1 AS num")
        assert len(result) == 1
        assert result.result_set[0][0] == 1

    def test_protocol_compliance(self, memgraph_backend):
        """MemgraphBackend should implement GraphBackend protocol."""
        assert isinstance(memgraph_backend, GraphBackend)

    def test_reinit_code_chunk_indexes_noop(self, memgraph_backend):
        """reinit_code_chunk_indexes should be a no-op for Memgraph."""
        # Should not raise - just a no-op
        memgraph_backend.reinit_code_chunk_indexes()


class TestGraphFactory:
    """Test graph factory auto-detection."""

    def test_get_backend_info(self):
        """Should return backend availability info."""
        info = get_backend_info()

        assert "memgraph" in info
        assert "falkordb" in info
        assert "kuzu" in info
        assert "active" in info

        # At least one should be installed
        assert info["memgraph"]["installed"] or info["falkordb"]["installed"] or info["kuzu"]["installed"]

    def test_create_kuzu_backend(self):
        """Should create KuzuDB backend when forced."""
        try:
            from simplemem_lite.db.graph_factory import create_graph_backend
        except ImportError:
            pytest.skip("Graph factory not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = create_graph_backend(
                backend="kuzu",
                kuzu_path=Path(tmpdir) / "test_db",
            )
            assert backend.backend_name == "kuzu"
            assert backend.health_check()

    def test_auto_detection(self):
        """Auto mode should select an available backend."""
        try:
            from simplemem_lite.db.graph_factory import create_graph_backend
        except ImportError:
            pytest.skip("Graph factory not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = create_graph_backend(
                backend="auto",
                kuzu_path=Path(tmpdir) / "test_db",
            )
            assert backend.backend_name in ("memgraph", "falkordb", "kuzu")
            assert backend.health_check()


class TestCypherDialectTranslation:
    """Test Cypher dialect translation for KuzuDB."""

    @pytest.fixture
    def kuzu_backend(self):
        """Create a temporary KuzuDB backend."""
        try:
            from simplemem_lite.db.kuzu_backend import KuzuDBBackend
        except ImportError:
            pytest.skip("KuzuDB not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = KuzuDBBackend(Path(tmpdir) / "test_kuzu")
            backend.init_schema()
            yield backend
            backend.close()

    def test_variable_length_path(self, kuzu_backend):
        """Variable-length paths should work."""
        # Create nodes first
        import time

        ts = int(time.time())
        kuzu_backend.merge_memory("m1", "First", "fact", "test", None, ts)
        kuzu_backend.merge_memory("m2", "Second", "fact", "test", None, ts)

        # Create relationship
        kuzu_backend.query(
            """
            MATCH (a:Memory {uuid: 'm1'}), (b:Memory {uuid: 'm2'})
            CREATE (a)-[:RELATES_TO {relation_type: 'test', weight: 1.0}]->(b)
            """
        )

        # Query with variable-length path
        result = kuzu_backend.query(
            "MATCH (a:Memory {uuid: 'm1'})-[*1..2]-(b:Memory) RETURN b.uuid"
        )
        assert len(result) >= 1
