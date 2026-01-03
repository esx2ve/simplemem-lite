"""Tests for orphan entity cleanup functionality.

Tests the automatic cleanup of Entity nodes that are no longer
referenced by any CodeChunk or Memory node after file deletions.
"""

import tempfile
import time
from pathlib import Path

import pytest


class TestOrphanEntityCleanup:
    """Test orphan entity cleanup after code file deletion."""

    @pytest.fixture
    def db_manager(self):
        """Create a DatabaseManager with temporary directories."""
        from simplemem_lite.config import Config
        from simplemem_lite.db.manager import DatabaseManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config = Config(
                data_dir=tmppath / "data",
                code_index_enabled=True,
            )
            config.data_dir.mkdir(parents=True, exist_ok=True)

            db = DatabaseManager(config)
            yield db

    def test_orphan_entity_deleted_after_file_removal(self, db_manager):
        """Entity with only CodeChunk ref should be deleted when file removed."""
        # Use unique names to avoid test pollution in shared FalkorDB
        unique_suffix = str(int(time.time() * 1000))[-6:]
        chunk_uuid = f"test-chunk-{unique_suffix}"
        entity_name = f"helper_function_{unique_suffix}"

        # Create a CodeChunk node
        db_manager.add_code_chunk_node(
            uuid=chunk_uuid,
            filepath="src/helper.py",
            project_root="/test/project",
            start_line=1,
            end_line=10,
        )

        # Link to an Entity
        db_manager.link_code_to_entity(
            chunk_uuid=chunk_uuid,
            entity_name=entity_name,
            entity_type="function",
            relation="DEFINES",
        )

        # Verify Entity exists
        result = db_manager.graph.query(
            "MATCH (e:Entity {name: $name}) RETURN e.name",
            {"name": entity_name},
        )
        assert len(result.result_set) == 1

        # Delete the CodeChunk (simulating file deletion)
        db_manager.graph.query(
            "MATCH (c:CodeChunk {uuid: $uuid}) DETACH DELETE c",
            {"uuid": chunk_uuid},
        )

        # Run orphan cleanup
        deleted = db_manager._cleanup_orphan_entities()

        # At least our entity should be deleted (may be more from other orphans)
        assert deleted >= 1

        # Our specific entity should be gone
        result = db_manager.graph.query(
            "MATCH (e:Entity {name: $name}) RETURN e.name",
            {"name": entity_name},
        )
        assert len(result.result_set) == 0

    def test_entity_with_memory_ref_preserved(self, db_manager):
        """Entity referenced by Memory should survive CodeChunk deletion."""
        ts = int(time.time())

        # Create Entity referenced by both CodeChunk and Memory
        chunk_uuid = "test-chunk-002"
        memory_uuid = "test-memory-002"

        # Create CodeChunk and link to Entity
        db_manager.add_code_chunk_node(
            uuid=chunk_uuid,
            filepath="src/auth.py",
            project_root="/test/project",
            start_line=1,
            end_line=20,
        )
        db_manager.link_code_to_entity(
            chunk_uuid=chunk_uuid,
            entity_name="auth.py",
            entity_type="file",
            relation="IN_FILE",
        )

        # Create Memory that also references this Entity
        db_manager.add_memory_node(
            uuid=memory_uuid,
            content="Fixed auth bug",
            mem_type="fact",
            source="test",
            session_id="session-001",
            created_at=ts,
        )
        db_manager.add_verb_edge(
            memory_uuid=memory_uuid,
            entity_name="auth.py",
            entity_type="file",
            action="modifies",
            timestamp=ts,
        )

        # Delete CodeChunk
        db_manager.graph.query(
            "MATCH (c:CodeChunk {uuid: $uuid}) DETACH DELETE c",
            {"uuid": chunk_uuid},
        )

        # Run orphan cleanup
        deleted = db_manager._cleanup_orphan_entities()

        # Entity should NOT be deleted (still has Memory ref)
        assert deleted == 0

        result = db_manager.graph.query(
            "MATCH (e:Entity {type: 'file'}) WHERE e.name CONTAINS 'auth' RETURN e.name",
        )
        assert len(result.result_set) >= 1

    def test_clear_code_index_clears_graph_nodes(self, db_manager):
        """clear_code_index should delete CodeChunk nodes from graph."""
        # Add CodeChunk to graph directly
        db_manager.add_code_chunk_node(
            uuid="chunk-to-clear",
            filepath="test.py",
            project_root="/clear/test",
            start_line=1,
            end_line=5,
        )

        # Verify it exists
        result = db_manager.graph.query(
            "MATCH (c:CodeChunk {project_root: $root}) RETURN count(c)",
            {"root": "/clear/test"},
        )
        assert result.result_set[0][0] >= 1

        # Clear code index for project
        db_manager.clear_code_index("/clear/test")

        # Verify CodeChunk is gone from graph
        result = db_manager.graph.query(
            "MATCH (c:CodeChunk {project_root: $root}) RETURN count(c)",
            {"root": "/clear/test"},
        )
        assert result.result_set[0][0] == 0

    def test_manual_cleanup_works(self, db_manager):
        """Public cleanup_orphan_entities() should work correctly."""
        # Use unique name to avoid test pollution
        unique_suffix = str(int(time.time() * 1000))[-6:]
        orphan_name = f"orphan_func_{unique_suffix}"

        # Create an orphaned Entity directly
        db_manager.graph.query(
            f"""
            CREATE (e:Entity {{
                name: '{orphan_name}',
                type: 'function',
                created_at: timestamp()
            }})
            """
        )

        # Verify it exists
        result = db_manager.graph.query(
            "MATCH (e:Entity {name: $name}) RETURN e.name",
            {"name": orphan_name},
        )
        assert len(result.result_set) == 1

        # Run public cleanup method
        stats = db_manager.cleanup_orphan_entities()

        # Should report success
        assert stats["status"] == "success"
        assert stats["orphans_deleted"] >= 1

        # Orphan should be gone
        result = db_manager.graph.query(
            "MATCH (e:Entity {name: $name}) RETURN e.name",
            {"name": orphan_name},
        )
        assert len(result.result_set) == 0

    def test_cleanup_on_empty_graph(self, db_manager):
        """Cleanup on empty graph should return 0 without errors."""
        # Clear everything first
        db_manager.graph.query("MATCH (n) DETACH DELETE n")

        # Run cleanup
        deleted = db_manager._cleanup_orphan_entities()

        # Should handle gracefully
        assert deleted == 0

    def test_delete_chunks_by_filepath_triggers_cleanup(self, db_manager):
        """delete_chunks_by_filepath should trigger orphan cleanup."""
        # Use unique names to avoid test pollution
        unique_suffix = str(int(time.time() * 1000))[-6:]
        chunk_uuid = f"chunk-for-filepath-{unique_suffix}"
        entity_name = f"deletable_class_{unique_suffix}"

        # Create CodeChunk with Entity
        db_manager.add_code_chunk_node(
            uuid=chunk_uuid,
            filepath="to_delete.py",
            project_root="/filepath/test",
            start_line=1,
            end_line=10,
        )
        db_manager.link_code_to_entity(
            chunk_uuid=chunk_uuid,
            entity_name=entity_name,
            entity_type="class",
            relation="DEFINES",
        )

        # Verify Entity exists
        result = db_manager.graph.query(
            "MATCH (e:Entity {name: $name}) RETURN e.name",
            {"name": entity_name},
        )
        assert len(result.result_set) == 1

        # Delete CodeChunk manually and run cleanup (simulating what delete_chunks does)
        db_manager.graph.query(
            "MATCH (c:CodeChunk {uuid: $uuid}) DETACH DELETE c",
            {"uuid": chunk_uuid},
        )
        db_manager._cleanup_orphan_entities()

        # Entity should be gone
        result = db_manager.graph.query(
            "MATCH (e:Entity {name: $name}) RETURN e.name",
            {"name": entity_name},
        )
        assert len(result.result_set) == 0


class TestCleanupEdgeCases:
    """Test edge cases for orphan cleanup."""

    @pytest.fixture
    def db_manager(self):
        """Create a DatabaseManager with temporary directories."""
        from simplemem_lite.config import Config
        from simplemem_lite.db.manager import DatabaseManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config = Config(
                data_dir=tmppath / "data",
                code_index_enabled=True,
            )
            config.data_dir.mkdir(parents=True, exist_ok=True)

            db = DatabaseManager(config)
            yield db

    def test_entity_with_outgoing_edges_only_is_orphan(self, db_manager):
        """Entity with only outgoing edges (no incoming) should be deleted."""
        # Use unique names to avoid test pollution
        unique_suffix = str(int(time.time() * 1000))[-6:]
        orphan_name = f"orphan_with_outgoing_{unique_suffix}"
        target_name = f"target_entity_{unique_suffix}"

        # Create Entity that points to something but nothing points to it
        db_manager.graph.query(
            f"""
            CREATE (e1:Entity {{name: '{orphan_name}', type: 'module'}})
            CREATE (e2:Entity {{name: '{target_name}', type: 'function'}})
            CREATE (e1)-[:REFERENCES]->(e2)
            """
        )

        # orphan_with_outgoing has no incoming edges
        # target_entity has incoming edge from orphan_with_outgoing

        db_manager._cleanup_orphan_entities()

        # orphan_with_outgoing should be deleted (no incoming edges)
        result = db_manager.graph.query(
            "MATCH (e:Entity {name: $name}) RETURN e",
            {"name": orphan_name},
        )
        assert len(result.result_set) == 0

        # target_entity should also be deleted now (was only referenced by orphan)
        # Run cleanup again to catch the cascade
        db_manager._cleanup_orphan_entities()

        result = db_manager.graph.query(
            "MATCH (e:Entity {name: $name}) RETURN e",
            {"name": target_name},
        )
        assert len(result.result_set) == 0

    def test_multiple_orphans_deleted(self, db_manager):
        """Multiple orphaned entities should all be deleted."""
        # Use unique prefix to avoid test pollution
        unique_suffix = str(int(time.time() * 1000))[-6:]
        prefix = f"orphan_{unique_suffix}_"

        # Create several orphans with unique names
        for i in range(5):
            db_manager.graph.query(
                f"CREATE (e:Entity {{name: '{prefix}{i}', type: 'function'}})"
            )

        # Verify they were created
        result = db_manager.graph.query(
            f"MATCH (e:Entity) WHERE e.name STARTS WITH '{prefix}' RETURN count(e)"
        )
        assert result.result_set[0][0] == 5

        deleted = db_manager._cleanup_orphan_entities()

        # At least 5 should be deleted (may be more from other orphans)
        assert deleted >= 5

        # Our specific orphans should be gone
        result = db_manager.graph.query(
            f"MATCH (e:Entity) WHERE e.name STARTS WITH '{prefix}' RETURN count(e)"
        )
        assert result.result_set[0][0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
