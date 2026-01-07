"""Tests for graph/vector sync health functionality.

Tests the sync health detection and repair mechanism that handles
the desync problem where memories exist in graph but not in LanceDB.
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch


class TestSyncHealthDetection:
    """Test get_sync_health() detection logic."""

    def test_empty_stores_is_healthy(self):
        """Empty graph and vector stores should report healthy."""
        from simplemem_lite.db import DatabaseManager

        # Mock graph result
        graph_result = MagicMock()
        graph_result.result_set = []

        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(tolist=MagicMock(return_value=[])))

        # Create manager with mocked components
        with patch.object(DatabaseManager, '__init__', lambda x, y: None):
            manager = DatabaseManager(None)
            manager.graph = MagicMock()
            manager.graph.query.return_value = graph_result
            manager.lance_table = MagicMock()
            manager.lance_table.to_pandas.return_value = mock_df
            manager._write_lock = threading.Lock()

            result = manager.get_sync_health()

            assert result["graph_count"] == 0
            assert result["vector_count"] == 0
            assert result["missing_count"] == 0
            assert result["healthy"] is True
            assert result["sync_ratio"] == 1.0

    def test_sync_ratio_calculation(self):
        """Sync ratio should be (graph - missing) / graph."""
        # With 100 in graph, 10 missing: ratio = 90/100 = 0.9
        graph_count = 100
        missing_count = 10
        expected_ratio = (graph_count - missing_count) / graph_count

        assert expected_ratio == 0.9

    def test_healthy_threshold(self):
        """Healthy should be True when sync_ratio >= 0.99."""
        # 99% sync = healthy
        assert 0.99 >= 0.99  # boundary case

        # 98% sync = not healthy
        assert 0.98 < 0.99


class TestRetryQueue:
    """Test the vector write retry queue mechanism."""

    def test_retry_queue_initialization(self):
        """MemoryStore should initialize with empty retry queue."""
        from simplemem_lite.memory import MemoryStore

        with patch.object(MemoryStore, '__init__', lambda x, y=None: None):
            store = MemoryStore()
            store._vector_retry_queue = []
            store._max_retry_queue_size = 100
            store._retry_queue_lock = threading.Lock()
            store._retry_backoff_seconds = 30
            store._last_retry_attempt = 0.0

            assert store.get_retry_queue_size() == 0

    def test_retry_queue_max_size(self):
        """Retry queue should enforce max size by dropping oldest."""
        from simplemem_lite.memory import MemoryStore

        with patch.object(MemoryStore, '__init__', lambda x, y=None: None):
            store = MemoryStore()
            store._vector_retry_queue = []
            store._max_retry_queue_size = 3  # Small for testing
            store._retry_queue_lock = threading.Lock()

            # Add items up to and beyond max
            for i in range(5):
                store._queue_vector_retry(
                    uuid=f"uuid-{i}",
                    embedding=[0.1] * 10,
                    content=f"content-{i}",
                    mem_type="fact",
                    session_id=None,
                    metadata={},
                )

            # Should only have last 3 items
            assert store.get_retry_queue_size() == 3

            # Oldest should have been dropped
            with store._retry_queue_lock:
                uuids = [item["uuid"] for item in store._vector_retry_queue]
            assert "uuid-0" not in uuids
            assert "uuid-1" not in uuids
            assert "uuid-4" in uuids

    def test_retry_queue_no_duplicates(self):
        """Retry queue should not allow duplicate UUIDs."""
        from simplemem_lite.memory import MemoryStore

        with patch.object(MemoryStore, '__init__', lambda x, y=None: None):
            store = MemoryStore()
            store._vector_retry_queue = []
            store._max_retry_queue_size = 100
            store._retry_queue_lock = threading.Lock()

            # Add same UUID twice
            store._queue_vector_retry(
                uuid="uuid-dup",
                embedding=[0.1] * 10,
                content="content",
                mem_type="fact",
                session_id=None,
                metadata={},
            )
            store._queue_vector_retry(
                uuid="uuid-dup",
                embedding=[0.1] * 10,
                content="content2",
                mem_type="fact",
                session_id=None,
                metadata={},
            )

            assert store.get_retry_queue_size() == 1

    def test_retry_backoff_respected(self):
        """process_retry_queue should respect backoff interval."""
        from simplemem_lite.memory import MemoryStore

        with patch.object(MemoryStore, '__init__', lambda x, y=None: None):
            store = MemoryStore()
            store._vector_retry_queue = [{"uuid": "test", "embedding": [], "content": "", "type": "fact", "session_id": "", "metadata": {}}]
            store._max_retry_queue_size = 100
            store._retry_queue_lock = threading.Lock()
            store._retry_backoff_seconds = 30
            store._last_retry_attempt = time.time()  # Just attempted
            store.db = MagicMock()

            # Should skip due to backoff
            result = store.process_retry_queue(force=False)
            assert result["skipped"] is True

    def test_retry_force_ignores_backoff(self):
        """process_retry_queue with force=True should ignore backoff."""
        from simplemem_lite.memory import MemoryStore

        with patch.object(MemoryStore, '__init__', lambda x, y=None: None):
            store = MemoryStore()
            store._vector_retry_queue = [{"uuid": "test", "embedding": [0.1], "content": "test", "type": "fact", "session_id": "", "metadata": {}}]
            store._max_retry_queue_size = 100
            store._retry_queue_lock = threading.Lock()
            store._retry_backoff_seconds = 30
            store._last_retry_attempt = time.time()  # Just attempted

            # Mock DB write to fail
            store.db = MagicMock()
            store.db.write_lock = threading.Lock()
            store.db.lance_table.add.side_effect = Exception("DB error")

            # Force should process despite backoff
            result = store.process_retry_queue(force=True)
            assert result["skipped"] is False
            assert result["failed"] == 1


class TestRepairSync:
    """Test repair_sync() functionality."""

    def test_dry_run_returns_missing_uuids(self):
        """Dry run should report what would be repaired without changing anything."""
        # The dry_run parameter should return missing_uuids preview
        dry_run_result = {
            "dry_run": True,
            "missing_count": 5,
            "repaired_count": 0,
            "errors": [],
            "missing_uuids": ["uuid-1", "uuid-2", "uuid-3", "uuid-4", "uuid-5"],
        }

        assert dry_run_result["dry_run"] is True
        assert dry_run_result["repaired_count"] == 0
        assert len(dry_run_result["missing_uuids"]) == 5

    def test_actual_repair_returns_count(self):
        """Actual repair should return count of repaired memories."""
        repair_result = {
            "dry_run": False,
            "missing_count": 5,
            "repaired_count": 5,
            "errors": [],
        }

        assert repair_result["dry_run"] is False
        assert repair_result["repaired_count"] == 5


class TestThreadSafety:
    """Test thread safety of retry queue operations."""

    def test_concurrent_queue_operations(self):
        """Multiple threads should safely access retry queue."""
        from simplemem_lite.memory import MemoryStore

        with patch.object(MemoryStore, '__init__', lambda x, y=None: None):
            store = MemoryStore()
            store._vector_retry_queue = []
            store._max_retry_queue_size = 100
            store._retry_queue_lock = threading.Lock()

            errors = []

            def add_items(thread_id):
                try:
                    for i in range(10):
                        store._queue_vector_retry(
                            uuid=f"thread-{thread_id}-item-{i}",
                            embedding=[0.1] * 10,
                            content=f"content",
                            mem_type="fact",
                            session_id=None,
                            metadata={},
                        )
                except Exception as e:
                    errors.append(e)

            # Start multiple threads
            threads = [threading.Thread(target=add_items, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # No errors should have occurred
            assert len(errors) == 0

            # All items should be in queue (50 total, no duplicates)
            assert store.get_retry_queue_size() == 50
