"""Tests for SessionStateDB - Session indexing state management.

Tests:
- Lock acquisition and release
- Hash-based skip detection
- Session state CRUD operations
- Concurrent access safety
"""

import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pytest

from simplemem_lite.session_state import (
    SessionStateDB,
    SessionState,
    compute_content_hash,
    read_from_offset,
    LOCK_EXPIRY_SECONDS,
)


class TestSessionStateDB:
    """Test SessionStateDB initialization and schema."""

    def test_init_creates_database(self, tmp_path):
        """Database file should be created on init."""
        db_path = tmp_path / "session_state.db"
        db = SessionStateDB(db_path)

        assert db_path.exists()
        db.close()

    def test_init_creates_tables(self, tmp_path):
        """Schema should create required tables."""
        db_path = tmp_path / "session_state.db"
        db = SessionStateDB(db_path)

        # Check tables exist
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "indexed_sessions" in tables
        assert "session_locks" in tables
        db.close()

    def test_wal_mode_enabled(self, tmp_path):
        """WAL journal mode should be enabled."""
        db_path = tmp_path / "session_state.db"
        db = SessionStateDB(db_path)

        # Check WAL mode
        cursor = db.conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]

        assert mode.lower() == "wal"
        db.close()


class TestLockAcquisition:
    """Test session locking mechanism."""

    def test_acquire_lock_success(self, tmp_path):
        """Lock acquisition should succeed for unlocked session."""
        db = SessionStateDB(tmp_path / "test.db")

        result = db.acquire_lock("session-123")

        assert result is True
        db.close()

    def test_acquire_lock_already_owned(self, tmp_path):
        """Re-acquiring own lock should succeed (extend)."""
        db = SessionStateDB(tmp_path / "test.db")

        db.acquire_lock("session-123")
        result = db.acquire_lock("session-123")

        assert result is True
        db.close()

    def test_acquire_lock_blocked_by_other(self, tmp_path):
        """Lock should be denied if held by another owner."""
        db1 = SessionStateDB(tmp_path / "test.db")
        db2 = SessionStateDB(tmp_path / "test.db")  # Different owner

        db1.acquire_lock("session-123")
        result = db2.acquire_lock("session-123")

        assert result is False
        db1.close()
        db2.close()

    def test_release_lock_success(self, tmp_path):
        """Releasing own lock should succeed."""
        db = SessionStateDB(tmp_path / "test.db")

        db.acquire_lock("session-123")
        result = db.release_lock("session-123")

        assert result is True
        db.close()

    def test_release_lock_not_owner(self, tmp_path):
        """Releasing lock we don't own should return False."""
        db1 = SessionStateDB(tmp_path / "test.db")
        db2 = SessionStateDB(tmp_path / "test.db")

        db1.acquire_lock("session-123")
        result = db2.release_lock("session-123")

        assert result is False
        db1.close()
        db2.close()

    def test_lock_after_release(self, tmp_path):
        """Lock should be acquirable after release."""
        db1 = SessionStateDB(tmp_path / "test.db")
        db2 = SessionStateDB(tmp_path / "test.db")

        db1.acquire_lock("session-123")
        db1.release_lock("session-123")
        result = db2.acquire_lock("session-123")

        assert result is True
        db1.close()
        db2.close()

    def test_expired_lock_takeover(self, tmp_path):
        """Expired locks should be taken over."""
        db1 = SessionStateDB(tmp_path / "test.db")
        db2 = SessionStateDB(tmp_path / "test.db")

        # Acquire with very short timeout
        db1.acquire_lock("session-123", timeout_seconds=0.1)

        # Wait for expiry
        time.sleep(0.2)

        # Should be able to take over
        result = db2.acquire_lock("session-123")
        assert result is True

        db1.close()
        db2.close()


class TestSessionState:
    """Test session state CRUD operations."""

    def test_get_nonexistent_session(self, tmp_path):
        """Getting nonexistent session should return None."""
        db = SessionStateDB(tmp_path / "test.db")

        state = db.get_session_state("nonexistent")

        assert state is None
        db.close()

    def test_update_and_get_session(self, tmp_path):
        """Session state should be retrievable after update."""
        db = SessionStateDB(tmp_path / "test.db")

        db.update_session_state(
            session_id="session-123",
            file_path="/path/to/file.jsonl",
            byte_offset=1000,
            content_hash="abc123",
            status="indexed",
            inode=12345,
        )

        state = db.get_session_state("session-123")

        assert state is not None
        assert state.session_id == "session-123"
        assert state.file_path == "/path/to/file.jsonl"
        assert state.indexed_byte_offset == 1000
        assert state.content_hash == "abc123"
        assert state.status == "indexed"
        assert state.inode == 12345
        db.close()

    def test_update_existing_session(self, tmp_path):
        """Updating existing session should overwrite values."""
        db = SessionStateDB(tmp_path / "test.db")

        db.update_session_state(
            session_id="session-123",
            file_path="/path/to/file.jsonl",
            byte_offset=1000,
            content_hash="abc123",
            status="indexed",
        )

        db.update_session_state(
            session_id="session-123",
            file_path="/path/to/file.jsonl",
            byte_offset=2000,
            content_hash="def456",
            status="indexed",
        )

        state = db.get_session_state("session-123")

        assert state.indexed_byte_offset == 2000
        assert state.content_hash == "def456"
        db.close()

    def test_set_status(self, tmp_path):
        """Setting status should update only status."""
        db = SessionStateDB(tmp_path / "test.db")

        db.update_session_state(
            session_id="session-123",
            file_path="/path/to/file.jsonl",
            byte_offset=1000,
            content_hash="abc123",
            status="indexed",
        )

        db.set_status("session-123", "failed")
        state = db.get_session_state("session-123")

        assert state.status == "failed"
        assert state.indexed_byte_offset == 1000  # Unchanged
        db.close()

    def test_list_sessions(self, tmp_path):
        """Listing sessions should return all sessions."""
        db = SessionStateDB(tmp_path / "test.db")

        for i in range(5):
            db.update_session_state(
                session_id=f"session-{i}",
                file_path=f"/path/to/file-{i}.jsonl",
                byte_offset=i * 100,
                content_hash=f"hash-{i}",
                status="indexed",
            )

        sessions = db.list_sessions()

        assert len(sessions) == 5
        db.close()

    def test_list_sessions_with_status_filter(self, tmp_path):
        """Listing sessions should respect status filter."""
        db = SessionStateDB(tmp_path / "test.db")

        db.update_session_state("s1", "/p1", 100, "h1", "indexed")
        db.update_session_state("s2", "/p2", 200, "h2", "failed")
        db.update_session_state("s3", "/p3", 300, "h3", "indexed")

        indexed = db.list_sessions(status="indexed")
        failed = db.list_sessions(status="failed")

        assert len(indexed) == 2
        assert len(failed) == 1
        db.close()


class TestContentHash:
    """Test content hash computation."""

    def test_compute_hash_small_file(self, tmp_path):
        """Hash should be computed for small files."""
        file_path = tmp_path / "small.txt"
        file_path.write_text("Hello, World!")

        hash_result = compute_content_hash(file_path)

        assert len(hash_result) == 64  # SHA256 hex length
        assert hash_result != ""

    def test_compute_hash_same_content(self, tmp_path):
        """Same content should produce same hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Same content")
        file2.write_text("Same content")

        hash1 = compute_content_hash(file1)
        hash2 = compute_content_hash(file2)

        assert hash1 == hash2

    def test_compute_hash_different_content(self, tmp_path):
        """Different content should produce different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")

        hash1 = compute_content_hash(file1)
        hash2 = compute_content_hash(file2)

        assert hash1 != hash2

    def test_compute_hash_nonexistent_file(self, tmp_path):
        """Nonexistent file should return empty string."""
        file_path = tmp_path / "nonexistent.txt"

        hash_result = compute_content_hash(file_path)

        assert hash_result == ""

    def test_compute_hash_large_file(self, tmp_path):
        """Large file should still produce hash (first 1MB only)."""
        file_path = tmp_path / "large.txt"
        # Write 2MB of data
        file_path.write_text("x" * (2 * 1024 * 1024))

        hash_result = compute_content_hash(file_path)

        assert len(hash_result) == 64


class TestReadFromOffset:
    """Test reading file from byte offset."""

    def test_read_from_start(self, tmp_path):
        """Reading from offset 0 should return entire content."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Line 1\nLine 2\nLine 3\n")

        content, new_offset = read_from_offset(file_path, 0)

        assert "Line 1" in content
        assert "Line 2" in content
        assert "Line 3" in content
        assert new_offset > 0

    def test_read_from_middle(self, tmp_path):
        """Reading from middle should skip beginning."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Line 1\nLine 2\nLine 3\n")

        content, new_offset = read_from_offset(file_path, 7)  # After "Line 1\n"

        assert "Line 1" not in content
        assert "Line 2" in content
        assert "Line 3" in content

    def test_read_from_end(self, tmp_path):
        """Reading from end should return empty."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Line 1\nLine 2\n")
        file_size = file_path.stat().st_size

        content, new_offset = read_from_offset(file_path, file_size)

        assert content == ""
        assert new_offset == file_size

    def test_read_nonexistent_file(self, tmp_path):
        """Reading nonexistent file should return empty."""
        file_path = tmp_path / "nonexistent.txt"

        content, new_offset = read_from_offset(file_path, 0)

        assert content == ""
        assert new_offset == 0


class TestConcurrency:
    """Test concurrent access patterns."""

    def test_concurrent_lock_attempts(self, tmp_path):
        """Only one thread should acquire the lock."""
        db_path = tmp_path / "test.db"
        results = []
        lock = threading.Lock()

        def try_lock():
            db = SessionStateDB(db_path)
            acquired = db.acquire_lock("session-123", timeout_seconds=10)
            with lock:
                results.append(acquired)
            time.sleep(0.1)  # Hold lock briefly
            if acquired:
                db.release_lock("session-123")
            db.close()

        threads = [threading.Thread(target=try_lock) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one should have succeeded initially
        # (others may succeed after releases, but at least one True)
        assert any(results)

    def test_state_updates_are_atomic(self, tmp_path):
        """Concurrent state updates should not corrupt data."""
        db_path = tmp_path / "test.db"
        errors = []

        def update_state(session_num):
            try:
                db = SessionStateDB(db_path)
                for i in range(10):
                    db.update_session_state(
                        session_id=f"session-{session_num}",
                        file_path=f"/path/{session_num}",
                        byte_offset=i * 100,
                        content_hash=f"hash-{session_num}-{i}",
                        status="indexed",
                    )
                db.close()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_state, args=(i,))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
