"""Session indexing state management with SQLite.

Provides:
- Session locks for race condition prevention
- Content hash tracking for change detection
- Line index tracking for incremental indexing
- WAL mode for concurrent access safety
"""

import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from simplemem_lite.log_config import get_logger

log = get_logger("session_state")

# Lock expiry time in seconds (stale lock cleanup)
LOCK_EXPIRY_SECONDS = 300  # 5 minutes


@dataclass
class SessionState:
    """State of an indexed session."""

    session_id: str
    file_path: str
    indexed_line_index: int  # Line index (not byte offset) for incremental reads
    content_hash: str
    status: str  # 'pending', 'processing', 'indexed', 'failed'
    indexed_at: float
    inode: int | None = None


class SessionStateDB:
    """SQLite-based session indexing state with WAL mode.

    Provides:
    - Per-session distributed locks
    - Content hash tracking for skip detection
    - Line index tracking for incremental reads
    - Concurrent access safety via WAL mode
    """

    def __init__(self, db_path: Path | str):
        """Initialize session state database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate unique owner ID for this process
        self._owner_id = f"{os.getpid()}-{time.time()}"

        log.info(f"Initializing SessionStateDB at {self.db_path}")
        self._init_connection()
        self._create_schema()
        self._cleanup_stale_locks()

    def _init_connection(self) -> None:
        """Initialize database connection with WAL mode."""
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row

        # Enable WAL mode for concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Set busy timeout to 30 seconds
        self.conn.execute("PRAGMA busy_timeout=30000")
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys=ON")

        log.debug("SQLite connection initialized with WAL mode")

    def _create_schema(self) -> None:
        """Create database schema if not exists."""
        self.conn.executescript("""
            -- Session indexing state
            CREATE TABLE IF NOT EXISTS indexed_sessions (
                session_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                indexed_line_index INTEGER DEFAULT 0,
                content_hash TEXT,
                inode INTEGER,
                status TEXT CHECK(status IN ('pending', 'processing', 'indexed', 'failed')) DEFAULT 'pending',
                indexed_at REAL,
                created_at REAL DEFAULT (unixepoch('now')),
                updated_at REAL DEFAULT (unixepoch('now'))
            );

            -- Session locks for distributed locking
            CREATE TABLE IF NOT EXISTS session_locks (
                session_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                acquired_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );

            -- Index for efficient queries
            CREATE INDEX IF NOT EXISTS idx_sessions_status ON indexed_sessions(status);
            CREATE INDEX IF NOT EXISTS idx_locks_expires ON session_locks(expires_at);
        """)
        self.conn.commit()

        # Migrate old column name if it exists (SQLite 3.25+)
        self._migrate_column_names()

        log.debug("Database schema created/verified")

    def _migrate_column_names(self) -> None:
        """Migrate old column names to new names for consistency."""
        try:
            # Check if old column exists
            cursor = self.conn.execute("PRAGMA table_info(indexed_sessions)")
            columns = {row[1] for row in cursor.fetchall()}

            if "indexed_byte_offset" in columns and "indexed_line_index" not in columns:
                log.info("Migrating indexed_byte_offset -> indexed_line_index")
                self.conn.execute(
                    "ALTER TABLE indexed_sessions RENAME COLUMN indexed_byte_offset TO indexed_line_index"
                )
                self.conn.commit()
                log.info("Column migration complete")
        except sqlite3.OperationalError as e:
            # SQLite < 3.25 doesn't support RENAME COLUMN - data is still usable
            log.warning(f"Column migration skipped (SQLite version): {e}")

    def _cleanup_stale_locks(self) -> None:
        """Remove expired locks on startup.

        This is a best-effort cleanup - if database is locked by another
        connection, skip cleanup (it will happen during normal operations).
        """
        now = time.time()
        try:
            cursor = self.conn.execute(
                "DELETE FROM session_locks WHERE expires_at < ?",
                (now,)
            )
            if cursor.rowcount > 0:
                self.conn.commit()
                log.info(f"Cleaned up {cursor.rowcount} stale locks")
        except sqlite3.OperationalError as e:
            # Database locked by another connection - skip cleanup
            # Stale locks will be handled during normal lock acquisition
            log.debug(f"Skipping stale lock cleanup (database busy): {e}")

    def acquire_lock(
        self,
        session_id: str,
        timeout_seconds: float = 30.0,
    ) -> bool:
        """Acquire an exclusive lock on a session.

        Args:
            session_id: Session to lock
            timeout_seconds: Lock timeout (auto-release after this time)

        Returns:
            True if lock acquired, False if already locked by another owner
        """
        now = time.time()
        expires_at = now + timeout_seconds

        # First, clean up any expired lock for this session
        self.conn.execute(
            "DELETE FROM session_locks WHERE session_id = ? AND expires_at < ?",
            (session_id, now)
        )

        # Try to acquire lock
        try:
            self.conn.execute(
                """
                INSERT INTO session_locks (session_id, owner_id, acquired_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, self._owner_id, now, expires_at)
            )
            self.conn.commit()
            log.debug(f"Lock acquired for session {session_id[:8]}... (owner={self._owner_id[:16]})")
            return True
        except sqlite3.IntegrityError:
            # Lock already exists - check if it's ours or expired
            row = self.conn.execute(
                "SELECT owner_id, expires_at FROM session_locks WHERE session_id = ?",
                (session_id,)
            ).fetchone()

            if row is None:
                # Race condition - try again
                return self.acquire_lock(session_id, timeout_seconds)

            if row["owner_id"] == self._owner_id:
                # We already own it - extend the lock
                self.conn.execute(
                    "UPDATE session_locks SET expires_at = ? WHERE session_id = ?",
                    (expires_at, session_id)
                )
                self.conn.commit()
                log.debug(f"Lock extended for session {session_id[:8]}...")
                return True

            if row["expires_at"] < now:
                # Lock expired - take it over
                self.conn.execute(
                    """
                    UPDATE session_locks
                    SET owner_id = ?, acquired_at = ?, expires_at = ?
                    WHERE session_id = ?
                    """,
                    (self._owner_id, now, expires_at, session_id)
                )
                self.conn.commit()
                log.info(f"Took over expired lock for session {session_id[:8]}...")
                return True

            # Lock held by another owner
            log.debug(f"Lock denied for session {session_id[:8]}... (held by {row['owner_id'][:16]})")
            return False

    def release_lock(self, session_id: str) -> bool:
        """Release a lock on a session.

        Args:
            session_id: Session to unlock

        Returns:
            True if lock was released, False if we didn't own it
        """
        cursor = self.conn.execute(
            "DELETE FROM session_locks WHERE session_id = ? AND owner_id = ?",
            (session_id, self._owner_id)
        )
        self.conn.commit()

        if cursor.rowcount > 0:
            log.debug(f"Lock released for session {session_id[:8]}...")
            return True
        else:
            log.debug(f"No lock to release for session {session_id[:8]}... (not owner)")
            return False

    def get_session_state(self, session_id: str) -> SessionState | None:
        """Get the current state of a session.

        Args:
            session_id: Session to look up

        Returns:
            SessionState if found, None otherwise
        """
        row = self.conn.execute(
            """
            SELECT session_id, file_path, indexed_line_index, content_hash,
                   inode, status, indexed_at
            FROM indexed_sessions
            WHERE session_id = ?
            """,
            (session_id,)
        ).fetchone()

        if row is None:
            return None

        return SessionState(
            session_id=row["session_id"],
            file_path=row["file_path"],
            indexed_line_index=row["indexed_line_index"] or 0,
            content_hash=row["content_hash"] or "",
            status=row["status"],
            indexed_at=row["indexed_at"] or 0,
            inode=row["inode"],
        )

    def update_session_state(
        self,
        session_id: str,
        file_path: str,
        line_index: int,
        content_hash: str,
        status: str = "indexed",
        inode: int | None = None,
    ) -> None:
        """Update or create session state.

        Args:
            session_id: Session identifier
            file_path: Path to trace file
            line_index: Lines processed so far (for incremental indexing)
            content_hash: SHA256 hash of content
            status: Processing status
            inode: File inode for rotation detection
        """
        now = time.time()
        self.conn.execute(
            """
            INSERT INTO indexed_sessions
                (session_id, file_path, indexed_line_index, content_hash, inode, status, indexed_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                file_path = excluded.file_path,
                indexed_line_index = excluded.indexed_line_index,
                content_hash = excluded.content_hash,
                inode = excluded.inode,
                status = excluded.status,
                indexed_at = excluded.indexed_at,
                updated_at = excluded.updated_at
            """,
            (session_id, file_path, line_index, content_hash, inode, status, now, now)
        )
        self.conn.commit()
        log.debug(f"Session state updated: {session_id[:8]}... line_index={line_index} status={status}")

    def set_status(self, session_id: str, status: str) -> None:
        """Update only the status of a session.

        Args:
            session_id: Session identifier
            status: New status
        """
        self.conn.execute(
            "UPDATE indexed_sessions SET status = ?, updated_at = ? WHERE session_id = ?",
            (status, time.time(), session_id)
        )
        self.conn.commit()

    def list_sessions(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SessionState]:
        """List indexed sessions.

        Args:
            status: Filter by status (optional)
            limit: Maximum number to return

        Returns:
            List of SessionState objects
        """
        if status:
            rows = self.conn.execute(
                """
                SELECT session_id, file_path, indexed_line_index, content_hash,
                       inode, status, indexed_at
                FROM indexed_sessions
                WHERE status = ?
                ORDER BY indexed_at DESC
                LIMIT ?
                """,
                (status, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT session_id, file_path, indexed_line_index, content_hash,
                       inode, status, indexed_at
                FROM indexed_sessions
                ORDER BY indexed_at DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

        return [
            SessionState(
                session_id=row["session_id"],
                file_path=row["file_path"],
                indexed_line_index=row["indexed_line_index"] or 0,
                content_hash=row["content_hash"] or "",
                status=row["status"],
                indexed_at=row["indexed_at"] or 0,
                inode=row["inode"],
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            log.debug("SessionStateDB connection closed")

    def __enter__(self) -> "SessionStateDB":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close connection."""
        self.close()


def compute_content_hash(file_path: Path, max_bytes: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of file content.

    For large files, only hashes the first max_bytes to avoid
    excessive I/O during change detection.

    Args:
        file_path: Path to file
        max_bytes: Maximum bytes to hash (default 1MB)

    Returns:
        Hex-encoded SHA256 hash
    """
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            data = f.read(max_bytes)
            hasher.update(data)
            # Also include file size in hash for large files
            if len(data) == max_bytes:
                hasher.update(str(file_path.stat().st_size).encode())
        return hasher.hexdigest()
    except (OSError, FileNotFoundError) as e:
        log.warning(f"Failed to compute hash for {file_path}: {e}")
        return ""
