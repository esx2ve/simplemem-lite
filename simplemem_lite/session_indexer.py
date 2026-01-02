"""Session Auto-Indexer for automatic Claude Code trace processing.

Provides background polling daemon with:
- Stability detection (mtime + size unchanged for N cycles)
- Incremental tailing via offset tracking
- Rate limiting and file size safeguards
- enabled_at boundary for historical session control
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from simplemem_lite.log_config import get_logger

if TYPE_CHECKING:
    from simplemem_lite.config import Config
    from simplemem_lite.job_manager import JobManager
    from simplemem_lite.projects import ProjectManager
    from simplemem_lite.session_state import SessionStateDB
    from simplemem_lite.traces import HierarchicalIndexer

log = get_logger("session_indexer")


@dataclass
class FileStabilityState:
    """Track file stability for change detection.

    A file is considered "stable" when its mtime and size remain
    unchanged for N consecutive poll cycles.

    Attributes:
        file_path: Path to the session file
        last_mtime: Last observed modification time
        last_size: Last observed file size in bytes
        stable_count: Number of consecutive unchanged cycles
        last_checked: Timestamp of last check
    """

    file_path: Path
    last_mtime: float = 0.0
    last_size: int = 0
    stable_count: int = 0
    last_checked: float = field(default_factory=time.time)


class SessionAutoIndexer:
    """Background daemon for automatic session indexing.

    Polls session files at configurable intervals, detects stable files,
    and queues them for incremental indexing. Uses P1's SessionStateDB
    for robust state tracking with locking and hash-based skip detection.

    Features:
    - Stability detection: mtime + size unchanged for N cycles
    - Rate limiting: max sessions per day
    - File size limit: skip oversized sessions
    - enabled_at boundary: only auto-process sessions after timestamp
    - Graceful shutdown with task cancellation
    """

    def __init__(
        self,
        config: "Config",
        session_state_db: "SessionStateDB",
        indexer: "HierarchicalIndexer",
        job_manager: "JobManager",
        project_manager: "ProjectManager",
    ):
        """Initialize the auto-indexer.

        Args:
            config: Application configuration
            session_state_db: SQLite state database (from P1)
            indexer: HierarchicalIndexer for trace processing
            job_manager: Background job manager
            project_manager: Project detection and management
        """
        self._config = config
        self._session_state_db = session_state_db
        self._indexer = indexer
        self._job_manager = job_manager
        self._project_manager = project_manager

        # Stability tracking per file
        self._stability_states: dict[str, FileStabilityState] = {}

        # Track active indexing jobs to prevent duplicate submissions
        self._active_sessions: set[str] = set()

        # Runtime state
        self._running = False
        self._task: asyncio.Task | None = None

        # Rate limiting
        self._sessions_indexed_today: int = 0
        self._last_reset_date: date = date.today()
        self._daily_limit_logged: bool = False  # Prevent log spam

        log.info(
            f"SessionAutoIndexer initialized: "
            f"poll_interval={config.auto_index_poll_interval}s, "
            f"max_file_mb={config.auto_index_max_file_mb}, "
            f"max_per_day={config.auto_index_max_per_day}, "
            f"stability_cycles={config.auto_index_stability_cycles}"
        )

    async def start(self) -> None:
        """Start the background polling loop.

        Creates an asyncio task that runs the poll loop until stopped.
        Safe to call multiple times (idempotent).
        """
        if self._running:
            log.warning("SessionAutoIndexer already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        log.info("SessionAutoIndexer started")

    async def stop(self) -> None:
        """Stop the background polling loop.

        Cancels the polling task and waits for cleanup.
        Safe to call multiple times (idempotent).
        """
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("SessionAutoIndexer stopped")

    def stop_sync(self) -> None:
        """Stop the background polling loop synchronously.

        For use in atexit handlers where async is not available.
        Cancels the task without awaiting (cleanup happens in background).
        """
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            # Don't await - let it clean up on its own
            self._task = None
        log.info("SessionAutoIndexer stopped (sync)")

    @property
    def is_running(self) -> bool:
        """Check if the indexer is currently running."""
        return self._running

    @property
    def sessions_indexed_today(self) -> int:
        """Get the count of sessions indexed today."""
        return self._sessions_indexed_today

    async def _poll_loop(self) -> None:
        """Main polling loop - runs every poll_interval seconds.

        Catches and logs exceptions to prevent loop termination.
        """
        log.info("Poll loop starting")
        while self._running:
            try:
                cycle_start = time.time()
                await self._poll_cycle()
                cycle_duration = time.time() - cycle_start
                log.debug(f"Poll cycle completed in {cycle_duration:.2f}s")
            except asyncio.CancelledError:
                log.info("Poll loop cancelled")
                break
            except Exception as e:
                log.error(f"Poll cycle error: {e}", exc_info=True)

            # Sleep until next poll
            await asyncio.sleep(self._config.auto_index_poll_interval)

        log.info("Poll loop exited")

    async def _poll_cycle(self) -> None:
        """Single poll cycle - scan files, update stability, queue indexing.

        Handles:
        - Daily counter reset
        - Rate limit checking
        - Session scanning and stability updates
        """
        # Reset daily counter if new day
        today = date.today()
        if today != self._last_reset_date:
            log.info(
                f"New day detected, resetting daily counter "
                f"(was {self._sessions_indexed_today})"
            )
            self._sessions_indexed_today = 0
            self._last_reset_date = today
            self._daily_limit_logged = False  # Reset log flag for new day

        # Check rate limit (only log once when limit is first reached)
        if self._sessions_indexed_today >= self._config.auto_index_max_per_day:
            if not self._daily_limit_logged:
                log.info(
                    f"Daily limit reached ({self._config.auto_index_max_per_day}), "
                    f"will skip poll cycles until tomorrow"
                )
                self._daily_limit_logged = True
            return

        # Scan session files
        sessions = self._indexer.parser.list_sessions()
        log.debug(f"Found {len(sessions)} sessions to check")

        # Build set of current session paths for stale cleanup
        current_paths = {session["path"] for session in sessions}

        stable_count = 0
        skipped_count = 0
        queued_count = 0

        for session in sessions:
            result = await self._check_session(session)
            if result == "stable":
                stable_count += 1
            elif result == "skipped":
                skipped_count += 1
            elif result == "queued":
                queued_count += 1

            # Stop if we hit the daily limit
            if self._sessions_indexed_today >= self._config.auto_index_max_per_day:
                log.info("Daily limit reached during cycle, stopping early")
                break

        # Cleanup stale stability states (files that no longer exist)
        stale_keys = [
            key for key in self._stability_states
            if key not in current_paths
        ]
        if stale_keys:
            for key in stale_keys:
                del self._stability_states[key]
            log.debug(f"Cleaned up {len(stale_keys)} stale stability states")

        log.info(
            f"Poll cycle summary: "
            f"checked={len(sessions)}, stable={stable_count}, "
            f"skipped={skipped_count}, queued={queued_count}, "
            f"indexed_today={self._sessions_indexed_today}"
        )

    async def _check_session(self, session: dict) -> str:
        """Check a single session for stability and queue if ready.

        Args:
            session: Session info dict from TraceParser.list_sessions()

        Returns:
            Status string: "skipped", "unstable", "stable", or "queued"
        """
        session_id = session.get("session_id", "unknown")
        file_path = Path(session["path"])
        size_kb = session.get("size_kb", 0)

        # Skip if already being indexed (prevents duplicate job submissions)
        if session_id in self._active_sessions:
            log.debug(f"Skipping session {session_id[:8]}...: already being indexed")
            return "skipped"

        # Skip if too large
        max_size_kb = self._config.auto_index_max_file_mb * 1024
        if size_kb > max_size_kb:
            log.debug(
                f"Skipping session {session_id[:8]}...: "
                f"size {size_kb}KB > limit {max_size_kb}KB"
            )
            return "skipped"

        # Check file stats
        try:
            stat = file_path.stat()
            current_mtime = stat.st_mtime
            current_size = stat.st_size
        except OSError as e:
            log.warning(f"Cannot stat {file_path}: {e}")
            return "skipped"

        # Skip if before enabled_at timestamp
        if self._config.auto_index_enabled_at > 0:
            if current_mtime < self._config.auto_index_enabled_at:
                log.debug(
                    f"Skipping session {session_id[:8]}...: "
                    f"mtime {current_mtime} < enabled_at {self._config.auto_index_enabled_at}"
                )
                return "skipped"

        # Update stability state
        state = self._update_stability(
            file_path, session_id, current_mtime, current_size
        )

        # Check if stable enough
        required_cycles = self._config.auto_index_stability_cycles
        if state.stable_count < required_cycles:
            log.debug(
                f"Session {session_id[:8]}... not stable yet: "
                f"{state.stable_count}/{required_cycles} cycles"
            )
            return "unstable"

        # File is stable - queue for indexing
        log.info(
            f"Session {session_id[:8]}... is stable "
            f"({state.stable_count} cycles), queueing for indexing"
        )
        await self._queue_for_indexing(session)
        return "queued"

    def _update_stability(
        self,
        file_path: Path,
        session_id: str,
        current_mtime: float,
        current_size: int,
    ) -> FileStabilityState:
        """Update stability state for a file.

        Tracks mtime and size across poll cycles. If unchanged,
        increments stable_count. If changed, resets to 0.

        Args:
            file_path: Path to the session file
            session_id: Session identifier for logging
            current_mtime: Current modification time
            current_size: Current file size

        Returns:
            Updated FileStabilityState
        """
        key = str(file_path)
        now = time.time()

        if key not in self._stability_states:
            # First time seeing this file
            state = FileStabilityState(
                file_path=file_path,
                last_mtime=current_mtime,
                last_size=current_size,
                stable_count=0,
                last_checked=now,
            )
            self._stability_states[key] = state
            log.debug(f"New session detected: {session_id[:8]}...")
            return state

        state = self._stability_states[key]

        # Check if file changed
        if current_mtime != state.last_mtime or current_size != state.last_size:
            # File changed - reset stability
            log.debug(
                f"Session {session_id[:8]}... changed: "
                f"mtime {state.last_mtime}->{current_mtime}, "
                f"size {state.last_size}->{current_size}"
            )
            state.last_mtime = current_mtime
            state.last_size = current_size
            state.stable_count = 0
        else:
            # File unchanged - increment stability
            state.stable_count += 1
            log.debug(
                f"Session {session_id[:8]}... unchanged: "
                f"stable_count={state.stable_count}"
            )

        state.last_checked = now
        return state

    async def _queue_for_indexing(self, session: dict) -> None:
        """Queue a session for background indexing.

        Uses JobManager to run the indexing asynchronously.
        Increments the daily counter on successful queue.

        Args:
            session: Session info dict
        """
        session_id = session.get("session_id", "unknown")
        project = session.get("project", "")

        try:
            # Mark as active before queueing (prevents race condition)
            self._active_sessions.add(session_id)

            # Detect project root from session
            project_root = self._project_manager.detect_project_root(project)
            if not project_root:
                project_root = project

            log.info(
                f"Queueing session {session_id[:8]}... for indexing "
                f"(project: {project_root})"
            )

            # Submit job to JobManager
            # Note: submit() takes a factory function + args, not an instantiated coroutine
            job_id = await self._job_manager.submit(
                "auto_index_session",
                self._index_session,
                session_id,
                project_root,
                session["path"],
            )

            log.info(f"Session {session_id[:8]}... queued as job {job_id[:8]}...")
            self._sessions_indexed_today += 1

            # Reset stability so we don't re-queue immediately
            key = str(Path(session["path"]))
            if key in self._stability_states:
                self._stability_states[key].stable_count = 0

        except Exception as e:
            # Remove from active set on failure
            self._active_sessions.discard(session_id)
            log.error(
                f"Failed to queue session {session_id[:8]}...: {e}",
                exc_info=True,
            )

    async def _index_session(
        self, session_id: str, project_root: str, transcript_path: str
    ) -> dict:
        """Index a session using the HierarchicalIndexer.

        This is the async coroutine submitted to JobManager.

        Args:
            session_id: Session UUID
            project_root: Detected project root path
            transcript_path: Path to session transcript

        Returns:
            Result dict from index_session_delta
        """
        log.info(
            f"Auto-indexing session {session_id[:8]}... "
            f"(project: {project_root})"
        )

        try:
            result = await self._indexer.index_session_delta(
                session_id=session_id,
                project_root=project_root,
                project_manager=self._project_manager,
                transcript_path=transcript_path,
            )

            processed = result.get("processed", 0)
            skipped = result.get("skipped", False)

            if skipped:
                log.info(
                    f"Session {session_id[:8]}... skipped "
                    f"(reason: {result.get('reason', 'unknown')})"
                )
            else:
                log.info(
                    f"Session {session_id[:8]}... indexed: "
                    f"{processed} messages processed"
                )

            return result

        except Exception as e:
            log.error(
                f"Error indexing session {session_id[:8]}...: {e}",
                exc_info=True,
            )
            return {"error": str(e), "session_id": session_id}

        finally:
            # Always remove from active set when done (success or failure)
            self._active_sessions.discard(session_id)
            log.debug(f"Session {session_id[:8]}... removed from active set")

    def get_status(self) -> dict:
        """Get current status of the auto-indexer.

        Returns:
            Status dict with running state, counters, and config
        """
        return {
            "running": self._running,
            "sessions_indexed_today": self._sessions_indexed_today,
            "daily_limit": self._config.auto_index_max_per_day,
            "tracked_sessions": len(self._stability_states),
            "active_sessions": len(self._active_sessions),
            "poll_interval": self._config.auto_index_poll_interval,
            "stability_cycles": self._config.auto_index_stability_cycles,
            "max_file_mb": self._config.auto_index_max_file_mb,
            "enabled_at": self._config.auto_index_enabled_at,
        }
