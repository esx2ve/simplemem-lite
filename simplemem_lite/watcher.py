"""File watcher for automatic code index updates.

P3: Auto-capture background worker that monitors file changes
and incrementally updates the code index.
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from simplemem_lite.logging import get_logger

if TYPE_CHECKING:
    from simplemem_lite.code_index import CodeIndexer

log = get_logger("watcher")


@dataclass
class FileEvent:
    """Represents a file change event."""

    path: Path
    event_type: str  # "created", "modified", "deleted", "moved"
    timestamp: float = field(default_factory=time.time)
    dest_path: Path | None = None  # For move events


class DebouncedQueue:
    """Queue with debouncing - waits for quiet period before yielding events.

    Coalesces rapid file changes into single events per file.
    """

    def __init__(self, quiet_period: float = 1.5):
        """Initialize the debounced queue.

        Args:
            quiet_period: Seconds to wait after last event before processing
        """
        self.quiet_period = quiet_period
        self._pending: dict[str, FileEvent] = {}  # path -> latest event
        self._lock = threading.Lock()
        self._last_event_time = 0.0

    def put(self, event: FileEvent) -> None:
        """Add an event to the queue (coalesces with pending events)."""
        with self._lock:
            path_key = str(event.path)

            # For deletes, just record it
            if event.event_type == "deleted":
                self._pending[path_key] = event
            # For moves, track as delete + create
            elif event.event_type == "moved" and event.dest_path:
                # Delete from old path
                self._pending[path_key] = FileEvent(
                    path=event.path,
                    event_type="deleted",
                    timestamp=event.timestamp,
                )
                # Create at new path
                dest_key = str(event.dest_path)
                self._pending[dest_key] = FileEvent(
                    path=event.dest_path,
                    event_type="created",
                    timestamp=event.timestamp,
                )
            else:
                # Create/modify: coalesce to latest
                self._pending[path_key] = event

            self._last_event_time = time.time()

    def get_ready_events(self) -> list[FileEvent]:
        """Get events that are ready (quiet period elapsed).

        Returns:
            List of events ready for processing (empty if still in quiet period)
        """
        with self._lock:
            if not self._pending:
                return []

            # Check if quiet period has elapsed
            elapsed = time.time() - self._last_event_time
            if elapsed < self.quiet_period:
                return []

            # Return all pending events and clear
            events = list(self._pending.values())
            self._pending.clear()
            return events

    def clear(self) -> None:
        """Clear all pending events."""
        with self._lock:
            self._pending.clear()


class CodeWatchHandler(FileSystemEventHandler):
    """Watchdog event handler that filters and queues file events."""

    def __init__(
        self,
        queue: DebouncedQueue,
        patterns: list[str],
        project_root: Path,
    ):
        """Initialize the handler.

        Args:
            queue: DebouncedQueue to add events to
            patterns: Glob patterns for files to watch (e.g., ["*.py", "*.ts"])
            project_root: Project root for relative path matching
        """
        super().__init__()
        self.queue = queue
        self.patterns = patterns
        self.project_root = project_root

    def _matches_patterns(self, path: Path) -> bool:
        """Check if path matches any of our patterns."""
        for pattern in self.patterns:
            if path.match(pattern):
                return True
        return False

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        path_str = str(path)
        # Ignore common non-code directories
        ignore_dirs = {
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            ".tox",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            ".eggs",
            "*.egg-info",
        }
        parts = path.parts
        for part in parts:
            if part in ignore_dirs or part.endswith(".egg-info"):
                return True
        return False

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._should_ignore(path):
            return
        if self._matches_patterns(path):
            log.debug(f"File created: {path}")
            self.queue.put(FileEvent(path=path, event_type="created"))

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._should_ignore(path):
            return
        if self._matches_patterns(path):
            log.debug(f"File modified: {path}")
            self.queue.put(FileEvent(path=path, event_type="modified"))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._should_ignore(path):
            return
        if self._matches_patterns(path):
            log.debug(f"File deleted: {path}")
            self.queue.put(FileEvent(path=path, event_type="deleted"))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path)
        if self._should_ignore(src_path) and self._should_ignore(dest_path):
            return
        src_matches = self._matches_patterns(src_path)
        dest_matches = self._matches_patterns(dest_path)
        if src_matches or dest_matches:
            log.debug(f"File moved: {src_path} -> {dest_path}")
            self.queue.put(FileEvent(
                path=src_path,
                event_type="moved",
                dest_path=dest_path,
            ))


class WatcherWorker(threading.Thread):
    """Background thread that processes file events from the queue."""

    def __init__(
        self,
        queue: DebouncedQueue,
        code_indexer: "CodeIndexer",
        project_root: Path,
        poll_interval: float = 0.5,
    ):
        """Initialize the worker.

        Args:
            queue: DebouncedQueue to consume events from
            code_indexer: CodeIndexer for updating the index
            project_root: Project root path
            poll_interval: How often to check for ready events (seconds)
        """
        super().__init__(daemon=True)
        self.queue = queue
        self.code_indexer = code_indexer
        self.project_root = project_root
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._stats = {
            "files_indexed": 0,
            "files_deleted": 0,
            "errors": 0,
            "last_error": None,
            "last_activity": None,
        }
        self._stats_lock = threading.Lock()

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()

    def get_stats(self) -> dict[str, Any]:
        """Get worker statistics."""
        with self._stats_lock:
            return dict(self._stats)

    def run(self) -> None:
        """Main worker loop."""
        log.info(f"WatcherWorker started for {self.project_root}")

        while not self._stop_event.is_set():
            try:
                # Get ready events (returns empty if still in quiet period)
                events = self.queue.get_ready_events()

                for event in events:
                    self._process_event(event)

                # Sleep before next poll
                self._stop_event.wait(self.poll_interval)

            except Exception as e:
                log.error(f"WatcherWorker error: {e}")
                with self._stats_lock:
                    self._stats["errors"] += 1
                    self._stats["last_error"] = str(e)
                # Brief pause on error to avoid tight loops
                self._stop_event.wait(1.0)

        log.info(f"WatcherWorker stopped for {self.project_root}")

    def _process_event(self, event: FileEvent) -> None:
        """Process a single file event."""
        log.debug(f"Processing event: {event.event_type} {event.path}")

        try:
            if event.event_type == "deleted":
                result = self.code_indexer.delete_file(event.path, self.project_root)
                with self._stats_lock:
                    self._stats["files_deleted"] += 1
                    self._stats["last_activity"] = time.time()
                log.info(f"Deleted from index: {result.get('filepath')} ({result.get('chunks_deleted', 0)} chunks)")

            elif event.event_type in ("created", "modified"):
                result = self.code_indexer.index_file(event.path, self.project_root)
                if result.get("error"):
                    log.warning(f"Index error: {result['error']}")
                    with self._stats_lock:
                        self._stats["errors"] += 1
                        self._stats["last_error"] = result["error"]
                else:
                    with self._stats_lock:
                        self._stats["files_indexed"] += 1
                        self._stats["last_activity"] = time.time()
                    log.info(f"Indexed: {result.get('filepath')} ({result.get('chunks_created', 0)} chunks)")

        except Exception as e:
            log.error(f"Error processing {event.event_type} for {event.path}: {e}")
            with self._stats_lock:
                self._stats["errors"] += 1
                self._stats["last_error"] = str(e)


@dataclass
class ProjectWatcher:
    """Holds state for a single watched project."""

    project_root: Path
    observer: Observer
    worker: WatcherWorker
    queue: DebouncedQueue
    started_at: float = field(default_factory=time.time)


class ProjectWatcherManager:
    """Manages watchdog observers for multiple projects."""

    def __init__(self, code_indexer: "CodeIndexer", patterns: list[str] | None = None):
        """Initialize the watcher manager.

        Args:
            code_indexer: CodeIndexer for updating indexes
            patterns: Glob patterns for files to watch
        """
        self.code_indexer = code_indexer
        self.patterns = patterns or ["*.py", "*.ts", "*.js", "*.tsx", "*.jsx"]
        self._watchers: dict[str, ProjectWatcher] = {}
        self._lock = threading.Lock()
        log.info(f"ProjectWatcherManager initialized with patterns: {self.patterns}")

    def start_watching(self, project_root: str | Path) -> dict[str, Any]:
        """Start watching a project directory.

        Args:
            project_root: Path to project root

        Returns:
            Status dict with success/error info
        """
        root = Path(project_root).resolve()
        root_key = str(root)

        if not root.exists():
            log.error(f"Project root does not exist: {root}")
            return {"error": f"Directory not found: {root}"}

        if not root.is_dir():
            log.error(f"Project root is not a directory: {root}")
            return {"error": f"Not a directory: {root}"}

        with self._lock:
            if root_key in self._watchers:
                log.warning(f"Already watching: {root}")
                return {"status": "already_watching", "project_root": root_key}

            log.info(f"Starting watcher for: {root}")

            # Create queue, handler, worker
            queue = DebouncedQueue(quiet_period=1.5)
            handler = CodeWatchHandler(queue, self.patterns, root)
            worker = WatcherWorker(queue, self.code_indexer, root)

            # Create and start observer
            observer = Observer()
            observer.schedule(handler, str(root), recursive=True)
            observer.start()

            # Start worker thread
            worker.start()

            # Store watcher state
            self._watchers[root_key] = ProjectWatcher(
                project_root=root,
                observer=observer,
                worker=worker,
                queue=queue,
            )

            log.info(f"Watcher started for: {root}")
            return {
                "status": "started",
                "project_root": root_key,
                "patterns": self.patterns,
            }

    def stop_watching(self, project_root: str | Path) -> dict[str, Any]:
        """Stop watching a project directory.

        Args:
            project_root: Path to project root

        Returns:
            Status dict with success/error info
        """
        root = Path(project_root).resolve()
        root_key = str(root)

        with self._lock:
            if root_key not in self._watchers:
                log.warning(f"Not watching: {root}")
                return {"error": f"Not watching: {root}"}

            watcher = self._watchers.pop(root_key)

            log.info(f"Stopping watcher for: {root}")

            # Stop worker
            watcher.worker.stop()
            watcher.worker.join(timeout=5.0)

            # Stop observer
            watcher.observer.stop()
            watcher.observer.join(timeout=5.0)

            # Clear queue
            watcher.queue.clear()

            stats = watcher.worker.get_stats()
            log.info(f"Watcher stopped for: {root}")

            return {
                "status": "stopped",
                "project_root": root_key,
                "stats": stats,
            }

    def stop_all(self) -> dict[str, Any]:
        """Stop all watchers.

        Returns:
            Status dict with counts
        """
        with self._lock:
            roots = list(self._watchers.keys())

        results = []
        for root in roots:
            result = self.stop_watching(root)
            results.append(result)

        return {
            "status": "stopped_all",
            "count": len(results),
            "results": results,
        }

    def get_status(self) -> dict[str, Any]:
        """Get status of all watchers.

        Returns:
            Dict with watched projects and their stats
        """
        with self._lock:
            status = {
                "watching": len(self._watchers),
                "patterns": self.patterns,
                "projects": {},
            }

            for root_key, watcher in self._watchers.items():
                worker_stats = watcher.worker.get_stats()
                status["projects"][root_key] = {
                    "started_at": watcher.started_at,
                    "uptime": time.time() - watcher.started_at,
                    "observer_alive": watcher.observer.is_alive(),
                    "worker_alive": watcher.worker.is_alive(),
                    **worker_stats,
                }

            return status

    def is_watching(self, project_root: str | Path) -> bool:
        """Check if a project is being watched.

        Args:
            project_root: Path to project root

        Returns:
            True if project is being watched
        """
        root = Path(project_root).resolve()
        root_key = str(root)
        with self._lock:
            return root_key in self._watchers
