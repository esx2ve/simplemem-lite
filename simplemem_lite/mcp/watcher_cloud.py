"""Cloud-integrated file watcher for automatic code index updates.

Unlike the local watcher (watcher.py) which uses CodeIndexer directly,
this version sends updates through the BackendClient to the cloud API.
"""

import asyncio
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.observers import Observer

from simplemem_lite.log_config import get_logger
from simplemem_lite.watcher import CodeWatchHandler, DebouncedQueue, FileEvent

if TYPE_CHECKING:
    from simplemem_lite.mcp.client import BackendClient
    from simplemem_lite.mcp.local_reader import LocalReader

log = get_logger("watcher_cloud")

# Default file patterns to watch
DEFAULT_PATTERNS = ["*.py", "*.ts", "*.js", "*.tsx", "*.jsx"]


class CloudWatcherWorker(threading.Thread):
    """Background thread that processes file events and sends to cloud backend."""

    def __init__(
        self,
        queue: DebouncedQueue,
        client: "BackendClient",
        reader: "LocalReader",
        project_root: Path,
        poll_interval: float = 0.5,
    ):
        """Initialize the worker.

        Args:
            queue: DebouncedQueue with file events
            client: BackendClient for sending updates to cloud
            reader: LocalReader for reading file contents
            project_root: Project root directory
            poll_interval: How often to check for events (seconds)
        """
        super().__init__(daemon=True, name=f"cloud-watcher-{project_root.name}")
        self.queue = queue
        self.client = client
        self.reader = reader
        self.project_root = project_root
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def run(self) -> None:
        """Main worker loop - process events from queue."""
        log.info(f"CloudWatcherWorker started for {self.project_root}")

        # Create event loop for async client calls
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            while not self._stop_event.is_set():
                # Check for ready events (debounce period elapsed)
                events = self.queue.get_ready_events()

                if events:
                    self._process_events(events)

                # Sleep before next check
                time.sleep(self.poll_interval)
        finally:
            self._loop.close()
            log.info(f"CloudWatcherWorker stopped for {self.project_root}")

    def stop(self) -> None:
        """Signal worker to stop."""
        self._stop_event.set()

    def _process_events(self, events: list[FileEvent]) -> None:
        """Process a batch of file events."""
        log.info(f"Processing {len(events)} file events")

        updates = []
        for event in events:
            try:
                relative_path = str(event.path.relative_to(self.project_root))
            except ValueError:
                relative_path = str(event.path)

            if event.event_type == "deleted":
                updates.append({
                    "path": relative_path,
                    "action": "delete",
                    "content": None,
                })
            elif event.event_type in ("created", "modified"):
                # Read file content
                try:
                    content = self.reader.read_file(event.path)
                    if content:
                        action = "add" if event.event_type == "created" else "modify"
                        updates.append({
                            "path": relative_path,
                            "action": action,
                            "content": content,
                        })
                except Exception as e:
                    log.error(f"Failed to read {event.path}: {e}")

        if updates:
            self._send_updates(updates)

    def _send_updates(self, updates: list[dict]) -> None:
        """Send updates to cloud backend."""
        if not self._loop:
            log.error("Event loop not available")
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.client.update_code(
                    project_root=str(self.project_root),
                    updates=updates,
                ),
                self._loop,
            )
            # Wait for result with timeout
            result = future.result(timeout=30)
            log.info(
                f"Cloud update: {result.get('files_updated', 0)} files, "
                f"+{result.get('chunks_created', 0)}/-{result.get('chunks_deleted', 0)} chunks"
            )
        except Exception as e:
            log.error(f"Failed to send updates to cloud: {e}")


class CloudWatcherManager:
    """Manages file watchers for multiple projects, sending updates to cloud."""

    def __init__(
        self,
        client: "BackendClient",
        reader: "LocalReader",
    ):
        """Initialize the manager.

        Args:
            client: BackendClient for cloud API calls
            reader: LocalReader for reading file contents
        """
        self.client = client
        self.reader = reader
        self._watchers: dict[str, dict] = {}  # project_root -> {observer, worker, queue}
        self._lock = threading.Lock()

    def start_watching(
        self,
        project_root: str | Path,
        patterns: list[str] | None = None,
    ) -> dict:
        """Start watching a project directory.

        Args:
            project_root: Directory to watch
            patterns: Glob patterns for files to watch (default: Python/JS/TS)

        Returns:
            Status dict with watching state
        """
        project_root = Path(project_root).resolve()
        project_key = str(project_root)
        patterns = patterns or DEFAULT_PATTERNS

        with self._lock:
            if project_key in self._watchers:
                return {
                    "status": "already_watching",
                    "project_root": project_key,
                    "patterns": patterns,
                }

            log.info(f"Starting watcher for {project_root} with patterns {patterns}")

            # Create debounced queue
            queue = DebouncedQueue(quiet_period=1.5)

            # Create handler
            handler = CodeWatchHandler(
                queue=queue,
                patterns=patterns,
                project_root=project_root,
            )

            # Create observer
            observer = Observer()
            observer.schedule(handler, str(project_root), recursive=True)
            observer.start()

            # Create worker
            worker = CloudWatcherWorker(
                queue=queue,
                client=self.client,
                reader=self.reader,
                project_root=project_root,
            )
            worker.start()

            self._watchers[project_key] = {
                "observer": observer,
                "worker": worker,
                "queue": queue,
                "patterns": patterns,
                "started_at": time.time(),
            }

            return {
                "status": "started",
                "project_root": project_key,
                "patterns": patterns,
            }

    def stop_watching(self, project_root: str | Path) -> dict:
        """Stop watching a project directory.

        Args:
            project_root: Directory to stop watching

        Returns:
            Status dict
        """
        project_root = Path(project_root).resolve()
        project_key = str(project_root)

        # Pop watcher from dict while holding lock (fast operation)
        with self._lock:
            if project_key not in self._watchers:
                return {
                    "status": "not_watching",
                    "project_root": project_key,
                }

            log.info(f"Stopping watcher for {project_root}")
            watcher = self._watchers.pop(project_key)

        # Stop observer and worker outside lock to avoid deadlock
        # (join() can block for up to 5s each)
        watcher["observer"].stop()
        watcher["observer"].join(timeout=5)

        watcher["worker"].stop()
        watcher["worker"].join(timeout=5)

        return {
            "status": "stopped",
            "project_root": project_key,
        }

    def get_status(self, project_root: str | Path | None = None) -> dict:
        """Get watcher status.

        Args:
            project_root: Optional specific project to check

        Returns:
            Status dict with watching state
        """
        with self._lock:
            if project_root:
                project_root = Path(project_root).resolve()
                project_key = str(project_root)

                if project_key in self._watchers:
                    watcher = self._watchers[project_key]
                    return {
                        "is_watching": True,
                        "project_root": project_key,
                        "patterns": watcher["patterns"],
                        "started_at": watcher["started_at"],
                    }
                else:
                    return {
                        "is_watching": False,
                        "project_root": project_key,
                    }
            else:
                # Return all watchers
                return {
                    "watching_count": len(self._watchers),
                    "projects": [
                        {
                            "project_root": key,
                            "patterns": w["patterns"],
                            "started_at": w["started_at"],
                        }
                        for key, w in self._watchers.items()
                    ],
                }

    def stop_all(self) -> None:
        """Stop all watchers (for graceful shutdown)."""
        with self._lock:
            for project_key in list(self._watchers.keys()):
                self.stop_watching(project_key)
