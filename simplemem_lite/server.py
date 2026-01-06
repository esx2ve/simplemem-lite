"""MCP Server for SimpleMem Lite.

Exposes memory operations via MCP protocol with tools, resources, and prompts.
Also runs an HTTP server for hook communication.
"""

import asyncio
import atexit
import json
import os
import secrets
import signal
import threading
import time
from collections import defaultdict
from datetime import datetime
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable

from mcp.server.fastmcp import Context, FastMCP

from simplemem_lite.bootstrap import Bootstrap
from simplemem_lite.code_index import CodeIndexer
from simplemem_lite.config import Config
from simplemem_lite.extractors import extract_with_actions
from simplemem_lite.job_manager import JobManager, init_job_manager
from simplemem_lite.log_config import get_logger
from simplemem_lite.memory import Memory, MemoryItem, MemoryStore
from simplemem_lite.projects import ProjectManager
from simplemem_lite.session_indexer import SessionAutoIndexer
from simplemem_lite.session_state import SessionStateDB
from simplemem_lite.traces import HierarchicalIndexer, TraceParser
from simplemem_lite.watcher import ProjectWatcherManager

log = get_logger("server")


# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY INJECTION CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════


class Dependencies:
    """Dependency injection container for the MCP server.

    Encapsulates all server dependencies for easier testing and configuration.
    Dependencies are initialized on first access (lazy initialization).

    Usage in tests:
        deps = Dependencies()
        deps.configure_for_testing(
            config=mock_config,
            store=mock_store,
        )
    """

    def __init__(self) -> None:
        """Initialize empty container. Dependencies are created on first access."""
        self._config: Config | None = None
        self._store: MemoryStore | None = None
        self._parser: TraceParser | None = None
        self._session_state_db: SessionStateDB | None = None
        self._indexer: HierarchicalIndexer | None = None
        self._code_indexer: CodeIndexer | None = None
        self._watcher_manager: ProjectWatcherManager | None = None
        self._project_manager: ProjectManager | None = None
        self._bootstrap: Bootstrap | None = None
        self._job_manager: JobManager | None = None
        self._auto_indexer: SessionAutoIndexer | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize all dependencies on first access."""
        if self._initialized:
            return

        log.info("SimpleMem Lite MCP server starting...")
        log.debug(f"HOME={os.environ.get('HOME', 'NOT SET')}")

        if self._config is None:
            self._config = Config()
            log.debug("Config initialized")

        if self._store is None:
            self._store = MemoryStore(self._config)
            log.debug("MemoryStore initialized")

        if self._parser is None:
            self._parser = TraceParser(self._config.claude_traces_dir)
            log.debug("TraceParser initialized")

        if self._session_state_db is None:
            session_db_path = self._config.data_dir / "session_state.db"
            self._session_state_db = SessionStateDB(session_db_path)
            log.debug("SessionStateDB initialized")

        if self._indexer is None:
            self._indexer = HierarchicalIndexer(
                self._store, self._config, self._session_state_db
            )
            log.debug("HierarchicalIndexer initialized")

        if self._code_indexer is None:
            self._code_indexer = CodeIndexer(self._store.db, self._config)
            log.debug("CodeIndexer initialized")

        if self._watcher_manager is None:
            self._watcher_manager = ProjectWatcherManager(
                self._code_indexer, self._config.code_patterns_list
            )
            log.debug("ProjectWatcherManager initialized")

        if self._project_manager is None:
            self._project_manager = ProjectManager(self._config)
            log.debug("ProjectManager initialized")

        if self._bootstrap is None:
            self._bootstrap = Bootstrap(
                self._config,
                self._project_manager,
                self._code_indexer,
                self._watcher_manager,
            )
            log.debug("Bootstrap initialized")

        if self._job_manager is None:
            self._job_manager = init_job_manager(self._config.data_dir)
            log.debug("JobManager initialized")

        # Initialize auto-indexer if enabled (P2: Active Session Handling)
        if self._auto_indexer is None and self._config.auto_index_enabled:
            self._auto_indexer = SessionAutoIndexer(
                config=self._config,
                session_state_db=self._session_state_db,
                indexer=self._indexer,
                job_manager=self._job_manager,
                project_manager=self._project_manager,
            )
            log.debug("SessionAutoIndexer initialized")

        self._initialized = True

    def configure_for_testing(
        self,
        config: Config | None = None,
        store: MemoryStore | None = None,
        parser: TraceParser | None = None,
        indexer: HierarchicalIndexer | None = None,
        code_indexer: CodeIndexer | None = None,
        watcher_manager: ProjectWatcherManager | None = None,
        project_manager: ProjectManager | None = None,
        bootstrap: Bootstrap | None = None,
        job_manager: JobManager | None = None,
        auto_indexer: SessionAutoIndexer | None = None,
    ) -> None:
        """Configure dependencies for testing.

        Allows injecting mock dependencies for isolated testing.
        Must be called BEFORE accessing any dependency properties.

        Args:
            config: Optional mock Config
            store: Optional mock MemoryStore
            parser: Optional mock TraceParser
            indexer: Optional mock HierarchicalIndexer
            code_indexer: Optional mock CodeIndexer
            watcher_manager: Optional mock ProjectWatcherManager
            project_manager: Optional mock ProjectManager
            bootstrap: Optional mock Bootstrap
            job_manager: Optional mock JobManager
            auto_indexer: Optional mock SessionAutoIndexer
        """
        if self._initialized:
            log.warning("configure_for_testing called after initialization - resetting")
            self._initialized = False

        if config is not None:
            self._config = config
        if store is not None:
            self._store = store
        if parser is not None:
            self._parser = parser
        if indexer is not None:
            self._indexer = indexer
        if code_indexer is not None:
            self._code_indexer = code_indexer
        if watcher_manager is not None:
            self._watcher_manager = watcher_manager
        if project_manager is not None:
            self._project_manager = project_manager
        if bootstrap is not None:
            self._bootstrap = bootstrap
        if job_manager is not None:
            self._job_manager = job_manager
        if auto_indexer is not None:
            self._auto_indexer = auto_indexer

    @property
    def config(self) -> Config:
        """Get Config instance."""
        self._ensure_initialized()
        assert self._config is not None
        return self._config

    @property
    def store(self) -> MemoryStore:
        """Get MemoryStore instance."""
        self._ensure_initialized()
        assert self._store is not None
        return self._store

    @property
    def parser(self) -> TraceParser:
        """Get TraceParser instance."""
        self._ensure_initialized()
        assert self._parser is not None
        return self._parser

    @property
    def session_state_db(self) -> SessionStateDB:
        """Get SessionStateDB instance."""
        self._ensure_initialized()
        assert self._session_state_db is not None
        return self._session_state_db

    @property
    def indexer(self) -> HierarchicalIndexer:
        """Get HierarchicalIndexer instance."""
        self._ensure_initialized()
        assert self._indexer is not None
        return self._indexer

    @property
    def code_indexer(self) -> CodeIndexer:
        """Get CodeIndexer instance."""
        self._ensure_initialized()
        assert self._code_indexer is not None
        return self._code_indexer

    @property
    def watcher_manager(self) -> ProjectWatcherManager:
        """Get ProjectWatcherManager instance."""
        self._ensure_initialized()
        assert self._watcher_manager is not None
        return self._watcher_manager

    @property
    def project_manager(self) -> ProjectManager:
        """Get ProjectManager instance."""
        self._ensure_initialized()
        assert self._project_manager is not None
        return self._project_manager

    @property
    def bootstrap(self) -> Bootstrap:
        """Get Bootstrap instance."""
        self._ensure_initialized()
        assert self._bootstrap is not None
        return self._bootstrap

    @property
    def job_manager(self) -> JobManager:
        """Get JobManager instance."""
        self._ensure_initialized()
        assert self._job_manager is not None
        return self._job_manager

    @property
    def auto_indexer(self) -> SessionAutoIndexer | None:
        """Get SessionAutoIndexer instance (if enabled).

        Returns None if auto_index_enabled is False in config.
        """
        self._ensure_initialized()
        return self._auto_indexer

    async def start_auto_indexer(self) -> bool:
        """Start the auto-indexer if configured and not already running.

        Returns:
            True if started, False if disabled or already running
        """
        self._ensure_initialized()
        if self._auto_indexer is None:
            return False
        if self._auto_indexer.is_running:
            return False
        await self._auto_indexer.start()
        return True

    async def stop_auto_indexer(self) -> bool:
        """Stop the auto-indexer if running.

        Returns:
            True if stopped, False if not running
        """
        if self._auto_indexer is None:
            return False
        if not self._auto_indexer.is_running:
            return False
        await self._auto_indexer.stop()
        return True


# Global dependency container
_deps = Dependencies()


def get_dependencies() -> Dependencies:
    """Get the global dependency container.

    For testing, use `_deps.configure_for_testing()` before accessing dependencies.

    Returns:
        The global Dependencies instance
    """
    return _deps


# Convenience accessors for the global dependencies
# Use these in tool functions: store, config, etc.
# For testing, configure via: _deps.configure_for_testing(...)


def _get_config() -> Config:
    """Get config from dependency container."""
    return _deps.config


def _get_store() -> MemoryStore:
    """Get store from dependency container."""
    return _deps.store


def _get_parser() -> TraceParser:
    """Get parser from dependency container."""
    return _deps.parser


def _get_indexer() -> HierarchicalIndexer:
    """Get indexer from dependency container."""
    return _deps.indexer


def _get_code_indexer() -> CodeIndexer:
    """Get code_indexer from dependency container."""
    return _deps.code_indexer


def _get_watcher_manager() -> ProjectWatcherManager:
    """Get watcher_manager from dependency container."""
    return _deps.watcher_manager


def _get_project_manager() -> ProjectManager:
    """Get project_manager from dependency container."""
    return _deps.project_manager


def _get_bootstrap() -> Bootstrap:
    """Get bootstrap from dependency container."""
    return _deps.bootstrap


def _get_job_manager() -> JobManager:
    """Get job_manager from dependency container."""
    return _deps.job_manager


def _cleanup_watchers() -> None:
    """Cleanup all watchers and background services on server shutdown."""
    try:
        # Stop auto-indexer if running
        if _deps._auto_indexer is not None:
            log.info("Stopping auto-indexer...")
            _deps._auto_indexer.stop_sync()
            log.info("Auto-indexer stopped")

        # Stop file watchers
        log.info("Cleaning up file watchers...")
        _deps.watcher_manager.stop_all()
        log.info("File watchers cleaned up")
    except (ValueError, OSError):
        # Suppress logging errors if stdout is closed (e.g., during pytest cleanup)
        pass


atexit.register(_cleanup_watchers)


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP SERVER FOR HOOK COMMUNICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Server state
_http_server: HTTPServer | None = None
_http_thread: threading.Thread | None = None
_auth_token: str | None = None


def _get_lock_file_path() -> Path:
    """Get path to the lock file."""
    return _deps.config.data_dir / "server.lock"


def _write_lock_file(port: int, token: str) -> None:
    """Write lock file with server info.

    Uses restricted permissions (0o600) to prevent token leakage on multi-user systems.
    """
    lock_data = {
        "port": port,
        "pid": os.getpid(),
        "token": token,
        "started_at": datetime.now().isoformat(),
        "host": _deps.config.http_host,
    }
    lock_path = _get_lock_file_path()
    # Security: Use restricted permissions (read/write by owner only)
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(lock_data, f, indent=2)
    except Exception:
        os.close(fd)
        raise
    log.info(f"Lock file written with secure permissions: {lock_path}")


def _remove_lock_file() -> None:
    """Remove lock file on shutdown."""
    lock_path = _get_lock_file_path()
    if lock_path.exists():
        lock_path.unlink()
        log.info(f"Lock file removed: {lock_path}")


def _read_lock_file() -> dict[str, Any] | None:
    """Read lock file if it exists."""
    lock_path = _get_lock_file_path()
    if lock_path.exists():
        try:
            return json.loads(lock_path.read_text())
        except Exception as e:
            log.warning(f"Failed to read lock file: {e}")
    return None


class RateLimiter:
    """Simple token bucket rate limiter for HTTP requests.

    Thread-safe implementation that tracks requests per client IP.
    """

    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.max_tokens = requests_per_minute
        self._tokens: dict[str, float] = defaultdict(lambda: self.max_tokens)
        self._last_update: dict[str, float] = defaultdict(time.time)
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed and consume a token.

        Returns:
            True if request is allowed, False if rate limited
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update[client_ip]
            self._last_update[client_ip] = now

            # Replenish tokens
            self._tokens[client_ip] = min(
                self.max_tokens,
                self._tokens[client_ip] + elapsed * self.rate
            )

            # Check and consume
            if self._tokens[client_ip] >= 1:
                self._tokens[client_ip] -= 1
                return True
            return False


# Global rate limiter instance (initialized with config)
_rate_limiter: RateLimiter | None = None


def _get_rate_limiter() -> RateLimiter:
    """Get or create the rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(_deps.config.http_rate_limit)
    return _rate_limiter


class HookHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for hook endpoints."""

    # Server reference set by HTTPServer
    server: "HookHTTPServer"

    def log_message(self, format: str, *args) -> None:
        """Override to use our logger."""
        log.debug(f"HTTP: {format % args}")

    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits.

        Returns:
            True if allowed, False if rate limited (response already sent)
        """
        client_ip = self.client_address[0]
        if not _get_rate_limiter().is_allowed(client_ip):
            log.warning(f"Rate limited request from {client_ip}")
            self._send_json_response(429, {"error": "Too many requests"})
            return False
        return True

    def _send_json_response(self, status: int, data: dict) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _verify_auth(self) -> bool:
        """Verify auth token from request.

        Uses constant-time comparison to prevent timing attacks.
        """
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # Security: Use constant-time comparison to prevent timing attacks
            if secrets.compare_digest(token, self.server.auth_token):
                return True
        log.warning("HTTP request with invalid/missing auth token")
        return False

    def _read_json_body(self) -> dict | None:
        """Read and parse JSON body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        try:
            body = self.rfile.read(content_length)
            return json.loads(body.decode())
        except Exception as e:
            log.error(f"Failed to parse JSON body: {e}")
            return None

    def do_GET(self) -> None:
        """Handle GET requests."""
        if not self._check_rate_limit():
            return

        if self.path == "/health":
            # Include database health check
            db_health = _deps.store.db.health_check()
            all_healthy = db_health["falkordb"]["healthy"] and db_health["lancedb"]["healthy"]
            self._send_json_response(200 if all_healthy else 503, {
                "status": "ok" if all_healthy else "degraded",
                "pid": os.getpid(),
                "uptime_seconds": (datetime.now() - self.server.started_at).total_seconds(),
                "databases": {
                    "falkordb": db_health["falkordb"],
                    "lancedb": db_health["lancedb"],
                },
            })
        else:
            self._send_json_response(404, {"error": "Not found"})

    def do_POST(self) -> None:
        """Handle POST requests."""
        if not self._check_rate_limit():
            return

        # All POST endpoints require auth
        if not self._verify_auth():
            self._send_json_response(401, {"error": "Unauthorized"})
            return

        body = self._read_json_body()
        if body is None:
            self._send_json_response(400, {"error": "Invalid JSON"})
            return

        if self.path == "/hook/session-start":
            self._handle_session_start(body)
        elif self.path == "/hook/stop":
            self._handle_stop(body)
        elif self.path == "/stats":
            self._handle_stats()
        else:
            self._send_json_response(404, {"error": "Not found"})

    def _handle_session_start(self, body: dict) -> None:
        """Handle session-start hook.

        Body expected: {cwd, session_id}
        Returns: {should_bootstrap, project_root, context}
        """
        cwd = body.get("cwd", "")
        session_id = body.get("session_id", "")
        log.info(f"Hook: session-start cwd={cwd}, session={session_id[:8] if session_id else 'none'}...")

        # Detect project root (use git root if available)
        project_root = _deps.project_manager.detect_project_root(cwd)

        # Get bootstrap status
        status = _deps.bootstrap.get_bootstrap_status(project_root)

        # Generate context injection
        context = _deps.bootstrap.generate_context_injection(project_root)

        self._send_json_response(200, {
            "status": "ok",
            "project_root": project_root,
            "session_id": session_id,
            "is_bootstrapped": status.get("is_bootstrapped", False),
            "should_ask": status.get("should_ask", False),
            "context": context,
        })

    def _handle_stop(self, body: dict) -> None:
        """Handle stop hook.

        Body expected: {cwd, session_id, transcript_path}
        Returns: {status, processed}
        """
        import asyncio

        cwd = body.get("cwd", "")
        session_id = body.get("session_id", "")
        transcript_path = body.get("transcript_path", "")
        log.info(f"Hook: stop cwd={cwd}, session={session_id[:8] if session_id else 'none'}...")

        if not session_id:
            self._send_json_response(200, {
                "status": "ok",
                "message": "No session_id provided, skipping trace processing",
            })
            return

        # Detect project root
        project_root = _deps.project_manager.detect_project_root(cwd)

        # Process trace delta asynchronously
        try:
            # Run the async delta indexer in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    _deps.indexer.index_session_delta(
                        session_id=session_id,
                        project_root=project_root,
                        project_manager=_deps.project_manager,
                        transcript_path=transcript_path if transcript_path else None,
                    )
                )
            finally:
                loop.close()

            log.info(f"Delta processing result: {result}")
            self._send_json_response(200, {
                "status": "ok",
                "session_id": session_id,
                "project_root": project_root,
                **result,
            })
        except Exception as e:
            log.error(f"Delta processing failed: {e}")
            self._send_json_response(200, {
                "status": "error",
                "session_id": session_id,
                "error": str(e),
            })

    def _handle_stats(self) -> None:
        """Handle stats request for statusline.

        Returns comprehensive stats for SimpleMem statusline display.
        """
        try:
            # Memory stats
            mem_stats = _deps.store.get_stats()

            # Code index stats
            code_stats = _deps.code_indexer.get_stats() if _deps.code_indexer else {}

            # Watcher stats
            watcher_status = _deps.watcher_manager.get_status() if _deps.watcher_manager else {}

            # Job stats
            job_stats = _deps.job_manager.get_active_stats() if _deps.job_manager else {}

            # Code index status (from job_manager for statusline)
            code_index_status = (
                _deps.job_manager.get_code_index_status()
                if _deps.job_manager
                else {}
            )

            # Todo count (query for pending todos)
            todo_count = 0
            try:
                todos = _deps.store.db.graph.query(
                    "MATCH (m:Memory {type: 'todo'}) "
                    "WHERE m.content CONTAINS 'pending' "
                    "RETURN count(m) AS count"
                )
                if todos.result_set:
                    todo_count = todos.result_set[0][0]
            except Exception:
                pass

            self._send_json_response(200, {
                "memories": mem_stats.get("total_memories", 0),
                "entities": mem_stats.get("entities", 0),
                "relations": mem_stats.get("total_relations", 0),
                "code_files": code_stats.get("unique_files", 0),
                "code_chunks": code_stats.get("chunk_count", 0),
                "watchers": watcher_status.get("watching_count", 0),
                "jobs_running": job_stats.get("active", 0),
                "job_current": job_stats.get("current"),
                "todos_pending": todo_count,
                "code_index": code_index_status,
            })
        except Exception as e:
            log.error(f"Stats endpoint failed: {e}")
            self._send_json_response(500, {"error": str(e)})


class HookHTTPServer(HTTPServer):
    """HTTP server with auth token storage."""

    def __init__(self, server_address: tuple, handler_class: type, auth_token: str):
        super().__init__(server_address, handler_class)
        self.auth_token = auth_token
        self.started_at = datetime.now()


def _start_http_server() -> tuple[HTTPServer, threading.Thread] | None:
    """Start HTTP server in a background thread.

    Returns:
        Tuple of (server, thread) if successful, None if disabled
    """
    global _auth_token

    if not _deps.config.http_enabled:
        log.info("HTTP server disabled by config")
        return None

    # Generate auth token
    _auth_token = secrets.token_urlsafe(32)

    # Create server (port 0 = let OS assign)
    try:
        server = HookHTTPServer(
            (_deps.config.http_host, _deps.config.http_port),
            HookHTTPHandler,
            _auth_token,
        )
        actual_port = server.server_address[1]
        log.info(f"HTTP server bound to {_deps.config.http_host}:{actual_port}")

        # Write lock file
        _write_lock_file(actual_port, _auth_token)

        # Start server thread
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        log.info(f"HTTP server started on port {actual_port}")

        return server, thread
    except Exception as e:
        log.error(f"Failed to start HTTP server: {e}")
        return None


def _stop_http_server() -> None:
    """Stop HTTP server and cleanup."""
    global _http_server, _http_thread

    try:
        if _http_server is not None:
            log.info("Stopping HTTP server...")
            _http_server.shutdown()
            _http_server = None

        if _http_thread is not None:
            _http_thread.join(timeout=5.0)
            _http_thread = None

        _remove_lock_file()
        log.info("HTTP server stopped")
    except (ValueError, OSError):
        # Suppress logging errors if stdout is closed (e.g., during pytest cleanup)
        # Still attempt cleanup even if logging fails
        if _http_server is not None:
            try:
                _http_server.shutdown()
            except Exception:
                pass
            _http_server = None
        if _http_thread is not None:
            try:
                _http_thread.join(timeout=5.0)
            except Exception:
                pass
            _http_thread = None
        _remove_lock_file()


atexit.register(_stop_http_server)

# Debug info
_debug_info = {
    "HOME": os.environ.get("HOME", "NOT SET"),
    "data_dir": str(_deps.config.data_dir),
    "claude_traces_dir": str(_deps.config.claude_traces_dir),
    "traces_dir_exists": _deps.config.claude_traces_dir.exists(),
}
log.info(f"Server initialized: traces_dir={_deps.config.claude_traces_dir}, exists={_deps.config.claude_traces_dir.exists()}")

# Create MCP server with detailed usage guidance
mcp = FastMCP(
    "simplemem-lite",
    instructions="""SimpleMem Lite: Long-term structured memory with cross-session learning.

## WHEN TO USE MEMORY

**At session START**: Recall relevant prior work
- Use `ask_memories("context for {task}")` for LLM-synthesized answers with citations
- Check what files/errors we've encountered before in similar work

**When encountering ERRORS**: Check past solutions
- `ask_memories("solution for {error}")` returns answers grounded in past sessions
- Cross-session patterns reveal if this error appeared in other projects

**Before complex DECISIONS**: Look for past approaches
- `reason_memories` finds conclusions via multi-hop graph traversal
- Shared entities (files, tools, errors) link knowledge across sessions

**At session END**: Store key learnings
- `store_memory` with type="lesson_learned" for valuable insights
- Linked entities enable future cross-session discovery

## KEY CAPABILITIES

- `ask_memories`: LLM-synthesized answers with citations [1][2]
- `reason_memories`: Multi-hop graph reasoning with proof chains
- `search_memories`: Hybrid vector + graph search
- `process_trace`: Index Claude Code sessions hierarchically
- Cross-session insights via shared entities (files, tools, errors)

## CODE SEARCH

- `search_code`: Semantic search over indexed codebases
- `index_directory`: Index a directory for code search
- `code_stats`: Get code index statistics
- `check_code_staleness`: Check if index needs refresh

Use `index_directory` once per project, then `search_code` to find implementations.

## FILE WATCHING (Auto-update)

- `start_code_watching`: Start watching a directory for changes
- `stop_code_watching`: Stop watching a directory
- `get_watcher_status`: Show all active watchers

Use `start_code_watching` after initial index to keep it up-to-date automatically.
""",
)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS (5 Core Actions)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def store_memory(
    text: str,
    type: str = "fact",
    source: str = "user",
    relations: list[dict] | None = None,
    project_id: str | None = None,
) -> str:
    """Store a memory with optional relationships.

    Args:
        text: The content to store
        type: Memory type (fact, session_summary, chunk_summary, message)
        source: Source of memory (user, claude_trace, extracted)
        relations: Optional list of {target_id, type} relationships
        project_id: Optional project identifier for cross-project isolation

    Returns:
        UUID of stored memory
    """
    log.info(f"Tool: store_memory called (type={type}, source={source}, project={project_id})")
    log.debug(f"Content preview: {text[:100]}...")
    metadata = {"type": type, "source": source}
    if project_id:
        metadata["project_id"] = project_id
    item = MemoryItem(
        content=text,
        metadata=metadata,
        relations=relations or [],
    )
    result = _deps.store.store(item)
    log.info(f"Tool: store_memory complete: {result[:8]}...")

    # ECL-LITE: Extract entities and link to memory (graph-first approach)
    try:
        extraction = await extract_with_actions(text, _deps.config)
        if not extraction.is_empty():
            linked_count = 0
            for entity in extraction.entities:
                success = _deps.store.add_verb_edge(
                    memory_id=result,
                    entity_name=entity.name,
                    entity_type=entity.type,
                    action=entity.action,
                )
                if success:
                    linked_count += 1
            log.info(f"ECL-LITE: Linked {linked_count} entities to memory {result[:8]}...")
    except Exception as e:
        log.warning(f"ECL-LITE entity extraction failed (non-fatal): {e}")

    return result


@mcp.tool()
async def search_memories(
    query: str,
    limit: int = 10,
    use_graph: bool = True,
    type_filter: str | None = None,
    project_id: str | None = None,
) -> list[dict]:
    """Hybrid search combining vector similarity and graph traversal.

    Searches summaries first for efficiency, then expands via graph
    to find related memories.

    Args:
        query: Search query text
        limit: Maximum results to return
        use_graph: Whether to expand results via graph relationships
        type_filter: Optional filter by memory type
        project_id: Optional filter by project (for cross-project isolation)

    Returns:
        List of matching memories with scores
    """
    log.info(f"Tool: search_memories called (query='{query[:50]}...', limit={limit}, project={project_id})")
    results = _deps.store.search(
        query=query,
        limit=limit,
        use_graph=use_graph,
        type_filter=type_filter,
        project_id=project_id,
    )
    log.info(f"Tool: search_memories complete: {len(results)} results")
    return [_memory_to_dict(m) for m in results]


@mcp.tool()
async def relate_memories(
    from_id: str,
    to_id: str,
    relation_type: str = "relates",
) -> bool:
    """Create a relationship between two memories.

    Args:
        from_id: Source memory UUID
        to_id: Target memory UUID
        relation_type: Type of relationship (contains, child_of, supports, follows, similar)

    Returns:
        True if relationship was created
    """
    log.info(f"Tool: relate_memories called ({from_id[:8]}... --[{relation_type}]--> {to_id[:8]}...)")
    result = _deps.store.relate(from_id, to_id, relation_type)
    log.info(f"Tool: relate_memories complete: {result}")
    return result


@mcp.tool()
async def process_trace(session_id: str, background: bool = True) -> dict:
    """Index a SINGLE Claude Code session trace with hierarchical summaries.

    Creates a hierarchy of memories:
    - session_summary (1) - Overall session summary
    - chunk_summary (5-15) - Summaries of activity chunks

    Uses cheap LLM (flash-lite) for summarization with progress updates.
    Runs in background by default to avoid MCP timeout on large sessions.

    FOR MULTIPLE SESSIONS: Use process_trace_batch() instead!
    It handles concurrency, progress tracking, and is more efficient.

    Args:
        session_id: Session UUID to index
        background: Run in background (default: True). Use job_status to check progress.

    Returns:
        If background=True: {job_id, status: "submitted"}
        If background=False: {session_summary_id, chunk_count, message_count} or error
    """
    log.info(f"Tool: process_trace called (session_id={session_id}, background={background})")
    log.debug(f"Using indexer with traces_dir={_deps.indexer.parser.traces_dir}")
    log.debug(f"traces_dir.exists()={_deps.indexer.parser.traces_dir.exists()}")

    if background:
        # Submit to background job manager
        async def _process_trace_job(
            sid: str, progress_callback: Callable[[int, str], None] | None = None
        ) -> dict:
            """Background job wrapper for process_trace."""
            log.info(f"Background job: process_trace starting for session {sid}")
            result = await _deps.indexer.index_session(
                sid, ctx=None, progress_callback=progress_callback
            )
            if result is None:
                log.error(f"Background job: process_trace failed - session not found: {sid}")
                raise ValueError(f"Session {sid} not found")
            log.info(f"Background job: process_trace complete: {len(result.chunk_summary_ids)} chunks")
            return {
                "session_summary_id": result.session_summary_id,
                "chunk_count": len(result.chunk_summary_ids),
                "message_count": len(result.message_ids),
            }

        job_id = await _deps.job_manager.submit("process_trace", _process_trace_job, session_id)
        log.info(f"Tool: process_trace submitted as background job {job_id}")
        return {"job_id": job_id, "status": "submitted", "message": f"Use job_status('{job_id}') to check progress"}

    # Synchronous execution (may timeout for large sessions)
    log.warning("Tool: process_trace running synchronously - may timeout for large sessions")
    result = await _deps.indexer.index_session(session_id, ctx=None)

    if result is None:
        log.error(f"Tool: process_trace failed - session not found: {session_id}")
        return {"error": f"Session {session_id} not found"}

    log.info(f"Tool: process_trace complete: {len(result.chunk_summary_ids)} chunks")
    return {
        "session_summary_id": result.session_summary_id,
        "chunk_count": len(result.chunk_summary_ids),
        "message_count": len(result.message_ids),
    }


@mcp.tool()
async def process_trace_batch(
    sessions: list[dict],
    max_concurrent: int = 3,
) -> dict:
    """Process MULTIPLE session traces - the preferred way to index many sessions.

    WHEN TO USE (prefer this over process_trace for multiple sessions):
    - Indexing historical sessions after discover_sessions()
    - Batch processing sessions from the last N days
    - Regular maintenance to index all unprocessed sessions

    HOW IT WORKS:
    1. Accepts session dicts directly from discover_sessions() output
    2. Queues up to max_concurrent * 10 sessions (default: 30)
    3. Runs full LLM summarization for each session
    4. Creates hierarchical memories (session_summary + chunk_summaries)
    5. Returns job IDs for progress tracking via job_status()

    WORKFLOW:
        # Discover unindexed sessions from last 7 days
        sessions = discover_sessions(days_back=7, include_indexed=False)

        # Process them all in batch
        result = process_trace_batch(sessions=sessions["sessions"])

        # Check individual job progress
        for session_id, job_id in result["job_ids"].items():
            status = job_status(job_id)

    Args:
        sessions: List of session dicts with 'session_id' and 'path' keys
                  (as returned by discover_sessions)
        max_concurrent: Maximum concurrent jobs (default: 3, max effective: 30 sessions)

    Returns:
        {
            "queued": ["session-id-1", "session-id-2", ...],
            "errors": [{"session_id": "...", "error": "..."}],
            "job_ids": {"session-id-1": "job-uuid-1", ...}
        }
    """
    log.info(f"Tool: process_trace_batch called with {len(sessions)} sessions")

    queued = []
    errors = []
    job_ids = {}

    for session in sessions[:max_concurrent * 10]:  # Limit total to prevent overload
        session_id = session.get("session_id")
        session_path = session.get("path")

        if not session_id or not session_path:
            errors.append({"session_id": session_id, "error": "Missing session_id or path"})
            continue

        try:
            # Submit job with the full path to bypass find_session lookup
            async def _process_with_path(
                sid: str, path: str, progress_callback: Callable[[int, str], None] | None = None
            ) -> dict:
                """Background job wrapper that uses provided path."""
                log.info(f"Background job: process_trace_batch starting for {sid}")
                result = await _deps.indexer.index_session(
                    sid, ctx=None, session_path=path, progress_callback=progress_callback
                )
                if result is None:
                    raise ValueError(f"Session {sid} indexing failed")
                log.info(f"Background job: process_trace_batch complete: {len(result.chunk_summary_ids)} chunks")
                return {
                    "session_summary_id": result.session_summary_id,
                    "chunk_count": len(result.chunk_summary_ids),
                    "message_count": len(result.message_ids),
                }

            job_id = await _deps.job_manager.submit(
                "process_trace_batch",
                _process_with_path,
                session_id,
                session_path,
            )
            queued.append(session_id)
            job_ids[session_id] = job_id
            log.info(f"Tool: process_trace_batch queued {session_id} as job {job_id}")

        except Exception as e:
            log.error(f"Tool: process_trace_batch failed for {session_id}: {e}")
            errors.append({"session_id": session_id, "error": str(e)})

    log.info(f"Tool: process_trace_batch complete: {len(queued)} queued, {len(errors)} errors")
    return {
        "queued": queued,
        "errors": errors,
        "job_ids": job_ids,
        "total_requested": len(sessions),
    }


@mcp.tool()
async def get_stats() -> dict:
    """Get memory store statistics.

    Returns:
        {total_memories, total_relations, types_breakdown}
    """
    log.info("Tool: get_stats called")
    result = _deps.store.get_stats()
    log.info(f"Tool: get_stats complete: {result['total_memories']} memories")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENT TODO SYSTEM (P4)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def create_todo(
    title: str,
    description: str | None = None,
    priority: str = "medium",
    project_id: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Create a persistent todo item.

    Unlike Claude's ephemeral TodoWrite, these todos persist across sessions
    and can be searched semantically. Use this for important tasks that
    should be tracked long-term.

    Args:
        title: Short description of the task
        description: Detailed description (optional)
        priority: Priority level - low, medium, high, or critical (default: medium)
        project_id: Project scope for filtering (optional)
        tags: List of tags for categorization (optional)

    Returns:
        Dict containing:
        - uuid: Created todo's unique identifier
        - title: The todo title
        - status: Current status (always "pending" for new todos)
        - priority: Priority level
        - created_at: Creation timestamp
    """
    from simplemem_lite.todo import TodoStore

    log.info(
        f"Tool: create_todo called "
        f"(title={title[:50]}..., priority={priority}, project={project_id})"
    )

    try:
        todo_store = TodoStore(_deps.store)
        todo = todo_store.create(
            title=title,
            description=description,
            priority=priority,
            project_id=project_id,
            tags=tags or [],
            source="user",
        )

        log.info(f"Tool: create_todo complete: {todo.uuid[:8]}...")

        return {
            "uuid": todo.uuid,
            "title": todo.title,
            "status": todo.status,
            "priority": todo.priority,
            "project_id": todo.project_id,
            "created_at": todo.created_at,
        }

    except ValueError as e:
        log.error(f"Tool: create_todo validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        log.error(f"Tool: create_todo failed: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def list_todos(
    project_id: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    limit: int = 20,
) -> dict:
    """List persistent todos with optional filters.

    Returns todos matching the specified criteria, sorted by priority
    and creation date.

    Args:
        project_id: Filter by project scope (optional)
        status: Filter by status - pending, in_progress, completed, cancelled, blocked (optional)
        priority: Filter by priority - low, medium, high, critical (optional)
        limit: Maximum number to return (default: 20)

    Returns:
        Dict containing:
        - todos: List of todo objects with uuid, title, status, priority, etc.
        - total_count: Number of todos returned
    """
    from simplemem_lite.todo import TodoStore

    log.info(
        f"Tool: list_todos called "
        f"(project={project_id}, status={status}, priority={priority}, limit={limit})"
    )

    try:
        todo_store = TodoStore(_deps.store)
        todos = todo_store.find(
            project_id=project_id,
            status=status,
            priority=priority,
            limit=limit,
        )

        result = {
            "todos": [t.to_dict() for t in todos],
            "total_count": len(todos),
        }

        log.info(f"Tool: list_todos complete: {len(todos)} todos found")
        return result

    except ValueError as e:
        log.error(f"Tool: list_todos validation error: {e}")
        return {"error": str(e), "todos": [], "total_count": 0}
    except Exception as e:
        log.error(f"Tool: list_todos failed: {e}", exc_info=True)
        return {"error": str(e), "todos": [], "total_count": 0}


@mcp.tool()
async def update_todo(
    todo_id: str,
    status: str | None = None,
    priority: str | None = None,
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Update a persistent todo's fields.

    Update any combination of fields. To mark a todo complete,
    use status="completed". To cancel, use status="cancelled".

    Args:
        todo_id: UUID of the todo to update
        status: New status - pending, in_progress, completed, cancelled, blocked (optional)
        priority: New priority - low, medium, high, critical (optional)
        title: New title (optional)
        description: New description (optional)
        tags: New tags list (optional)

    Returns:
        Dict containing:
        - uuid: Updated todo's UUID (may change due to re-storage)
        - title: Current title
        - status: Current status
        - priority: Current priority
        - updated_at: Update timestamp
        - success: Whether update succeeded
    """
    from simplemem_lite.todo import TodoStore

    log.info(
        f"Tool: update_todo called "
        f"(todo_id={todo_id[:8]}..., status={status}, priority={priority})"
    )

    try:
        todo_store = TodoStore(_deps.store)
        todo = todo_store.update(
            todo_id=todo_id,
            status=status,
            priority=priority,
            title=title,
            description=description,
            tags=tags,
        )

        if not todo:
            log.warning(f"Tool: update_todo not found: {todo_id[:8]}...")
            return {"error": "Todo not found", "success": False}

        log.info(f"Tool: update_todo complete: {todo.uuid[:8]}...")

        return {
            "uuid": todo.uuid,
            "title": todo.title,
            "status": todo.status,
            "priority": todo.priority,
            "updated_at": todo.updated_at,
            "completed_at": todo.completed_at,
            "success": True,
        }

    except ValueError as e:
        log.error(f"Tool: update_todo validation error: {e}")
        return {"error": str(e), "success": False}
    except Exception as e:
        log.error(f"Tool: update_todo failed: {e}", exc_info=True)
        return {"error": str(e), "success": False}


@mcp.tool()
async def promote_todo(
    title: str,
    description: str | None = None,
    priority: str = "medium",
    project_id: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Promote an ephemeral todo to persistent storage.

    Use this to persist an important task from Claude's ephemeral TodoWrite
    to SimpleMem's persistent todo system. The todo will be searchable
    across sessions and linked to relevant context.

    Args:
        title: The todo title (copy from TodoWrite)
        description: Detailed description (optional)
        priority: Priority level - low, medium, high, critical (default: medium)
        project_id: Project scope for filtering (optional)
        tags: List of tags for categorization (optional)

    Returns:
        Dict containing:
        - uuid: Created todo's unique identifier
        - title: The todo title
        - status: Current status (always "pending")
        - priority: Priority level
        - source: Always "promoted"
        - created_at: Creation timestamp
    """
    from simplemem_lite.todo import TodoStore

    log.info(
        f"Tool: promote_todo called "
        f"(title={title[:50]}..., priority={priority}, project={project_id})"
    )

    try:
        todo_store = TodoStore(_deps.store)
        todo = todo_store.promote(
            title=title,
            description=description,
            priority=priority,
            project_id=project_id,
            tags=tags or [],
        )

        log.info(f"Tool: promote_todo complete: {todo.uuid[:8]}...")

        return {
            "uuid": todo.uuid,
            "title": todo.title,
            "status": todo.status,
            "priority": todo.priority,
            "project_id": todo.project_id,
            "source": todo.source,
            "created_at": todo.created_at,
        }

    except ValueError as e:
        log.error(f"Tool: promote_todo validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        log.error(f"Tool: promote_todo failed: {e}", exc_info=True)
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND JOB MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def job_status(job_id: str) -> dict:
    """Get status of a background job.

    Use this to check progress of long-running operations like process_trace
    or index_directory that were submitted with background=True.

    Args:
        job_id: Job ID returned when submitting the background job

    Returns:
        {id, type, status, progress, message, result, error, timestamps}
        status is one of: pending, running, completed, failed, cancelled
    """
    log.info(f"Tool: job_status called (job_id={job_id})")
    result = _deps.job_manager.get_status(job_id)
    if result is None:
        log.warning(f"Tool: job_status - job not found: {job_id}")
        return {"error": f"Job {job_id} not found"}
    log.info(f"Tool: job_status complete: {job_id} status={result['status']} progress={result['progress']}%")
    return result


@mcp.tool()
async def list_jobs(include_completed: bool = True, limit: int = 20) -> dict:
    """List all background jobs.

    Args:
        include_completed: Include completed/failed/cancelled jobs (default: True)
        limit: Maximum number of jobs to return (default: 20)

    Returns:
        {jobs: [{id, type, status, progress, message}]}
    """
    log.info(f"Tool: list_jobs called (include_completed={include_completed}, limit={limit})")
    jobs = _deps.job_manager.list_jobs(include_completed=include_completed, limit=limit)
    log.info(f"Tool: list_jobs complete: {len(jobs)} jobs returned")
    return {"jobs": jobs}


@mcp.tool()
async def cancel_job(job_id: str) -> dict:
    """Cancel a running background job.

    Args:
        job_id: Job ID to cancel

    Returns:
        {cancelled: bool, message: str}
    """
    log.info(f"Tool: cancel_job called (job_id={job_id})")
    success = await _deps.job_manager.cancel(job_id)
    if success:
        log.info(f"Tool: cancel_job - successfully cancelled job {job_id}")
        return {"cancelled": True, "message": f"Job {job_id} cancelled"}
    else:
        log.warning(f"Tool: cancel_job - failed to cancel job {job_id}")
        return {"cancelled": False, "message": f"Job {job_id} not found or not running"}


@mcp.tool()
async def reset_all(confirm: bool = False) -> dict:
    """DEBUG: Completely reset all memories and relationships.

    WARNING: This is destructive and irreversible!
    Deletes ALL stored memories, embeddings, and graph relationships.

    Args:
        confirm: Must be True to actually perform the reset

    Returns:
        {memories_deleted, relations_deleted} or error if not confirmed
    """
    log.warning(f"Tool: reset_all called (confirm={confirm})")

    if not confirm:
        log.info("Tool: reset_all aborted - confirmation required")
        return {
            "error": "Reset not confirmed",
            "message": "Set confirm=True to actually delete all data. This is irreversible!",
        }

    result = _deps.store.reset_all()
    log.warning(f"Tool: reset_all complete: {result}")
    return result


@mcp.tool()
async def reason_memories(
    query: str,
    max_hops: int = 2,
    min_score: float = 0.1,
) -> dict:
    """Multi-hop reasoning over memory graph.

    Combines vector search with graph traversal and semantic path scoring
    to answer complex questions that require following chains of evidence.

    Example queries:
    - "What debugging patterns work for database issues?"
    - "How did the authentication feature evolve?"
    - "Find solutions related to connection timeouts"

    Args:
        query: Natural language query
        max_hops: Maximum path length for traversal (1-3)
        min_score: Minimum score threshold for results

    Returns:
        {conclusions: [{uuid, content, type, score, proof_chain, hops}]}
    """
    log.info(f"Tool: reason_memories called (query='{query[:50]}...', max_hops={max_hops})")

    results = _deps.store.reason(
        query=query,
        max_hops=max_hops,
        min_score=min_score,
    )

    log.info(f"Tool: reason_memories complete: {len(results)} conclusions")

    # Count cross-session results
    cross_session_count = sum(1 for r in results if r.get("cross_session"))

    return {
        "conclusions": [
            {
                "uuid": r["uuid"],
                "content": r["content"][:500],  # Truncate for response size
                "type": r["type"],
                "score": round(r["score"], 3),
                "pagerank": r.get("pagerank", 0.0),
                "proof_chain": r["proof_chain"],
                "hops": r["hops"],
                "cross_session": r.get("cross_session", False),
                "bridge_entity": r.get("bridge_entity"),
            }
            for r in results[:10]  # Limit to top 10
        ],
        "cross_session_count": cross_session_count,
    }


@mcp.tool()
async def ask_memories(
    query: str,
    max_memories: int = 8,
    max_hops: int = 2,
    project_id: str | None = None,
) -> dict:
    """Ask a question and get an LLM-synthesized answer from memory graph.

    Retrieves relevant memories via multi-hop graph traversal, then uses
    an LLM to synthesize a coherent answer grounded in the evidence.

    The answer includes citations [1], [2], etc. referencing specific memories.
    Cross-session insights (patterns found across different work sessions) are
    highlighted as especially valuable.

    Example queries:
    - "What was the solution to the database connection issue?"
    - "How did we implement the authentication feature?"
    - "What patterns have worked for debugging async code?"

    Args:
        query: Natural language question
        max_memories: Maximum memories to include in context (default: 8)
        max_hops: Maximum graph traversal depth (default: 2)
        project_id: Optional project identifier for cross-project isolation

    Returns:
        {answer, memories_used, cross_session_insights, confidence, sources}
    """
    log.info(f"Tool: ask_memories called (query='{query[:50]}...', project={project_id})")

    result = await _deps.store.ask_memories(
        query=query,
        max_memories=max_memories,
        max_hops=max_hops,
        project_id=project_id,
    )

    log.info(f"Tool: ask_memories complete: confidence={result['confidence']}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CODE SEARCH TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def search_code(
    query: str,
    limit: int = 10,
    project_id: str | None = None,
) -> dict:
    """Search indexed code for implementations, patterns, and functionality.

    PURPOSE: Find relevant code snippets using semantic search. Unlike grep/ripgrep
    which match exact text, this finds code by MEANING - "authentication handler"
    will find login functions even if they don't contain those exact words.

    WHEN TO USE:
    - Finding implementations: "user authentication", "database connection pool"
    - Finding patterns: "error handling pattern", "retry logic"
    - Understanding structure: "API endpoints", "middleware functions"
    - Before implementing: Check if similar code exists
    - Debugging: Find code related to an error

    WHEN NOT TO USE:
    - For exact text matches (use grep/ripgrep instead)
    - For files that haven't been indexed (use index_directory first)
    - For memory/insight search (use search_memories instead)

    PREREQUISITE: The codebase must be indexed first using index_directory().
    If no results found, the codebase may not be indexed.

    EXAMPLES:
        # Find authentication code
        search_code(query="user login authentication handler")

        # Find database patterns
        search_code(
            query="connection pool database initialization",
            limit=20
        )

        # Search in specific project
        search_code(
            query="API rate limiting middleware",
            project_id="git:github.com/user/myproject"
        )

        # Find error handling
        search_code(query="exception handling retry logic")

    Args:
        query: Natural language description of code you're looking for.
               Be descriptive: "user authentication JWT token validation"
               is better than just "auth".
        limit: Maximum results (default: 10). Increase for broader search.
        project_id: Filter to specific project (preferred). Auto-inferred
                    from cwd if not specified.

    Returns:
        On success: {
            "results": [
                {
                    "file_path": "/path/to/file.py",
                    "line_start": 45,
                    "line_end": 78,
                    "content": "def authenticate_user(...)...",
                    "score": 0.89
                },
                ...
            ]
        }
        On error: {"error": "...", "results": []}
    """
    log.info(f"Tool: search_code called (query={query[:50]}..., limit={limit}, project_id={project_id})")
    results = _deps.code_indexer.search(query, limit, project_id)
    log.info(f"Tool: search_code complete: {len(results)} results")
    return {"results": results, "count": len(results)}


@mcp.tool()
async def index_directory(
    path: str,
    patterns: list[str] | None = None,
    clear_existing: bool = True,
    background: bool = True,
) -> dict:
    """Index a directory for code search.

    Scans the directory for source files matching the patterns,
    splits them into semantic chunks, and adds to the search index.
    Runs in background by default to avoid MCP timeout on large codebases.

    Args:
        path: Directory path to index
        patterns: Optional glob patterns (default: ['**/*.py', '**/*.ts', '**/*.js', '**/*.tsx', '**/*.jsx'])
        clear_existing: Whether to clear existing index for this directory (default: True)
        background: Run in background (default: True). Use job_status to check progress.

    Returns:
        If background=True: {job_id, status: "submitted"}
        If background=False: Indexing statistics including files indexed and chunks created
    """
    log.info(f"Tool: index_directory called (path={path}, background={background})")

    if background:
        # Submit to background job manager
        async def _index_directory_job(p: str, pats: list[str] | None, clear: bool) -> dict:
            """Background job wrapper for index_directory.

            Uses async version that yields to event loop, preventing server unresponsiveness.
            """
            log.info(f"Background job: index_directory starting for path {p}")
            # Use async version that yields to event loop during indexing
            result = await _deps.code_indexer.index_directory_async(p, pats, clear)
            log.info(f"Background job: index_directory complete: {result.get('files_indexed', 0)} files")
            return result

        job_id = await _deps.job_manager.submit("index_directory", _index_directory_job, path, patterns, clear_existing)
        log.info(f"Tool: index_directory submitted as background job {job_id}")
        return {"job_id": job_id, "status": "submitted", "message": f"Use job_status('{job_id}') to check progress"}

    # Synchronous execution
    log.debug("Tool: index_directory running synchronously")
    result = _deps.code_indexer.index_directory(path, patterns, clear_existing)
    log.info(f"Tool: index_directory complete: {result.get('files_indexed', 0)} files, {result.get('chunks_created', 0)} chunks")
    return result


@mcp.tool()
async def code_stats(project_id: str | None = None) -> dict:
    """Get statistics about the code index.

    Args:
        project_id: Optional - filter to specific project (preferred)
        project_root: Optional - filter by path (deprecated, use project_id)

    Returns:
        Statistics including chunk count and unique files
    """
    log.info(f"Tool: code_stats called (project_id={project_id})")
    stats = _deps.store.db.get_code_stats(project_id)
    log.info(f"Tool: code_stats complete: {stats}")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# P1: ENTITY LINKING (CODE ↔ MEMORY BRIDGE)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def code_related_memories(
    chunk_uuid: str,
    limit: int = 10,
) -> dict:
    """Find memories related to a code chunk via shared entities.

    Uses the entity graph to find memories that reference the same entities
    (files, functions, modules) as the given code chunk.

    Args:
        chunk_uuid: UUID of the code chunk (from search_code results)
        limit: Maximum memories to return (default: 10)

    Returns:
        Dict with related memories and their shared entities
    """
    log.info(f"Tool: code_related_memories called (chunk={chunk_uuid[:8]}...)")
    memories = _deps.store.db.get_code_related_memories(chunk_uuid, limit)
    log.info(f"Tool: code_related_memories complete: {len(memories)} memories found")
    return {
        "chunk_uuid": chunk_uuid,
        "related_memories": memories,
        "count": len(memories),
    }


@mcp.tool()
async def memory_related_code(
    memory_uuid: str,
    limit: int = 10,
) -> dict:
    """Find code chunks related to a memory via shared entities.

    Uses the entity graph to find code chunks that reference the same entities
    (files, functions, modules) as the given memory.

    Args:
        memory_uuid: UUID of the memory (from search_memories results)
        limit: Maximum code chunks to return (default: 10)

    Returns:
        Dict with related code chunks and their shared entities
    """
    log.info(f"Tool: memory_related_code called (memory={memory_uuid[:8]}...)")
    code_chunks = _deps.store.db.get_memory_related_code(memory_uuid, limit)
    log.info(f"Tool: memory_related_code complete: {len(code_chunks)} chunks found")
    return {
        "memory_uuid": memory_uuid,
        "related_code": code_chunks,
        "count": len(code_chunks),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# P2: STALENESS DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def check_code_staleness(project_root: str) -> dict:
    """Check if the code index is stale and needs refreshing.

    Uses git to detect changes since last indexing. Returns staleness status
    with details about changed files.

    Args:
        project_root: Absolute path to the project root directory

    Returns:
        Dict with:
        - is_indexed: Whether project has been indexed
        - is_stale: Whether index needs refresh
        - is_git_repo: Whether project is a git repository
        - indexed_hash: Commit hash when last indexed
        - current_hash: Current HEAD commit hash
        - changed_files: Files changed since last index
        - uncommitted_files: Uncommitted changes
        - reason: Human-readable explanation
    """
    log.info(f"Tool: check_code_staleness called (project={project_root})")
    result = _deps.code_indexer.check_staleness(project_root)
    log.info(f"Tool: check_code_staleness complete: is_stale={result.get('is_stale')}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# P3: FILE WATCHER (AUTO-CAPTURE)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def start_code_watching(project_root: str) -> dict:
    """Start watching a project directory for file changes.

    Automatically updates the code index when files are created, modified,
    or deleted. Uses debouncing to coalesce rapid changes.

    Args:
        project_root: Absolute path to the project root directory

    Returns:
        Dict with status and project info
    """
    log.info(f"Tool: start_code_watching called (project={project_root})")
    result = _deps.watcher_manager.start_watching(project_root)
    log.info(f"Tool: start_code_watching complete: status={result.get('status', result.get('error'))}")
    return result


@mcp.tool()
async def stop_code_watching(project_root: str) -> dict:
    """Stop watching a project directory.

    Args:
        project_root: Absolute path to the project root directory

    Returns:
        Dict with status and final statistics
    """
    log.info(f"Tool: stop_code_watching called (project={project_root})")
    result = _deps.watcher_manager.stop_watching(project_root)
    log.info(f"Tool: stop_code_watching complete: status={result.get('status', result.get('error'))}")
    return result


@mcp.tool()
async def get_watcher_status() -> dict:
    """Get status of all active file watchers.

    Returns:
        Dict with:
        - watching: Number of projects being watched
        - patterns: File patterns being monitored
        - projects: Dict of project status with stats per project
    """
    log.info("Tool: get_watcher_status called")
    result = _deps.watcher_manager.get_status()
    log.info(f"Tool: get_watcher_status complete: watching={result.get('watching', 0)} projects")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# P4: PROJECT BOOTSTRAP (AUTO-DETECT & SETUP)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def bootstrap_project(
    project_root: str,
    index_code: bool = True,
    start_watcher: bool = True,
) -> dict:
    """Bootstrap a project for SimpleMem features.

    Detects project type, indexes code files, and starts file watcher.
    This enables code search, memory features, and automatic index updates.

    Args:
        project_root: Absolute path to the project root directory
        index_code: Whether to index code files (default: True)
        start_watcher: Whether to start file watcher (default: True)

    Returns:
        Dict with bootstrap results including detected project info
    """
    log.info(f"Tool: bootstrap_project called (project={project_root})")
    result = _deps.bootstrap.bootstrap_project(project_root, index_code, start_watcher)
    log.info(f"Tool: bootstrap_project complete: success={result.get('success')}")
    return result


@mcp.tool()
async def get_project_status(project_root: str) -> dict:
    """Get bootstrap and watcher status for a project.

    Args:
        project_root: Absolute path to the project root directory

    Returns:
        Dict with:
        - is_known: Whether project is tracked
        - is_bootstrapped: Whether project has been bootstrapped
        - is_watching: Whether file watcher is active
        - project_name: Detected project name
        - should_ask: Whether to prompt for bootstrap
        - deferred_context: Context from pending session (if any)
    """
    log.info(f"Tool: get_project_status called (project={project_root})")

    # Check for pending session from deferred hook execution
    pending_file = Path.home() / ".simplemem_lite" / "pending_session.json"
    deferred_context = None
    if pending_file.exists():
        try:
            import json
            pending_data = json.loads(pending_file.read_text())
            pending_cwd = pending_data.get("cwd", "")
            # Use project_root from pending if not provided, or validate match
            effective_root = project_root or _deps.project_manager.detect_project_root(pending_cwd)
            deferred_context = _deps.bootstrap.generate_context_injection(effective_root)
            pending_file.unlink()  # Clean up pending file
            log.info(f"Processed deferred session context for {effective_root}")
        except Exception as e:
            log.warning(f"Failed to process pending session: {e}")

    result = _deps.bootstrap.get_bootstrap_status(project_root)
    if deferred_context:
        result["deferred_context"] = deferred_context
    log.info(f"Tool: get_project_status complete: bootstrapped={result.get('is_bootstrapped')}")
    return result


@mcp.tool()
async def set_project_preference(
    project_root: str,
    never_ask: bool = True,
) -> dict:
    """Set user preference for a project.

    Use this to mark a project as "never ask about bootstrap".

    Args:
        project_root: Absolute path to the project root directory
        never_ask: If True, never prompt about bootstrapping this project

    Returns:
        Updated project state
    """
    log.info(f"Tool: set_project_preference called (project={project_root}, never_ask={never_ask})")
    state = _deps.project_manager.set_never_ask(project_root, never_ask)
    log.info(f"Tool: set_project_preference complete: never_ask={state.never_ask}")
    return {
        "project_root": state.project_root,
        "never_ask": state.never_ask,
        "is_bootstrapped": state.is_bootstrapped,
    }


@mcp.tool()
async def list_tracked_projects() -> dict:
    """List all tracked projects.

    Returns:
        List of projects with their bootstrap status
    """
    log.info("Tool: list_tracked_projects called")
    projects = _deps.project_manager.list_projects()
    log.info(f"Tool: list_tracked_projects complete: {len(projects)} projects")
    return {"projects": projects, "count": len(projects)}


# ═══════════════════════════════════════════════════════════════════════════════
# P5: GRAPH POWER TOOLS (RAW CYPHER ACCESS)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def get_graph_schema() -> dict:
    """Get the complete graph schema for zero-discovery query generation.

    Returns the full FalkorDB graph schema including:
    - All node labels with their properties, types, and indexes
    - All relationship types with descriptions and properties
    - Common query templates ready to use

    Use this BEFORE writing any Cypher queries to avoid wasting tokens
    on schema discovery. The schema includes everything needed to write
    accurate queries on the first attempt.

    Returns:
        Dict containing:
        - node_labels: List of node types with properties
        - relationship_types: List of edge types with metadata
        - common_queries: Ready-to-use query templates
    """
    log.info("Tool: get_graph_schema called")
    try:
        schema = _deps.store.db.get_schema()
        log.info(
            f"Tool: get_graph_schema complete: "
            f"{len(schema.get('node_labels', []))} nodes, "
            f"{len(schema.get('relationship_types', []))} relationships"
        )
        return schema
    except Exception as e:
        log.error(f"Tool: get_graph_schema failed: {e}")
        return {"error": str(e), "node_labels": [], "relationship_types": [], "common_queries": []}


@mcp.tool()
async def run_cypher_query(
    query: str,
    params: dict | None = None,
    max_results: int = 100,
) -> dict:
    """Execute a validated Cypher query against the FalkorDB graph.

    IMPORTANT: Use get_graph_schema first to understand available
    node types, relationships, and properties before writing queries.

    Security features (enforced automatically):
    - Read-only by default: CREATE, MERGE, DELETE, SET, REMOVE blocked
    - LIMIT injection: Queries without LIMIT get one added
    - Result truncation: Output capped at max_results

    Args:
        query: Cypher query string (e.g., "MATCH (m:Memory) RETURN m.uuid, m.type LIMIT 10")
        params: Optional query parameters for parameterized queries
        max_results: Maximum rows to return (default: 100, max: 1000)

    Returns:
        Dict containing:
        - results: List of result rows as dicts
        - row_count: Number of rows returned
        - truncated: True if results were limited
        - execution_time_ms: Query execution time

    Example queries (after checking schema):
        - "MATCH (m:Memory)-[:RELATES_TO]->(e:Entity) WHERE e.name CONTAINS 'auth' RETURN m.uuid, m.content LIMIT 20"
        - "MATCH path = (m:Memory)-[*1..2]-(other) WHERE m.uuid = $uuid RETURN path"
        - "MATCH (e:Entity {type: 'file'}) RETURN e.name, e.version ORDER BY e.version DESC LIMIT 10"
    """
    log.info(f"Tool: run_cypher_query called (max_results={max_results})")
    log.debug(f"Query: {query[:200]}...")

    # Clamp max_results
    max_results = min(max(1, max_results), 1000)

    try:
        result = _deps.store.db.execute_validated_cypher(
            query=query,
            params=params,
            max_results=max_results,
            allow_mutations=False,  # Always read-only via MCP
        )
        log.info(
            f"Tool: run_cypher_query complete: "
            f"{result['row_count']} rows in {result['execution_time_ms']}ms"
        )
        return result
    except ValueError as e:
        # Mutation blocked or validation error
        log.warning(f"Tool: run_cypher_query validation failed: {e}")
        return {"error": str(e), "results": [], "row_count": 0, "truncated": False}
    except Exception as e:
        log.error(f"Tool: run_cypher_query failed: {e}")
        return {"error": str(e), "results": [], "row_count": 0, "truncated": False}


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL SESSION DISCOVERY (P3)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def discover_sessions(
    days_back: int = 30,
    group_by: str | None = None,
    include_indexed: bool = True,
) -> dict:
    """Discover available Claude Code sessions for potential indexing.

    Lightweight scan that reads file metadata only (no LLM calls).
    Use this to explore historical sessions before batch indexing.

    Args:
        days_back: Only include sessions modified within this many days (default: 30)
        group_by: Optional grouping - "project" or "date" (default: None, flat list)
        include_indexed: Include already-indexed sessions in results (default: True)

    Returns:
        Dict containing:
        - sessions: List of session metadata (or grouped dict if group_by specified)
        - total_count: Total sessions found
        - indexed_count: How many are already indexed
        - unindexed_count: How many are not yet indexed
    """
    import asyncio
    import time
    from collections import defaultdict
    from datetime import datetime

    log.info(f"Tool: discover_sessions called (days_back={days_back}, group_by={group_by})")

    # Validate group_by parameter
    if group_by is not None and group_by not in ("project", "date"):
        return {"error": "group_by must be 'project' or 'date'", "sessions": [], "total_count": 0}

    try:
        # Get all sessions from TraceParser (offload to thread to avoid blocking)
        all_sessions = await asyncio.to_thread(_deps.indexer.parser.list_sessions)
        log.debug(f"Found {len(all_sessions)} total sessions")

        # Filter by days_back
        cutoff_time = time.time() - (days_back * 24 * 60 * 60)
        sessions = [s for s in all_sessions if s["modified"] >= cutoff_time]
        log.debug(f"After days_back filter: {len(sessions)} sessions")

        # Batch fetch indexed session IDs (avoids N+1 queries)
        all_indexed = _deps.session_state_db.list_sessions(status="indexed", limit=10000)
        indexed_ids = {s.session_id for s in all_indexed}

        # Enrich session data
        enriched = []
        for s in sessions:
            is_indexed = s["session_id"] in indexed_ids
            if not include_indexed and is_indexed:
                continue

            enriched.append({
                "session_id": s["session_id"],
                "project": s["project"],
                "path": s["path"],
                "size_kb": s["size_kb"],
                "modified_at": s["modified"],
                "modified_date": datetime.fromtimestamp(s["modified"]).strftime("%Y-%m-%d"),
                "is_indexed": is_indexed,
            })

        # Group if requested
        if group_by == "project":
            grouped = defaultdict(list)
            for s in enriched:
                grouped[s["project"]].append(s)
            result_sessions = dict(grouped)
        elif group_by == "date":
            grouped = defaultdict(list)
            for s in enriched:
                grouped[s["modified_date"]].append(s)
            result_sessions = dict(grouped)
        else:
            result_sessions = enriched

        indexed_count = len(indexed_ids)
        unindexed_count = len(sessions) - indexed_count

        log.info(
            f"Tool: discover_sessions complete: "
            f"{len(enriched)} sessions returned, {indexed_count} indexed, {unindexed_count} unindexed"
        )

        return {
            "sessions": result_sessions,
            "total_count": len(sessions),
            "indexed_count": indexed_count,
            "unindexed_count": unindexed_count,
            "days_back": days_back,
            "group_by": group_by,
        }

    except Exception as e:
        log.error(f"Tool: discover_sessions failed: {e}", exc_info=True)
        return {"error": str(e), "sessions": [], "total_count": 0}


@mcp.tool()
async def index_sessions_batch(
    session_ids: list[str] | None = None,
    days_back: int | None = None,
    max_sessions: int = 10,
    skip_indexed: bool = True,
) -> dict:
    """Queue historical sessions for background indexing.

    Use discover_sessions first to find available sessions, then use this
    tool to batch index them. Reuses existing indexing infrastructure.

    Args:
        session_ids: Specific session IDs to index (optional)
        days_back: Index all unindexed sessions from last N days (optional)
        max_sessions: Maximum sessions to queue in this batch (default: 10)
        skip_indexed: Skip sessions that are already indexed (default: True)

    Returns:
        Dict containing:
        - queued: List of session IDs that were queued
        - skipped: List of session IDs that were skipped (already indexed)
        - errors: List of session IDs that failed to queue
        - job_ids: Map of session_id -> job_id for tracking
    """
    import time

    log.info(
        f"Tool: index_sessions_batch called "
        f"(session_ids={len(session_ids) if session_ids else 'None'}, "
        f"days_back={days_back}, max_sessions={max_sessions})"
    )

    try:
        # Get all sessions from TraceParser (offload to thread to avoid blocking)
        all_sessions = await asyncio.to_thread(_deps.indexer.parser.list_sessions)

        # Determine which sessions to process
        if session_ids:
            # Use provided session IDs
            sessions_map = {s["session_id"]: s for s in all_sessions}
            sessions_to_process = [
                sessions_map[sid] for sid in session_ids if sid in sessions_map
            ]
        elif days_back is not None:
            # Use days_back filter
            cutoff_time = time.time() - (days_back * 24 * 60 * 60)
            sessions_to_process = [s for s in all_sessions if s["modified"] >= cutoff_time]
        else:
            return {
                "error": "Must provide either session_ids or days_back",
                "queued": [],
                "skipped": [],
                "errors": [],
            }

        log.debug(f"Sessions to consider: {len(sessions_to_process)}")

        # Batch fetch indexed session IDs (avoids N+1 queries)
        indexed_ids: set[str] = set()
        if skip_indexed:
            all_indexed = _deps.session_state_db.list_sessions(status="indexed", limit=10000)
            indexed_ids = {s.session_id for s in all_indexed}
            log.debug(f"Found {len(indexed_ids)} already indexed sessions")

        queued = []
        skipped = []
        errors = []
        job_ids = {}

        for session in sessions_to_process:
            if len(queued) >= max_sessions:
                log.info(f"Reached max_sessions limit ({max_sessions})")
                break

            session_id = session["session_id"]

            # Check if already indexed (using pre-fetched set)
            if skip_indexed and session_id in indexed_ids:
                skipped.append(session_id)
                continue

            # Detect project root
            project = session.get("project", "")
            project_root = _deps.project_manager.detect_project_root(project)
            if not project_root:
                project_root = project

            try:
                # Queue for indexing via JobManager
                # Note: submit() takes a factory function + args, not an instantiated coroutine
                job_id = await _deps.job_manager.submit(
                    "batch_index_session",
                    _deps.indexer.index_session_delta,
                    session_id=session_id,
                    project_root=project_root,
                    project_manager=_deps.project_manager,
                    transcript_path=session["path"],
                )
                queued.append(session_id)
                job_ids[session_id] = job_id
                log.info(f"Queued session {session_id[:8]}... as job {job_id[:8]}...")

            except Exception as e:
                log.error(f"Failed to queue session {session_id[:8]}...: {e}")
                errors.append(session_id)

        log.info(
            f"Tool: index_sessions_batch complete: "
            f"{len(queued)} queued, {len(skipped)} skipped, {len(errors)} errors"
        )

        return {
            "queued": queued,
            "skipped": skipped,
            "errors": errors,
            "job_ids": job_ids,
            "queued_count": len(queued),
            "skipped_count": len(skipped),
        }

    except Exception as e:
        log.error(f"Tool: index_sessions_batch failed: {e}", exc_info=True)
        return {"error": str(e), "queued": [], "skipped": [], "errors": []}


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSOLIDATION
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def consolidate_project(
    project_id: str | None = None,
    operations: list[str] | None = None,
    dry_run: bool = False,
    confidence_threshold: float = 0.9,
) -> dict[str, Any]:
    """Run graph consolidation for a project.

    Performs LLM-assisted graph maintenance:
    - Entity deduplication (merge `main.py` ↔ `./main.py`)
    - Memory merging (combine near-duplicate memories)
    - Supersession detection (mark older memories as superseded)

    Uses blocking + LSH strategy for O(n) complexity instead of O(n²),
    with batch LLM calls to gemini-flash for cheap, fast classification.

    WHEN TO USE:
    - After significant work to clean up duplicate entities
    - Before querying to improve retrieval quality
    - Periodically (weekly) for active projects
    - When graph feels "messy" with redundant information

    HOW IT WORKS:
    1. Candidate generation: Uses embeddings to find similar pairs (no LLM)
    2. LLM scoring: Batch prompts to gemini-flash for classification
    3. Execution: Auto-merge high confidence (>=0.9), queue medium (0.7-0.9)

    Args:
        project_id: Project to consolidate (auto-inferred from cwd if not specified)
        operations: List of operations to run (default: all):
            - "entity_dedup": Merge duplicate entities (file paths, tool names)
            - "memory_merge": Combine near-duplicate memories
            - "supersession": Detect and mark memory supersession
        dry_run: If True, report candidates without executing changes
        confidence_threshold: Auto-merge threshold (default 0.9).
            Values >= this are auto-merged, 0.7-threshold go to review queue.

    Returns:
        Consolidation report with:
        - project_id: Project that was consolidated
        - operations_run: Which operations were performed
        - dry_run: Whether this was a preview
        - entity_dedup: {candidates_found, merges_executed, merges_queued}
        - memory_merge: {candidates_found, merges_executed, merges_queued}
        - supersession: {candidates_found, supersessions_executed, supersessions_queued}
        - errors: Any errors encountered
        - review_queue_count: Items requiring manual review

    Examples:
        # Full consolidation (auto-inferred project)
        consolidate_project()

        # Preview what would be consolidated (dry run)
        consolidate_project(dry_run=True)

        # Only deduplicate entities
        consolidate_project(operations=["entity_dedup"])

        # Lower threshold for more aggressive merging
        consolidate_project(confidence_threshold=0.85)

        # Specific project
        consolidate_project(project_id="config:simplemem")
    """
    from simplemem_lite.backend.consolidation import consolidate_project as run_consolidation

    log.info(
        f"Tool: consolidate_project called with project_id={project_id}, "
        f"operations={operations}, dry_run={dry_run}, threshold={confidence_threshold}"
    )

    try:
        # Resolve project_id if not provided
        resolved_project_id = project_id
        if not resolved_project_id:
            resolved_project_id = _deps.project_manager.get_project_id()
            if not resolved_project_id:
                return {
                    "error": "SIMPLEMEM_NOT_BOOTSTRAPPED",
                    "message": "Project not bootstrapped. Run bootstrap_project() first.",
                    "suggested_action": "bootstrap_project",
                }
            log.info(f"Auto-resolved project_id: {resolved_project_id}")

        # Run consolidation
        report = await run_consolidation(
            project_id=resolved_project_id,
            operations=operations,
            dry_run=dry_run,
            confidence_threshold=confidence_threshold,
        )

        result = report.to_dict()
        log.info(f"Tool: consolidate_project complete: {result}")
        return result

    except Exception as e:
        log.error(f"Tool: consolidate_project failed: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def get_review_queue(
    project_id: str | None = None,
    limit: int = 20,
    type_filter: str | None = None,
) -> dict[str, Any]:
    """Get pending consolidation candidates requiring human review.

    Returns candidates with confidence 0.7-0.9 that need manual approval
    before merge/supersession is executed.

    WHEN TO USE:
    - After running consolidate_project() to see what needs review
    - To check pending items before approving/rejecting
    - To understand what consolidation would do with lower thresholds

    Args:
        project_id: Filter to specific project (auto-inferred if not specified)
        limit: Maximum candidates to return (default: 20)
        type_filter: Filter by type: entity_dedup, memory_merge, supersession

    Returns:
        {
            items: List of candidates with uuid, type, confidence, reason, decision_data
            count: Number of candidates returned
            project_id: Project these candidates belong to
        }

    Examples:
        # Get all pending items for current project
        get_review_queue()

        # Get only entity deduplication candidates
        get_review_queue(type_filter="entity_dedup")

        # Get more results
        get_review_queue(limit=50)
    """
    log.info(f"Tool: get_review_queue called with project_id={project_id}, limit={limit}, type_filter={type_filter}")

    try:
        # Resolve project_id if not provided
        resolved_project_id = project_id
        if not resolved_project_id:
            resolved_project_id = _deps.project_manager.get_project_id()
            if not resolved_project_id:
                return {
                    "error": "SIMPLEMEM_NOT_BOOTSTRAPPED",
                    "message": "Project not bootstrapped. Run bootstrap_project() first.",
                }

        candidates = _deps.store.db.get_review_candidates(
            project_id=resolved_project_id,
            status="pending",
            type_filter=type_filter,
            limit=limit,
        )

        return {
            "items": candidates,
            "count": len(candidates),
            "project_id": resolved_project_id,
        }

    except Exception as e:
        log.error(f"Tool: get_review_queue failed: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def approve_review_item(candidate_id: str) -> dict[str, Any]:
    """Approve and execute a pending consolidation candidate.

    Executes the merge/supersession that was queued for review.
    Idempotent - approving twice returns success without re-executing.

    WHEN TO USE:
    - After reviewing a candidate from get_review_queue()
    - When you agree the items should be merged/superseded

    Args:
        candidate_id: UUID of the ReviewCandidate to approve

    Returns:
        {
            status: "approved" | "already_resolved" | "error"
            candidate_id: The candidate that was approved
            type: entity_dedup | memory_merge | supersession
        }

    Examples:
        # Approve a specific candidate
        approve_review_item(candidate_id="abc-123-def")
    """
    import time

    log.info(f"Tool: approve_review_item called with candidate_id={candidate_id}")

    try:
        candidate = _deps.store.db.get_review_candidate(candidate_id)

        if not candidate:
            return {
                "error": "CANDIDATE_NOT_FOUND",
                "candidate_id": candidate_id,
            }

        if candidate["status"] != "pending":
            return {
                "status": "already_resolved",
                "candidate_id": candidate_id,
                "previous_status": candidate["status"],
            }

        decision_data = candidate["decision_data"]

        # Execute the merge based on type
        if candidate["type"] == "entity_dedup":
            entity_a = decision_data.get("entity_a_name")
            entity_b = decision_data.get("entity_b_name")
            canonical = decision_data.get("canonical_name") or entity_a
            entity_type = decision_data.get("entity_type", "file")
            deprecated = entity_b if entity_a == canonical else entity_a

            log.info(f"Executing entity merge: {deprecated} -> {canonical}")

            _deps.store.db.graph.query(
                """
                MATCH (m)-[r]->(e:Entity {name: $deprecated_name, type: $type})
                MATCH (canonical:Entity {name: $canonical_name, type: $type})
                CREATE (m)-[r2:REFERENCES]->(canonical)
                DELETE r
                """,
                {
                    "deprecated_name": deprecated,
                    "canonical_name": canonical,
                    "type": entity_type,
                },
            )

            _deps.store.db.graph.query(
                """
                MATCH (e:Entity {name: $name, type: $type})
                WHERE NOT EXISTS((e)-[]-())
                DELETE e
                """,
                {"name": deprecated, "type": entity_type},
            )

        elif candidate["type"] == "memory_merge":
            mem_a_uuid = decision_data.get("memory_a_uuid")
            mem_b_uuid = decision_data.get("memory_b_uuid")
            merged_content = decision_data.get("merged_content")

            result = _deps.store.db.graph.query(
                """
                MATCH (a:Memory {uuid: $uuid_a})
                MATCH (b:Memory {uuid: $uuid_b})
                RETURN a.created_at, b.created_at
                """,
                {"uuid_a": mem_a_uuid, "uuid_b": mem_b_uuid},
            )

            if result.result_set:
                time_a, time_b = result.result_set[0]
                if (time_a or 0) >= (time_b or 0):
                    newer_uuid, older_uuid = mem_a_uuid, mem_b_uuid
                else:
                    newer_uuid, older_uuid = mem_b_uuid, mem_a_uuid
            else:
                newer_uuid, older_uuid = mem_a_uuid, mem_b_uuid

            log.info(f"Executing memory merge: {older_uuid[:8]}... -> {newer_uuid[:8]}...")

            if merged_content:
                _deps.store.db.graph.query(
                    """
                    MATCH (m:Memory {uuid: $uuid})
                    SET m.content = $content,
                        m.merged_from = $older_uuid
                    """,
                    {
                        "uuid": newer_uuid,
                        "content": merged_content,
                        "older_uuid": older_uuid,
                    },
                )

            _deps.store.db.mark_merged(older_uuid, newer_uuid)

        elif candidate["type"] == "supersession":
            newer_uuid = decision_data.get("newer_uuid")
            older_uuid = decision_data.get("older_uuid")
            supersession_type = decision_data.get("supersession_type", "full_replace")

            log.info(f"Executing supersession: {newer_uuid[:8]}... supersedes {older_uuid[:8]}...")

            _deps.store.db.add_supersession(
                newer_uuid=newer_uuid,
                older_uuid=older_uuid,
                confidence=candidate["confidence"],
                supersession_type=supersession_type,
            )

        _deps.store.db.update_candidate_status(candidate_id, "approved", int(time.time()))

        return {
            "status": "approved",
            "candidate_id": candidate_id,
            "type": candidate["type"],
        }

    except Exception as e:
        log.error(f"Tool: approve_review_item failed: {e}", exc_info=True)
        return {"error": str(e), "candidate_id": candidate_id}


@mcp.tool()
async def reject_review_item(
    candidate_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Reject a candidate and mark pair to skip in future runs.

    Creates a REJECTED_PAIR edge so future consolidation runs will
    automatically skip this pair.

    WHEN TO USE:
    - After reviewing a candidate from get_review_queue()
    - When you disagree that items should be merged/superseded
    - When the suggested merge is incorrect

    Args:
        candidate_id: UUID of the ReviewCandidate to reject
        reason: Optional reason for rejection (stored for reference)

    Returns:
        {
            status: "rejected" | "already_resolved" | "error"
            candidate_id: The candidate that was rejected
            type: entity_dedup | memory_merge | supersession
            reason: The reason provided (if any)
        }

    Examples:
        # Reject a candidate
        reject_review_item(candidate_id="abc-123-def")

        # Reject with reason
        reject_review_item(candidate_id="abc-123-def", reason="These are actually different concepts")
    """
    import time

    log.info(f"Tool: reject_review_item called with candidate_id={candidate_id}, reason={reason}")

    try:
        candidate = _deps.store.db.get_review_candidate(candidate_id)

        if not candidate:
            return {
                "error": "CANDIDATE_NOT_FOUND",
                "candidate_id": candidate_id,
            }

        if candidate["status"] != "pending":
            return {
                "status": "already_resolved",
                "candidate_id": candidate_id,
                "previous_status": candidate["status"],
            }

        # Add rejected pair edge for future skip
        _deps.store.db.add_rejected_pair(
            candidate["source_id"],
            candidate["target_id"],
            candidate_id,
        )

        _deps.store.db.update_candidate_status(candidate_id, "rejected", int(time.time()))

        return {
            "status": "rejected",
            "candidate_id": candidate_id,
            "type": candidate["type"],
            "reason": reason,
        }

    except Exception as e:
        log.error(f"Tool: reject_review_item failed: {e}", exc_info=True)
        return {"error": str(e), "candidate_id": candidate_id}


@mcp.tool()
async def reindex_memories(
    project_id: str | None = None,
    background: bool = True,
) -> dict[str, Any]:
    """Re-generate embeddings for memories in a project.

    PURPOSE: Fix embedding model mismatches or regenerate missing embeddings.
    Use when vector search returns empty despite memories existing in graph.

    This is necessary when:
    - Memories were embedded with one model (e.g., MiniLM) but searches use another (e.g., Gemini)
    - Embeddings were deleted from LanceDB but memories still exist in Memgraph
    - Switching embedding models and need to re-embed all content

    WHEN TO USE:
    - search_memories returns empty for a project you know has data
    - After changing SIMPLEMEM_EMBEDDING_MODEL config
    - After manually deleting vectors from LanceDB

    Args:
        project_id: Project to reindex. Uses current project if not specified.
                    MUST be provided for cloud/remote operations.
        background: Run in background job (default: True). Large projects
                    may take minutes; background prevents timeout.

    Returns:
        If background=True: {"job_id": "...", "status": "submitted"}
        If background=False: {"reindexed": N, "errors": 0, "project_id": "..."}

    Examples:
        # Reindex 3dtex project synchronously
        reindex_memories(project_id="config:3dtex", background=False)

        # Reindex in background (for large projects)
        reindex_memories(project_id="config:3dtex", background=True)
        # Check progress with: job_status(job_id="...")
    """
    log.info(f"Tool: reindex_memories called with project_id={project_id}, background={background}")

    # Resolve project_id if not provided
    resolved_project_id = project_id
    if not resolved_project_id:
        resolved_project_id = await _resolve_project_id(None)
        if not resolved_project_id:
            return {
                "error": "PROJECT_ID_REQUIRED",
                "message": "project_id is required. Provide explicitly or ensure you're in a bootstrapped project.",
            }

    log.info(f"Tool: reindex_memories resolved project_id={resolved_project_id}")

    if background:
        # Submit to background job manager
        async def _reindex_task(pid: str) -> dict:
            """Background job wrapper for reindex_memories."""
            log.info(f"Background job: reindex_memories starting for project {pid}")
            result = _deps.store.reindex_memories(pid)
            log.info(f"Background job: reindex_memories complete: {result.get('reindexed', 0)} reindexed")
            return result

        job_id = await _deps.job_manager.submit("reindex_memories", _reindex_task, resolved_project_id)
        log.info(f"Tool: reindex_memories submitted as background job {job_id}")
        return {
            "job_id": job_id,
            "status": "submitted",
            "project_id": resolved_project_id,
            "message": f"Reindex submitted. Check progress with job_status('{job_id}')",
        }

    # Synchronous execution
    try:
        result = _deps.store.reindex_memories(resolved_project_id)
        return result
    except Exception as e:
        log.error(f"Tool: reindex_memories failed: {e}", exc_info=True)
        return {"error": str(e), "project_id": resolved_project_id}


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCES (4 Browsable Data Sources)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.resource("memory://recent")
def list_recent_memories() -> str:
    """Browse 20 most recent memories."""
    memories = _deps.store.list_recent(limit=20)
    return json.dumps(
        [
            {
                "uuid": m.uuid,
                "type": m.type,
                "preview": m.content[:100],
                "created_at": m.created_at,
            }
            for m in memories
        ],
        indent=2,
    )


@mcp.resource("memory://{uuid}")
def get_memory_resource(uuid: str) -> str:
    """Read a specific memory and its relationships."""
    memory = _deps.store.get(uuid)
    if memory is None:
        return json.dumps({"error": "Memory not found"})

    related = _deps.store.get_related(uuid, hops=1)

    return json.dumps(
        {
            "memory": _memory_to_dict(memory),
            "related": [_memory_to_dict(r) for r in related],
        },
        indent=2,
    )


@mcp.resource("traces://sessions")
def list_trace_sessions() -> str:
    """Browse available Claude Code session traces."""
    sessions = _deps.parser.list_sessions()[:50]
    return json.dumps(sessions, indent=2)


@mcp.resource("graph://explore/{uuid}")
def explore_graph(uuid: str) -> str:
    """2-hop graph exploration from a memory."""
    connections = _deps.store.get_related(uuid, hops=2)
    return json.dumps([_memory_to_dict(c) for c in connections], indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# P1: ENTITY-CENTRIC RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.resource("entities://files")
def list_file_entities() -> str:
    """Browse all tracked files with action counts.

    Returns files sorted by activity (reads + modifies), with:
    - name: File path (canonicalized)
    - version: Current version (incremented on modifies)
    - reads: Number of read operations
    - modifies: Number of modify operations
    - sessions_count: Number of sessions this file appeared in
    """
    files = _deps.store.db.get_entities_by_type("file", limit=100)
    return json.dumps(files, indent=2)


@mcp.resource("entities://files/{name}")
def get_file_history(name: str) -> str:
    """Get complete history of a specific file.

    Returns:
    - entity: File metadata (name, version, timestamps)
    - memories: All memories linked to this file with actions
    - sessions: List of sessions that touched this file
    - related_errors: Errors encountered while working with this file
    """
    history = _deps.store.db.get_entity_history(name, "file", limit=50)
    return json.dumps(history, indent=2)


@mcp.resource("entities://tools")
def list_tool_usage() -> str:
    """Browse tool usage patterns across sessions.

    Returns tools sorted by execution count, with:
    - tool: Tool name (canonicalized)
    - executions: Number of times executed
    - sessions_count: Number of sessions that used this tool
    """
    tools = _deps.store.db.get_tool_usage(limit=100)
    return json.dumps(tools, indent=2)


@mcp.resource("entities://tools/{name}")
def get_tool_history(name: str) -> str:
    """Get complete history of a specific tool.

    Returns:
    - entity: Tool metadata
    - memories: All memories where this tool was executed
    - sessions: Sessions that used this tool
    - related_errors: Errors triggered by this tool
    """
    history = _deps.store.db.get_entity_history(name, "tool", limit=50)
    return json.dumps(history, indent=2)


@mcp.resource("entities://errors")
def list_error_patterns() -> str:
    """Browse common errors and their occurrence patterns.

    Returns errors sorted by occurrence count, with:
    - error: Error name/pattern (canonicalized)
    - occurrences: Number of times this error was triggered
    - sessions_count: Number of sessions that encountered this error
    """
    errors = _deps.store.db.get_error_patterns(limit=100)
    return json.dumps(errors, indent=2)


@mcp.resource("entities://errors/{pattern}")
def get_error_history(pattern: str) -> str:
    """Get complete history of a specific error pattern.

    Returns:
    - entity: Error metadata
    - memories: All memories where this error was triggered
    - sessions: Sessions that encountered this error
    """
    history = _deps.store.db.get_entity_history(pattern, "error", limit=50)
    return json.dumps(history, indent=2)


@mcp.resource("insights://cross-session")
def get_cross_session_insights() -> str:
    """Browse entities that appear across multiple sessions.

    These are bridge entities - valuable for cross-session insights
    as they link different work sessions together.

    Returns entities sorted by session count (appearing in 2+ sessions), with:
    - name: Entity name
    - type: Entity type (file, tool, error, command)
    - sessions_count: Number of sessions this entity appeared in
    - session_ids: List of session IDs
    """
    entities = _deps.store.db.get_cross_session_entities(min_sessions=2, limit=50)
    return json.dumps(entities, indent=2)


@mcp.resource("insights://project/{project_id}")
def get_project_insights(project_id: str) -> str:
    """Get all learnings and insights for a specific project.

    Returns:
    - project: Project metadata (path, session count, timestamps)
    - sessions: Session summaries belonging to this project
    - top_files: Most frequently touched files
    - errors: Errors encountered in this project
    """
    insights = _deps.store.db.get_project_insights(project_id, limit=50)
    return json.dumps(insights, indent=2)


@mcp.resource("insights://projects")
def list_projects() -> str:
    """Browse all tracked projects.

    Returns projects sorted by session count.
    """
    projects = _deps.store.db.get_projects(limit=50)
    return json.dumps(projects, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS (6 Workflow-Integrated Templates)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.prompt(title="Start with Context")
def start_with_context(task: str) -> str:
    """Gather relevant context before starting a task."""
    return f'''Before starting: "{task}"

1. Use ask_memories to find relevant prior work:
   ask_memories("context for {task}")

2. Check for related files we've worked with before

3. Look for similar problems and their solutions

Synthesize: What do we already know that helps here?

Example output format:
"Based on memories [1][2], this task relates to prior work on X.
Key insight: [specific pattern or solution that worked].
Files involved: [relevant files from past sessions]."'''


@mcp.prompt(title="Smart Debug")
def smart_debug(error: str, file_context: str = "") -> str:
    """Debug using full memory graph and cross-session insights."""
    file_note = f"\nContext: Error occurred in {file_context}" if file_context else ""
    return f'''Debugging error: {error}{file_note}

Steps:
1. Use ask_memories("solutions for: {error}") to get LLM-synthesized answer with citations

2. Check for similar error patterns across sessions - cross-session insights are especially valuable

3. If file context provided, look for that file's modification history

4. Provide solution based on what worked before, citing specific memories [1][2]

Focus on actionable solutions with evidence from past sessions.'''


@mcp.prompt(title="Store Session Learnings")
def store_session_learnings() -> str:
    """Capture key insights from the current session."""
    return '''Before ending, preserve what we learned:

Identify and store each of:
1. **Problem solved**: What was the main challenge?
2. **Solution that worked**: What approach succeeded?
3. **Gotchas discovered**: Any non-obvious issues?
4. **Patterns worth remembering**: Reusable techniques?

For each insight, use store_memory with:
- Descriptive content capturing the learning
- type="lesson_learned" for cross-session discovery
- Include relevant file/tool/command context

Example:
store_memory(
    text="Fixed auth token expiry by checking refresh logic before API calls. Key: always validate token age, not just presence.",
    type="lesson_learned"
)'''


@mcp.prompt(title="What Do We Know About")
def entity_recall(entity_name: str, entity_type: str = "file") -> str:
    """Recall everything we know about a specific entity."""
    return f'''Retrieve all knowledge about {entity_type}: {entity_name}

1. Search memories mentioning this entity:
   search_memories("{entity_name}")

2. Use reason_memories for deeper connections:
   reason_memories("history and patterns for {entity_name}")

3. Check cross-session patterns - has this entity appeared in other projects?

Synthesize into a brief summary:
- What operations were performed (reads/modifies)?
- What issues were encountered?
- What solutions worked?
- Any patterns or best practices?'''


@mcp.prompt(title="Reflect on Session")
def reflect_on_session(session_id: str) -> str:
    """Index and analyze a Claude Code session."""
    return f'''Analyze Claude Code session {session_id}:

1. First, index the session:
   process_trace(session_id="{session_id}")

2. Then explore what was learned:
   ask_memories("key learnings from session {session_id}")

3. Identify patterns worth preserving:
   - Problems solved and their solutions
   - Errors encountered and fixes applied
   - Files modified and why
   - Reusable patterns discovered

Store significant learnings as type="lesson_learned" for future reference.'''


@mcp.prompt(title="Project Overview")
def project_overview(project: str) -> str:
    """Comprehensive summary of project knowledge."""
    return f'''Summarize knowledge about project: "{project}"

1. Get synthesized overview:
   ask_memories("overview of {project} project")

2. Find cross-session patterns:
   reason_memories("patterns and decisions in {project}")

3. Compile:
   - **Architecture**: Key design decisions
   - **Common Issues**: Recurring problems and solutions
   - **Patterns**: Coding conventions and best practices
   - **Dependencies**: Key integrations and their quirks

Focus on actionable insights that help when working on this project.'''


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _memory_to_dict(memory: Memory) -> dict:
    """Convert Memory to JSON-serializable dict."""
    return {
        "uuid": memory.uuid,
        "content": memory.content,
        "type": memory.type,
        "created_at": memory.created_at,
        "score": memory.score,
        "session_id": memory.session_id,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Run the MCP server with HTTP hook support.

    Supports two transport modes:
    - stdio (default): For local MCP clients (Claude Desktop, Claude Code)
    - sse: For remote MCP clients over HTTP (Fly.io deployment)

    Set SIMPLEMEM_TRANSPORT=sse for remote hosting.
    """
    global _http_server, _http_thread

    transport = os.environ.get("SIMPLEMEM_TRANSPORT", "stdio").lower()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    log.info(f"Starting SimpleMem Lite server (transport={transport})...")

    # Start HTTP server for hook communication (only for stdio mode)
    # In SSE mode, hooks would need a different mechanism
    if transport == "stdio":
        result = _start_http_server()
        if result:
            _http_server, _http_thread = result
            log.info("HTTP hook server ready")
        else:
            log.warning("HTTP hook server not started (hooks will not work)")

    # Run MCP server (blocks until shutdown)
    log.info(f"Starting MCP server run loop (transport={transport})")
    try:
        if transport == "sse":
            log.info(f"SSE endpoint: http://{host}:{port}/sse")
            mcp.run(transport="sse", host=host, port=port)
        else:
            mcp.run()
    finally:
        log.info("MCP server stopped")
        if transport == "stdio":
            _stop_http_server()


if __name__ == "__main__":
    main()
