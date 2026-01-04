"""Singleton daemon for SimpleMem Lite.

Owns all database access (LanceDB + FalkorDB) to prevent corruption
from concurrent MCP server processes. Communicates via Unix socket.

Architecture:
    Claude Code 1 --> MCP Server 1 --\
                                      +--> Daemon --> LanceDB + FalkorDB
    Claude Code 2 --> MCP Server 2 --/

Logs:
    - ~/.simplemem_lite/logs/simplemem_YYYY-MM-DD.log (DEBUG+)
    - ~/.simplemem_lite/logs/latest.log (TRACE)
"""

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from simplemem_lite.config import Config
from simplemem_lite.job_manager import JobManager
from simplemem_lite.log_config import get_logger

log = get_logger("daemon")

# Daemon socket path
DAEMON_SOCKET_PATH = Path.home() / ".simplemem_lite" / "daemon.sock"
DAEMON_LOCK_PATH = Path.home() / ".simplemem_lite" / "daemon.lock"


class SimplememDaemon:
    """Singleton daemon that owns all database access.

    Provides JSON-RPC interface over Unix socket for MCP servers.
    """

    def __init__(self, config: Config | None = None):
        """Initialize the daemon with all subsystems.

        Args:
            config: Optional config, uses defaults if not provided
        """
        log.info("Initializing SimplememDaemon...")
        self.config = config or Config()
        self._shutdown_event = asyncio.Event()
        self._server: asyncio.Server | None = None
        self._client_count = 0
        self._client_lock = asyncio.Lock()  # Protect client count access
        self._request_count = 0
        self._start_time = time.time()

        # Lazy initialization of subsystems (done in start())
        self._memory_store = None
        self._code_indexer = None
        self._watcher_manager = None
        self._project_manager = None
        self._bootstrap = None
        self._trace_indexer = None
        self._job_manager = None

        # Method dispatch table
        self._handlers: dict[str, callable] = {}

        log.debug(f"Daemon config: data_dir={self.config.data_dir}")

    def _init_subsystems(self) -> None:
        """Initialize all subsystems (called on first request or startup)."""
        if self._memory_store is not None:
            log.trace("Subsystems already initialized")
            return

        log.info("Initializing daemon subsystems...")

        # Import here to avoid circular imports and defer heavy initialization
        from simplemem_lite.memory import MemoryStore
        from simplemem_lite.code_index import CodeIndexer
        from simplemem_lite.watcher import ProjectWatcherManager
        from simplemem_lite.projects import ProjectManager
        from simplemem_lite.bootstrap import Bootstrap
        from simplemem_lite.traces import HierarchicalIndexer

        log.debug("Creating MemoryStore...")
        self._memory_store = MemoryStore(self.config)

        log.debug("Creating CodeIndexer...")
        self._code_indexer = CodeIndexer(self._memory_store.db, self.config)

        log.debug("Creating ProjectWatcherManager...")
        self._watcher_manager = ProjectWatcherManager(
            self._code_indexer,
            self.config.code_patterns_list
        )

        log.debug("Creating ProjectManager...")
        self._project_manager = ProjectManager(self.config)

        log.debug("Creating Bootstrap...")
        self._bootstrap = Bootstrap(
            self.config,
            self._project_manager,
            self._code_indexer,
            self._watcher_manager,
        )

        log.debug("Creating HierarchicalIndexer (trace indexer)...")
        self._trace_indexer = HierarchicalIndexer(self._memory_store, self.config)

        log.debug("Creating JobManager...")
        self._job_manager = JobManager()

        # Register all handlers
        self._register_handlers()

        log.info(f"Daemon subsystems initialized: {len(self._handlers)} handlers registered")

    def _register_handlers(self) -> None:
        """Register all RPC method handlers."""
        # Memory operations
        self._handlers["store_memory"] = self._handle_store_memory
        self._handlers["search_memories"] = self._handle_search_memories
        self._handlers["relate_memories"] = self._handle_relate_memories
        self._handlers["reason_memories"] = self._handle_reason_memories
        self._handlers["ask_memories"] = self._handle_ask_memories
        self._handlers["get_stats"] = self._handle_get_stats
        self._handlers["reset_all"] = self._handle_reset_all

        # Code indexing
        self._handlers["search_code"] = self._handle_search_code
        self._handlers["index_directory"] = self._handle_index_directory
        self._handlers["code_stats"] = self._handle_code_stats
        self._handlers["check_code_staleness"] = self._handle_check_code_staleness
        self._handlers["code_related_memories"] = self._handle_code_related_memories
        self._handlers["memory_related_code"] = self._handle_memory_related_code

        # Watcher operations
        self._handlers["start_code_watching"] = self._handle_start_code_watching
        self._handlers["stop_code_watching"] = self._handle_stop_code_watching
        self._handlers["get_watcher_status"] = self._handle_get_watcher_status

        # Project operations
        self._handlers["bootstrap_project"] = self._handle_bootstrap_project
        self._handlers["get_project_status"] = self._handle_get_project_status
        self._handlers["set_project_preference"] = self._handle_set_project_preference
        self._handlers["list_tracked_projects"] = self._handle_list_tracked_projects

        # Trace operations
        self._handlers["process_trace"] = self._handle_process_trace

        # Job management
        self._handlers["job_status"] = self._handle_job_status
        self._handlers["list_jobs"] = self._handle_list_jobs
        self._handlers["cancel_job"] = self._handle_cancel_job

        # Daemon-specific
        self._handlers["ping"] = self._handle_ping
        self._handlers["daemon_status"] = self._handle_daemon_status

        log.debug(f"Registered handlers: {list(self._handlers.keys())}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # HANDLER IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════════════

    async def _handle_ping(self, params: dict) -> dict:
        """Health check handler."""
        return {"status": "ok", "timestamp": time.time()}

    async def _handle_daemon_status(self, params: dict) -> dict:
        """Return daemon status and statistics."""
        return {
            "status": "running",
            "uptime": time.time() - self._start_time,
            "client_count": self._client_count,
            "request_count": self._request_count,
            "handlers": list(self._handlers.keys()),
        }

    # ═══════════════════════════════════════════════════════════════════════════════
    # JOB MANAGEMENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════════

    async def _handle_job_status(self, params: dict) -> dict:
        """Get status of a background job."""
        job_id = params["job_id"]
        log.debug(f"job_status: {job_id}")
        return await self._job_manager.get_status(job_id)

    async def _handle_list_jobs(self, params: dict) -> dict:
        """List all background jobs."""
        include_completed = params.get("include_completed", True)
        limit = params.get("limit", 20)
        log.debug(f"list_jobs: include_completed={include_completed}, limit={limit}")
        jobs = await self._job_manager.list_jobs(
            include_completed=include_completed,
            limit=limit,
        )
        return {"jobs": jobs}

    async def _handle_cancel_job(self, params: dict) -> dict:
        """Cancel a running background job."""
        job_id = params["job_id"]
        log.debug(f"cancel_job: {job_id}")
        cancelled = await self._job_manager.cancel(job_id)
        return {"cancelled": cancelled, "job_id": job_id}

    async def _handle_store_memory(self, params: dict) -> dict:
        """Store a memory."""
        log.debug(f"store_memory: type={params.get('type', 'fact')}")
        from simplemem_lite.memory import MemoryItem

        item = MemoryItem(
            content=params["text"],
            metadata={
                "type": params.get("type", "fact"),
                "source": params.get("source", "user"),
                "session_id": params.get("session_id"),
            },
            relations=[
                {"target_id": r["target_id"], "type": r.get("type", "relates")}
                for r in params.get("relations", [])
            ],
        )

        uuid = self._memory_store.store(item)
        log.info(f"Memory stored: {uuid[:8]}...")
        return {"uuid": uuid}

    async def _handle_search_memories(self, params: dict) -> dict:
        """Search memories."""
        log.debug(f"search_memories: query='{params['query'][:50]}...'")

        results = self._memory_store.search(
            query=params["query"],
            limit=params.get("limit", 10),
            use_graph=params.get("use_graph", True),
            type_filter=params.get("type_filter"),
            project_id=params.get("project_id"),
        )

        log.debug(f"search_memories: found {len(results)} results")
        return {
            "results": [
                {
                    "uuid": m.uuid,
                    "content": m.content,
                    "type": m.type,
                    "score": m.score,
                    "session_id": m.session_id,
                }
                for m in results
            ]
        }

    async def _handle_relate_memories(self, params: dict) -> dict:
        """Create relationship between memories."""
        log.debug(f"relate_memories: {params['from_id'][:8]}... -> {params['to_id'][:8]}...")

        success = self._memory_store.relate(
            from_id=params["from_id"],
            to_id=params["to_id"],
            relation_type=params.get("relation_type", "relates"),
        )
        return {"success": success}

    async def _handle_reason_memories(self, params: dict) -> dict:
        """Multi-hop reasoning over memory graph."""
        project_id = params.get("project_id")
        log.debug(f"reason_memories: query='{params['query'][:50]}...', project={project_id}")

        results = self._memory_store.reason(
            query=params["query"],
            max_hops=params.get("max_hops", 2),
            min_score=params.get("min_score", 0.1),
            project_id=project_id,
        )

        log.debug(f"reason_memories: found {len(results)} conclusions")
        return {"conclusions": results}

    async def _handle_ask_memories(self, params: dict) -> dict:
        """LLM-synthesized answer from memories."""
        project_id = params.get("project_id")
        log.debug(f"ask_memories: query='{params['query'][:50]}...', project={project_id}")

        result = await self._memory_store.ask_memories(
            query=params["query"],
            max_memories=params.get("max_memories", 8),
            max_hops=params.get("max_hops", 2),
            project_id=project_id,
        )

        log.info(f"ask_memories: confidence={result.get('confidence')}")
        return result

    async def _handle_get_stats(self, params: dict) -> dict:
        """Get memory store statistics."""
        log.trace("get_stats")
        return self._memory_store.get_stats()

    async def _handle_reset_all(self, params: dict) -> dict:
        """Reset all data (dangerous!)."""
        if not params.get("confirm"):
            log.warning("reset_all called without confirm=True")
            return {"error": "Must set confirm=True to reset"}

        log.warning("RESET_ALL: Wiping all data!")
        result = self._memory_store.reset_all()
        log.warning(f"RESET_ALL complete: {result}")
        return result

    async def _handle_search_code(self, params: dict) -> dict:
        """Search code index."""
        log.debug(f"search_code: query='{params['query'][:50]}...'")

        results = self._code_indexer.search(
            query=params["query"],
            limit=params.get("limit", 10),
            project_root=params.get("project_root"),
        )

        log.debug(f"search_code: found {len(results)} results")
        return {"results": results}

    async def _handle_index_directory(self, params: dict) -> dict:
        """Index a directory for code search.

        Runs in background by default to return immediately.
        Use job_status to check progress.
        """
        path = params["path"]
        patterns = params.get("patterns")
        clear_existing = params.get("clear_existing", True)
        background = params.get("background", True)

        log.info(f"index_directory: {path} (background={background})")

        if background:
            # Submit to background job manager - returns immediately
            async def _index_job(p: str, pats: list[str] | None, clear: bool) -> dict:
                """Background job wrapper using async indexer."""
                log.info(f"Background job: index_directory starting for {p}")
                result = await self._code_indexer.index_directory_async(p, pats, clear)
                log.info(f"Background job: index_directory complete: {result.get('files_indexed', 0)} files")
                return result

            job_id = await self._job_manager.submit(
                "index_directory", _index_job, path, patterns, clear_existing
            )
            log.info(f"index_directory submitted as background job {job_id}")
            return {
                "job_id": job_id,
                "status": "submitted",
                "message": f"Use job_status('{job_id}') to check progress",
            }

        # Synchronous execution (not recommended for large codebases)
        log.debug("index_directory running synchronously")
        result = await self._code_indexer.index_directory_async(
            path, patterns, clear_existing
        )
        log.info(f"index_directory complete: {result.get('files_indexed', 0)} files, {result.get('chunks_created', 0)} chunks")
        return result

    async def _handle_code_stats(self, params: dict) -> dict:
        """Get code index statistics."""
        log.trace("code_stats")
        return self._code_indexer.db.get_code_stats(params.get("project_root"))

    async def _handle_check_code_staleness(self, params: dict) -> dict:
        """Check if code index is stale."""
        log.debug(f"check_code_staleness: {params['project_root']}")
        return self._code_indexer.check_staleness(params["project_root"])

    async def _handle_code_related_memories(self, params: dict) -> dict:
        """Find memories related to code chunk."""
        log.debug(f"code_related_memories: chunk={params['chunk_uuid'][:8]}...")

        results = self._memory_store.db.get_code_related_memories(
            chunk_uuid=params["chunk_uuid"],
            limit=params.get("limit", 10),
        )
        return {"memories": results}

    async def _handle_memory_related_code(self, params: dict) -> dict:
        """Find code chunks related to memory."""
        log.debug(f"memory_related_code: memory={params['memory_uuid'][:8]}...")

        results = self._memory_store.db.get_memory_related_code(
            memory_uuid=params["memory_uuid"],
            limit=params.get("limit", 10),
        )
        return {"code_chunks": results}

    async def _handle_start_code_watching(self, params: dict) -> dict:
        """Start watching a project for file changes."""
        log.info(f"start_code_watching: {params['project_root']}")
        return self._watcher_manager.start_watching(params["project_root"])

    async def _handle_stop_code_watching(self, params: dict) -> dict:
        """Stop watching a project."""
        log.info(f"stop_code_watching: {params['project_root']}")
        return self._watcher_manager.stop_watching(params["project_root"])

    async def _handle_get_watcher_status(self, params: dict) -> dict:
        """Get status of all watchers."""
        log.trace("get_watcher_status")
        return self._watcher_manager.get_status()

    async def _handle_bootstrap_project(self, params: dict) -> dict:
        """Bootstrap a project for SimpleMem features.

        Runs indexing in background by default to return immediately.
        Use job_status to check progress.
        """
        project_root = params["project_root"]
        index_code = params.get("index_code", True)
        start_watcher = params.get("start_watcher", True)
        background = params.get("background", True)

        log.info(f"bootstrap_project: {project_root} (background={background})")

        if background and index_code:
            # Submit indexing to background job - return immediately
            async def _bootstrap_job(root: str, watcher: bool) -> dict:
                """Background job for bootstrap with async indexing."""
                log.info(f"Background job: bootstrap_project starting for {root}")

                # Detect project info (fast)
                info = self._bootstrap.detect_project_info(root)
                results: dict = {
                    "project_root": root,
                    "success": False,
                    "project_info": {
                        "name": info.name,
                        "type": info.project_type,
                        "description": info.description,
                        "frameworks": info.frameworks,
                        "source": info.source,
                    },
                }

                # Index code using async version
                try:
                    index_result = await self._code_indexer.index_directory_async(root)
                    results["index"] = {
                        "files_indexed": index_result.get("files_indexed", 0),
                        "chunks_created": index_result.get("chunks_created", 0),
                    }
                    log.info(f"Background job: indexed {results['index']['files_indexed']} files")
                except Exception as e:
                    log.error(f"Background job: indexing failed: {e}")
                    results["index"] = {"error": str(e)}

                # Start watcher
                if watcher:
                    try:
                        watcher_result = self._watcher_manager.start_watching(root)
                        results["watcher"] = watcher_result
                    except Exception as e:
                        log.error(f"Background job: watcher failed: {e}")
                        results["watcher"] = {"error": str(e)}

                # Mark project as bootstrapped
                self._project_manager.mark_bootstrapped(root)
                results["success"] = True
                log.info(f"Background job: bootstrap_project complete for {root}")
                return results

            job_id = await self._job_manager.submit(
                "bootstrap_project", _bootstrap_job, project_root, start_watcher
            )
            log.info(f"bootstrap_project submitted as background job {job_id}")
            return {
                "job_id": job_id,
                "status": "submitted",
                "message": f"Use job_status('{job_id}') to check progress",
                "project_root": project_root,
            }

        # Synchronous execution (not recommended)
        log.debug("bootstrap_project running synchronously")
        result = self._bootstrap.bootstrap_project(
            project_root=project_root,
            index_code=index_code,
            start_watcher=start_watcher,
        )

        log.info(f"bootstrap_project complete: {project_root}")
        return result

    async def _handle_get_project_status(self, params: dict) -> dict:
        """Get project bootstrap status."""
        log.trace(f"get_project_status: {params['project_root']}")
        return self._bootstrap.get_bootstrap_status(params["project_root"])

    async def _handle_set_project_preference(self, params: dict) -> dict:
        """Set project preference (e.g., never_ask)."""
        log.debug(f"set_project_preference: {params['project_root']}")

        if params.get("never_ask"):
            self._project_manager.set_never_ask(params["project_root"])

        return self._bootstrap.get_bootstrap_status(params["project_root"])

    async def _handle_list_tracked_projects(self, params: dict) -> dict:
        """List all tracked projects."""
        log.trace("list_tracked_projects")
        return {"projects": self._project_manager.list_projects()}

    async def _handle_process_trace(self, params: dict) -> dict:
        """Process a Claude Code session trace."""
        log.info(f"process_trace: session={params['session_id']}")

        result = await self._trace_indexer.index_session(
            session_id=params["session_id"],
        )

        if result:
            log.info(f"process_trace complete: {result.chunk_count} chunks")
            return {
                "session_summary_id": result.session_summary_id,
                "chunk_count": result.chunk_count,
                "message_count": result.message_count,
            }
        else:
            log.warning(f"process_trace failed for session {params['session_id']}")
            return {"error": "Failed to process trace"}

    # ═══════════════════════════════════════════════════════════════════════════════
    # SERVER LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════════

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a client connection."""
        client_id = id(writer)
        async with self._client_lock:
            self._client_count += 1
        peer = writer.get_extra_info("peername") or "unknown"
        log.info(f"Client connected: {client_id} from {peer}")

        try:
            while not self._shutdown_event.is_set():
                # Read length-prefixed message
                try:
                    length_bytes = await asyncio.wait_for(
                        reader.readexactly(4),
                        timeout=300.0,  # 5 minute timeout
                    )
                except asyncio.TimeoutError:
                    log.debug(f"Client {client_id} timeout, closing")
                    break
                except asyncio.IncompleteReadError:
                    log.debug(f"Client {client_id} disconnected")
                    break

                length = int.from_bytes(length_bytes, "big")
                if length > 10 * 1024 * 1024:  # 10 MB max
                    log.error(f"Client {client_id} sent oversized message: {length} bytes")
                    break

                data = await reader.readexactly(length)

                # Parse JSON-RPC request
                try:
                    request = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError as e:
                    log.error(f"Client {client_id} sent invalid JSON: {e}")
                    await self._send_error(writer, None, -32700, "Parse error")
                    continue

                log.trace(f"Request from {client_id}: {request.get('method')}")

                # Process request
                response = await self._process_request(request)

                # Send response
                response_bytes = json.dumps(response).encode("utf-8")
                writer.write(len(response_bytes).to_bytes(4, "big"))
                writer.write(response_bytes)
                await writer.drain()

                self._request_count += 1

        except Exception as e:
            log.error(f"Client {client_id} error: {e}")
        finally:
            async with self._client_lock:
                self._client_count -= 1
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            log.info(f"Client {client_id} disconnected (active: {self._client_count})")

    async def _process_request(self, request: dict) -> dict:
        """Process a JSON-RPC request."""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        log.debug(f"Processing: {method} (id={request_id})")

        # Ensure subsystems are initialized
        self._init_subsystems()

        # Find handler
        handler = self._handlers.get(method)
        if handler is None:
            log.warning(f"Unknown method: {method}")
            return self._make_error(request_id, -32601, f"Method not found: {method}")

        # Execute handler
        try:
            start_time = time.time()
            result = await handler(params)
            elapsed = time.time() - start_time
            log.debug(f"Completed: {method} in {elapsed*1000:.1f}ms")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as e:
            log.error(f"Handler error for {method}: {e}", exc_info=True)
            return self._make_error(request_id, -32000, str(e))

    def _make_error(self, request_id: Any, code: int, message: str) -> dict:
        """Create a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    async def _send_error(self, writer: asyncio.StreamWriter, request_id: Any, code: int, message: str) -> None:
        """Send an error response."""
        response = self._make_error(request_id, code, message)
        response_bytes = json.dumps(response).encode("utf-8")
        writer.write(len(response_bytes).to_bytes(4, "big"))
        writer.write(response_bytes)
        await writer.drain()

    async def start(self) -> None:
        """Start the daemon server."""
        log.info("Starting SimpleMem daemon...")

        # Initialize subsystems eagerly (before accepting connections)
        # This ensures predictable latency for all requests
        self._init_subsystems()

        # Clean up stale socket
        if DAEMON_SOCKET_PATH.exists():
            log.debug(f"Removing stale socket: {DAEMON_SOCKET_PATH}")
            DAEMON_SOCKET_PATH.unlink()

        # Create socket directory
        DAEMON_SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Start Unix socket server
        self._server = await asyncio.start_unix_server(
            self.handle_client,
            path=str(DAEMON_SOCKET_PATH),
        )

        # Set socket permissions (user-only)
        os.chmod(DAEMON_SOCKET_PATH, 0o600)

        log.info(f"Daemon listening on: {DAEMON_SOCKET_PATH}")

        # Serve until shutdown
        async with self._server:
            await self._shutdown_event.wait()

        log.info("Daemon stopped")

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        log.info("Stopping daemon...")

        # Stop watchers
        if self._watcher_manager:
            log.debug("Stopping all watchers...")
            self._watcher_manager.stop_all()

        # Signal shutdown
        self._shutdown_event.set()

        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Clean up socket
        if DAEMON_SOCKET_PATH.exists():
            DAEMON_SOCKET_PATH.unlink()

        log.info("Daemon shutdown complete")


def write_lock_file(pid: int) -> None:
    """Write daemon lock file with PID."""
    lock_data = {
        "pid": pid,
        "socket": str(DAEMON_SOCKET_PATH),
        "started_at": time.time(),
    }
    DAEMON_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    DAEMON_LOCK_PATH.write_text(json.dumps(lock_data, indent=2))
    log.debug(f"Lock file written: {DAEMON_LOCK_PATH}")


def remove_lock_file() -> None:
    """Remove daemon lock file."""
    if DAEMON_LOCK_PATH.exists():
        DAEMON_LOCK_PATH.unlink()
        log.debug(f"Lock file removed: {DAEMON_LOCK_PATH}")


async def main() -> None:
    """Daemon entry point."""
    log.info("=" * 60)
    log.info("SimpleMem Daemon starting...")
    log.info(f"PID: {os.getpid()}")
    log.info(f"Socket: {DAEMON_SOCKET_PATH}")
    log.info(f"Lock: {DAEMON_LOCK_PATH}")
    log.info("=" * 60)

    # Write lock file
    write_lock_file(os.getpid())

    daemon = SimplememDaemon()

    # Set up signal handlers
    loop = asyncio.get_running_loop()

    def handle_signal(signum: int) -> None:
        sig_name = signal.Signals(signum).name
        log.info(f"Received {sig_name}, initiating shutdown...")
        asyncio.create_task(daemon.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal, sig)

    try:
        await daemon.start()
    finally:
        remove_lock_file()
        log.info("Daemon exited")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error(f"Daemon crashed: {e}", exc_info=True)
        sys.exit(1)
