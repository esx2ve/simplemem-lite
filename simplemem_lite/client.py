"""Thin async client for SimpleMem Lite daemon.

This is the client that MCP servers use to communicate with the
singleton daemon. It provides a simple async API over JSON-RPC.

Usage:
    from simplemem_lite.client import DaemonClient

    client = DaemonClient()
    result = await client.store_memory(text="...", type="fact")
    await client.close()

    # Or use as context manager:
    async with DaemonClient() as client:
        result = await client.search_memories(query="...")
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from simplemem_lite.log_config import get_logger

log = get_logger("client")

# Socket path (must match daemon.py)
DAEMON_SOCKET_PATH = Path.home() / ".simplemem_lite" / "daemon.sock"

# Default timeouts (seconds)
DEFAULT_TIMEOUT = 30.0
LONG_TIMEOUT = 300.0  # For operations like index_directory


class DaemonError(Exception):
    """Error from daemon or communication failure."""

    def __init__(self, message: str, code: int = -1):
        super().__init__(message)
        self.code = code


class ConnectionError(DaemonError):
    """Failed to connect to daemon."""
    pass


class DaemonClient:
    """Async client for SimpleMem daemon.

    Communicates with the daemon over Unix socket using JSON-RPC 2.0.
    Thread-safe and supports concurrent calls from multiple coroutines.

    All tool methods return the result directly or raise DaemonError.
    """

    def __init__(self, socket_path: Path | str | None = None):
        """Initialize the client.

        Args:
            socket_path: Path to daemon socket (uses default if not provided)
        """
        self.socket_path = Path(socket_path) if socket_path else DAEMON_SOCKET_PATH
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()
        self._request_id = 0
        log.trace(f"DaemonClient initialized: socket={self.socket_path}")

    async def __aenter__(self) -> "DaemonClient":
        """Async context manager entry."""
        await self._ensure_connected()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_connected(self) -> None:
        """Ensure connection to daemon is established."""
        if self._writer is not None and not self._writer.is_closing():
            return

        log.debug(f"Connecting to daemon at {self.socket_path}")

        if not self.socket_path.exists():
            log.error(f"Daemon socket not found: {self.socket_path}")
            raise ConnectionError(f"Daemon socket not found: {self.socket_path}")

        try:
            self._reader, self._writer = await asyncio.open_unix_connection(
                str(self.socket_path)
            )
            log.debug("Connected to daemon")
        except Exception as e:
            log.error(f"Failed to connect to daemon: {e}")
            raise ConnectionError(f"Failed to connect to daemon: {e}")

    async def close(self) -> None:
        """Close the connection to daemon."""
        if self._writer is not None:
            log.trace("Closing daemon connection")
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as e:
                log.debug(f"Error closing connection: {e}")
            finally:
                self._writer = None
                self._reader = None

    async def call(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Any:
        """Make an RPC call to the daemon.

        Args:
            method: RPC method name
            params: Method parameters
            timeout: Request timeout in seconds

        Returns:
            Result from daemon

        Raises:
            DaemonError: On RPC error or communication failure
            asyncio.TimeoutError: On timeout
        """
        async with self._lock:
            await self._ensure_connected()

            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }

            log.debug(f"RPC call: {method} (id={self._request_id})")
            log.trace(f"RPC params: {params}")

            try:
                # Send request (length-prefixed)
                request_bytes = json.dumps(request).encode("utf-8")
                self._writer.write(len(request_bytes).to_bytes(4, "big"))
                self._writer.write(request_bytes)
                await self._writer.drain()

                # Read response (length-prefixed)
                length_bytes = await asyncio.wait_for(
                    self._reader.readexactly(4),
                    timeout=timeout,
                )
                length = int.from_bytes(length_bytes, "big")

                if length > 50 * 1024 * 1024:  # 50 MB max response
                    raise DaemonError(f"Response too large: {length} bytes")

                response_bytes = await asyncio.wait_for(
                    self._reader.readexactly(length),
                    timeout=timeout,
                )

                response = json.loads(response_bytes.decode("utf-8"))
                log.trace(f"RPC response: {response}")

                # Check for error
                if "error" in response:
                    error = response["error"]
                    log.warning(f"RPC error: {error}")
                    raise DaemonError(
                        error.get("message", "Unknown error"),
                        error.get("code", -1),
                    )

                result = response.get("result")
                log.debug(f"RPC {method} completed successfully")
                return result

            except asyncio.IncompleteReadError as e:
                log.error(f"Connection lost during RPC: {e}")
                await self.close()
                raise ConnectionError("Connection lost during RPC")

            except asyncio.TimeoutError:
                log.error(f"RPC timeout for {method} (timeout={timeout}s)")
                raise

            except json.JSONDecodeError as e:
                log.error(f"Invalid JSON response: {e}")
                raise DaemonError(f"Invalid JSON response: {e}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # DAEMON OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════════

    async def ping(self) -> dict:
        """Health check - verify daemon is responsive.

        Returns:
            {"status": "ok", "timestamp": ...}
        """
        return await self.call("ping", {}, timeout=5.0)

    async def daemon_status(self) -> dict:
        """Get daemon status and statistics.

        Returns:
            Status dict with uptime, client count, etc.
        """
        return await self.call("daemon_status", {}, timeout=5.0)

    # ═══════════════════════════════════════════════════════════════════════════════
    # MEMORY OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════════

    async def store_memory(
        self,
        text: str,
        type: str = "fact",
        source: str = "user",
        relations: list[dict] | None = None,
        project_id: str | None = None,
    ) -> dict:
        """Store a memory.

        Args:
            text: Memory content
            type: Memory type (fact, session_summary, chunk_summary, lesson_learned)
            source: Source (user, claude_trace, extracted)
            relations: Optional relationships [{target_id, type}]
            project_id: Optional project for isolation

        Returns:
            {"uuid": "..."}
        """
        params = {
            "text": text,
            "type": type,
            "source": source,
        }
        if relations:
            params["relations"] = relations
        if project_id:
            params["project_id"] = project_id

        return await self.call("store_memory", params)

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        use_graph: bool = True,
        type_filter: str | None = None,
        project_id: str | None = None,
    ) -> dict:
        """Search memories with hybrid vector + graph search.

        Args:
            query: Search query
            limit: Max results
            use_graph: Whether to expand via graph
            type_filter: Optional type filter
            project_id: Optional project filter

        Returns:
            {"results": [...]}
        """
        params = {
            "query": query,
            "limit": limit,
            "use_graph": use_graph,
        }
        if type_filter:
            params["type_filter"] = type_filter
        if project_id:
            params["project_id"] = project_id

        return await self.call("search_memories", params)

    async def relate_memories(
        self,
        from_id: str,
        to_id: str,
        relation_type: str = "relates",
    ) -> dict:
        """Create relationship between memories.

        Returns:
            {"success": True/False}
        """
        return await self.call("relate_memories", {
            "from_id": from_id,
            "to_id": to_id,
            "relation_type": relation_type,
        })

    async def reason_memories(
        self,
        query: str,
        max_hops: int = 2,
        min_score: float = 0.1,
        project_id: str | None = None,
    ) -> dict:
        """Multi-hop reasoning over memory graph.

        Returns:
            {"conclusions": [...]}
        """
        params = {
            "query": query,
            "max_hops": max_hops,
            "min_score": min_score,
        }
        if project_id:
            params["project_id"] = project_id
        return await self.call("reason_memories", params)

    async def ask_memories(
        self,
        query: str,
        max_memories: int = 8,
        max_hops: int = 2,
        project_id: str | None = None,
    ) -> dict:
        """LLM-synthesized answer from memories.

        Returns:
            {"answer": "...", "memories_used": [...], "confidence": ...}
        """
        params = {
            "query": query,
            "max_memories": max_memories,
            "max_hops": max_hops,
        }
        if project_id:
            params["project_id"] = project_id
        return await self.call("ask_memories", params, timeout=60.0)  # LLM calls can be slow

    async def get_stats(self) -> dict:
        """Get memory store statistics.

        Returns:
            {"total_memories": ..., "total_relations": ..., ...}
        """
        return await self.call("get_stats", {})

    async def reset_all(self, confirm: bool = False) -> dict:
        """Reset all data (dangerous!).

        Args:
            confirm: Must be True to proceed

        Returns:
            {"memories_deleted": ..., "relations_deleted": ...}
        """
        return await self.call("reset_all", {"confirm": confirm})

    # ═══════════════════════════════════════════════════════════════════════════════
    # CODE SEARCH OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════════

    async def search_code(
        self,
        query: str,
        limit: int = 10,
        project_root: str | None = None,
    ) -> dict:
        """Search code index.

        Returns:
            {"results": [...]}
        """
        params = {"query": query, "limit": limit}
        if project_root:
            params["project_root"] = project_root
        return await self.call("search_code", params)

    async def index_directory(
        self,
        path: str,
        patterns: list[str] | None = None,
        clear_existing: bool = True,
    ) -> dict:
        """Index a directory for code search.

        Returns:
            {"files_indexed": ..., "chunks_created": ...}
        """
        params = {"path": path, "clear_existing": clear_existing}
        if patterns:
            params["patterns"] = patterns
        return await self.call("index_directory", params, timeout=LONG_TIMEOUT)

    async def code_stats(self, project_root: str | None = None) -> dict:
        """Get code index statistics."""
        params = {}
        if project_root:
            params["project_root"] = project_root
        return await self.call("code_stats", params)

    async def check_code_staleness(self, project_root: str) -> dict:
        """Check if code index is stale."""
        return await self.call("check_code_staleness", {"project_root": project_root})

    async def code_related_memories(
        self,
        chunk_uuid: str,
        limit: int = 10,
    ) -> dict:
        """Find memories related to a code chunk."""
        return await self.call("code_related_memories", {
            "chunk_uuid": chunk_uuid,
            "limit": limit,
        })

    async def memory_related_code(
        self,
        memory_uuid: str,
        limit: int = 10,
    ) -> dict:
        """Find code chunks related to a memory."""
        return await self.call("memory_related_code", {
            "memory_uuid": memory_uuid,
            "limit": limit,
        })

    # ═══════════════════════════════════════════════════════════════════════════════
    # WATCHER OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════════

    async def start_code_watching(self, project_root: str) -> dict:
        """Start watching a project for file changes."""
        return await self.call("start_code_watching", {"project_root": project_root})

    async def stop_code_watching(self, project_root: str) -> dict:
        """Stop watching a project."""
        return await self.call("stop_code_watching", {"project_root": project_root})

    async def get_watcher_status(self) -> dict:
        """Get status of all watchers."""
        return await self.call("get_watcher_status", {})

    # ═══════════════════════════════════════════════════════════════════════════════
    # PROJECT OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════════

    async def bootstrap_project(
        self,
        project_root: str,
        index_code: bool = True,
        start_watcher: bool = True,
    ) -> dict:
        """Bootstrap a project for SimpleMem features."""
        return await self.call("bootstrap_project", {
            "project_root": project_root,
            "index_code": index_code,
            "start_watcher": start_watcher,
        }, timeout=LONG_TIMEOUT)

    async def get_project_status(self, project_root: str) -> dict:
        """Get project bootstrap status."""
        return await self.call("get_project_status", {"project_root": project_root})

    async def set_project_preference(
        self,
        project_root: str,
        never_ask: bool = True,
    ) -> dict:
        """Set project preference."""
        return await self.call("set_project_preference", {
            "project_root": project_root,
            "never_ask": never_ask,
        })

    async def list_tracked_projects(self) -> dict:
        """List all tracked projects."""
        return await self.call("list_tracked_projects", {})

    # ═══════════════════════════════════════════════════════════════════════════════
    # TRACE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════════

    async def process_trace(self, session_id: str) -> dict:
        """Process a Claude Code session trace."""
        return await self.call("process_trace", {
            "session_id": session_id,
        }, timeout=LONG_TIMEOUT)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNCHRONOUS WRAPPER (for compatibility)
# ═══════════════════════════════════════════════════════════════════════════════


class SyncDaemonClient:
    """Synchronous wrapper around DaemonClient.

    For use in non-async contexts. Creates event loop internally.
    """

    def __init__(self, socket_path: Path | str | None = None):
        self._async_client = DaemonClient(socket_path)
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run(self, coro):
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def close(self) -> None:
        self._run(self._async_client.close())
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # Forward all methods synchronously
    def ping(self) -> dict:
        return self._run(self._async_client.ping())

    def daemon_status(self) -> dict:
        return self._run(self._async_client.daemon_status())

    def store_memory(self, **kwargs) -> dict:
        return self._run(self._async_client.store_memory(**kwargs))

    def search_memories(self, **kwargs) -> dict:
        return self._run(self._async_client.search_memories(**kwargs))

    def relate_memories(self, **kwargs) -> dict:
        return self._run(self._async_client.relate_memories(**kwargs))

    def reason_memories(self, query: str, max_hops: int = 2, min_score: float = 0.1, project_id: str | None = None) -> dict:
        return self._run(self._async_client.reason_memories(query, max_hops, min_score, project_id))

    def ask_memories(self, query: str, max_memories: int = 8, max_hops: int = 2, project_id: str | None = None) -> dict:
        return self._run(self._async_client.ask_memories(query, max_memories, max_hops, project_id))

    def get_stats(self) -> dict:
        return self._run(self._async_client.get_stats())

    def reset_all(self, confirm: bool = False) -> dict:
        return self._run(self._async_client.reset_all(confirm))

    def search_code(self, **kwargs) -> dict:
        return self._run(self._async_client.search_code(**kwargs))

    def index_directory(self, **kwargs) -> dict:
        return self._run(self._async_client.index_directory(**kwargs))

    def code_stats(self, **kwargs) -> dict:
        return self._run(self._async_client.code_stats(**kwargs))

    def check_code_staleness(self, project_root: str) -> dict:
        return self._run(self._async_client.check_code_staleness(project_root))

    def code_related_memories(self, **kwargs) -> dict:
        return self._run(self._async_client.code_related_memories(**kwargs))

    def memory_related_code(self, **kwargs) -> dict:
        return self._run(self._async_client.memory_related_code(**kwargs))

    def start_code_watching(self, project_root: str) -> dict:
        return self._run(self._async_client.start_code_watching(project_root))

    def stop_code_watching(self, project_root: str) -> dict:
        return self._run(self._async_client.stop_code_watching(project_root))

    def get_watcher_status(self) -> dict:
        return self._run(self._async_client.get_watcher_status())

    def bootstrap_project(self, **kwargs) -> dict:
        return self._run(self._async_client.bootstrap_project(**kwargs))

    def get_project_status(self, project_root: str) -> dict:
        return self._run(self._async_client.get_project_status(project_root))

    def set_project_preference(self, **kwargs) -> dict:
        return self._run(self._async_client.set_project_preference(**kwargs))

    def list_tracked_projects(self) -> dict:
        return self._run(self._async_client.list_tracked_projects())

    def process_trace(self, session_id: str) -> dict:
        return self._run(self._async_client.process_trace(session_id))
