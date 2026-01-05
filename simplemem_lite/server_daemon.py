"""MCP Server for SimpleMem Lite (Daemon-backed).

Thin MCP server that proxies all operations to the singleton daemon.
This eliminates concurrency issues when multiple MCP servers run simultaneously.

Architecture:
    Claude Code 1 --> MCP Server 1 --\
                                      +--> Daemon --> LanceDB + FalkorDB
    Claude Code 2 --> MCP Server 2 --/
"""

import atexit
import json
import os
import secrets
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from simplemem_lite.client import DaemonClient, DaemonError, ConnectionError
from simplemem_lite.config import Config
from simplemem_lite.daemon_manager import DaemonManager, ensure_daemon_running
from simplemem_lite.log_config import get_logger

log = get_logger("server")

# Initialize config and daemon manager
log.info("SimpleMem Lite MCP server starting (daemon-backed)...")
log.debug(f"HOME={os.environ.get('HOME', 'NOT SET')}")

config = Config()
log.debug("Config initialized")

# Ensure daemon is running before accepting requests
log.info("Ensuring daemon is running...")
daemon_manager = DaemonManager()
daemon_result = daemon_manager.ensure_running()
log.info(f"Daemon status: {daemon_result}")

# Create shared client for this server
_client: DaemonClient | None = None


async def get_client() -> DaemonClient:
    """Get the daemon client, creating if needed.

    Validates connection with a ping and reconnects if stale.
    Also ensures daemon is running if connection fails.
    """
    global _client

    if _client is None:
        # Ensure daemon is running before first connect
        daemon_manager.ensure_running()
        _client = DaemonClient()
        return _client

    # Validate existing connection with ping
    try:
        await _client.ping()
        return _client
    except Exception as e:
        log.warning(f"Client connection stale, reconnecting: {e}")
        try:
            await _client.close()
        except Exception:
            pass

        # Ensure daemon is running before reconnect
        daemon_manager.ensure_running()
        _client = DaemonClient()
        return _client


def _cleanup_client() -> None:
    """Cleanup client on server shutdown."""
    global _client
    if _client is not None:
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_client.close())
            loop.close()
        except Exception as e:
            log.debug(f"Error closing client: {e}")
        _client = None
    log.info("Client cleaned up")


atexit.register(_cleanup_client)


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP SERVER FOR HOOK COMMUNICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Server state
_http_server: HTTPServer | None = None
_http_thread: threading.Thread | None = None
_auth_token: str | None = None


def _get_lock_file_path() -> Path:
    """Get path to the lock file."""
    return config.data_dir / "server.lock"


def _write_lock_file(port: int, token: str) -> None:
    """Write lock file with server info."""
    lock_data = {
        "port": port,
        "pid": os.getpid(),
        "token": token,
        "started_at": datetime.now().isoformat(),
        "host": config.http_host,
    }
    lock_path = _get_lock_file_path()
    lock_path.write_text(json.dumps(lock_data, indent=2))
    log.info(f"Lock file written: {lock_path}")


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


class HookHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for hook endpoints."""

    server: "HookHTTPServer"

    def log_message(self, format: str, *args) -> None:
        """Override to use our logger."""
        log.debug(f"HTTP: {format % args}")

    def _send_json_response(self, status: int, data: dict) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _verify_auth(self) -> bool:
        """Verify auth token from request."""
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token == self.server.auth_token:
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
        if self.path == "/health":
            self._send_json_response(200, {
                "status": "ok",
                "pid": os.getpid(),
                "uptime_seconds": (datetime.now() - self.server.started_at).total_seconds(),
                "mode": "daemon-backed",
            })
        else:
            self._send_json_response(404, {"error": "Not found"})

    def do_POST(self) -> None:
        """Handle POST requests."""
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
        else:
            self._send_json_response(404, {"error": "Not found"})

    def _handle_session_start(self, body: dict) -> None:
        """Handle session-start hook via daemon."""
        import asyncio

        cwd = body.get("cwd", "")
        session_id = body.get("session_id", "")
        log.info(f"Hook: session-start cwd={cwd}, session={session_id[:8] if session_id else 'none'}...")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                client = DaemonClient()
                result = loop.run_until_complete(
                    client.get_project_status(cwd)
                )
                loop.run_until_complete(client.close())

                self._send_json_response(200, {
                    "status": "ok",
                    "project_root": cwd,
                    "session_id": session_id,
                    **result,
                })
            finally:
                loop.close()
        except Exception as e:
            log.error(f"Session-start hook error: {e}")
            self._send_json_response(200, {
                "status": "error",
                "error": str(e),
            })

    def _handle_stop(self, body: dict) -> None:
        """Handle stop hook via daemon."""
        import asyncio

        session_id = body.get("session_id", "")
        log.info(f"Hook: stop session={session_id[:8] if session_id else 'none'}...")

        if not session_id:
            self._send_json_response(200, {
                "status": "ok",
                "message": "No session_id provided",
            })
            return

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                client = DaemonClient()
                result = loop.run_until_complete(
                    client.process_trace(session_id)
                )
                loop.run_until_complete(client.close())

                self._send_json_response(200, {
                    "status": "ok",
                    "session_id": session_id,
                    **result,
                })
            finally:
                loop.close()
        except Exception as e:
            log.error(f"Stop hook error: {e}")
            self._send_json_response(200, {
                "status": "error",
                "session_id": session_id,
                "error": str(e),
            })


class HookHTTPServer(HTTPServer):
    """HTTP server with auth token storage."""

    def __init__(self, server_address: tuple, handler_class: type, auth_token: str):
        super().__init__(server_address, handler_class)
        self.auth_token = auth_token
        self.started_at = datetime.now()


def _start_http_server() -> tuple[HTTPServer, threading.Thread] | None:
    """Start HTTP server in a background thread."""
    global _auth_token

    if not config.http_enabled:
        log.info("HTTP server disabled by config")
        return None

    _auth_token = secrets.token_urlsafe(32)

    try:
        server = HookHTTPServer(
            (config.http_host, config.http_port),
            HookHTTPHandler,
            _auth_token,
        )
        actual_port = server.server_address[1]
        log.info(f"HTTP server bound to {config.http_host}:{actual_port}")

        _write_lock_file(actual_port, _auth_token)

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

    if _http_server is not None:
        log.info("Stopping HTTP server...")
        _http_server.shutdown()
        _http_server = None

    if _http_thread is not None:
        _http_thread.join(timeout=5.0)
        _http_thread = None

    _remove_lock_file()
    log.info("HTTP server stopped")


atexit.register(_stop_http_server)

# Create MCP server
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
# TOOLS - All proxy to daemon
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
    client = await get_client()
    result = await client.store_memory(
        text=text,
        type=type,
        source=source,
        relations=relations,
        project_id=project_id,
    )
    log.info(f"Tool: store_memory complete: {result.get('uuid', '')[:8]}...")
    return result.get("uuid", "")


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
    client = await get_client()
    result = await client.search_memories(
        query=query,
        limit=limit,
        use_graph=use_graph,
        type_filter=type_filter,
        project_id=project_id,
    )
    results = result.get("results", [])
    log.info(f"Tool: search_memories complete: {len(results)} results")
    return results


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
    client = await get_client()
    result = await client.relate_memories(from_id, to_id, relation_type)
    log.info(f"Tool: relate_memories complete: {result.get('success')}")
    return result.get("success", False)


@mcp.tool()
async def process_trace(session_id: str, ctx: Context) -> dict:
    """Index a Claude Code session trace with hierarchical summaries.

    Creates a hierarchy of memories:
    - session_summary (1) - Overall session summary
    - chunk_summary (5-15) - Summaries of activity chunks

    Uses cheap LLM (flash-lite) for summarization with progress updates.

    Args:
        session_id: Session UUID to index
        ctx: MCP context for progress reporting

    Returns:
        {session_summary_id, chunk_count, message_count} or error
    """
    log.info(f"Tool: process_trace called (session_id={session_id})")

    await ctx.report_progress(0, 100)
    await ctx.info(f"Starting to process session {session_id[:8]}...")

    client = await get_client()
    result = await client.process_trace(session_id)

    if result.get("error"):
        log.error(f"Tool: process_trace failed: {result['error']}")
        return result

    await ctx.report_progress(100, 100)
    log.info(f"Tool: process_trace complete: {result.get('chunk_count', 0)} chunks")
    return result


@mcp.tool()
async def get_stats() -> dict:
    """Get memory store statistics.

    Returns:
        {total_memories, total_relations, types_breakdown}
    """
    log.info("Tool: get_stats called")
    client = await get_client()
    result = await client.get_stats()
    log.info(f"Tool: get_stats complete: {result.get('total_memories', 0)} memories")
    return result


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

    client = await get_client()
    result = await client.reset_all(confirm=True)
    log.warning(f"Tool: reset_all complete: {result}")
    return result


@mcp.tool()
async def reason_memories(
    query: str,
    max_hops: int = 2,
    min_score: float = 0.1,
    project_id: str | None = None,
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
        project_id: Optional project identifier for cross-project isolation

    Returns:
        {conclusions: [{uuid, content, type, score, proof_chain, hops}]}
    """
    log.info(f"Tool: reason_memories called (query='{query[:50]}...', max_hops={max_hops}, project_id={project_id})")
    client = await get_client()
    result = await client.reason_memories(query, max_hops, min_score, project_id)
    conclusions = result.get("conclusions", [])
    log.info(f"Tool: reason_memories complete: {len(conclusions)} conclusions")
    return result


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
    log.info(f"Tool: ask_memories called (query='{query[:50]}...', project_id={project_id})")
    client = await get_client()
    result = await client.ask_memories(query, max_memories, max_hops, project_id)
    log.info(f"Tool: ask_memories complete: confidence={result.get('confidence')}")
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
    """Search the code index for relevant code snippets.

    Use this to find code implementations, patterns, or specific functionality.
    Results are ranked by semantic similarity to the query.

    Args:
        query: Natural language description of what you're looking for
        limit: Maximum number of results (default: 10)
        project_id: Optional - filter to specific project

    Returns:
        List of matching code chunks with file paths and line numbers
    """
    log.info(f"Tool: search_code called (query={query[:50]}..., limit={limit})")
    client = await get_client()
    result = await client.search_code(query, limit, project_id)
    results = result.get("results", [])
    log.info(f"Tool: search_code complete: {len(results)} results")
    return {"results": results, "count": len(results)}


@mcp.tool()
async def index_directory(
    path: str,
    patterns: list[str] | None = None,
    clear_existing: bool = True,
) -> dict:
    """Index a directory for code search.

    Scans the directory for source files matching the patterns,
    splits them into semantic chunks, and adds to the search index.

    Args:
        path: Directory path to index
        patterns: Optional glob patterns (default: ['**/*.py', '**/*.ts', '**/*.js', '**/*.tsx', '**/*.jsx'])
        clear_existing: Whether to clear existing index for this directory (default: True)

    Returns:
        Indexing statistics including files indexed and chunks created
    """
    log.info(f"Tool: index_directory called (path={path})")
    client = await get_client()
    result = await client.index_directory(path, patterns, clear_existing)
    log.info(f"Tool: index_directory complete: {result.get('files_indexed', 0)} files, {result.get('chunks_created', 0)} chunks")
    return result


@mcp.tool()
async def code_stats(project_id: str | None = None) -> dict:
    """Get statistics about the code index.

    Args:
        project_id: Optional - filter to specific project

    Returns:
        Statistics including chunk count and unique files
    """
    log.info(f"Tool: code_stats called (project_id={project_id})")
    client = await get_client()
    result = await client.code_stats(project_id)
    log.info(f"Tool: code_stats complete: {result}")
    return result


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
    client = await get_client()
    result = await client.code_related_memories(chunk_uuid, limit)
    log.info(f"Tool: code_related_memories complete: {result.get('count', 0)} memories found")
    return result


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
    client = await get_client()
    result = await client.memory_related_code(memory_uuid, limit)
    log.info(f"Tool: memory_related_code complete: {result.get('count', 0)} chunks found")
    return result


@mcp.tool()
async def check_code_staleness(project_id: str) -> dict:
    """Check if the code index is stale and needs refreshing.

    Uses git to detect changes since last indexing. Returns staleness status
    with details about changed files.

    Args:
        project_id: Project identifier

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
    log.info(f"Tool: check_code_staleness called (project={project_id})")
    client = await get_client()
    result = await client.check_code_staleness(project_id)
    log.info(f"Tool: check_code_staleness complete: is_stale={result.get('is_stale')}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FILE WATCHER TOOLS
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
    client = await get_client()
    result = await client.start_code_watching(project_root)
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
    client = await get_client()
    result = await client.stop_code_watching(project_root)
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
    client = await get_client()
    result = await client.get_watcher_status()
    log.info(f"Tool: get_watcher_status complete: watching={result.get('watching', 0)} projects")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT BOOTSTRAP TOOLS
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
    client = await get_client()
    result = await client.bootstrap_project(project_root, index_code, start_watcher)
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
    client = await get_client()
    result = await client.get_project_status(project_root)
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
    client = await get_client()
    result = await client.set_project_preference(project_root, never_ask)
    log.info(f"Tool: set_project_preference complete")
    return result


@mcp.tool()
async def list_tracked_projects() -> dict:
    """List all tracked projects.

    Returns:
        List of projects with their bootstrap status
    """
    log.info("Tool: list_tracked_projects called")
    client = await get_client()
    result = await client.list_tracked_projects()
    log.info(f"Tool: list_tracked_projects complete: {result.get('count', 0)} projects")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCES (minimal set for daemon mode)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.resource("memory://stats")
def get_memory_stats() -> str:
    """Get memory store statistics."""
    import asyncio

    async def fetch():
        client = DaemonClient()
        try:
            return await client.get_stats()
        finally:
            await client.close()

    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(fetch())
        loop.close()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("daemon://status")
def get_daemon_status() -> str:
    """Get daemon status and health."""
    status = daemon_manager.get_status()
    return json.dumps(status, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.prompt(title="Start with Context")
def start_with_context(task: str) -> str:
    """Gather relevant context before starting a task."""
    return f'''Before starting: "{task}"

1. Use ask_memories to find relevant prior work:
   ask_memories("context for {task}")

2. Check for related files we've worked with before

3. Look for similar problems and their solutions

Synthesize: What do we already know that helps here?'''


@mcp.prompt(title="Smart Debug")
def smart_debug(error: str, file_context: str = "") -> str:
    """Debug using full memory graph and cross-session insights."""
    file_note = f"\nContext: Error occurred in {file_context}" if file_context else ""
    return f'''Debugging error: {error}{file_note}

Steps:
1. Use ask_memories("solutions for: {error}") to get LLM-synthesized answer with citations

2. Check for similar error patterns across sessions

3. If file context provided, look for that file's modification history

4. Provide solution based on what worked before, citing specific memories [1][2]'''


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
- type="lesson_learned" for cross-session discovery'''


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Run the MCP server with daemon backend."""
    global _http_server, _http_thread

    log.info("Starting SimpleMem Lite server (daemon-backed)...")

    # Start HTTP server for hook communication
    result = _start_http_server()
    if result:
        _http_server, _http_thread = result
        log.info("HTTP hook server ready")
    else:
        log.warning("HTTP hook server not started (hooks will not work)")

    # Run MCP server (blocks until shutdown)
    log.info("Starting MCP server run loop")
    try:
        mcp.run()
    finally:
        log.info("MCP server stopped")
        _stop_http_server()
        _cleanup_client()


if __name__ == "__main__":
    main()
