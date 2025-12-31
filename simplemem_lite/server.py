"""MCP Server for SimpleMem Lite.

Exposes memory operations via MCP protocol with tools, resources, and prompts.
"""

import json
import os
from dataclasses import asdict

from mcp.server.fastmcp import Context, FastMCP

from simplemem_lite.code_index import CodeIndexer
from simplemem_lite.config import Config
from simplemem_lite.logging import get_logger
from simplemem_lite.memory import Memory, MemoryItem, MemoryStore
from simplemem_lite.traces import HierarchicalIndexer, TraceParser

log = get_logger("server")

# Initialize global state
log.info("SimpleMem Lite MCP server starting...")
log.debug(f"HOME={os.environ.get('HOME', 'NOT SET')}")

config = Config()
log.debug("Config initialized")

store = MemoryStore(config)
log.debug("MemoryStore initialized")

parser = TraceParser(config.claude_traces_dir)
log.debug("TraceParser initialized")

indexer = HierarchicalIndexer(store, config)
log.debug("HierarchicalIndexer initialized")

code_indexer = CodeIndexer(store.db, config)
log.debug("CodeIndexer initialized")

# Debug info
_debug_info = {
    "HOME": os.environ.get("HOME", "NOT SET"),
    "data_dir": str(config.data_dir),
    "claude_traces_dir": str(config.claude_traces_dir),
    "traces_dir_exists": config.claude_traces_dir.exists(),
}
log.info(f"Server initialized: traces_dir={config.claude_traces_dir}, exists={config.claude_traces_dir.exists()}")

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

Use `index_directory` once per project, then `search_code` to find implementations.
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
    result = store.store(item)
    log.info(f"Tool: store_memory complete: {result[:8]}...")
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
    results = store.search(
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
    result = store.relate(from_id, to_id, relation_type)
    log.info(f"Tool: relate_memories complete: {result}")
    return result


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
    log.debug(f"Using indexer with traces_dir={indexer.parser.traces_dir}")
    log.debug(f"traces_dir.exists()={indexer.parser.traces_dir.exists()}")

    await ctx.report_progress(0, 100)
    await ctx.info(f"Starting to process session {session_id[:8]}...")

    result = await indexer.index_session(session_id, ctx)

    if result is None:
        log.error(f"Tool: process_trace failed - session not found: {session_id}")
        return {"error": f"Session {session_id} not found"}

    await ctx.report_progress(100, 100)
    log.info(f"Tool: process_trace complete: {len(result.chunk_summary_ids)} chunks")
    return {
        "session_summary_id": result.session_summary_id,
        "chunk_count": len(result.chunk_summary_ids),
        "message_count": len(result.message_ids),
    }


@mcp.tool()
async def get_stats() -> dict:
    """Get memory store statistics.

    Returns:
        {total_memories, total_relations, types_breakdown}
    """
    log.info("Tool: get_stats called")
    result = store.get_stats()
    log.info(f"Tool: get_stats complete: {result['total_memories']} memories")
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

    result = store.reset_all()
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

    results = store.reason(
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

    Returns:
        {answer, memories_used, cross_session_insights, confidence, sources}
    """
    log.info(f"Tool: ask_memories called (query='{query[:50]}...')")

    result = await store.ask_memories(
        query=query,
        max_memories=max_memories,
        max_hops=max_hops,
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
    project_root: str | None = None,
) -> dict:
    """Search the code index for relevant code snippets.

    Use this to find code implementations, patterns, or specific functionality.
    Results are ranked by semantic similarity to the query.

    Args:
        query: Natural language description of what you're looking for
        limit: Maximum number of results (default: 10)
        project_root: Optional - filter to specific project directory

    Returns:
        List of matching code chunks with file paths and line numbers
    """
    log.info(f"Tool: search_code called (query={query[:50]}..., limit={limit})")
    results = code_indexer.search(query, limit, project_root)
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
    result = code_indexer.index_directory(path, patterns, clear_existing)
    log.info(f"Tool: index_directory complete: {result.get('files_indexed', 0)} files, {result.get('chunks_created', 0)} chunks")
    return result


@mcp.tool()
async def code_stats(project_root: str | None = None) -> dict:
    """Get statistics about the code index.

    Args:
        project_root: Optional - filter to specific project

    Returns:
        Statistics including chunk count and unique files
    """
    log.info(f"Tool: code_stats called (project_root={project_root})")
    stats = store.db.get_code_stats(project_root)
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
    memories = store.db.get_code_related_memories(chunk_uuid, limit)
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
    code_chunks = store.db.get_memory_related_code(memory_uuid, limit)
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
    result = code_indexer.check_staleness(project_root)
    log.info(f"Tool: check_code_staleness complete: is_stale={result.get('is_stale')}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCES (4 Browsable Data Sources)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.resource("memory://recent")
def list_recent_memories() -> str:
    """Browse 20 most recent memories."""
    memories = store.list_recent(limit=20)
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
    memory = store.get(uuid)
    if memory is None:
        return json.dumps({"error": "Memory not found"})

    related = store.get_related(uuid, hops=1)

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
    sessions = parser.list_sessions()[:50]
    return json.dumps(sessions, indent=2)


@mcp.resource("graph://explore/{uuid}")
def explore_graph(uuid: str) -> str:
    """2-hop graph exploration from a memory."""
    connections = store.get_related(uuid, hops=2)
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
    files = store.db.get_entities_by_type("file", limit=100)
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
    history = store.db.get_entity_history(name, "file", limit=50)
    return json.dumps(history, indent=2)


@mcp.resource("entities://tools")
def list_tool_usage() -> str:
    """Browse tool usage patterns across sessions.

    Returns tools sorted by execution count, with:
    - tool: Tool name (canonicalized)
    - executions: Number of times executed
    - sessions_count: Number of sessions that used this tool
    """
    tools = store.db.get_tool_usage(limit=100)
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
    history = store.db.get_entity_history(name, "tool", limit=50)
    return json.dumps(history, indent=2)


@mcp.resource("entities://errors")
def list_error_patterns() -> str:
    """Browse common errors and their occurrence patterns.

    Returns errors sorted by occurrence count, with:
    - error: Error name/pattern (canonicalized)
    - occurrences: Number of times this error was triggered
    - sessions_count: Number of sessions that encountered this error
    """
    errors = store.db.get_error_patterns(limit=100)
    return json.dumps(errors, indent=2)


@mcp.resource("entities://errors/{pattern}")
def get_error_history(pattern: str) -> str:
    """Get complete history of a specific error pattern.

    Returns:
    - entity: Error metadata
    - memories: All memories where this error was triggered
    - sessions: Sessions that encountered this error
    """
    history = store.db.get_entity_history(pattern, "error", limit=50)
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
    entities = store.db.get_cross_session_entities(min_sessions=2, limit=50)
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
    insights = store.db.get_project_insights(project_id, limit=50)
    return json.dumps(insights, indent=2)


@mcp.resource("insights://projects")
def list_projects() -> str:
    """Browse all tracked projects.

    Returns projects sorted by session count.
    """
    projects = store.db.get_projects(limit=50)
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
    """Run the MCP server."""
    log.info("Starting MCP server run loop")
    mcp.run()
    log.info("MCP server stopped")


if __name__ == "__main__":
    main()
