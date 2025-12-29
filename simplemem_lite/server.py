"""MCP Server for SimpleMem Lite.

Exposes memory operations via MCP protocol with tools, resources, and prompts.
"""

import json
import os
from dataclasses import asdict

from mcp.server.fastmcp import Context, FastMCP

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

# Debug info
_debug_info = {
    "HOME": os.environ.get("HOME", "NOT SET"),
    "data_dir": str(config.data_dir),
    "claude_traces_dir": str(config.claude_traces_dir),
    "traces_dir_exists": config.claude_traces_dir.exists(),
}
log.info(f"Server initialized: traces_dir={config.claude_traces_dir}, exists={config.claude_traces_dir.exists()}")

# Create MCP server
mcp = FastMCP(
    "simplemem-lite",
    instructions="Minimal hybrid memory MCP server with Claude Code trace processing",
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
) -> str:
    """Store a memory with optional relationships.

    Args:
        text: The content to store
        type: Memory type (fact, session_summary, chunk_summary, message)
        source: Source of memory (user, claude_trace, extracted)
        relations: Optional list of {target_id, type} relationships

    Returns:
        UUID of stored memory
    """
    log.info(f"Tool: store_memory called (type={type}, source={source})")
    log.debug(f"Content preview: {text[:100]}...")
    item = MemoryItem(
        content=text,
        metadata={"type": type, "source": source},
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
) -> list[dict]:
    """Hybrid search combining vector similarity and graph traversal.

    Searches summaries first for efficiency, then expands via graph
    to find related memories.

    Args:
        query: Search query text
        limit: Maximum results to return
        use_graph: Whether to expand results via graph relationships
        type_filter: Optional filter by memory type

    Returns:
        List of matching memories with scores
    """
    log.info(f"Tool: search_memories called (query='{query[:50]}...', limit={limit})")
    results = store.search(
        query=query,
        limit=limit,
        use_graph=use_graph,
        type_filter=type_filter,
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
# PROMPTS (4 Interaction Templates)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.prompt(title="Reflect on Session")
def reflect_on_session(session_id: str) -> str:
    """Analyze a Claude Code session and extract learnings."""
    return f"""Analyze Claude Code session {session_id} and extract learnings.

Steps:
1. First, use the process_trace tool to index this session
2. Then search for key insights using search_memories

Extract and store:
- Key problems solved
- Solutions that worked
- Errors encountered and how they were fixed
- Patterns worth remembering

Store each learning as a separate memory with appropriate relationships."""


@mcp.prompt(title="Debug with History")
def debug_with_history(error: str) -> str:
    """Debug using past solutions from memory."""
    return f"""I'm encountering this error:

{error}

Please:
1. Search memories for similar errors using search_memories
2. Look for past solutions and their context
3. Check if there are related memories via graph connections
4. Suggest solutions based on what worked before"""


@mcp.prompt(title="Project Summary")
def summarize_project(project: str) -> str:
    """Summarize everything known about a project."""
    return f"""Summarize what we know about the "{project}" project based on stored memories.

Use search_memories to find relevant information about:
- Key architectural decisions
- Common issues and solutions
- Important patterns or conventions
- Dependencies and integrations

Provide a comprehensive overview of the project knowledge."""


@mcp.prompt(title="Find Similar Solutions")
def find_similar_solutions(problem: str) -> str:
    """Search for past solutions to similar problems."""
    return f"""I need to solve this problem:

{problem}

Please:
1. Search memories for similar problems using search_memories
2. Expand search via graph to find related solutions
3. Show me the most relevant past solutions with their context
4. Suggest which approach might work best for my current situation"""


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
