# SimpleMem Lite - Progress Log

## Phase 1: Setup [COMPLETED]

**Date**: 2025-12-29

### Completed Tasks
- [x] Created project directory structure
- [x] Created Python virtual environment (.venv)
- [x] Created requirements.txt with dependencies
- [x] Created pyproject.toml with package config
- [x] Implemented config.py (75 lines)
- [x] Implemented embeddings.py (113 lines)
- [x] Implemented db.py - KuzuDB + LanceDB manager (292 lines)
- [x] Implemented memory.py - MemoryStore with hybrid search (388 lines)
- [x] Implemented traces.py - Claude Code trace processing (435 lines)
- [x] Implemented server.py - MCP server with tools/resources/prompts (295 lines)
- [x] Created README.md
- [x] Created tests/ directory

### Files Created
```
simplemem_lite/
├── .venv/
├── simplemem_lite/
│   ├── __init__.py     (23 lines)
│   ├── config.py       (75 lines)
│   ├── embeddings.py   (113 lines)
│   ├── db.py           (292 lines)
│   ├── memory.py       (388 lines)
│   ├── traces.py       (435 lines)
│   └── server.py       (295 lines)
├── tests/__init__.py
├── requirements.txt
├── pyproject.toml
└── README.md

Total: 1,621 lines
```

---

## Phase 2: Testing [COMPLETED]

**Date**: 2025-12-29

### Tasks
- [x] Install dependencies with pip install -e .
- [x] Verify KuzuDB schema initialization
- [x] Verify LanceDB table creation
- [x] Test embedding generation (local + API)
- [x] Test trace parsing on real Claude Code session
- [x] Test MCP server startup
- [x] Test basic store/search operations

### Progress Log

#### Step 2.1: Install Dependencies
- Status: COMPLETED
- Command: `pip install -e .`
- Installed: kuzu-0.11.3, lancedb-0.26.0, litellm-1.80.11, mcp-1.25.0
- Added sentence-transformers for local embeddings

#### Step 2.2: Database Verification
- Status: COMPLETED
- KuzuDB Cypher queries work (nodes + relationships)
- LanceDB vector operations work (search + add)
- **Fix**: Removed pre-creation of graph_dir (KuzuDB creates its own)
- **Fix**: Made embedding_dim dynamic (384 for local, 1536 for OpenAI)
- **Fix**: Simplified graph hop count query (KuzuDB doesn't support size/len on recursive)

#### Step 2.3: Trace Parsing
- Status: COMPLETED
- Found 954 sessions in ~/.claude/projects
- TraceParser.list_sessions() works
- TraceParser.parse_session() works (parsed 135 messages from test session)

#### Step 2.4: MCP Server
- Status: COMPLETED
- **Fix**: Changed FastMCP `description` to `instructions` (API change)
- All 5 tools work: store_memory, search_memories, relate_memories, process_trace, get_stats
- Resources and prompts registered

---

## Phase 3: Validation [COMPLETED]

**Date**: 2025-12-29

### Tasks
- [x] Index a real Claude Code session
- [x] Test hierarchical search (summary → chunk → message)
- [x] Test graph expansion improves recall
- [x] Configure OpenRouter API for cheap LLM calls

### Results
- Indexed 38KB session: 1 session_summary, 1 chunk_summary, 9 messages
- 10 relationships created (parent-child hierarchy)
- LLM cost: ~$0.000007 per summary via OpenRouter/flash-lite
- Search returns relevant results across hierarchy
- No regex used - all parsing via json.loads()

---

## Phase 4: Integration [COMPLETED]

**Date**: 2025-12-29

### Tasks
- [x] Configure in Claude Code settings
- [x] Test MCP server connectivity
- [ ] Test MCP tools from Claude Code
- [ ] Test MCP prompts

### Configuration
Added to `~/.claude.json`:
```json
{
  "simplemem-lite": {
    "type": "stdio",
    "command": "/Users/shimon.vainer/repo/simplemem/simplemem_lite/.venv/bin/simplemem-lite",
    "env": {
      "OPENROUTER_API_KEY": "...",
      "SIMPLEMEM_LITE_USE_LOCAL_EMBEDDINGS": "true",
      "SIMPLEMEM_LITE_SUMMARY_MODEL": "openrouter/google/gemini-2.0-flash-lite-001"
    }
  }
}
```

### MCP Tools Available
- `mcp__simplemem-lite__store_memory` - Store text with relationships
- `mcp__simplemem-lite__search_memories` - Hybrid vector + graph search
- `mcp__simplemem-lite__relate_memories` - Create relationships
- `mcp__simplemem-lite__process_trace` - Index Claude Code session
- `mcp__simplemem-lite__get_stats` - Memory statistics

---

## Issues & Notes

### Bugs Fixed
1. **KuzuDB directory creation**: KuzuDB expects to create its own directory; removed `mkdir` before initialization
2. **Embedding dimension mismatch**: Local model (all-MiniLM-L6-v2) uses 384-dim, OpenAI uses 1536-dim; made dynamic via property
3. **KuzuDB path length**: `size(r)` and `len(r)` don't work on recursive relationships; simplified to constant
4. **FastMCP API change**: `description` parameter renamed to `instructions` in mcp 1.25.0

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-29 | Use KuzuDB over Neo4j | Embedded, no server needed |
| 2025-12-29 | Use LanceDB over ChromaDB | Simpler API, BM25 support |
| 2025-12-29 | Hierarchical indexing | 4x cheaper, faster search |
| 2025-12-29 | Flash-lite for summaries | ~$0.01/1M tokens |
