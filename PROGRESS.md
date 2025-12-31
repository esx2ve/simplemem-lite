# SimpleMem-Lite + Code Search Integration Progress

## Session: 2024-12-31

### Goal
Integrate LLM-queryable codebase index into SimpleMem-Lite infrastructure (not as separate "coco_*" module).

### Architecture Decision (from consensus)
- **CocoIndex paradigm**: "What IS the code" (current state via semantic search)
- **SimpleMem paradigm**: "What we LEARNED" (history of decisions, debugging, patterns)
- **Unified MCP server**: Single interface with query router

### Phase Plan
| Phase | Description | Status |
|-------|-------------|--------|
| P0 | Manual routing - add code search tool | ✅ Complete |
| P1 | Entity linking shared namespace | ✅ Complete |
| P2 | Staleness detection via git hash | ✅ Complete |
| P3 | Auto-capture background worker | ⏳ Pending |
| P4 | Smart routing with classifier | ⏳ Pending |

---

## Progress Log

### 03:15 - Session Start
- Searched SimpleMem for prior context (found 5 relevant memories from earlier session)
- Key insights recalled:
  - Symbol binding requires AST-based anchoring
  - Context budgeting: <4k tokens
  - Automatic learning capture is killer feature

### 03:17 - MCP Connection Issue
- SimpleMem-lite failed to reconnect initially
- Diagnosed: Server works fine, was transient issue
- Stored fact: MCP config is at ~/.claude.json

### 03:18 - Starting P0 Implementation
- Adding code search capability to existing SimpleMem infrastructure
- Will use SentenceTransformer embeddings (same as existing)
- Target: `search_code` and `index_directory` MCP tools

### 03:19 - Bug Fixes (sub-agent)
- Fixed FalkorDB Cypher error: `EXISTS { }` not supported, replaced with `OPTIONAL MATCH`
- Location: db.py:1435 `get_memories_in_project`
- Confirmed no top_k/limit mismatch - both use `limit` consistently

### 03:20 - P0 Implementation Complete
- Created `code_index.py`: CodeIndexer class with index_directory() and search()
- Added to `db.py`:
  - `CODE_TABLE_NAME = "code_chunks"`
  - `add_code_chunks()`, `search_code()`, `clear_code_index()`, `get_code_stats()`
- Added to `config.py`:
  - `code_index_enabled`, `code_index_patterns`, `code_chunk_size`, `code_chunk_overlap`
- Added to `server.py`:
  - MCP tools: `search_code`, `index_directory`, `code_stats`
  - Updated instructions to include code search

### 03:24 - Testing
- Indexed simplemem_lite package: 10 files, 178 chunks
- Search "embedding generation" correctly returns embeddings.py
- All logging working correctly

### 03:25 - Online Code Indexing Design (pal:chat with gemini-2.5-pro)
Designed hybrid watcher + git-polling model for P1:
- **Primary**: watchdog file watcher for real-time updates
- **Secondary**: Git-based staleness check (rev-parse HEAD, git status --porcelain)

Key implementation decisions:
1. **In-process threading** with work queue (not subprocess)
   - Watchdog observers in separate threads → put events on queue
   - Single worker thread does embedding + LanceDB writes
2. **Batch deletes** using SQL IN clause to avoid LanceDB fragmentation
3. **ProjectWatcherManager** class for lifecycle management
   - `start_watching(path)`, `stop_watching(path)`, `stop_all()`
4. **Debouncing**: 1-2 second quiet period before processing
5. **Use LiteLLM consistently** - already done via `embed_batch()`

### 03:30 - SentenceTransformer Caching Fix
- Fixed model reloading issue in `embeddings.py`
- Added `_get_local_model()` function with global cache
- **Performance improvement**: 26x speedup (6.3s → 0.03s for cached calls)
- Model loads once per session, reused for all subsequent calls

### 03:32 - P1 Entity Linking Implementation
Added entity linking to bridge code chunks ↔ memories via shared entities:

**db.py additions:**
- `add_code_chunk_node()` - creates CodeChunk nodes in FalkorDB
- `link_code_to_entity()` - creates REFERENCES edges to Entity nodes
- `get_code_related_memories()` - finds memories via shared entities
- `get_memory_related_code()` - finds code via shared entities
- Added CodeChunk indexes: uuid, filepath

**code_index.py additions:**
- `_extract_entities()` - regex-based extraction for Python/JS/TS
  - Extracts: imports, classes, functions, file references
- `_link_chunk_entities()` - creates graph nodes and links during indexing

**server.py additions:**
- MCP tool: `code_related_memories(chunk_uuid)` - find memories for code
- MCP tool: `memory_related_code(memory_uuid)` - find code for memories

### 03:36 - P1 Testing Complete
- Re-indexed 10 files → 190 chunks, 98 entities, 324 REFERENCES edges
- Verified cross-referencing:
  - Memory → Code: Found 4 embeddings.py chunks via shared file entity
  - Code → Memory: Found 1 related memory via shared file entity
- Entity types working: class, function, module, file

### 10:16 - P2 Staleness Detection Implementation
Implemented git-based staleness detection for code index:

**db.py additions:**
- `set_project_index_metadata()` - stores ProjectIndex node with commit hash, timestamps
- `get_project_index_metadata()` - retrieves project index state
- Added ProjectIndex index for efficient lookups

**code_index.py additions:**
- `_get_git_info()` - runs git rev-parse, git status --porcelain
- `_get_changed_files()` - runs git diff --name-only with pattern filtering
- `check_staleness()` - comprehensive staleness report
- Updated `index_directory()` to save metadata after indexing

**server.py additions:**
- MCP tool: `check_code_staleness(project_root)` - returns staleness status

**Code Review Fixes (gemini-2.5-pro):**
- Added missing ProjectIndex database index
- Simplified pattern matching using Path.match() instead of repetitive if/elif

**Test Results:**
- Indexed 10 files → commit hash saved
- Staleness detection correctly identifies uncommitted changes
- Non-indexed projects report as stale

---

## Implementation Notes

### Files to Modify
- `simplemem_lite/server.py` - Add new MCP tools
- `simplemem_lite/db.py` - Add code index table if needed
- `simplemem_lite/config.py` - Add code index paths

### Key Decisions
1. Use existing LanceDB for code vectors (not separate Postgres)
2. Leverage existing SentenceTransformer in embeddings.py
3. Add as optional feature (graceful degradation if no code indexed)

---

## Learnings to Store (End of Session)
- [x] SentenceTransformer caching pattern: global `_local_model` cache with lazy loading
- [x] P1 Entity linking: regex extraction → graph nodes → REFERENCES edges
- [x] Cross-referencing: bidirectional lookup via shared Entity nodes
- [x] FalkorDB: MERGE for upserts, collect(DISTINCT) for aggregation
