# SimpleMem Lite

Minimal hybrid memory MCP server with Claude Code trace processing.

## Features

- **Hybrid Search**: Vector similarity (LanceDB) + Graph traversal (KuzuDB)
- **Hierarchical Indexing**: Session → Chunk → Message summaries
- **MCP Native**: Tools, Resources, and Prompts
- **Zero Infrastructure**: All embedded databases, no external services

## Quick Start

```bash
# Create venv and install
cd simplemem_lite
source .venv/bin/activate
pip install -e .

# Run MCP server
simplemem-lite
```

## Configuration

Environment variables (prefix: `SIMPLEMEM_LITE_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `~/.simplemem_lite` | Database storage |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | LiteLLM embedding model |
| `SUMMARY_MODEL` | `gemini/gemini-2.0-flash-lite` | Cheap LLM for summaries |
| `CLAUDE_TRACES_DIR` | `~/.claude/projects` | Claude Code traces |
| `USE_LOCAL_EMBEDDINGS` | `false` | Use sentence-transformers |

## MCP Interface

### Tools (5)

- `store_memory` - Store text with relationships
- `search_memories` - Hybrid vector + graph search
- `relate_memories` - Create relationships
- `process_trace` - Index Claude Code session
- `get_stats` - Memory statistics

### Resources (4)

- `memory://recent` - Recent memories
- `memory://{uuid}` - Specific memory + relations
- `traces://sessions` - Available Claude sessions
- `graph://explore/{uuid}` - Graph exploration

### Prompts (4)

- `Reflect on Session` - Extract learnings from a session
- `Debug with History` - Debug using past solutions
- `Project Summary` - Summarize project knowledge
- `Find Similar Solutions` - Search for past solutions

## Architecture

```
simplemem_lite/
├── config.py      # Configuration (~50 lines)
├── embeddings.py  # LiteLLM wrapper (~80 lines)
├── db.py          # KuzuDB + LanceDB (~150 lines)
├── memory.py      # MemoryStore (~250 lines)
├── traces.py      # Trace processing (~200 lines)
└── server.py      # MCP server (~200 lines)

Total: ~930 lines
```

## Dependencies

- `mcp` - MCP protocol
- `lancedb` - Embedded vector database
- `kuzu` - Embedded graph database
- `litellm` - LLM API wrapper
- `pyarrow` - Data serialization

## License

MIT
