# SimpleMem-Lite Backend

Cloud backend API for SimpleMem - provides memory storage, vector search, and graph reasoning.

## Architecture

```
┌─────────────────┐         ┌──────────────────────────┐
│  Claude Code    │         │  SimpleMem-Lite Backend  │
│                 │  HTTPS  │  (Fly.io)                │
│  simplemem-mcp  │ ──────► │                          │
│  (thin client)  │         │  - LanceDB (vectors)     │
│                 │         │  - KuzuDB (graph)        │
└─────────────────┘         │  - LiteLLM (embeddings)  │
                            └──────────────────────────┘
```

**For users**: Install `simplemem-mcp` - the lightweight MCP client:
```bash
uvx simplemem-mcp serve
```

**This repo**: The backend API that powers SimpleMem Cloud.

## Deployment

### Fly.io (Production)

```bash
# Deploy
fly deploy

# Check status
fly status

# View logs
fly logs
```

### Local Development

```bash
# Install dependencies
uv sync --all-extras

# Run backend locally
uvicorn simplemem_lite.backend.app:app --port 8420

# Or use the CLI
simplemem serve
```

## API Endpoints

### Memories
- `POST /api/v1/memories/store` - Store a memory
- `POST /api/v1/memories/search` - Hybrid vector + graph search
- `POST /api/v1/memories/relate` - Create relationships
- `POST /api/v1/memories/ask` - LLM-synthesized answers
- `POST /api/v1/memories/reason` - Multi-hop graph reasoning
- `GET /api/v1/memories/stats` - Memory statistics

### Traces
- `POST /api/v1/traces/process` - Process Claude Code session
- `GET /api/v1/traces/job/{job_id}` - Job status
- `GET /api/v1/traces/jobs` - List jobs
- `POST /api/v1/traces/job/{job_id}/cancel` - Cancel job

### Code
- `POST /api/v1/code/index` - Index code files
- `POST /api/v1/code/update` - Incremental update
- `POST /api/v1/code/search` - Semantic code search
- `GET /api/v1/code/stats` - Code index statistics

### Graph
- `GET /api/v1/graph/schema` - Graph schema
- `POST /api/v1/graph/query` - Execute Cypher query

### Health
- `GET /health` - Health check

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8420` | Server port |
| `SIMPLEMEM_LITE_DATA_DIR` | `~/.simplemem_lite` | Database storage |
| `SIMPLEMEM_LITE_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `SIMPLEMEM_LITE_SUMMARY_MODEL` | `gemini/gemini-2.0-flash-lite` | Summary model |

## Development

```bash
# Run tests
pytest tests/

# Type checking
pyright

# Linting
ruff check .
```

## License

MIT
