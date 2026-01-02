# SimpleMem Lite

Minimal hybrid memory MCP server with Claude Code trace processing.

## Features

- **Hybrid Search**: Vector similarity (LanceDB) + Graph traversal (KuzuDB/FalkorDB)
- **Hierarchical Indexing**: Session → Chunk → Message summaries
- **MCP Native**: Tools, Resources, and Prompts
- **Zero Infrastructure**: All embedded databases, no external services
- **HPC Ready**: Works on clusters without Docker (Apptainer support)

## Installation

### Quick Install with uv (Recommended)

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/shimonvainer/simplemem.git
cd simplemem/simplemem_lite

# Basic installation (verbose)
uv pip install -e . -v

# With local embeddings (sentence-transformers)
uv pip install -e ".[local]" -v

# With FalkorDB support
uv pip install -e ".[falkordb]" -v

# Full installation (all optional deps)
uv pip install -e ".[all]" -v
```

### Using uv sync (for development)

```bash
# Uses uv.lock for reproducible installs (verbose)
uv sync -v

# With optional dependencies
uv sync --extra local -v
uv sync --extra dev -v
uv sync --all-extras -v
```

### Verify Installation

After installation, run the comprehensive verification:

```bash
# Full verification with verbose output
simplemem verify

# Quick check
simplemem verify --no-verbose
```

This checks:
- ✓ Core package imports
- ✓ All dependencies (required + optional)
- ✓ Graph backends (KuzuDB, FalkorDB)
- ✓ Container runtimes (Apptainer, Docker)
- ✓ CLI entry points in PATH
- ✓ Configuration directories

Example output:
```
╭──────────────────────────────────────────╮
│            SimpleMem Lite                │
╰──────────────────────────────────────────╯
Verifying SimpleMem Installation...

Core Package
  ✓ simplemem_lite package (v0.1.0)
  ✓ Config (data_dir=~/.simplemem_lite)
  ✓ DatabaseManager
  ✓ MemoryStore
  ✓ Embeddings module
  ✓ Trace processing

Core Dependencies
  ✓ mcp (MCP protocol)
  ✓ lancedb (Vector database v0.26.0)
  ✓ kuzu (Embedded graph database v0.11.3)
  ✓ litellm (LLM API wrapper)
  ✓ typer (CLI framework v0.21.0)
  ✓ rich (Terminal output)
  ...

Optional Dependencies
  ○ falkordb (Not installed - install with: uv pip install -e '.[falkordb]')
  ○ sentence-transformers (Not installed - install with: uv pip install -e '.[local]')

Graph Backends
  ✓ KUZU (Available and working)
  Active backend: kuzu

Container Runtimes (for FalkorDB)
  ○ Apptainer (Not found - optional, for HPC)
  ○ Docker (Not found - optional)

==================================================

Verification Complete
  Passed:  20
  Failed:   0
  Warnings: 5

✓ Installation verified successfully!
```

### Alternative: pip

```bash
cd simplemem_lite
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -e . -v
```

## Quick Start

```bash
# 1. Install to Claude Code (user scope)
simplemem install

# This registers:
# - MCP server in ~/.claude.json
# - Skills in ~/.claude/skills/
# - Agents in ~/.claude/agents/
# - Auto-allow permissions

# 2. Restart Claude Code to load the MCP server

# 3. Verify installation
simplemem verify
```

### For HPC Clusters

```bash
# On HPC, specify the venv path explicitly
simplemem install --venv /path/to/simplemem/.venv

# Or with custom data directory on large storage
simplemem install --venv /path/to/venv --data-dir /weka/home-user/simplemem/data
```

### CLI Commands

```bash
simplemem --help              # Show all commands
simplemem status              # Show backend status
simplemem verify              # Verify installation
simplemem install             # Install to Claude Code
simplemem install --dry-run   # Preview installation
simplemem start               # Start FalkorDB (Apptainer/Docker)
simplemem stop                # Stop FalkorDB
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

### Core (installed by default)
- `mcp` - MCP protocol
- `lancedb` - Embedded vector database
- `kuzu` - Embedded graph database (zero-config)
- `litellm` - LLM API wrapper
- `typer` + `rich` - CLI framework

### Optional
- `sentence-transformers` - Local embeddings (`[local]`)
- `falkordb` - FalkorDB client (`[falkordb]`)

## HPC Clusters

SimpleMem works on HPC clusters without Docker using Apptainer:

```bash
# Start FalkorDB via Apptainer (auto-detected)
simplemem start

# Or explicitly use Apptainer
simplemem start --backend apptainer

# Use KuzuDB only (no container needed)
# Just don't start FalkorDB - KuzuDB is embedded
```

See [docs/APPTAINER.md](docs/APPTAINER.md) for detailed HPC setup instructions.

## License

MIT
