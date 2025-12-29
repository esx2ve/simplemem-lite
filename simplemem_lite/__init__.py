"""SimpleMem Lite - Minimal hybrid memory MCP server.

A lightweight implementation of hybrid memory (vector + graph) with:
- LanceDB for vector storage
- KuzuDB for graph relationships
- FastMCP for MCP server
- Hierarchical Claude Code trace processing
"""

__version__ = "0.1.0"

from simplemem_lite.config import Config
from simplemem_lite.memory import MemoryStore, MemoryItem, Memory
from simplemem_lite.traces import TraceParser, HierarchicalIndexer

__all__ = [
    "Config",
    "MemoryStore",
    "MemoryItem",
    "Memory",
    "TraceParser",
    "HierarchicalIndexer",
]
