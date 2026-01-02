"""Thin MCP layer for SimpleMem-Lite.

This package provides a thin MCP server that:
1. Reads local files on-demand (traces, code files)
2. Compresses payloads with gzip + base64
3. Proxies requests to the backend API (local or cloud)
4. Auto-starts local backend if not running (configurable)

Architecture:
    MCP (always local) → Backend API (local OR cloud)
    - MCP handles: stdio protocol, local file access, compression, auto-start
    - Backend handles: LanceDB vectors, KuzuDB graph, LLM calls

Configuration:
    - SIMPLEMEM_BACKEND_URL: Backend URL (default: http://localhost:8420)
    - SIMPLEMEM_AUTO_START: Enable auto-start (default: true)
    - PORT: Backend port when auto-starting (default: 8420)
"""

from simplemem_lite.mcp.client import BackendClient
from simplemem_lite.mcp.launcher import BackendLauncher, ensure_backend_running, stop_backend
from simplemem_lite.mcp.local_reader import LocalReader

__all__ = [
    "BackendClient",
    "BackendLauncher",
    "LocalReader",
    "ensure_backend_running",
    "stop_backend",
]
