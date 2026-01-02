"""SimpleMem-Lite Backend API.

FastAPI backend that handles all heavy processing:
- Memory operations (store, search, reason)
- Trace processing (receives trace content, not paths)
- Code indexing (receives file contents)
- LanceDB + KuzuDB database operations

Can run locally (default) or deployed to cloud (Fly.io).
"""

from simplemem_lite.backend.app import create_app

__all__ = ["create_app"]
