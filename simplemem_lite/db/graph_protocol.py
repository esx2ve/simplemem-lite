"""Graph database protocol for SimpleMem Lite.

Defines abstract interface for graph database backends (FalkorDB, KuzuDB).
Enables runtime switching between backends based on environment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class QueryResult:
    """Unified query result from any graph backend.

    Attributes:
        result_set: List of result rows (each row is a list of values)
        header: Column names if available
        stats: Query statistics (nodes created, relationships created, etc.)
    """
    result_set: list[list[Any]]
    header: list[str] | None = None
    stats: dict[str, Any] | None = None

    def __iter__(self):
        """Allow iteration over result set."""
        return iter(self.result_set)

    def __len__(self):
        """Return number of result rows."""
        return len(self.result_set)

    def __bool__(self):
        """Check if result has any rows."""
        return len(self.result_set) > 0


@runtime_checkable
class GraphBackend(Protocol):
    """Protocol for graph database backends.

    All graph backends must implement these methods to be compatible
    with SimpleMem's DatabaseManager.
    """

    @property
    def backend_name(self) -> str:
        """Return the backend name (e.g., 'falkordb', 'kuzu')."""
        ...

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> QueryResult:
        """Execute a Cypher query.

        Args:
            cypher: Cypher query string
            params: Optional query parameters (use $param syntax)

        Returns:
            QueryResult with result_set, header, and stats
        """
        ...

    def health_check(self) -> bool:
        """Check if the backend is healthy and connected.

        Returns:
            True if backend is operational
        """
        ...

    def close(self) -> None:
        """Close the database connection and release resources."""
        ...

    def init_schema(self) -> None:
        """Initialize the graph schema (tables, indexes).

        Called once on first connection. Should be idempotent.
        """
        ...


class BaseGraphBackend(ABC):
    """Abstract base class for graph backends with common functionality."""

    GRAPH_NAME = "simplemem"

    # Node labels
    LABEL_MEMORY = "Memory"
    LABEL_ENTITY = "Entity"
    LABEL_CODE_CHUNK = "CodeChunk"
    LABEL_PROJECT_INDEX = "ProjectIndex"
    LABEL_GOAL = "Goal"
    LABEL_PROJECT = "Project"

    # Relationship types
    REL_RELATES_TO = "RELATES_TO"
    REL_REFERENCES = "REFERENCES"
    REL_READS = "READS"
    REL_MODIFIES = "MODIFIES"
    REL_EXECUTES = "EXECUTES"
    REL_TRIGGERED = "TRIGGERED"
    REL_CONTAINS = "CONTAINS"
    REL_CHILD_OF = "CHILD_OF"
    REL_FOLLOWS = "FOLLOWS"
    REL_HAS_GOAL = "HAS_GOAL"
    REL_ACHIEVES = "ACHIEVES"
    REL_BELONGS_TO = "BELONGS_TO"

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name."""
        pass

    @abstractmethod
    def query(self, cypher: str, params: dict[str, Any] | None = None) -> QueryResult:
        """Execute a Cypher query."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check backend health."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    def init_schema(self) -> None:
        """Initialize schema."""
        pass

    def reinit_code_chunk_indexes(self) -> None:
        """Drop and recreate CodeChunk indexes.

        CRITICAL for FalkorDB: Must call after mass-deleting CodeChunk nodes
        to prevent SIGSEGV crashes in Schema_AddNodeToIndex.

        Default implementation is a no-op (not all backends need this).
        """
        pass

    def _translate_cypher(self, cypher: str) -> str:
        """Translate Cypher dialect differences if needed.

        Override in subclasses for dialect-specific translations.

        Args:
            cypher: Standard Cypher query

        Returns:
            Backend-specific Cypher query
        """
        return cypher
