"""Factory functions for creating test graph data.

These factories create consistent test data for consolidation testing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock


def create_entity(name: str, entity_type: str = "file") -> dict[str, Any]:
    """Create an entity dict for testing."""
    return {"name": name, "type": entity_type}


def create_memory(
    uuid: str,
    content: str,
    memory_type: str = "fact",
    created_at: float | None = None,
    session_id: str = "test-session",
) -> dict[str, Any]:
    """Create a memory dict for testing."""
    return {
        "uuid": uuid,
        "content": content,
        "type": memory_type,
        "created_at": created_at or datetime.now().timestamp(),
        "session_id": session_id,
    }


def create_entity_pair_result(
    entities: list[tuple[str, str]]
) -> MagicMock:
    """Create a mock query result for entity queries.

    Args:
        entities: List of (name, type) tuples

    Returns:
        Mock result with result_set attribute
    """
    result = MagicMock()
    result.result_set = [[name, entity_type] for name, entity_type in entities]
    return result


def create_memory_query_result(
    memories: list[dict[str, Any]]
) -> MagicMock:
    """Create a mock query result for memory queries.

    Args:
        memories: List of memory dicts

    Returns:
        Mock result with result_set attribute
    """
    result = MagicMock()
    result.result_set = [
        [
            m["uuid"],
            m["content"],
            m["type"],
            m["created_at"],
            m.get("session_id", "test-session"),
        ]
        for m in memories
    ]
    return result


def create_shared_entity_result(entity_names: list[str]) -> MagicMock:
    """Create a mock query result for shared entity queries."""
    result = MagicMock()
    result.result_set = [[name] for name in entity_names]
    return result


def create_empty_result() -> MagicMock:
    """Create an empty mock query result."""
    result = MagicMock()
    result.result_set = []
    return result


# Pre-built test scenarios


def duplicate_file_entities() -> list[tuple[str, str]]:
    """File entities with likely duplicates."""
    return [
        ("main.py", "file"),
        ("./main.py", "file"),
        ("src/main.py", "file"),
        ("test.py", "file"),
        ("utils.py", "file"),
    ]


def similar_memories() -> list[dict[str, Any]]:
    """Memories that are candidates for merging."""
    base_time = datetime.now().timestamp()
    return [
        create_memory(
            "mem-001",
            "Fixed database connection timeout by increasing pool size",
            "lesson_learned",
            base_time,
        ),
        create_memory(
            "mem-002",
            "Database connection pool timeout fixed with larger pool",
            "lesson_learned",
            base_time + 3600,  # 1 hour later
        ),
        create_memory(
            "mem-003",
            "Implemented user authentication with JWT tokens",
            "decision",
            base_time + 7200,  # 2 hours later
        ),
    ]


def supersession_candidates() -> list[dict[str, Any]]:
    """Memories that are candidates for supersession."""
    base_time = datetime.now().timestamp()
    return [
        create_memory(
            "mem-old",
            "Initial approach to database caching",
            "lesson_learned",
            base_time - 86400 * 7,  # 7 days ago
        ),
        create_memory(
            "mem-new",
            "Updated caching strategy after testing",
            "lesson_learned",
            base_time,
        ),
    ]


class MockGraphStore:
    """Configurable mock graph store for testing."""

    def __init__(self):
        self.db = MagicMock()
        self.db.graph = MagicMock()
        self.db.mark_merged = MagicMock()
        self.db.add_supersession = MagicMock()
        self._query_responses = {}

    def set_entity_response(self, entities: list[tuple[str, str]]):
        """Configure entity query response."""
        self._query_responses["entity"] = create_entity_pair_result(entities)

    def set_memory_response(self, memories: list[dict[str, Any]]):
        """Configure memory query response."""
        self._query_responses["memory"] = create_memory_query_result(memories)

    def set_shared_entity_response(self, entities: list[str]):
        """Configure shared entity query response."""
        self._query_responses["shared"] = create_shared_entity_result(entities)

    def configure_query_routing(self):
        """Set up query routing based on configured responses."""
        def route_query(query: str, params: dict) -> MagicMock:
            if "e.name AS name, e.type AS type" in query:
                return self._query_responses.get("entity", create_empty_result())
            elif "m.uuid, m.content, m.type" in query:
                return self._query_responses.get("memory", create_empty_result())
            elif "DISTINCT e.name" in query:
                return self._query_responses.get("shared", create_empty_result())
            return create_empty_result()

        self.db.graph.query.side_effect = route_query
        return self
