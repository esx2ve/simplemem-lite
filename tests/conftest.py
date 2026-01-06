"""Shared pytest fixtures for SimpleMem-Lite tests."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_embeddings():
    """Mock embedding function with controlled vectors for deterministic tests.

    Returns vectors that allow precise control over cosine similarity:
    - Identical vectors: similarity = 1.0
    - Orthogonal vectors: similarity = 0.0
    - Similar vectors: 0.85-0.95 similarity
    """
    EMBEDDING_MAP = {
        # Entity names - similar pairs
        "main.py": [1.0, 0.0, 0.0, 0.0],
        "./main.py": [0.99, 0.12, 0.0, 0.0],  # ~0.99 similarity with main.py
        "src/main.py": [0.95, 0.31, 0.0, 0.0],  # ~0.95 similarity
        "test.py": [0.0, 1.0, 0.0, 0.0],  # Orthogonal to main.py
        "utils.py": [0.0, 0.0, 1.0, 0.0],  # Orthogonal
        # Memory content - various similarities
        "debug database connection": [0.8, 0.6, 0.0, 0.0],
        "debugging database connections": [0.79, 0.61, 0.0, 0.0],  # ~0.99
        "fix authentication bug": [0.0, 0.8, 0.6, 0.0],  # Different topic
        "improve login flow": [0.1, 0.75, 0.65, 0.0],  # ~0.95 with auth
        # Edge cases
        "": [0.0, 0.0, 0.0, 0.0],  # Zero vector
        "unique_entity": [0.5, 0.5, 0.5, 0.5],  # No similar pair
    }

    def _embed_batch(texts: list[str]) -> list[list[float]]:
        """Return embeddings for given texts, with fallback."""
        results = []
        for text in texts:
            if text in EMBEDDING_MAP:
                results.append(EMBEDDING_MAP[text])
            else:
                # Default embedding based on hash (deterministic but "random")
                h = hash(text) % 1000
                results.append([
                    (h % 100) / 100,
                    ((h // 10) % 100) / 100,
                    ((h // 100) % 100) / 100,
                    0.5,
                ])
        return results

    return _embed_batch


@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses."""
    def _create_response(content: str):
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message.content = content
        return mock
    return _create_response


@pytest.fixture
def mock_llm_entity_dedup(mock_llm_response):
    """Mock LLM for entity deduplication scoring."""
    responses = {
        "same": '[{"pair": 1, "same": true, "confidence": 0.95, "canonical": "main.py", "reason": "Same file, different paths"}]',
        "different": '[{"pair": 1, "same": false, "confidence": 0.1, "canonical": null, "reason": "Unrelated entities"}]',
        "multiple": '''[
            {"pair": 1, "same": true, "confidence": 0.95, "canonical": "main.py", "reason": "Same file"},
            {"pair": 2, "same": false, "confidence": 0.2, "canonical": null, "reason": "Different"}
        ]''',
    }

    async def _respond(scenario: str = "same", *args, **kwargs):
        return mock_llm_response(responses.get(scenario, responses["same"]))

    return _respond


@pytest.fixture
def mock_llm_memory_merge(mock_llm_response):
    """Mock LLM for memory merge scoring."""
    responses = {
        "merge": '[{"pair": 1, "merge": true, "confidence": 0.92, "merged": "Combined insight about database connections and debugging", "reason": "Nearly identical content"}]',
        "no_merge": '[{"pair": 1, "merge": false, "confidence": 0.3, "merged": null, "reason": "Different topics"}]',
    }

    async def _respond(scenario: str = "merge", *args, **kwargs):
        return mock_llm_response(responses.get(scenario, responses["merge"]))

    return _respond


@pytest.fixture
def mock_llm_supersession(mock_llm_response):
    """Mock LLM for supersession scoring."""
    responses = {
        "supersedes": '[{"pair": 1, "supersedes": true, "confidence": 0.88, "type": "full_replace", "reason": "Newer provides updated solution"}]',
        "partial": '[{"pair": 1, "supersedes": true, "confidence": 0.75, "type": "partial_update", "reason": "Newer adds context"}]',
        "none": '[{"pair": 1, "supersedes": false, "confidence": 0.1, "type": "none", "reason": "Different aspects"}]',
    }

    async def _respond(scenario: str = "supersedes", *args, **kwargs):
        return mock_llm_response(responses.get(scenario, responses["supersedes"]))

    return _respond


@pytest.fixture
def mock_graph_store():
    """Mock graph database store for testing without Memgraph."""
    store = MagicMock()
    store.db = MagicMock()
    store.db.graph = MagicMock()

    # Default empty result set
    empty_result = MagicMock()
    empty_result.result_set = []
    store.db.graph.query = MagicMock(return_value=empty_result)

    return store


@pytest.fixture
def sample_entities() -> list[dict[str, Any]]:
    """Sample entity data for testing."""
    return [
        {"name": "main.py", "type": "file"},
        {"name": "./main.py", "type": "file"},
        {"name": "src/main.py", "type": "file"},
        {"name": "test.py", "type": "file"},
        {"name": "utils.py", "type": "file"},
        {"name": "requests", "type": "tool"},
        {"name": "http_client", "type": "tool"},
    ]


@pytest.fixture
def sample_memories() -> list[dict[str, Any]]:
    """Sample memory data for testing."""
    return [
        {
            "uuid": "mem-001",
            "content": "debug database connection",
            "type": "lesson_learned",
            "created_at": 1704067200,  # 2024-01-01
            "session_id": "sess-1",
        },
        {
            "uuid": "mem-002",
            "content": "debugging database connections",
            "type": "lesson_learned",
            "created_at": 1704153600,  # 2024-01-02
            "session_id": "sess-2",
        },
        {
            "uuid": "mem-003",
            "content": "fix authentication bug",
            "type": "lesson_learned",
            "created_at": 1704240000,  # 2024-01-03
            "session_id": "sess-3",
        },
        {
            "uuid": "mem-004",
            "content": "improve login flow",
            "type": "decision",
            "created_at": 1704326400,  # 2024-01-04
            "session_id": "sess-4",
        },
    ]


@pytest.fixture
def project_id() -> str:
    """Standard test project ID."""
    return "config:test-project"
