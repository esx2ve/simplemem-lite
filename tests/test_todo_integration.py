"""Integration tests for Persistent Todo System.

Tests the full flow with real MemoryStore to catch issues
that unit tests with mocks miss.
"""

import tempfile
from pathlib import Path

import pytest

from simplemem_lite.config import Config
from simplemem_lite.memory import MemoryStore
from simplemem_lite.todo import TodoStore


class TestTodoIntegration:
    """Integration tests with real MemoryStore."""

    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create config with temp directories."""
        # Use temp directories to avoid polluting real data
        config = Config()
        config._data_dir = tmp_path / "data"
        config._data_dir.mkdir(parents=True, exist_ok=True)
        return config

    @pytest.fixture
    def memory_store(self, temp_config):
        """Create real MemoryStore."""
        store = MemoryStore(temp_config)
        yield store
        # Cleanup handled by tmp_path fixture

    @pytest.fixture
    def todo_store(self, memory_store):
        """Create TodoStore with real MemoryStore."""
        return TodoStore(memory_store)

    def test_create_then_list_todos(self, todo_store):
        """BUG REPRO: Created todo should appear in list.

        This reproduces the bug where create_todo returns a UUID
        but list_todos returns empty.
        """
        # Create a todo
        todo = todo_store.create(
            title="Test task",
            description="Test description",
            priority="high",
            project_id="test-project",
            tags=["test"],
        )

        assert todo.uuid is not None
        assert todo.title == "Test task"
        print(f"Created todo: {todo.uuid}")

        # Now list todos - THIS IS WHERE THE BUG IS
        todos = todo_store.find(project_id="test-project")

        print(f"Found {len(todos)} todos")
        for t in todos:
            print(f"  - {t.uuid}: {t.title}")

        # This assertion fails with the current implementation
        assert len(todos) >= 1, "Created todo should appear in list!"
        assert any(t.uuid == todo.uuid for t in todos), f"Created todo {todo.uuid} not found in list"

    def test_create_then_list_without_project_filter(self, todo_store):
        """Test listing without project filter."""
        # Create a todo
        todo = todo_store.create(
            title="No project todo",
            priority="medium",
        )

        print(f"Created todo: {todo.uuid}")

        # List all todos
        todos = todo_store.find()

        print(f"Found {len(todos)} todos (no filter)")
        assert len(todos) >= 1, "Created todo should appear in list!"

    def test_create_multiple_then_filter(self, todo_store):
        """Test creating multiple and filtering."""
        # Create todos with different projects
        todo1 = todo_store.create(title="Project A task 1", project_id="project-a")
        todo2 = todo_store.create(title="Project A task 2", project_id="project-a")
        todo3 = todo_store.create(title="Project B task 1", project_id="project-b")

        print(f"Created: {todo1.uuid}, {todo2.uuid}, {todo3.uuid}")

        # List all
        all_todos = todo_store.find()
        print(f"All todos: {len(all_todos)}")
        assert len(all_todos) >= 3

        # Filter by project A
        project_a_todos = todo_store.find(project_id="project-a")
        print(f"Project A todos: {len(project_a_todos)}")
        assert len(project_a_todos) >= 2

        # Filter by project B
        project_b_todos = todo_store.find(project_id="project-b")
        print(f"Project B todos: {len(project_b_todos)}")
        assert len(project_b_todos) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
