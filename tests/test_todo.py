"""Tests for Persistent Todo System (P4).

Tests:
- Todo dataclass and serialization
- TodoStore CRUD operations
- MCP tool integration
- Validation and error handling
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from simplemem_lite.todo import (
    TODO_PRIORITIES,
    TODO_SOURCES,
    TODO_STATUSES,
    Todo,
    TodoStore,
)


class TestTodoDataclass:
    """Test Todo dataclass."""

    def test_default_values(self):
        """Todo should have sensible defaults."""
        todo = Todo(uuid="test-uuid", title="Test task")

        assert todo.uuid == "test-uuid"
        assert todo.title == "Test task"
        assert todo.description is None
        assert todo.status == "pending"
        assert todo.priority == "medium"
        assert todo.project_id is None
        assert todo.tags == []
        assert todo.source == "user"
        assert todo.completed_at is None
        assert todo.created_at > 0
        assert todo.updated_at > 0

    def test_custom_values(self):
        """Todo should preserve custom values."""
        todo = Todo(
            uuid="custom-uuid",
            title="Custom task",
            description="Detailed description",
            status="in_progress",
            priority="high",
            project_id="my-project",
            tags=["bug", "urgent"],
            source="promoted",
            created_at=1234567890.0,
            updated_at=1234567900.0,
            completed_at=None,
        )

        assert todo.description == "Detailed description"
        assert todo.status == "in_progress"
        assert todo.priority == "high"
        assert todo.project_id == "my-project"
        assert todo.tags == ["bug", "urgent"]
        assert todo.source == "promoted"

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        todo = Todo(
            uuid="test-uuid",
            title="Test task",
            description="Description",
            status="pending",
            priority="high",
            project_id="project-1",
            tags=["tag1"],
            source="user",
            created_at=1000.0,
            updated_at=2000.0,
        )

        result = todo.to_dict()

        assert result["uuid"] == "test-uuid"
        assert result["title"] == "Test task"
        assert result["description"] == "Description"
        assert result["status"] == "pending"
        assert result["priority"] == "high"
        assert result["project_id"] == "project-1"
        assert result["tags"] == ["tag1"]
        assert result["source"] == "user"
        assert result["created_at"] == 1000.0
        assert result["updated_at"] == 2000.0
        assert result["completed_at"] is None

    def test_from_memory(self):
        """from_memory should reconstruct Todo from Memory object."""
        mock_memory = MagicMock()
        mock_memory.uuid = "mem-uuid"
        mock_memory.content = "TODO: Fix the bug\nDetailed steps to fix"
        mock_memory.created_at = 1000.0
        mock_memory.metadata = {
            "todo_status": "in_progress",
            "todo_priority": "high",
            "project_id": "project-1",
            "todo_tags": ["bug"],
            "todo_source": "promoted",
            "todo_updated_at": 2000.0,
            "todo_completed_at": None,
        }

        todo = Todo.from_memory(mock_memory)

        assert todo.uuid == "mem-uuid"
        assert todo.title == "Fix the bug"
        assert todo.description == "Detailed steps to fix"
        assert todo.status == "in_progress"
        assert todo.priority == "high"
        assert todo.project_id == "project-1"
        assert todo.tags == ["bug"]
        assert todo.source == "promoted"


class TestTodoConstants:
    """Test todo constants."""

    def test_valid_statuses(self):
        """Should have expected status values."""
        assert "pending" in TODO_STATUSES
        assert "in_progress" in TODO_STATUSES
        assert "completed" in TODO_STATUSES
        assert "cancelled" in TODO_STATUSES
        assert "blocked" in TODO_STATUSES

    def test_valid_priorities(self):
        """Should have expected priority values."""
        assert "low" in TODO_PRIORITIES
        assert "medium" in TODO_PRIORITIES
        assert "high" in TODO_PRIORITIES
        assert "critical" in TODO_PRIORITIES

    def test_valid_sources(self):
        """Should have expected source values."""
        assert "user" in TODO_SOURCES
        assert "promoted" in TODO_SOURCES
        assert "extracted" in TODO_SOURCES
        assert "claude" in TODO_SOURCES


class TestTodoStoreCreate:
    """Test TodoStore.create()."""

    @pytest.fixture
    def mock_memory_store(self):
        """Create mock MemoryStore."""
        store = MagicMock()
        store.store.return_value = "generated-uuid"
        return store

    def test_create_basic_todo(self, mock_memory_store):
        """Should create todo with defaults."""
        todo_store = TodoStore(mock_memory_store)

        todo = todo_store.create(title="Basic task")

        assert todo.title == "Basic task"
        assert todo.uuid == "generated-uuid"
        assert todo.status == "pending"
        assert todo.priority == "medium"
        assert todo.source == "user"
        mock_memory_store.store.assert_called_once()

    def test_create_todo_with_all_fields(self, mock_memory_store):
        """Should create todo with all fields."""
        todo_store = TodoStore(mock_memory_store)

        todo = todo_store.create(
            title="Complex task",
            description="With description",
            priority="critical",
            project_id="my-project",
            tags=["urgent", "bug"],
            source="claude",
        )

        assert todo.title == "Complex task"
        assert todo.description == "With description"
        assert todo.priority == "critical"
        assert todo.project_id == "my-project"
        assert todo.tags == ["urgent", "bug"]
        assert todo.source == "claude"

    def test_create_validates_priority(self, mock_memory_store):
        """Should reject invalid priority."""
        todo_store = TodoStore(mock_memory_store)

        with pytest.raises(ValueError, match="Invalid priority"):
            todo_store.create(title="Task", priority="invalid")

    def test_create_validates_source(self, mock_memory_store):
        """Should reject invalid source."""
        todo_store = TodoStore(mock_memory_store)

        with pytest.raises(ValueError, match="Invalid source"):
            todo_store.create(title="Task", source="invalid")


class TestTodoStoreList:
    """Test TodoStore.list()."""

    @pytest.fixture
    def mock_memory_store(self):
        """Create mock MemoryStore with sample todos."""
        store = MagicMock()

        # Create mock memories
        def make_memory(uuid, title, status, priority, project_id):
            mem = MagicMock()
            mem.uuid = uuid
            mem.content = f"TODO: {title}"
            mem.created_at = time.time()
            mem.metadata = {
                "todo_status": status,
                "todo_priority": priority,
                "project_id": project_id,
                "todo_tags": [],
                "todo_source": "user",
                "todo_updated_at": time.time(),
            }
            return mem

        store.search.return_value = [
            make_memory("uuid-1", "Task 1", "pending", "high", "project-a"),
            make_memory("uuid-2", "Task 2", "completed", "low", "project-a"),
            make_memory("uuid-3", "Task 3", "pending", "medium", "project-b"),
        ]

        return store

    def test_list_all_todos(self, mock_memory_store):
        """Should list all todos without filters."""
        todo_store = TodoStore(mock_memory_store)

        todos = todo_store.find()

        assert len(todos) == 3
        mock_memory_store.search.assert_called_once()

    def test_list_filters_by_status(self, mock_memory_store):
        """Should filter by status."""
        todo_store = TodoStore(mock_memory_store)

        todos = todo_store.find(status="pending")

        # Should only return pending todos
        assert all(t.status == "pending" for t in todos)

    def test_list_filters_by_project(self, mock_memory_store):
        """Should filter by project_id."""
        todo_store = TodoStore(mock_memory_store)

        todos = todo_store.find(project_id="project-a")

        # Should only return project-a todos
        assert all(t.project_id == "project-a" for t in todos)

    def test_list_filters_by_priority(self, mock_memory_store):
        """Should filter by priority."""
        todo_store = TodoStore(mock_memory_store)

        todos = todo_store.find(priority="high")

        # Should only return high priority todos
        assert all(t.priority == "high" for t in todos)

    def test_list_validates_status(self, mock_memory_store):
        """Should reject invalid status filter."""
        todo_store = TodoStore(mock_memory_store)

        with pytest.raises(ValueError, match="Invalid status"):
            todo_store.find(status="invalid")

    def test_list_validates_priority(self, mock_memory_store):
        """Should reject invalid priority filter."""
        todo_store = TodoStore(mock_memory_store)

        with pytest.raises(ValueError, match="Invalid priority"):
            todo_store.find(priority="invalid")


class TestTodoStoreUpdate:
    """Test TodoStore.update()."""

    @pytest.fixture
    def mock_memory_store(self):
        """Create mock MemoryStore."""
        store = MagicMock()

        # Mock existing todo
        existing = MagicMock()
        existing.uuid = "existing-uuid"
        existing.content = "TODO: Original title"
        existing.created_at = 1000.0
        existing.metadata = {
            "todo_status": "pending",
            "todo_priority": "medium",
            "project_id": "project-1",
            "todo_tags": [],
            "todo_source": "user",
            "todo_updated_at": 1000.0,
        }

        store.search.return_value = [existing]
        store.store.return_value = "new-uuid"

        return store

    def test_update_status(self, mock_memory_store):
        """Should update status."""
        todo_store = TodoStore(mock_memory_store)

        todo = todo_store.update("existing-uuid", status="in_progress")

        assert todo is not None
        assert todo.status == "in_progress"

    def test_update_priority(self, mock_memory_store):
        """Should update priority."""
        todo_store = TodoStore(mock_memory_store)

        todo = todo_store.update("existing-uuid", priority="critical")

        assert todo is not None
        assert todo.priority == "critical"

    def test_update_sets_completed_at(self, mock_memory_store):
        """Completing should set completed_at."""
        todo_store = TodoStore(mock_memory_store)

        todo = todo_store.update("existing-uuid", status="completed")

        assert todo is not None
        assert todo.status == "completed"
        assert todo.completed_at is not None

    def test_update_validates_status(self, mock_memory_store):
        """Should reject invalid status."""
        todo_store = TodoStore(mock_memory_store)

        with pytest.raises(ValueError, match="Invalid status"):
            todo_store.update("existing-uuid", status="invalid")


class TestTodoStorePromote:
    """Test TodoStore.promote()."""

    @pytest.fixture
    def mock_memory_store(self):
        """Create mock MemoryStore."""
        store = MagicMock()
        store.store.return_value = "promoted-uuid"
        return store

    def test_promote_sets_source(self, mock_memory_store):
        """Promote should set source to 'promoted'."""
        todo_store = TodoStore(mock_memory_store)

        todo = todo_store.promote(title="Promoted task")

        assert todo.source == "promoted"
        assert todo.title == "Promoted task"

    def test_promote_with_options(self, mock_memory_store):
        """Promote should accept all create options."""
        todo_store = TodoStore(mock_memory_store)

        todo = todo_store.promote(
            title="Complex promoted task",
            description="With description",
            priority="high",
            project_id="my-project",
            tags=["important"],
        )

        assert todo.source == "promoted"
        assert todo.priority == "high"
        assert todo.project_id == "my-project"


class TestMCPToolsIntegration:
    """Test MCP tool integration."""

    @pytest.fixture
    def mock_deps(self):
        """Create mock dependencies."""
        deps = MagicMock()
        deps.store.store.return_value = "mcp-uuid"
        deps.store.search.return_value = []
        return deps

    @pytest.mark.asyncio
    async def test_create_todo_tool(self, mock_deps):
        """create_todo MCP tool should work."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import create_todo

            result = await create_todo(
                title="MCP task",
                priority="high",
                project_id="test-project",
            )

            assert "uuid" in result
            assert result["title"] == "MCP task"
            assert result["priority"] == "high"

    @pytest.mark.asyncio
    async def test_list_todos_tool(self, mock_deps):
        """list_todos MCP tool should work."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import list_todos

            result = await list_todos(project_id="test-project")

            assert "todos" in result
            assert "total_count" in result
            assert isinstance(result["todos"], list)

    @pytest.mark.asyncio
    async def test_promote_todo_tool(self, mock_deps):
        """promote_todo MCP tool should work."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import promote_todo

            result = await promote_todo(
                title="Promoted MCP task",
                priority="critical",
            )

            assert "uuid" in result
            assert result["source"] == "promoted"

    @pytest.mark.asyncio
    async def test_create_todo_validation_error(self, mock_deps):
        """create_todo should return error for invalid input."""
        with patch("simplemem_lite.server._deps", mock_deps):
            from simplemem_lite.server import create_todo

            result = await create_todo(
                title="Task",
                priority="invalid",
            )

            assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
