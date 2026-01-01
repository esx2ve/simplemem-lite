"""Persistent Todo System for SimpleMem Lite.

Provides CRUD operations for persistent todos that integrate with
Claude's ephemeral TodoWrite tool via a "promote" model.

Todos are stored as memories with type="todo" and additional metadata
for status, priority, tags, etc. This leverages existing embedding
and graph infrastructure.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from simplemem_lite.log_config import get_logger

if TYPE_CHECKING:
    from simplemem_lite.memory import MemoryItem, MemoryStore

log = get_logger("todo")

# Valid status values
TODO_STATUSES = ("pending", "in_progress", "completed", "cancelled", "blocked")

# Valid priority values
TODO_PRIORITIES = ("low", "medium", "high", "critical")

# Valid source values
TODO_SOURCES = ("user", "promoted", "extracted", "claude")


@dataclass
class Todo:
    """Persistent todo item.

    Attributes:
        uuid: Unique identifier
        title: Short description of the task
        description: Detailed description (optional)
        status: Current status (pending, in_progress, completed, cancelled, blocked)
        priority: Priority level (low, medium, high, critical)
        project_id: Project scope for isolation (optional)
        tags: List of tags for categorization
        source: Origin of todo (user, promoted, extracted, claude)
        created_at: Creation timestamp
        updated_at: Last update timestamp
        completed_at: Completion timestamp (if completed)
    """

    uuid: str
    title: str
    description: str | None = None
    status: str = "pending"
    priority: str = "medium"
    project_id: str | None = None
    tags: list[str] = field(default_factory=list)
    source: str = "user"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "uuid": self.uuid,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "project_id": self.project_id,
            "tags": self.tags,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_memory(cls, memory: Any) -> "Todo":
        """Create Todo from a Memory object.

        Parses the memory content and metadata to reconstruct the Todo.
        """
        # Parse title and description from content
        content = memory.content
        lines = content.split("\n", 1)
        title = lines[0].replace("TODO: ", "") if lines else content
        description = lines[1] if len(lines) > 1 else None

        # Extract todo-specific metadata
        # Metadata may be stored as part of memory or in relations
        metadata = getattr(memory, "metadata", {}) or {}

        return cls(
            uuid=memory.uuid,
            title=title,
            description=description,
            status=metadata.get("todo_status", "pending"),
            priority=metadata.get("todo_priority", "medium"),
            project_id=metadata.get("project_id"),
            tags=metadata.get("todo_tags", []),
            source=metadata.get("todo_source", "user"),
            created_at=memory.created_at,
            updated_at=metadata.get("todo_updated_at", memory.created_at),
            completed_at=metadata.get("todo_completed_at"),
        )


class TodoStore:
    """CRUD operations for persistent todos.

    Wraps MemoryStore to store todos as memories with type="todo".
    Provides filtering, updating, and project isolation.

    Example:
        >>> from simplemem_lite.memory import MemoryStore
        >>> from simplemem_lite.config import Config
        >>> memory_store = MemoryStore(Config())
        >>> todos = TodoStore(memory_store)
        >>> todo = todos.create("Fix the login bug", priority="high")
        >>> todos.update(todo.uuid, status="in_progress")
    """

    def __init__(self, memory_store: "MemoryStore"):
        """Initialize TodoStore with a MemoryStore backend.

        Args:
            memory_store: MemoryStore instance for persistence
        """
        self.memory = memory_store
        log.info("TodoStore initialized")

    def create(
        self,
        title: str,
        description: str | None = None,
        priority: str = "medium",
        project_id: str | None = None,
        tags: list[str] | None = None,
        source: str = "user",
    ) -> Todo:
        """Create a new persistent todo.

        Args:
            title: Short task description
            description: Detailed description (optional)
            priority: Priority level (low, medium, high, critical)
            project_id: Project scope for isolation
            tags: List of tags for categorization
            source: Origin (user, promoted, extracted, claude)

        Returns:
            Created Todo with generated UUID

        Raises:
            ValueError: If priority or source is invalid
        """
        # Validate inputs
        if priority not in TODO_PRIORITIES:
            raise ValueError(f"Invalid priority: {priority}. Must be one of {TODO_PRIORITIES}")
        if source not in TODO_SOURCES:
            raise ValueError(f"Invalid source: {source}. Must be one of {TODO_SOURCES}")

        now = time.time()
        todo_uuid = str(uuid4())
        tags = tags or []

        # Build content for embedding (title + description)
        content = f"TODO: {title}"
        if description:
            content += f"\n{description}"

        # Store as memory with type="todo"
        from simplemem_lite.memory import MemoryItem

        memory_item = MemoryItem(
            content=content,
            metadata={
                "type": "todo",
                "project_id": project_id,
                "todo_status": "pending",
                "todo_priority": priority,
                "todo_tags": tags,
                "todo_source": source,
                "todo_updated_at": now,
                "todo_completed_at": None,
            },
        )

        # Store and get the UUID
        stored_uuid = self.memory.store(memory_item)

        todo = Todo(
            uuid=stored_uuid,
            title=title,
            description=description,
            status="pending",
            priority=priority,
            project_id=project_id,
            tags=tags,
            source=source,
            created_at=now,
            updated_at=now,
        )

        log.info(
            f"Created todo {stored_uuid[:8]}...: {title[:50]}... "
            f"(priority={priority}, project={project_id})"
        )

        return todo

    def get(self, todo_id: str) -> Todo | None:
        """Get a todo by ID.

        Args:
            todo_id: UUID of the todo

        Returns:
            Todo if found, None otherwise
        """
        # Search by exact UUID match
        results = self.memory.search(
            query="",  # Empty query for metadata-only search
            limit=1,
            type_filter="todo",
        )

        # Find matching UUID
        for memory in results:
            if memory.uuid == todo_id:
                return Todo.from_memory(memory)

        log.debug(f"Todo not found: {todo_id[:8]}...")
        return None

    def find(
        self,
        project_id: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 50,
    ) -> "list[Todo]":
        """List todos with optional filters.

        Args:
            project_id: Filter by project (optional)
            status: Filter by status (optional)
            priority: Filter by priority (optional)
            limit: Maximum number to return

        Returns:
            List of matching Todo objects
        """
        # Validate filters
        if status and status not in TODO_STATUSES:
            raise ValueError(f"Invalid status: {status}. Must be one of {TODO_STATUSES}")
        if priority and priority not in TODO_PRIORITIES:
            raise ValueError(f"Invalid priority: {priority}. Must be one of {TODO_PRIORITIES}")

        # Search all todos (use semantic search with empty query)
        results = self.memory.search(
            query="TODO task",  # Generic query to find todos
            limit=limit * 2,  # Fetch more to filter
            type_filter="todo",
        )

        todos = []
        for memory in results:
            try:
                todo = Todo.from_memory(memory)
            except Exception as e:
                log.warning(f"Failed to parse todo {memory.uuid}: {e}")
                continue

            # Apply filters
            if project_id and todo.project_id != project_id:
                continue
            if status and todo.status != status:
                continue
            if priority and todo.priority != priority:
                continue

            todos.append(todo)
            if len(todos) >= limit:
                break

        log.debug(
            f"Listed {len(todos)} todos "
            f"(project={project_id}, status={status}, priority={priority})"
        )

        return todos

    def update(
        self,
        todo_id: str,
        status: str | None = None,
        priority: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> Todo | None:
        """Update a todo's fields.

        Args:
            todo_id: UUID of the todo to update
            status: New status (optional)
            priority: New priority (optional)
            title: New title (optional)
            description: New description (optional)
            tags: New tags (optional)

        Returns:
            Updated Todo if found, None otherwise

        Raises:
            ValueError: If status or priority is invalid
        """
        # Validate inputs
        if status and status not in TODO_STATUSES:
            raise ValueError(f"Invalid status: {status}. Must be one of {TODO_STATUSES}")
        if priority and priority not in TODO_PRIORITIES:
            raise ValueError(f"Invalid priority: {priority}. Must be one of {TODO_PRIORITIES}")

        # Get existing todo
        todo = self.get(todo_id)
        if not todo:
            log.warning(f"Todo not found for update: {todo_id[:8]}...")
            return None

        now = time.time()

        # Apply updates
        if title is not None:
            todo.title = title
        if description is not None:
            todo.description = description
        if status is not None:
            todo.status = status
            if status == "completed":
                todo.completed_at = now
        if priority is not None:
            todo.priority = priority
        if tags is not None:
            todo.tags = tags

        todo.updated_at = now

        # Re-store with updated content and metadata
        content = f"TODO: {todo.title}"
        if todo.description:
            content += f"\n{todo.description}"

        from simplemem_lite.memory import MemoryItem

        memory_item = MemoryItem(
            content=content,
            metadata={
                "type": "todo",
                "project_id": todo.project_id,
                "todo_status": todo.status,
                "todo_priority": todo.priority,
                "todo_tags": todo.tags,
                "todo_source": todo.source,
                "todo_updated_at": now,
                "todo_completed_at": todo.completed_at,
            },
        )

        # Store as new memory (todo_id changes)
        new_uuid = self.memory.store(memory_item)
        todo.uuid = new_uuid

        log.info(
            f"Updated todo {todo_id[:8]}... -> {new_uuid[:8]}...: "
            f"status={todo.status}, priority={todo.priority}"
        )

        return todo

    def complete(self, todo_id: str, learnings: str | None = None) -> Todo | None:
        """Complete a todo, optionally capturing learnings.

        Args:
            todo_id: UUID of the todo to complete
            learnings: Optional learnings to store as a memory

        Returns:
            Completed Todo if found, None otherwise
        """
        todo = self.update(todo_id, status="completed")
        if not todo:
            return None

        # Store learnings as separate memory linked to todo
        if learnings:
            from simplemem_lite.memory import MemoryItem

            learning_item = MemoryItem(
                content=f"Lesson from completing: {todo.title}\n\n{learnings}",
                metadata={
                    "type": "lesson_learned",
                    "source": "todo_completion",
                    "project_id": todo.project_id,
                },
                relations=[{"target_id": todo.uuid, "type": "derived_from"}],
            )
            learning_uuid = self.memory.store(learning_item)
            log.info(f"Stored learnings for todo {todo_id[:8]}... as {learning_uuid[:8]}...")

        return todo

    def promote(
        self,
        title: str,
        description: str | None = None,
        priority: str = "medium",
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Todo:
        """Promote an ephemeral todo to persistent storage.

        Same as create() but with source="promoted" to indicate
        it came from Claude's ephemeral TodoWrite tool.

        Args:
            title: Short task description
            description: Detailed description (optional)
            priority: Priority level (low, medium, high, critical)
            project_id: Project scope for isolation
            tags: List of tags for categorization

        Returns:
            Created Todo with source="promoted"
        """
        log.info(f"Promoting ephemeral todo: {title[:50]}...")
        return self.create(
            title=title,
            description=description,
            priority=priority,
            project_id=project_id,
            tags=tags,
            source="promoted",
        )
