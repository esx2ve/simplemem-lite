"""Project state management for SimpleMem Lite.

Tracks project bootstrap status, user preferences, and trace processing cursors.
"""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from simplemem_lite.config import Config
from simplemem_lite.log_config import get_logger

log = get_logger("projects")


@dataclass
class ProjectState:
    """State for a tracked project."""

    # Core identification
    project_root: str  # Absolute path to project root
    git_root: str | None = None  # Git root (may differ from project_root)

    # Bootstrap status
    is_bootstrapped: bool = False  # Has bootstrap been run?
    never_ask: bool = False  # User said "never ask again"

    # Session tracking
    last_session_id: str | None = None

    # Trace processing cursor (for delta processing)
    last_processed_index: int = 0  # Last message index processed
    last_trace_inode: int | None = None  # Inode to detect file rotation

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Project metadata (from detection pipeline)
    project_name: str | None = None
    project_type: str | None = None  # python, typescript, etc.
    description: str | None = None

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()


class ProjectManager:
    """Manages project state persistence and detection."""

    def __init__(self, config: Config):
        """Initialize the project manager.

        Args:
            config: SimpleMem configuration
        """
        self.config = config
        self._projects_file = config.data_dir / "projects.json"
        self._cache: dict[str, ProjectState] = {}
        self._load_projects()
        log.info(f"ProjectManager initialized with {len(self._cache)} projects")

    def _load_projects(self) -> None:
        """Load projects from disk."""
        if not self._projects_file.exists():
            log.debug("No projects file found, starting fresh")
            return

        try:
            data = json.loads(self._projects_file.read_text())
            for root, state_dict in data.items():
                self._cache[root] = ProjectState(**state_dict)
            log.debug(f"Loaded {len(self._cache)} projects from disk")
        except Exception as e:
            log.error(f"Failed to load projects file: {e}")
            self._cache = {}

    def _save_projects(self) -> None:
        """Persist projects to disk."""
        try:
            data = {root: asdict(state) for root, state in self._cache.items()}
            self._projects_file.write_text(json.dumps(data, indent=2))
            log.debug(f"Saved {len(self._cache)} projects to disk")
        except Exception as e:
            log.error(f"Failed to save projects file: {e}")

    def get_project_state(self, project_root: str) -> ProjectState | None:
        """Get state for a project.

        Args:
            project_root: Absolute path to project root

        Returns:
            ProjectState if found, None otherwise
        """
        normalized = str(Path(project_root).resolve())
        return self._cache.get(normalized)

    def set_project_state(self, state: ProjectState) -> None:
        """Set state for a project.

        Args:
            state: ProjectState to save
        """
        normalized = str(Path(state.project_root).resolve())
        state.project_root = normalized
        state.touch()
        self._cache[normalized] = state
        self._save_projects()
        log.info(f"Saved project state: {normalized}")

    def get_or_create_project(self, project_root: str) -> ProjectState:
        """Get or create project state.

        Args:
            project_root: Absolute path to project root

        Returns:
            Existing or new ProjectState
        """
        normalized = str(Path(project_root).resolve())
        existing = self._cache.get(normalized)
        if existing:
            return existing

        # Create new project state
        git_root = self.detect_git_root(normalized)
        state = ProjectState(
            project_root=normalized,
            git_root=git_root,
        )
        self.set_project_state(state)
        return state

    def mark_bootstrapped(
        self,
        project_root: str,
        project_name: str | None = None,
        project_type: str | None = None,
        description: str | None = None,
    ) -> ProjectState:
        """Mark a project as bootstrapped.

        Args:
            project_root: Absolute path to project root
            project_name: Optional project name
            project_type: Optional project type
            description: Optional project description

        Returns:
            Updated ProjectState
        """
        state = self.get_or_create_project(project_root)
        state.is_bootstrapped = True
        if project_name:
            state.project_name = project_name
        if project_type:
            state.project_type = project_type
        if description:
            state.description = description
        self.set_project_state(state)
        log.info(f"Project marked as bootstrapped: {project_root}")
        return state

    def set_never_ask(self, project_root: str, never_ask: bool = True) -> ProjectState:
        """Set the never-ask preference for a project.

        Args:
            project_root: Absolute path to project root
            never_ask: Whether to never ask about bootstrapping

        Returns:
            Updated ProjectState
        """
        state = self.get_or_create_project(project_root)
        state.never_ask = never_ask
        self.set_project_state(state)
        log.info(f"Project never_ask set to {never_ask}: {project_root}")
        return state

    def update_trace_cursor(
        self,
        project_root: str,
        session_id: str,
        processed_index: int,
        trace_inode: int | None = None,
    ) -> ProjectState:
        """Update the trace processing cursor.

        Args:
            project_root: Absolute path to project root
            session_id: Current session ID
            processed_index: Last processed message index
            trace_inode: File inode (for rotation detection)

        Returns:
            Updated ProjectState
        """
        state = self.get_or_create_project(project_root)
        state.last_session_id = session_id
        state.last_processed_index = processed_index
        if trace_inode is not None:
            state.last_trace_inode = trace_inode
        self.set_project_state(state)
        log.debug(f"Trace cursor updated: {project_root} -> index={processed_index}")
        return state

    def should_ask_bootstrap(self, project_root: str) -> bool:
        """Check if we should ask about bootstrapping.

        Args:
            project_root: Absolute path to project root

        Returns:
            True if we should ask, False if bootstrapped or never_ask
        """
        state = self.get_project_state(project_root)
        if state is None:
            return True  # Unknown project, should ask
        if state.is_bootstrapped:
            return False  # Already bootstrapped
        if state.never_ask:
            return False  # User said never ask
        return True  # Not bootstrapped and user hasn't declined

    def list_projects(self) -> list[dict[str, Any]]:
        """List all tracked projects.

        Returns:
            List of project summaries
        """
        return [
            {
                "project_root": state.project_root,
                "project_name": state.project_name,
                "is_bootstrapped": state.is_bootstrapped,
                "never_ask": state.never_ask,
                "updated_at": state.updated_at,
            }
            for state in self._cache.values()
        ]

    @staticmethod
    def detect_git_root(cwd: str) -> str | None:
        """Detect git root from a working directory.

        Args:
            cwd: Current working directory

        Returns:
            Git root path if in a git repo, None otherwise
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_root = result.stdout.strip()
                log.debug(f"Detected git root: {git_root} for cwd: {cwd}")
                return git_root
        except subprocess.TimeoutExpired:
            log.warning(f"Git root detection timed out for: {cwd}")
        except FileNotFoundError:
            log.debug("Git command not found")
        except Exception as e:
            log.debug(f"Git root detection failed: {e}")
        return None

    @staticmethod
    def detect_project_root(cwd: str) -> str:
        """Detect project root from a working directory.

        Uses git root if available, otherwise falls back to cwd.

        Args:
            cwd: Current working directory

        Returns:
            Project root path
        """
        git_root = ProjectManager.detect_git_root(cwd)
        if git_root:
            return git_root
        # Fall back to cwd
        return str(Path(cwd).resolve())
