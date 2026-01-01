"""Project manager tests for SimpleMem Lite.

Tests critical project management logic:
- ProjectState dataclass
- ProjectManager persistence
- Bootstrap and preference tracking
"""

import tempfile
from pathlib import Path

import pytest


class TestProjectState:
    """Test ProjectState dataclass."""

    def test_project_state_defaults(self):
        """ProjectState should have sensible defaults."""
        from simplemem_lite.projects import ProjectState

        state = ProjectState(project_root="/test/project")

        assert state.project_root == "/test/project"
        assert state.is_bootstrapped is False
        assert state.never_ask is False
        assert state.last_processed_index == 0
        assert state.git_root is None

    def test_project_state_touch_updates_timestamp(self):
        """touch() should update the updated_at timestamp."""
        from simplemem_lite.projects import ProjectState
        import time

        state = ProjectState(project_root="/test")
        original = state.updated_at

        time.sleep(0.01)  # Small delay
        state.touch()

        assert state.updated_at > original


class TestProjectManager:
    """Test ProjectManager operations."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create config with temp data directory."""
        from simplemem_lite.config import Config

        # Config is a dataclass, pass data_dir as a field
        return Config(data_dir=tmp_path)

    def test_manager_starts_empty(self, config):
        """New manager should have no projects."""
        from simplemem_lite.projects import ProjectManager

        manager = ProjectManager(config)

        projects = manager.list_projects()
        assert projects == []

    def test_get_or_create_project(self, config):
        """get_or_create_project should create new project."""
        from simplemem_lite.projects import ProjectManager

        manager = ProjectManager(config)

        state = manager.get_or_create_project("/test/project")

        assert state.project_root == str(Path("/test/project").resolve())
        assert state.is_bootstrapped is False

    def test_get_existing_project(self, config):
        """get_or_create_project should return existing project."""
        from simplemem_lite.projects import ProjectManager

        manager = ProjectManager(config)

        # Create first
        state1 = manager.get_or_create_project("/test/project")
        state1.is_bootstrapped = True
        manager.set_project_state(state1)

        # Get again
        state2 = manager.get_or_create_project("/test/project")

        assert state2.is_bootstrapped is True

    def test_mark_bootstrapped(self, config):
        """mark_bootstrapped should update project state."""
        from simplemem_lite.projects import ProjectManager

        manager = ProjectManager(config)

        state = manager.mark_bootstrapped(
            "/test/project",
            project_name="Test Project",
            project_type="python",
        )

        assert state.is_bootstrapped is True
        assert state.project_name == "Test Project"
        assert state.project_type == "python"

    def test_set_never_ask(self, config):
        """set_never_ask should update preference."""
        from simplemem_lite.projects import ProjectManager

        manager = ProjectManager(config)

        state = manager.set_never_ask("/test/project", never_ask=True)

        assert state.never_ask is True

    def test_should_ask_bootstrap_new_project(self, config):
        """should_ask_bootstrap returns True for new projects."""
        from simplemem_lite.projects import ProjectManager

        manager = ProjectManager(config)

        assert manager.should_ask_bootstrap("/unknown/project") is True

    def test_should_ask_bootstrap_bootstrapped(self, config):
        """should_ask_bootstrap returns False for bootstrapped projects."""
        from simplemem_lite.projects import ProjectManager

        manager = ProjectManager(config)
        manager.mark_bootstrapped("/test/project")

        assert manager.should_ask_bootstrap("/test/project") is False

    def test_should_ask_bootstrap_never_ask(self, config):
        """should_ask_bootstrap returns False if never_ask is set."""
        from simplemem_lite.projects import ProjectManager

        manager = ProjectManager(config)
        manager.set_never_ask("/test/project", never_ask=True)

        assert manager.should_ask_bootstrap("/test/project") is False

    def test_update_trace_cursor(self, config):
        """update_trace_cursor should update cursor state."""
        from simplemem_lite.projects import ProjectManager

        manager = ProjectManager(config)

        state = manager.update_trace_cursor(
            "/test/project",
            session_id="session-123",
            processed_index=42,
            trace_inode=12345,
        )

        assert state.last_session_id == "session-123"
        assert state.last_processed_index == 42
        assert state.last_trace_inode == 12345

    def test_list_projects(self, config):
        """list_projects should return all tracked projects."""
        from simplemem_lite.projects import ProjectManager

        manager = ProjectManager(config)
        manager.mark_bootstrapped("/project1", project_name="Project 1")
        manager.set_never_ask("/project2")

        projects = manager.list_projects()

        assert len(projects) == 2
        names = [p["project_name"] for p in projects]
        assert "Project 1" in names


class TestProjectPersistence:
    """Test project state persistence."""

    def test_projects_persist_to_disk(self, tmp_path):
        """Projects should be saved to disk."""
        from simplemem_lite.config import Config
        from simplemem_lite.projects import ProjectManager

        config = Config(data_dir=tmp_path)
        manager = ProjectManager(config)
        manager.mark_bootstrapped("/test/project")

        # Check file exists
        projects_file = tmp_path / "projects.json"
        assert projects_file.exists()

    def test_projects_survive_restart(self, tmp_path):
        """Projects should be loaded on manager restart."""
        from simplemem_lite.config import Config
        from simplemem_lite.projects import ProjectManager

        config = Config(data_dir=tmp_path)

        # First manager - create project
        manager1 = ProjectManager(config)
        manager1.mark_bootstrapped("/test/project", project_name="Test")

        # Second manager - should load project
        manager2 = ProjectManager(config)
        state = manager2.get_project_state(str(Path("/test/project").resolve()))

        assert state is not None
        assert state.is_bootstrapped is True
        assert state.project_name == "Test"


class TestGitRootDetection:
    """Test git root detection."""

    def test_detect_git_root_non_repo(self, tmp_path):
        """Should return None for non-git directories."""
        from simplemem_lite.projects import ProjectManager

        result = ProjectManager.detect_git_root(str(tmp_path))
        assert result is None

    def test_detect_project_root_non_repo(self, tmp_path):
        """Should fall back to cwd for non-git directories."""
        from simplemem_lite.projects import ProjectManager

        result = ProjectManager.detect_project_root(str(tmp_path))
        assert result == str(tmp_path.resolve())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
