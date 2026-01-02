"""Tests for MCP thin layer local reader."""

import json
from pathlib import Path

import pytest

from simplemem_lite.mcp.local_reader import LocalReader


@pytest.fixture
def temp_traces_dir(tmp_path):
    """Create a temporary traces directory structure."""
    traces_dir = tmp_path / ".claude" / "projects"
    traces_dir.mkdir(parents=True)

    # Create test project with session trace
    project_dir = traces_dir / "test-project"
    project_dir.mkdir()

    # Create a session trace file
    trace_file = project_dir / "session-123.jsonl"
    trace_entries = [
        {"type": "user", "uuid": "u1", "content": "Hello"},
        {"type": "assistant", "uuid": "a1", "content": "Hi there"},
        {"type": "tool_use", "uuid": "t1", "name": "read", "input": {}},
    ]
    with open(trace_file, "w") as f:
        for entry in trace_entries:
            f.write(json.dumps(entry) + "\n")

    return traces_dir


@pytest.fixture
def temp_code_dir(tmp_path):
    """Create a temporary code directory structure."""
    code_dir = tmp_path / "project"
    code_dir.mkdir()

    # Create source files
    (code_dir / "main.py").write_text("print('hello')")
    (code_dir / "utils.py").write_text("def util(): pass")

    # Create nested structure
    src_dir = code_dir / "src"
    src_dir.mkdir()
    (src_dir / "app.py").write_text("class App: pass")

    # Create files that should be skipped
    venv_dir = code_dir / ".venv"
    venv_dir.mkdir()
    (venv_dir / "skip.py").write_text("# should be skipped")

    return code_dir


@pytest.fixture
def reader(temp_traces_dir):
    """Create a LocalReader with temp traces directory."""
    return LocalReader(traces_dir=temp_traces_dir)


class TestLocalReaderTraces:
    """Tests for trace file operations."""

    def test_discover_sessions(self, reader, temp_traces_dir):
        """discover_sessions should find trace files."""
        sessions = reader.discover_sessions(days_back=30)

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "session-123"
        assert sessions[0]["project"] == "test-project"
        assert "path" in sessions[0]
        assert "modified" in sessions[0]

    def test_discover_sessions_empty_dir(self, tmp_path):
        """discover_sessions should return empty list for missing dir."""
        reader = LocalReader(traces_dir=tmp_path / "nonexistent")
        sessions = reader.discover_sessions()
        assert sessions == []

    def test_find_session_path(self, reader):
        """find_session_path should locate trace file."""
        path = reader.find_session_path("session-123")

        assert path is not None
        assert path.name == "session-123.jsonl"
        assert path.exists()

    def test_find_session_path_not_found(self, reader):
        """find_session_path should return None for missing session."""
        path = reader.find_session_path("nonexistent-session")
        assert path is None

    def test_read_trace_file(self, reader):
        """read_trace_file should parse JSONL content."""
        entries = reader.read_trace_file("session-123")

        assert entries is not None
        assert len(entries) == 3
        assert entries[0]["type"] == "user"
        assert entries[1]["type"] == "assistant"

    def test_read_trace_file_not_found(self, reader):
        """read_trace_file should return None for missing session."""
        entries = reader.read_trace_file("nonexistent")
        assert entries is None

    def test_get_session_metadata(self, reader):
        """get_session_metadata should return session info."""
        metadata = reader.get_session_metadata("session-123")

        assert metadata is not None
        assert metadata["session_id"] == "session-123"
        assert metadata["project"] == "test-project"
        assert metadata["line_count"] == 3
        assert "size_kb" in metadata
        assert "modified_iso" in metadata

    def test_get_session_metadata_not_found(self, reader):
        """get_session_metadata should return None for missing session."""
        metadata = reader.get_session_metadata("nonexistent")
        assert metadata is None


class TestLocalReaderCode:
    """Tests for code file operations."""

    def test_scan_code_files(self, temp_code_dir):
        """scan_code_files should find matching files."""
        reader = LocalReader()
        files = list(reader.scan_code_files(temp_code_dir, patterns=["**/*.py"]))

        # Should find main.py, utils.py, src/app.py but NOT .venv/skip.py
        assert len(files) == 3
        paths = [f["path"] for f in files]
        assert "main.py" in paths
        assert "utils.py" in paths
        assert "src/app.py" in paths

    def test_scan_code_files_respects_skip_dirs(self, temp_code_dir):
        """scan_code_files should skip excluded directories."""
        reader = LocalReader()
        files = list(reader.scan_code_files(temp_code_dir))

        # .venv should be skipped
        for f in files:
            assert ".venv" not in f["path"]

    def test_scan_code_files_max_files(self, temp_code_dir):
        """scan_code_files should respect max_files limit."""
        reader = LocalReader()
        files = list(reader.scan_code_files(temp_code_dir, max_files=1))
        assert len(files) == 1

    def test_read_code_files(self, temp_code_dir):
        """read_code_files should return list of file dicts."""
        reader = LocalReader()
        files = reader.read_code_files(temp_code_dir, patterns=["**/*.py"])

        assert isinstance(files, list)
        assert len(files) == 3

        # Check content is included
        main_file = next(f for f in files if f["path"] == "main.py")
        assert "print('hello')" in main_file["content"]

    def test_read_code_files_skips_large_files(self, temp_code_dir):
        """read_code_files should skip files exceeding size limit."""
        # Create a large file
        large_file = temp_code_dir / "large.py"
        large_file.write_text("x" * 600 * 1024)  # 600KB

        reader = LocalReader()
        files = reader.read_code_files(temp_code_dir, max_file_size_kb=500)

        paths = [f["path"] for f in files]
        assert "large.py" not in paths

    def test_read_single_file(self, temp_code_dir):
        """read_single_file should return file content."""
        reader = LocalReader()
        content = reader.read_single_file(temp_code_dir / "main.py")

        assert content is not None
        assert "print('hello')" in content

    def test_read_single_file_not_found(self, temp_code_dir):
        """read_single_file should return None for missing file."""
        reader = LocalReader()
        content = reader.read_single_file(temp_code_dir / "nonexistent.py")
        assert content is None


class TestLocalReaderUtilities:
    """Tests for utility methods."""

    def test_check_directory_exists(self, temp_code_dir):
        """check_directory_exists should verify directory."""
        reader = LocalReader()

        assert reader.check_directory_exists(temp_code_dir) is True
        assert reader.check_directory_exists(temp_code_dir / "nonexistent") is False

    def test_get_directory_info(self, temp_code_dir):
        """get_directory_info should return directory details."""
        # Add .git dir to make it a git repo
        (temp_code_dir / ".git").mkdir()

        reader = LocalReader()
        info = reader.get_directory_info(temp_code_dir)

        assert info is not None
        assert info["exists"] is True
        assert info["is_git"] is True
        assert info["file_count"] >= 3

    def test_get_directory_info_nonexistent(self, tmp_path):
        """get_directory_info should handle nonexistent directory."""
        reader = LocalReader()
        info = reader.get_directory_info(tmp_path / "nonexistent")

        assert info is not None
        assert info["exists"] is False
