"""Tests for hybrid project_id generation.

Tests cover:
1. Git URL normalization
2. Git remote detection
3. Config file loading
4. Content hash generation
5. Hierarchical fallback chain
6. Project ID parsing
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

import pytest

from simplemem_lite.projects_utils import (
    normalize_git_url,
    get_git_remote_url,
    load_simplemem_config,
    hash_project_markers,
    get_project_id,
    get_project_id_legacy,
    parse_project_id,
    infer_project_from_session_path,
    normalize_project_id,
    extract_project_name,
)


class TestNormalizeGitUrl:
    """Tests for normalize_git_url function."""

    def test_ssh_github(self):
        """SSH GitHub URL should normalize correctly."""
        assert normalize_git_url("git@github.com:user/repo.git") == "github.com/user/repo"

    def test_https_github(self):
        """HTTPS GitHub URL should normalize correctly."""
        assert normalize_git_url("https://github.com/user/repo.git") == "github.com/user/repo"

    def test_https_no_git_suffix(self):
        """HTTPS URL without .git should normalize correctly."""
        assert normalize_git_url("https://github.com/user/repo") == "github.com/user/repo"

    def test_ssh_gitlab(self):
        """SSH GitLab URL should normalize correctly."""
        assert normalize_git_url("git@gitlab.com:org/project.git") == "gitlab.com/org/project"

    def test_ssh_protocol_prefix(self):
        """SSH URL with ssh:// prefix should normalize correctly."""
        assert normalize_git_url("ssh://git@bitbucket.org/team/repo") == "bitbucket.org/team/repo"

    def test_https_with_auth(self):
        """HTTPS URL with auth should strip auth."""
        assert normalize_git_url("https://user@github.com/org/repo.git") == "github.com/org/repo"

    def test_nested_path(self):
        """URL with nested path should preserve structure."""
        assert normalize_git_url("git@github.com:org/team/repo.git") == "github.com/org/team/repo"

    def test_empty_url(self):
        """Empty URL should return empty string."""
        assert normalize_git_url("") == ""

    def test_already_normalized(self):
        """Already normalized URL should pass through."""
        assert normalize_git_url("github.com/user/repo") == "github.com/user/repo"


class TestGetGitRemoteUrl:
    """Tests for get_git_remote_url function."""

    def test_git_repo_with_remote(self):
        """Git repo with remote should return normalized URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(
                ["git", "remote", "add", "origin", "git@github.com:test/repo.git"],
                cwd=tmppath,
                capture_output=True,
            )

            result = get_git_remote_url(tmppath)
            assert result == "github.com/test/repo"

    def test_git_repo_no_remote(self):
        """Git repo without remote should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)

            result = get_git_remote_url(tmppath)
            assert result is None

    def test_not_git_repo(self):
        """Non-git directory should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_git_remote_url(Path(tmpdir))
            assert result is None

    def test_git_timeout(self):
        """Git timeout should return None gracefully."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=2)
            result = get_git_remote_url(Path("/tmp"))
            assert result is None


class TestLoadSimplemConfigConfig:
    """Tests for load_simplemem_config function."""

    def test_valid_config(self):
        """Valid config file should load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config = {
                "version": 1,
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "test-project",
            }
            (tmppath / ".simplemem.json").write_text(json.dumps(config))

            result = load_simplemem_config(tmppath)
            assert result is not None
            assert result["project_id"] == "550e8400-e29b-41d4-a716-446655440000"

    def test_no_config_file(self):
        """Missing config file should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_simplemem_config(Path(tmpdir))
            assert result is None

    def test_invalid_json(self):
        """Invalid JSON should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / ".simplemem.json").write_text("not valid json")

            result = load_simplemem_config(tmppath)
            assert result is None

    def test_missing_project_id(self):
        """Config without project_id should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config = {"version": 1, "name": "test"}
            (tmppath / ".simplemem.json").write_text(json.dumps(config))

            result = load_simplemem_config(tmppath)
            assert result is None


class TestHashProjectMarkers:
    """Tests for hash_project_markers function."""

    def test_package_json(self):
        """package.json should be hashed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "package.json").write_text('{"name": "test"}')

            result = hash_project_markers(tmppath)
            assert result is not None
            assert len(result) == 16  # SHA256 prefix

    def test_pyproject_toml(self):
        """pyproject.toml should be hashed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "pyproject.toml").write_text('[project]\nname = "test"')

            result = hash_project_markers(tmppath)
            assert result is not None
            assert len(result) == 16

    def test_priority_order(self):
        """package.json should take priority over pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "package.json").write_text('{"name": "npm-project"}')
            (tmppath / "pyproject.toml").write_text('[project]\nname = "py-project"')

            # Get hash for package.json alone
            with tempfile.TemporaryDirectory() as tmpdir2:
                tmppath2 = Path(tmpdir2)
                (tmppath2 / "package.json").write_text('{"name": "npm-project"}')
                expected = hash_project_markers(tmppath2)

            result = hash_project_markers(tmppath)
            assert result == expected

    def test_no_markers(self):
        """Directory without markers should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = hash_project_markers(Path(tmpdir))
            assert result is None


class TestGetProjectId:
    """Tests for get_project_id function."""

    def test_git_repo(self):
        """Git repo should return git: prefixed ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(
                ["git", "remote", "add", "origin", "git@github.com:user/myrepo.git"],
                cwd=tmppath,
                capture_output=True,
            )

            result = get_project_id(tmppath)
            assert result == "git:github.com/user/myrepo"

    def test_config_file(self):
        """Config file should return uuid: prefixed ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config = {"version": 1, "project_id": "my-unique-id"}
            (tmppath / ".simplemem.json").write_text(json.dumps(config))

            result = get_project_id(tmppath)
            assert result == "uuid:my-unique-id"

    def test_config_file_already_prefixed(self):
        """Config with prefixed ID should preserve prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config = {"version": 1, "project_id": "git:github.com/linked/repo"}
            (tmppath / ".simplemem.json").write_text(json.dumps(config))

            result = get_project_id(tmppath)
            assert result == "git:github.com/linked/repo"

    def test_content_hash(self):
        """Project marker should return hash: prefixed ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "package.json").write_text('{"name": "test"}')

            result = get_project_id(tmppath)
            assert result.startswith("hash:")
            assert len(result) == 5 + 16  # "hash:" + 16 char hash

    def test_fallback_path(self):
        """Empty directory should return path: prefixed ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_project_id(tmpdir)
            assert result.startswith("path:")
            assert tmpdir in result

    def test_hierarchy_git_over_config(self):
        """Git should take priority over config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Add both git and config
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(
                ["git", "remote", "add", "origin", "git@github.com:user/repo.git"],
                cwd=tmppath,
                capture_output=True,
            )
            config = {"version": 1, "project_id": "config-id"}
            (tmppath / ".simplemem.json").write_text(json.dumps(config))

            result = get_project_id(tmppath)
            assert result.startswith("git:")


class TestParseProjectId:
    """Tests for parse_project_id function."""

    def test_git_prefix(self):
        """git: prefix should parse correctly."""
        id_type, value = parse_project_id("git:github.com/user/repo")
        assert id_type == "git"
        assert value == "github.com/user/repo"

    def test_uuid_prefix(self):
        """uuid: prefix should parse correctly."""
        id_type, value = parse_project_id("uuid:550e8400-e29b-41d4-a716-446655440000")
        assert id_type == "uuid"
        assert value == "550e8400-e29b-41d4-a716-446655440000"

    def test_hash_prefix(self):
        """hash: prefix should parse correctly."""
        id_type, value = parse_project_id("hash:a1b2c3d4e5f67890")
        assert id_type == "hash"
        assert value == "a1b2c3d4e5f67890"

    def test_path_prefix(self):
        """path: prefix should parse correctly."""
        id_type, value = parse_project_id("path:/Users/dev/project")
        assert id_type == "path"
        assert value == "/Users/dev/project"

    def test_legacy_no_prefix(self):
        """Legacy ID without prefix should be treated as path."""
        id_type, value = parse_project_id("/Users/dev/project")
        assert id_type == "path"
        assert value == "/Users/dev/project"


class TestInferProjectFromSessionPath:
    """Tests for infer_project_from_session_path function."""

    def test_valid_session_path_with_git(self):
        """Session path to git repo should return git: ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock session structure
            session_dir = Path(tmpdir) / ".claude" / "projects" / f"-{tmpdir[1:].replace('/', '-')}"
            session_dir.mkdir(parents=True)
            session_file = session_dir / "abc123.jsonl"
            session_file.write_text("{}")

            # Initialize git in the project dir
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "remote", "add", "origin", "git@github.com:test/project.git"],
                cwd=tmpdir,
                capture_output=True,
            )

            result = infer_project_from_session_path(session_file)
            assert result == "git:github.com/test/project"

    def test_nonexistent_path(self):
        """Session path to non-existent project should return path: ID."""
        session_path = Path("/home/user/.claude/projects/-nonexistent-path/session.jsonl")
        result = infer_project_from_session_path(session_path)
        assert result == "path:/nonexistent/path"

    def test_not_session_path(self):
        """Non-session path should return None."""
        result = infer_project_from_session_path("/some/random/path.jsonl")
        assert result is None


class TestNormalizeProjectId:
    """Tests for normalize_project_id function."""

    def test_already_prefixed(self):
        """Already prefixed ID should pass through."""
        assert normalize_project_id("git:github.com/user/repo") == "git:github.com/user/repo"

    def test_legacy_path(self):
        """Legacy path should be converted to prefixed ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = normalize_project_id(tmpdir)
            assert result is not None
            assert result.startswith("path:") or result.startswith("hash:")

    def test_fallback_path(self):
        """Fallback path should be used when project_id is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = normalize_project_id(None, fallback_path=tmpdir)
            assert result is not None

    def test_none_no_fallback(self):
        """None without fallback should return None."""
        assert normalize_project_id(None) is None


class TestExtractProjectName:
    """Tests for extract_project_name function."""

    def test_git_id(self):
        """Git ID should extract repo name."""
        assert extract_project_name("git:github.com/user/myproject") == "myproject"

    def test_path_id(self):
        """Path ID should extract directory name."""
        assert extract_project_name("path:/Users/shimon/repo/3dtex") == "3dtex"

    def test_uuid_id(self):
        """UUID ID should return the UUID (no path structure)."""
        result = extract_project_name("uuid:550e8400-e29b-41d4-a716-446655440000")
        assert result == "550e8400-e29b-41d4-a716-446655440000"

    def test_legacy_path(self):
        """Legacy path should extract directory name."""
        assert extract_project_name("/Users/dev/project") == "project"


class TestGetProjectIdLegacy:
    """Tests for backwards compatibility."""

    def test_returns_plain_path(self):
        """Legacy function should return path without prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_project_id_legacy(tmpdir)
            assert not result.startswith("git:")
            assert not result.startswith("uuid:")
            assert not result.startswith("hash:")
            assert not result.startswith("path:")
            assert tmpdir in result or Path(tmpdir).resolve().as_posix() in result
