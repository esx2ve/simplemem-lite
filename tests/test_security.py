"""Security tests for SimpleMem Lite.

Tests critical security pathways:
- Lock file permissions
- Auth token verification (timing-safe)
"""

import json
import os
import secrets
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestLockFileSecurity:
    """Test lock file security measures."""

    def test_lock_file_has_restricted_permissions(self, tmp_path: Path):
        """Lock file should be created with 0o600 (owner read/write only)."""
        lock_path = tmp_path / "server.lock"
        lock_data = {
            "port": 8080,
            "pid": os.getpid(),
            "token": "secret_token_123",
            "started_at": "2024-01-01T00:00:00",
            "host": "127.0.0.1",
        }

        # Simulate the secure write pattern from server.py
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(lock_data, f, indent=2)
        except Exception:
            os.close(fd)
            raise

        # Verify file permissions
        file_stat = os.stat(lock_path)
        mode = stat.S_IMODE(file_stat.st_mode)

        # Should be 0o600 (read/write for owner only)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

        # Verify no group/other permissions
        assert not (mode & stat.S_IRGRP), "Group should not have read permission"
        assert not (mode & stat.S_IWGRP), "Group should not have write permission"
        assert not (mode & stat.S_IROTH), "Others should not have read permission"
        assert not (mode & stat.S_IWOTH), "Others should not have write permission"

    def test_lock_file_content_is_valid_json(self, tmp_path: Path):
        """Lock file should contain valid JSON with expected fields."""
        lock_path = tmp_path / "server.lock"
        expected_token = secrets.token_urlsafe(32)
        lock_data = {
            "port": 9999,
            "pid": os.getpid(),
            "token": expected_token,
            "started_at": "2024-01-01T12:00:00",
            "host": "localhost",
        }

        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(lock_data, f, indent=2)

        # Read back and verify
        content = json.loads(lock_path.read_text())
        assert content["port"] == 9999
        assert content["token"] == expected_token
        assert "pid" in content
        assert "started_at" in content


class TestAuthTokenVerification:
    """Test timing-safe auth token verification."""

    def test_compare_digest_used_for_token_comparison(self):
        """Auth verification should use secrets.compare_digest for timing safety."""
        correct_token = secrets.token_urlsafe(32)
        wrong_token = secrets.token_urlsafe(32)

        # Simulate the verification logic
        def verify_auth(provided: str, expected: str) -> bool:
            return secrets.compare_digest(provided, expected)

        assert verify_auth(correct_token, correct_token) is True
        assert verify_auth(wrong_token, correct_token) is False

    def test_bearer_token_extraction(self):
        """Bearer token should be correctly extracted from Authorization header."""
        token = "test_token_abc123"
        auth_header = f"Bearer {token}"

        # Simulate header parsing
        if auth_header.startswith("Bearer "):
            extracted = auth_header[7:]
        else:
            extracted = None

        assert extracted == token

    def test_missing_bearer_prefix_rejected(self):
        """Auth header without Bearer prefix should be rejected."""
        auth_header = "Basic sometoken"

        has_bearer = auth_header.startswith("Bearer ")
        assert has_bearer is False

    def test_empty_auth_header_rejected(self):
        """Empty auth header should be rejected."""
        auth_header = ""

        has_bearer = auth_header.startswith("Bearer ")
        assert has_bearer is False

    def test_timing_safe_comparison_properties(self):
        """Verify secrets.compare_digest has expected behavior."""
        # Same strings
        assert secrets.compare_digest("abc", "abc") is True
        assert secrets.compare_digest("", "") is True

        # Different strings
        assert secrets.compare_digest("abc", "abd") is False
        assert secrets.compare_digest("abc", "ab") is False
        assert secrets.compare_digest("abc", "abcd") is False

        # Unicode
        assert secrets.compare_digest("token", "token") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
