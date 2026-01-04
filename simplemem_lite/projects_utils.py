"""Project ID utilities for 1:1 project-id to project-path mapping.

The canonical project_id is the absolute path of the project root.
This ensures:
- 1:1 mapping between project paths and IDs
- Easy discovery: project_id IS the path
- No encoding/decoding ambiguity
"""

import re
from pathlib import Path

from simplemem_lite.log_config import get_logger

log = get_logger("projects_utils")


def get_project_id(path: str | Path) -> str:
    """Convert a path to a canonical project_id.

    The project_id is simply the absolute, resolved path.
    This ensures 1:1 mapping and makes discovery trivial.

    Args:
        path: Project root path (absolute or relative)

    Returns:
        Canonical absolute path as project_id

    Example:
        >>> get_project_id("~/repo/myproject")
        '/Users/shimon/repo/myproject'
        >>> get_project_id(".")
        '/Users/shimon/current/dir'
    """
    return str(Path(path).expanduser().resolve())


def infer_project_from_session_path(session_path: str | Path) -> str | None:
    """Infer project_id from a Claude Code session path.

    Claude Code stores sessions at:
    ~/.claude/projects/{encoded-path}/{session-id}.jsonl

    The encoded-path uses hyphens for separators, making decoding lossy.
    However, we can attempt to reconstruct the original path.

    Args:
        session_path: Path to the session JSONL file

    Returns:
        Inferred absolute path as project_id, or None if inference fails

    Example:
        >>> infer_project_from_session_path(
        ...     "~/.claude/projects/-Users-shimon-repo-3dtex/abc123.jsonl"
        ... )
        '/Users/shimon/repo/3dtex'
    """
    session_path = Path(session_path).expanduser().resolve()

    # Check if this is a Claude session path
    if "projects" not in session_path.parts:
        log.debug(f"Not a Claude session path: {session_path}")
        return None

    # Find the encoded path component (parent of the .jsonl file)
    # Structure: ~/.claude/projects/{encoded-path}/{session}.jsonl
    try:
        projects_idx = session_path.parts.index("projects")
        if projects_idx + 1 >= len(session_path.parts) - 1:
            log.debug(f"Malformed session path: {session_path}")
            return None

        encoded_path = session_path.parts[projects_idx + 1]
    except (ValueError, IndexError):
        log.debug(f"Could not extract encoded path from: {session_path}")
        return None

    # Decode: replace leading hyphen with /, then hyphens with /
    # Note: This is lossy if original path contained hyphens
    decoded = _decode_project_path(encoded_path)

    if decoded and Path(decoded).exists():
        log.debug(f"Inferred project_id: {decoded}")
        return decoded

    # Path doesn't exist - might be lossy decoding issue
    log.debug(f"Decoded path doesn't exist: {decoded}")
    return None


def _decode_project_path(encoded: str) -> str | None:
    """Decode a Claude-encoded project path.

    Claude encodes paths by replacing / with -.
    Leading - indicates root /.

    Args:
        encoded: Encoded path component (e.g., "-Users-shimon-repo-myproject")

    Returns:
        Decoded absolute path, or None if invalid

    Limitations:
        - Lossy: can't distinguish original hyphens from path separators
        - Best effort: returns most likely interpretation
    """
    if not encoded:
        return None

    # Leading hyphen indicates absolute path (starts with /)
    if encoded.startswith("-"):
        # Replace hyphens with slashes
        decoded = "/" + encoded[1:].replace("-", "/")
    else:
        # Relative path (shouldn't happen in practice)
        decoded = encoded.replace("-", "/")

    return decoded


def normalize_project_id(project_id: str | None, fallback_path: str | None = None) -> str | None:
    """Normalize a project_id to canonical form.

    Handles various input formats:
    - Already canonical absolute path: returns as-is
    - Relative path: resolves to absolute
    - None with fallback: uses fallback path

    Args:
        project_id: Input project_id (may be None or relative)
        fallback_path: Path to use if project_id is None

    Returns:
        Canonical absolute path as project_id, or None if no valid input
    """
    if project_id:
        return get_project_id(project_id)

    if fallback_path:
        return get_project_id(fallback_path)

    return None


def extract_project_name(project_id: str) -> str:
    """Extract a human-readable project name from project_id.

    Args:
        project_id: Canonical absolute path

    Returns:
        Last component of the path (project directory name)

    Example:
        >>> extract_project_name("/Users/shimon/repo/3dtex")
        '3dtex'
    """
    return Path(project_id).name
