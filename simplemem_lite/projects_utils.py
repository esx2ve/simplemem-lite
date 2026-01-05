"""Project ID utilities with hybrid identification strategies.

Hierarchical project identification (most stable first):
1. Git remote URL - stable across machines and paths
2. Config file UUID (.simplemem.json) - explicit user control
3. Content hash of project markers - semi-stable
4. Resolved absolute path - fallback for backwards compatibility

ID Format Prefixes:
- git:github.com/user/repo - Git remote based
- uuid:550e8400-e29b-... - Config file UUID
- hash:a1b2c3d4e5f6... - Content hash
- path:/Users/dev/project - Fallback path
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from simplemem_lite.log_config import get_logger

log = get_logger("projects_utils")

# Project marker files in priority order
PROJECT_MARKERS = [
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "CMakeLists.txt",
    "Makefile",
]

# Git command timeout in seconds
GIT_TIMEOUT = 2


def normalize_git_url(url: str) -> str:
    """Normalize git URL to consistent format.

    Handles various git URL formats and normalizes to: host/owner/repo

    Args:
        url: Git remote URL in any format

    Returns:
        Normalized URL without protocol, auth, or .git suffix

    Examples:
        >>> normalize_git_url("git@github.com:user/repo.git")
        'github.com/user/repo'
        >>> normalize_git_url("https://github.com/user/repo.git")
        'github.com/user/repo'
        >>> normalize_git_url("ssh://git@bitbucket.org/team/repo")
        'bitbucket.org/team/repo'
        >>> normalize_git_url("https://gitlab.company.com:8443/group/repo.git")
        'gitlab.company.com:8443/group/repo'
    """
    if not url:
        return ""

    # Remove .git suffix
    url = re.sub(r"\.git$", "", url.strip())

    # HTTPS format: https://github.com/user/repo (check FIRST - more specific)
    # Supports optional port: https://gitlab.company.com:8443/group/repo
    https_match = re.match(r"^https?://(?:[\w.-]+@)?([\w.-]+(?::\d+)?)/(.+)$", url)
    if https_match:
        host, path = https_match.groups()
        return f"{host}/{path}"

    # SSH with explicit protocol: ssh://git@bitbucket.org/team/repo
    ssh_protocol_match = re.match(r"^ssh://(?:[\w.-]+@)?([\w.-]+(?::\d+)?)/(.+)$", url)
    if ssh_protocol_match:
        host, path = ssh_protocol_match.groups()
        return f"{host}/{path}"

    # SSH shorthand format: git@github.com:user/repo (colon separator, no protocol)
    ssh_shorthand_match = re.match(r"^(?:[\w.-]+@)?([\w.-]+):(.+)$", url)
    if ssh_shorthand_match:
        host, path = ssh_shorthand_match.groups()
        # Don't match if path looks like a port number followed by path (would be ssh://)
        if not re.match(r"^\d+/", path):
            return f"{host}/{path}"

    # Already normalized or unknown format
    return url


def get_git_remote_url(path: Path) -> str | None:
    """Extract normalized git remote origin URL.

    Args:
        path: Directory to check for git repo

    Returns:
        Normalized git remote URL or None if not a git repo
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
        if result.returncode == 0 and result.stdout.strip():
            raw_url = result.stdout.strip()
            normalized = normalize_git_url(raw_url)
            log.debug(f"Git remote: {raw_url} -> {normalized}")
            return normalized
    except subprocess.TimeoutExpired:
        log.warning(f"Git command timed out for {path}")
    except FileNotFoundError:
        log.debug("Git not installed")
    except Exception as e:
        log.debug(f"Git remote detection failed: {e}")

    return None


def load_simplemem_config(path: Path) -> dict | None:
    """Load .simplemem.json config file if it exists.

    Args:
        path: Project root directory

    Returns:
        Parsed config dict or None if not found/invalid

    Config Schema:
        {
            "version": 1,
            "project_id": "uuid:550e8400-e29b-41d4-a716-446655440000",
            "name": "my-project",
            "created": "2025-01-05T00:00:00Z"
        }
    """
    config_path = path / ".simplemem.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
            if isinstance(config, dict) and "project_id" in config:
                log.debug(f"Loaded config from {config_path}")
                return config
    except (json.JSONDecodeError, IOError) as e:
        log.warning(f"Failed to load config {config_path}: {e}")

    return None


def _extract_stable_identity(marker: str, content: bytes) -> str | None:
    """Extract stable identity fields from marker file content.

    Parses marker files to extract only stable fields (name, repository)
    that won't change with version bumps or dependency updates.

    Args:
        marker: Marker filename (e.g., "package.json")
        content: Raw file content

    Returns:
        Stable identity string or None if parsing fails
    """
    try:
        text = content.decode("utf-8")

        if marker == "package.json":
            data = json.loads(text)
            # Use name + repository for stable identity
            name = data.get("name", "")
            repo = data.get("repository", {})
            if isinstance(repo, dict):
                repo = repo.get("url", "")
            return f"npm:{name}:{repo}" if name else None

        elif marker == "pyproject.toml":
            # Simple TOML parsing for name field
            import tomllib
            data = tomllib.loads(text)
            name = data.get("project", {}).get("name") or data.get("tool", {}).get("poetry", {}).get("name")
            return f"pypi:{name}" if name else None

        elif marker == "Cargo.toml":
            # Simple TOML parsing for package name
            import tomllib
            data = tomllib.loads(text)
            name = data.get("package", {}).get("name")
            return f"cargo:{name}" if name else None

        elif marker == "go.mod":
            # Extract module path from first line
            for line in text.split("\n"):
                if line.startswith("module "):
                    module = line.split(" ", 1)[1].strip()
                    return f"go:{module}"
            return None

        elif marker == "pom.xml":
            # Extract groupId:artifactId from XML
            import re
            group = re.search(r"<groupId>([^<]+)</groupId>", text)
            artifact = re.search(r"<artifactId>([^<]+)</artifactId>", text)
            if group and artifact:
                return f"maven:{group.group(1)}:{artifact.group(1)}"
            return None

    except Exception as e:
        log.debug(f"Failed to parse {marker} for stable identity: {e}")

    return None


def hash_project_markers(path: Path) -> str | None:
    """Generate hash from stable identity fields in project marker files.

    Extracts only stable fields (name, repository) from marker files,
    ignoring version numbers and dependencies that change frequently.
    Falls back to full content hash if parsing fails.

    Args:
        path: Project root directory

    Returns:
        SHA256 hash prefix (16 chars) or None if no markers found
    """
    for marker in PROJECT_MARKERS:
        marker_path = path / marker
        if marker_path.exists():
            try:
                content = marker_path.read_bytes()

                # Try to extract stable identity first
                stable_identity = _extract_stable_identity(marker, content)
                if stable_identity:
                    hash_digest = hashlib.sha256(stable_identity.encode()).hexdigest()[:16]
                    log.debug(f"Hashed stable identity from {marker}: {stable_identity} -> {hash_digest}")
                    return hash_digest

                # Fallback to full content hash (less stable but still works)
                hash_digest = hashlib.sha256(content).hexdigest()[:16]
                log.debug(f"Hashed full content of {marker}: {hash_digest}")
                return hash_digest

            except IOError as e:
                log.debug(f"Failed to read {marker_path}: {e}")
                continue

    return None


def get_project_id(path: str | Path) -> str:
    """Generate hierarchical project ID using multiple strategies.

    Tries strategies in order of stability:
    1. Git remote URL (most stable, cross-machine)
    2. Config file UUID (.simplemem.json)
    3. Content hash of project markers
    4. Resolved absolute path (fallback)

    Args:
        path: Project root path (absolute or relative)

    Returns:
        Project ID with format prefix (git:, uuid:, hash:, path:)

    Examples:
        >>> get_project_id("/repo/myproject")  # with git remote
        'git:github.com/user/myproject'
        >>> get_project_id("/repo/myproject")  # with .simplemem.json
        'uuid:550e8400-e29b-41d4-a716-446655440000'
        >>> get_project_id("/repo/myproject")  # with package.json
        'hash:a1b2c3d4e5f67890'
        >>> get_project_id("/repo/myproject")  # fallback
        'path:/repo/myproject'
    """
    resolved_path = Path(path).expanduser().resolve()

    # Strategy 1: Git remote URL (most stable)
    git_url = get_git_remote_url(resolved_path)
    if git_url:
        return f"git:{git_url}"

    # Strategy 2: Config file UUID
    config = load_simplemem_config(resolved_path)
    if config and "project_id" in config:
        project_id = config["project_id"]
        # If already prefixed, use as-is; otherwise add uuid: prefix
        if not any(project_id.startswith(p) for p in ["git:", "uuid:", "hash:", "path:"]):
            project_id = f"uuid:{project_id}"
        return project_id

    # Strategy 3: Content hash
    content_hash = hash_project_markers(resolved_path)
    if content_hash:
        return f"hash:{content_hash}"

    # Strategy 4: Fallback to resolved path
    return f"path:{resolved_path}"


def get_project_id_legacy(path: str | Path) -> str:
    """Legacy project ID generation (absolute path only).

    Kept for backwards compatibility during migration.

    Args:
        path: Project root path

    Returns:
        Canonical absolute path as project_id (no prefix)
    """
    return str(Path(path).expanduser().resolve())


def parse_project_id(project_id: str) -> tuple[str, str]:
    """Parse project ID into type and value.

    Args:
        project_id: Full project ID with optional prefix

    Returns:
        Tuple of (type, value) where type is git/uuid/hash/path

    Examples:
        >>> parse_project_id("git:github.com/user/repo")
        ('git', 'github.com/user/repo')
        >>> parse_project_id("/Users/dev/project")
        ('path', '/Users/dev/project')
    """
    for prefix in ["git:", "uuid:", "hash:", "path:"]:
        if project_id.startswith(prefix):
            return prefix[:-1], project_id[len(prefix):]

    # No prefix = legacy path format
    return "path", project_id


def infer_project_from_session_path(session_path: str | Path) -> str | None:
    """Infer project_id from a Claude Code session path.

    Claude Code stores sessions at:
    ~/.claude/projects/{encoded-path}/{session-id}.jsonl

    The encoded-path uses hyphens for separators, making decoding lossy.
    However, we can attempt to reconstruct the original path.

    Args:
        session_path: Path to the session JSONL file

    Returns:
        Inferred project_id (with prefix) or None if inference fails

    Example:
        >>> infer_project_from_session_path(
        ...     "~/.claude/projects/-Users-shimon-repo-3dtex/abc123.jsonl"
        ... )
        'git:github.com/user/3dtex'  # if git repo
        'path:/Users/shimon/repo/3dtex'  # fallback
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

    if decoded:
        # Try to resolve and generate proper project_id
        try:
            resolved = Path(decoded).resolve()
            if resolved.exists():
                # Path exists locally - use full hierarchical ID generation
                return get_project_id(resolved)
            else:
                # Path doesn't exist locally - return as path: prefix
                log.debug(f"Decoded path doesn't exist locally: {decoded}")
                return f"path:{decoded}"
        except Exception:
            log.debug(f"Could not resolve path: {decoded}")
            return f"path:{decoded}"

    # Decoding failed
    log.debug(f"Failed to decode path: {encoded_path}")
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
    - Already has prefix (git:, uuid:, etc.): returns as-is
    - Absolute path without prefix: generates proper ID
    - Relative path: resolves and generates proper ID
    - None with fallback: uses fallback path

    Args:
        project_id: Input project_id (may be None or relative)
        fallback_path: Path to use if project_id is None

    Returns:
        Normalized project_id with prefix, or None if no valid input
    """
    if project_id:
        # Check if already prefixed
        if any(project_id.startswith(p) for p in ["git:", "uuid:", "hash:", "path:"]):
            return project_id
        # Treat as path and generate proper ID
        return get_project_id(project_id)

    if fallback_path:
        return get_project_id(fallback_path)

    return None


def extract_project_name(project_id: str) -> str:
    """Extract a human-readable project name from project_id.

    Args:
        project_id: Project ID with optional prefix

    Returns:
        Human-readable project name

    Examples:
        >>> extract_project_name("git:github.com/user/myproject")
        'myproject'
        >>> extract_project_name("path:/Users/shimon/repo/3dtex")
        '3dtex'
    """
    _, value = parse_project_id(project_id)

    # For git URLs, get the repo name
    if "/" in value:
        return value.split("/")[-1]

    # For paths and UUIDs, get the last component
    return Path(value).name if value else "unknown"
