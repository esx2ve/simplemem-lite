"""Code indexing for SimpleMem Lite.

Provides semantic search over codebases using the same embedding infrastructure.
P1: Entity linking bridges code chunks with memory graph.
P2: Git-based staleness detection for code index freshness.
"""

import re
import subprocess
import uuid as uuid_lib
from pathlib import Path
from typing import Any

import pathspec

from simplemem_lite.config import Config
from simplemem_lite.db import DatabaseManager
from simplemem_lite.embeddings import embed, embed_batch
from simplemem_lite.logging import get_logger

log = get_logger("code_index")

# Regex patterns for Python entity extraction
_PYTHON_PATTERNS = {
    "import": re.compile(r"^(?:from\s+(\S+)\s+)?import\s+([^#\n]+)", re.MULTILINE),
    "class": re.compile(r"^class\s+(\w+)", re.MULTILINE),
    "function": re.compile(r"^def\s+(\w+)", re.MULTILINE),
    "async_function": re.compile(r"^async\s+def\s+(\w+)", re.MULTILINE),
}

# JS/TS patterns
_JS_PATTERNS = {
    "import": re.compile(r"import\s+(?:{[^}]+}|[^;]+)\s+from\s+['\"]([^'\"]+)['\"]"),
    "class": re.compile(r"^(?:export\s+)?class\s+(\w+)", re.MULTILINE),
    "function": re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)", re.MULTILINE),
    "const_func": re.compile(r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>", re.MULTILINE),
}

# Built-in gitignore-style patterns (always excluded)
# These use gitignore/glob syntax, not exact directory names
_BUILTIN_IGNORE_PATTERNS = """
# Version control
.git/
.svn/
.hg/

# Python virtual environments (catches venv, .venv, myenv, texify_venv, etc.)
*venv*/
*env/
.venv/
venv/
ENV/
env.bak/
venv.bak/

# Python bytecode and caches
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.mypy_cache/
.ruff_cache/
.tox/
.nox/
*.egg-info/
*.egg
dist/
build/
eggs/
.eggs/

# Python site-packages (always noise)
**/site-packages/
**/lib/python*/

# Node.js
node_modules/
.npm/
.yarn/
bower_components/
.pnpm-store/

# IDE/Editor
.idea/
.vscode/
.vs/
*.swp
*.swo
*~

# Build outputs
target/
out/
bin/
obj/
.next/
.nuxt/

# Coverage/Test
coverage/
.coverage
htmlcov/
.nyc_output/

# Misc
.DS_Store
Thumbs.db
*.log
"""


class CodeIndexer:
    """Indexes code files for semantic search.

    Uses the existing LanceDB and embedding infrastructure to provide
    code search capabilities alongside memory search.
    """

    def __init__(self, db: DatabaseManager, config: Config):
        """Initialize the code indexer.

        Args:
            db: Database manager instance
            config: SimpleMem Lite configuration
        """
        self.db = db
        self.config = config
        # Cache pathspec objects per project root
        self._pathspec_cache: dict[str, pathspec.PathSpec] = {}
        log.info("CodeIndexer initialized")

    def _load_gitignore(self, root: Path) -> list[str]:
        """Load .gitignore patterns from project root and parent directories.

        Args:
            root: Project root directory

        Returns:
            List of gitignore pattern lines
        """
        patterns = []
        gitignore_path = root / ".gitignore"
        if gitignore_path.exists():
            try:
                content = gitignore_path.read_text(encoding="utf-8", errors="ignore")
                patterns.extend(content.splitlines())
                log.debug(f"Loaded {len(patterns)} patterns from {gitignore_path}")
            except Exception as e:
                log.warning(f"Failed to read .gitignore: {e}")
        return patterns

    def _build_pathspec(self, root: Path) -> pathspec.PathSpec:
        """Build a PathSpec combining built-in patterns and .gitignore.

        Results are cached per project root for efficiency.

        Args:
            root: Project root directory

        Returns:
            PathSpec object for matching files to exclude
        """
        root_str = str(root.resolve())
        if root_str in self._pathspec_cache:
            return self._pathspec_cache[root_str]

        # Combine built-in patterns with project .gitignore
        all_patterns = []

        # Add built-in patterns (strip comments and empty lines for efficiency)
        for line in _BUILTIN_IGNORE_PATTERNS.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                all_patterns.append(line)

        # Add project-specific .gitignore
        gitignore_patterns = self._load_gitignore(root)
        for line in gitignore_patterns:
            line = line.strip()
            if line and not line.startswith("#"):
                all_patterns.append(line)

        spec = pathspec.PathSpec.from_lines("gitwildmatch", all_patterns)
        self._pathspec_cache[root_str] = spec
        log.info(f"Built pathspec with {len(all_patterns)} patterns for {root}")
        return spec

    def _should_exclude(self, path: Path, root: Path) -> bool:
        """Check if a path should be excluded from indexing using pathspec.

        Uses gitignore-style pattern matching for robust exclusion of
        virtual environments, node_modules, build outputs, etc.

        Args:
            path: File or directory path to check
            root: Project root for loading .gitignore

        Returns:
            True if path should be excluded
        """
        spec = self._build_pathspec(root)

        # Get path relative to root for matching
        try:
            rel_path = path.relative_to(root)
        except ValueError:
            # Path is not under root - shouldn't happen but be safe
            return False

        # Check if path matches any exclusion pattern
        # We check both the file path and directory components
        rel_str = str(rel_path)

        # pathspec.match_file returns True if the path matches any pattern
        if spec.match_file(rel_str):
            log.trace(f"Excluding (pattern match): {rel_str}")
            return True

        # Also check if any parent directory matches (for directory patterns)
        for parent in rel_path.parents:
            parent_str = str(parent)
            if parent_str and parent_str != "." and spec.match_file(parent_str + "/"):
                log.trace(f"Excluding (parent match): {rel_str} (parent: {parent_str})")
                return True

        return False

    def index_directory(
        self,
        root_path: str | Path,
        patterns: list[str] | None = None,
        clear_existing: bool = True,
    ) -> dict[str, Any]:
        """Index all matching files in a directory.

        Args:
            root_path: Root directory to index
            patterns: Glob patterns to match (default: from config)
            clear_existing: Whether to clear existing index for this root

        Returns:
            Dict with indexing stats
        """
        root = Path(root_path).resolve()
        if not root.exists():
            log.error(f"Directory does not exist: {root}")
            return {"error": f"Directory not found: {root}", "files_indexed": 0}

        patterns = patterns or self.config.code_patterns_list
        log.info(f"Indexing directory: {root} with patterns: {patterns}")

        if clear_existing:
            cleared = self.db.clear_code_index(str(root))
            log.info(f"Cleared {cleared} existing chunks")

        # Find all matching files, excluding common non-source directories
        files = []
        for pattern in patterns:
            for f in root.glob(pattern):
                if f.is_file() and not self._should_exclude(f, root):
                    files.append(f)

        # Remove duplicates
        files = sorted(set(files))
        log.info(f"Found {len(files)} files to index")

        if not files:
            return {"files_indexed": 0, "chunks_created": 0, "project_root": str(root)}

        # Process files in batches
        total_chunks = 0
        files_indexed = 0
        errors = []

        for file_path in files:
            try:
                chunks = self._index_file(file_path, root)
                total_chunks += chunks
                files_indexed += 1
                if files_indexed % 10 == 0:
                    log.debug(f"Progress: {files_indexed}/{len(files)} files indexed")
            except Exception as e:
                log.error(f"Failed to index {file_path}: {e}")
                errors.append({"file": str(file_path), "error": str(e)})

        log.info(f"Indexing complete: {files_indexed} files, {total_chunks} chunks")

        # P2: Save project index metadata for staleness detection
        git_info = self._get_git_info(root)
        commit_hash = git_info.get("commit_hash") if git_info.get("is_git_repo") else None
        self.db.set_project_index_metadata(
            project_root=str(root),
            commit_hash=commit_hash,
            file_count=files_indexed,
            chunk_count=total_chunks,
        )

        return {
            "files_indexed": files_indexed,
            "chunks_created": total_chunks,
            "project_root": str(root),
            "commit_hash": commit_hash,
            "is_git_repo": git_info.get("is_git_repo", False),
            "errors": errors if errors else None,
        }

    def _index_file(self, file_path: Path, project_root: Path) -> int:
        """Index a single file.

        Args:
            file_path: Path to the file
            project_root: Project root directory

        Returns:
            Number of chunks created
        """
        log.trace(f"Indexing file: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.warning(f"Failed to read {file_path}: {e}")
            raise

        # Skip empty files
        if not content.strip():
            return 0

        # Split into chunks
        chunks = self._split_code(content, str(file_path))
        if not chunks:
            return 0

        # Generate embeddings for all chunks
        texts = [c["content"] for c in chunks]
        embeddings = embed_batch(texts, self.config)

        # Prepare records for database
        records = []
        relative_path = str(file_path.relative_to(project_root))
        for chunk, embedding in zip(chunks, embeddings):
            records.append({
                "uuid": str(uuid_lib.uuid4()),
                "vector": embedding,
                "content": chunk["content"],
                "filepath": relative_path,
                "project_root": str(project_root),
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
            })

        # Add to vector database
        self.db.add_code_chunks(records)

        # P1: Link chunks to entities in graph (for cross-referencing with memories)
        total_entities = 0
        for record in records:
            entities_linked = self._link_chunk_entities(
                chunk_uuid=record["uuid"],
                filepath=record["filepath"],
                project_root=record["project_root"],
                start_line=record["start_line"],
                end_line=record["end_line"],
                chunk_content=record["content"],
            )
            total_entities += entities_linked

        log.debug(f"Linked {total_entities} entities for {len(records)} chunks in {relative_path}")
        return len(records)

    def _split_code(self, content: str, filepath: str) -> list[dict[str, Any]]:
        """Split code into semantic chunks.

        Uses simple line-based chunking with overlap.
        Future: Could use tree-sitter for AST-aware chunking.

        Args:
            content: File content
            filepath: File path (for logging)

        Returns:
            List of chunks with content, start_line, end_line
        """
        lines = content.split("\n")
        chunk_size = self.config.code_chunk_size
        overlap = self.config.code_chunk_overlap

        # Estimate chars per line (for chunk sizing)
        avg_line_len = len(content) / max(len(lines), 1)
        lines_per_chunk = max(10, int(chunk_size / max(avg_line_len, 20)))
        overlap_lines = max(2, int(overlap / max(avg_line_len, 20)))

        chunks = []
        i = 0
        while i < len(lines):
            end = min(i + lines_per_chunk, len(lines))
            chunk_lines = lines[i:end]
            chunk_content = "\n".join(chunk_lines)

            # Skip very small chunks (less than 50 chars)
            if len(chunk_content.strip()) >= 50:
                chunks.append({
                    "content": chunk_content,
                    "start_line": i + 1,  # 1-indexed
                    "end_line": end,
                })

            # Move forward, keeping some overlap
            i = end - overlap_lines if end < len(lines) else end

        log.trace(f"Split {filepath} into {len(chunks)} chunks")
        return chunks

    def _extract_entities(self, content: str, filepath: str) -> list[dict[str, str]]:
        """Extract entities from code content for graph linking.

        Args:
            content: Code content
            filepath: File path (determines language detection)

        Returns:
            List of {name, type, relation} dicts
        """
        entities = []
        suffix = Path(filepath).suffix.lower()

        # Always add file entity
        entities.append({
            "name": filepath,
            "type": "file",
            "relation": "IN_FILE",
        })

        # Select patterns based on language
        if suffix == ".py":
            patterns = _PYTHON_PATTERNS
        elif suffix in (".js", ".ts", ".jsx", ".tsx"):
            patterns = _JS_PATTERNS
        else:
            return entities  # Only file entity for unknown languages

        # Extract imports
        if "import" in patterns:
            for match in patterns["import"].finditer(content):
                if suffix == ".py":
                    # Python: from X import Y or import X
                    module = match.group(1) or match.group(2).split(",")[0].split()[0]
                    if module:
                        entities.append({
                            "name": module.strip(),
                            "type": "module",
                            "relation": "IMPORTS",
                        })
                else:
                    # JS/TS: import ... from "module"
                    module = match.group(1)
                    if module:
                        entities.append({
                            "name": module,
                            "type": "module",
                            "relation": "IMPORTS",
                        })

        # Extract classes
        if "class" in patterns:
            for match in patterns["class"].finditer(content):
                entities.append({
                    "name": match.group(1),
                    "type": "class",
                    "relation": "DEFINES",
                })

        # Extract functions
        for key in ("function", "async_function", "const_func"):
            if key in patterns:
                for match in patterns[key].finditer(content):
                    entities.append({
                        "name": match.group(1),
                        "type": "function",
                        "relation": "DEFINES",
                    })

        log.trace(f"Extracted {len(entities)} entities from {filepath}")
        return entities

    def _link_chunk_entities(
        self,
        chunk_uuid: str,
        filepath: str,
        project_root: str,
        start_line: int,
        end_line: int,
        chunk_content: str,
    ) -> int:
        """Create graph node for chunk and link to extracted entities.

        Args:
            chunk_uuid: UUID of the chunk
            filepath: Relative file path
            project_root: Project root path
            start_line: Start line number
            end_line: End line number
            chunk_content: Content of the chunk

        Returns:
            Number of entities linked
        """
        # Create CodeChunk node in graph
        self.db.add_code_chunk_node(
            uuid=chunk_uuid,
            filepath=filepath,
            project_root=project_root,
            start_line=start_line,
            end_line=end_line,
        )

        # Extract and link entities from this chunk
        entities = self._extract_entities(chunk_content, filepath)
        for entity in entities:
            self.db.link_code_to_entity(
                chunk_uuid=chunk_uuid,
                entity_name=entity["name"],
                entity_type=entity["type"],
                relation=entity["relation"],
            )

        return len(entities)

    # ═══════════════════════════════════════════════════════════════════════════════
    # P3: SINGLE-FILE INDEXING (FOR INCREMENTAL UPDATES)
    # ═══════════════════════════════════════════════════════════════════════════════

    def index_file(
        self,
        file_path: str | Path,
        project_root: str | Path,
    ) -> dict[str, Any]:
        """Index or re-index a single file (incremental update).

        Deletes existing chunks for this file, then re-indexes it.
        Used by the file watcher for real-time updates.

        Args:
            file_path: Absolute path to the file
            project_root: Project root directory

        Returns:
            Dict with chunks_created, filepath, and any errors
        """
        file_path = Path(file_path).resolve()
        project_root = Path(project_root).resolve()

        if not file_path.exists():
            log.warning(f"File does not exist: {file_path}")
            return {"error": f"File not found: {file_path}", "chunks_created": 0}

        if not file_path.is_file():
            log.warning(f"Not a file: {file_path}")
            return {"error": f"Not a file: {file_path}", "chunks_created": 0}

        try:
            relative_path = str(file_path.relative_to(project_root))
        except ValueError:
            log.error(f"File {file_path} is not under project root {project_root}")
            return {"error": "File not under project root", "chunks_created": 0}

        log.info(f"Indexing single file: {relative_path}")

        # Delete existing chunks for this file
        deleted = self.db.delete_chunks_by_filepath(str(project_root), relative_path)
        if deleted > 0:
            log.debug(f"Deleted {deleted} existing chunks for {relative_path}")

        # Re-index the file
        try:
            chunks_created = self._index_file(file_path, project_root)
            log.info(f"Indexed {relative_path}: {chunks_created} chunks created")
            return {
                "filepath": relative_path,
                "project_root": str(project_root),
                "chunks_created": chunks_created,
                "chunks_deleted": deleted,
            }
        except Exception as e:
            log.error(f"Failed to index {relative_path}: {e}")
            return {
                "filepath": relative_path,
                "project_root": str(project_root),
                "error": str(e),
                "chunks_created": 0,
            }

    def delete_file(
        self,
        file_path: str | Path,
        project_root: str | Path,
    ) -> dict[str, Any]:
        """Remove a file from the index.

        Used by the file watcher when files are deleted.

        Args:
            file_path: Absolute path to the file (or what it was)
            project_root: Project root directory

        Returns:
            Dict with chunks_deleted count
        """
        file_path = Path(file_path)
        project_root = Path(project_root).resolve()

        try:
            relative_path = str(file_path.relative_to(project_root))
        except ValueError:
            # If file_path is already relative, use it directly
            relative_path = str(file_path)

        log.info(f"Deleting file from index: {relative_path}")

        deleted = self.db.delete_chunks_by_filepath(str(project_root), relative_path)
        log.info(f"Deleted {deleted} chunks for {relative_path}")

        return {
            "filepath": relative_path,
            "project_root": str(project_root),
            "chunks_deleted": deleted,
        }

    def search(
        self,
        query: str,
        limit: int = 10,
        project_root: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the code index.

        Args:
            query: Search query
            limit: Maximum results
            project_root: Optional filter by project

        Returns:
            List of matching code chunks with scores
        """
        log.info(f"Code search: '{query[:50]}...' limit={limit}")

        # Generate query embedding
        query_vector = embed(query, self.config)

        # Search
        results = self.db.search_code(query_vector, limit, project_root)

        # Format results
        formatted = []
        for r in results:
            formatted.append({
                "filepath": r.get("filepath", ""),
                "content": r.get("content", ""),
                "start_line": r.get("start_line", 0),
                "end_line": r.get("end_line", 0),
                "project_root": r.get("project_root", ""),
                "score": 1.0 - r.get("_distance", 0.0),  # Convert distance to similarity
            })

        log.info(f"Code search returned {len(formatted)} results")
        return formatted

    # ═══════════════════════════════════════════════════════════════════════════════
    # P2: GIT-BASED STALENESS DETECTION
    # ═══════════════════════════════════════════════════════════════════════════════

    def _get_git_info(self, path: Path) -> dict[str, Any]:
        """Get git status for a directory.

        Args:
            path: Directory path to check

        Returns:
            Dict with is_git_repo, commit_hash, has_uncommitted_changes, uncommitted_files
        """
        try:
            # Check if this is a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return {"is_git_repo": False}

            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            commit_hash = result.stdout.strip() if result.returncode == 0 else None

            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            uncommitted_files = []
            if result.returncode == 0 and result.stdout.strip():
                uncommitted_files = [
                    line[3:] for line in result.stdout.strip().split("\n") if line
                ]

            log.debug(f"Git info for {path}: hash={commit_hash[:8] if commit_hash else 'N/A'}, uncommitted={len(uncommitted_files)}")
            return {
                "is_git_repo": True,
                "commit_hash": commit_hash,
                "has_uncommitted_changes": len(uncommitted_files) > 0,
                "uncommitted_files": uncommitted_files,
            }

        except subprocess.TimeoutExpired:
            log.warning(f"Git command timed out for {path}")
            return {"is_git_repo": False, "error": "timeout"}
        except FileNotFoundError:
            log.warning("Git not found in PATH")
            return {"is_git_repo": False, "error": "git_not_found"}
        except Exception as e:
            log.error(f"Git info failed: {e}")
            return {"is_git_repo": False, "error": str(e)}

    def _get_changed_files(self, project_root: Path, since_hash: str) -> list[str]:
        """Get files changed since a specific commit.

        Args:
            project_root: Project root path
            since_hash: Commit hash to compare from

        Returns:
            List of changed file paths (relative to project root)
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", since_hash, "HEAD"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                log.warning(f"Git diff failed: {result.stderr}")
                return []

            changed = [f for f in result.stdout.strip().split("\n") if f]

            # Filter to only code files matching our patterns
            patterns = self.config.code_patterns_list
            code_files = []
            for f_str in changed:
                f_path = Path(f_str)
                if any(f_path.match(p) for p in patterns):
                    code_files.append(f_str)

            log.debug(f"Changed files since {since_hash[:8]}: {len(code_files)} code files")
            return code_files

        except subprocess.TimeoutExpired:
            log.warning("Git diff timed out")
            return []
        except Exception as e:
            log.error(f"Get changed files failed: {e}")
            return []

    def check_staleness(self, project_root: str) -> dict[str, Any]:
        """Check if code index is stale for a project.

        Args:
            project_root: Absolute path to project root

        Returns:
            Dict with staleness status and details
        """
        root = Path(project_root).resolve()
        log.info(f"Checking staleness for: {root}")

        # Get stored index metadata
        metadata = self.db.get_project_index_metadata(str(root))
        if metadata is None:
            return {
                "is_indexed": False,
                "is_stale": True,  # Not indexed = stale
                "reason": "Project not indexed",
                "project_root": str(root),
            }

        # Get current git info
        git_info = self._get_git_info(root)
        if not git_info.get("is_git_repo"):
            return {
                "is_indexed": True,
                "is_stale": False,  # Can't determine staleness without git
                "is_git_repo": False,
                "indexed_at": metadata.get("indexed_at"),
                "file_count": metadata.get("file_count"),
                "chunk_count": metadata.get("chunk_count"),
                "project_root": str(root),
                "reason": "Not a git repository - staleness cannot be detected",
            }

        indexed_hash = metadata.get("last_commit_hash")
        current_hash = git_info.get("commit_hash")

        # Check if commits differ
        is_stale = indexed_hash != current_hash
        changed_files = []

        if is_stale and indexed_hash and current_hash:
            changed_files = self._get_changed_files(root, indexed_hash)

        # Also consider uncommitted changes
        uncommitted = git_info.get("uncommitted_files", [])
        has_uncommitted = git_info.get("has_uncommitted_changes", False)

        result = {
            "is_indexed": True,
            "is_stale": is_stale or has_uncommitted,
            "is_git_repo": True,
            "indexed_hash": indexed_hash,
            "current_hash": current_hash,
            "indexed_at": metadata.get("indexed_at"),
            "file_count": metadata.get("file_count"),
            "chunk_count": metadata.get("chunk_count"),
            "changed_files": changed_files,
            "uncommitted_files": uncommitted if has_uncommitted else [],
            "project_root": str(root),
        }

        if is_stale:
            result["reason"] = f"Index from commit {indexed_hash[:8] if indexed_hash else 'unknown'}, current is {current_hash[:8] if current_hash else 'unknown'}"
        elif has_uncommitted:
            result["reason"] = f"{len(uncommitted)} uncommitted changes"
        else:
            result["reason"] = "Index is up to date"

        log.info(f"Staleness check: is_stale={result['is_stale']}, reason={result['reason']}")
        return result
