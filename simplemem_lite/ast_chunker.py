"""AST-aware code chunking using Tree-sitter.

Provides semantic code chunking that respects language syntax,
extracting functions, classes, and methods as atomic units with
rich metadata for improved search and retrieval.

Supports: Python, JavaScript, TypeScript, Go, Rust, Java, Ruby, PHP, C/C++
Falls back to line-based chunking for unsupported languages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from simplemem_lite.log_config import get_logger, log_timing

log = get_logger("ast_chunker")

# Language detection by file extension
LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".hxx": "cpp",
}

# AST node types to extract as chunks per language
CHUNK_NODE_TYPES: dict[str, list[str]] = {
    "python": [
        "function_definition",
        "class_definition",
        "decorated_definition",
    ],
    "javascript": [
        "function_declaration",
        "class_declaration",
        "arrow_function",
        "method_definition",
        "export_statement",
    ],
    "typescript": [
        "function_declaration",
        "class_declaration",
        "arrow_function",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
    ],
    "tsx": [
        "function_declaration",
        "class_declaration",
        "arrow_function",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
    ],
    "rust": [
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
    ],
    "go": [
        "function_declaration",
        "method_declaration",
        "type_declaration",
    ],
    "java": [
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
    ],
    "ruby": [
        "method",
        "class",
        "module",
    ],
    "php": [
        "function_definition",
        "class_declaration",
        "method_declaration",
    ],
    "c": [
        "function_definition",
        "struct_specifier",
    ],
    "cpp": [
        "function_definition",
        "class_specifier",
        "struct_specifier",
    ],
}


@dataclass
class CodeChunk:
    """Represents a semantic code chunk extracted via AST parsing.

    Attributes:
        content: The actual code content
        start_line: 1-indexed start line in source file
        end_line: 1-indexed end line in source file
        node_type: Normalized type (function, class, method, module, interface, type)
        function_name: Name of function/method if applicable
        class_name: Name of containing or defining class if applicable
        language: Programming language
        filepath: Source file path
    """
    content: str
    start_line: int
    end_line: int
    node_type: str  # function | class | method | module | interface | type
    function_name: str | None
    class_name: str | None
    language: str
    filepath: str


def detect_language(filepath: str) -> str | None:
    """Detect programming language from file extension.

    Args:
        filepath: Path to the source file

    Returns:
        Language identifier or None if unknown
    """
    suffix = Path(filepath).suffix.lower()
    return LANGUAGE_MAP.get(suffix)


def chunk_file(content: str, filepath: str, chunk_size: int = 1200, overlap: int = 150) -> list[CodeChunk]:
    """Parse file with Tree-sitter and extract semantic chunks.

    Uses AST parsing for supported languages to extract functions,
    classes, and methods as atomic units. Falls back to line-based
    chunking for unsupported languages or on parse failure.

    Args:
        content: File content as string
        filepath: File path (for language detection)
        chunk_size: Target chunk size for fallback (chars)
        overlap: Overlap between chunks for fallback (chars)

    Returns:
        List of CodeChunk objects with metadata
    """
    if not content or not content.strip():
        log.debug(f"Empty content for {filepath}, skipping")
        return []

    language = detect_language(filepath)

    if not language:
        log.debug(f"Unknown language for {filepath}, using line-based fallback")
        return _fallback_line_chunks(content, filepath, chunk_size, overlap)

    try:
        from tree_sitter_languages import get_parser
    except ImportError:
        log.warning("tree-sitter-languages not installed, using line-based fallback")
        return _fallback_line_chunks(content, filepath, chunk_size, overlap)

    try:
        with log_timing(f"Tree-sitter parsing {filepath}", log):
            parser = get_parser(language)
            tree = parser.parse(bytes(content, "utf8"))
    except Exception as e:
        log.warning(f"Tree-sitter parser failed for {language} ({filepath}): {e}")
        return _fallback_line_chunks(content, filepath, chunk_size, overlap)

    chunks: list[CodeChunk] = []
    node_types = CHUNK_NODE_TYPES.get(language, [])

    log.info(f"Parsing {filepath} with language={language}")

    _extract_chunks_recursive(
        tree.root_node,
        chunks,
        content,
        filepath,
        language,
        node_types,
        parent_class=None,
    )

    # If no chunks extracted (e.g., file has no functions/classes)
    if not chunks and content.strip():
        # For large files, use line-based fallback to avoid single massive chunk
        if len(content) > chunk_size * 2:
            log.debug(f"No semantic chunks in large file {filepath}, using line-based fallback")
            return _fallback_line_chunks(content, filepath, chunk_size, overlap)

        # Small files without functions/classes: treat as one module chunk
        log.debug(f"No semantic chunks in {filepath}, creating module-level chunk")
        chunks.append(CodeChunk(
            content=content,
            start_line=1,
            end_line=content.count('\n') + 1,
            node_type="module",
            function_name=None,
            class_name=None,
            language=language,
            filepath=filepath,
        ))

    log.debug(f"Tree-sitter parsed {filepath}: {len(chunks)} chunks extracted")

    # Log individual chunks at TRACE level
    for chunk in chunks:
        log.trace(
            f"Chunk extracted: {chunk.node_type} "
            f"'{chunk.function_name or chunk.class_name or 'anonymous'}' "
            f"lines {chunk.start_line}-{chunk.end_line} ({len(chunk.content)} chars)"
        )

    return chunks


def _extract_chunks_recursive(
    node,
    chunks: list[CodeChunk],
    content: str,
    filepath: str,
    language: str,
    target_types: list[str],
    parent_class: str | None,
) -> None:
    """Recursively walk AST and extract target node types.

    Args:
        node: Tree-sitter node to process
        chunks: List to append extracted chunks to
        content: Full file content
        filepath: Source file path
        language: Programming language
        target_types: Node types to extract
        parent_class: Name of parent class (for methods)
    """
    if node.type in target_types:
        # Extract metadata based on node type
        function_name = None
        class_name = parent_class
        node_type = _normalize_node_type(node.type, language)

        # Get name from child nodes
        name_node = node.child_by_field_name("name")
        # Rust impl blocks use "type" field instead of "name"
        if not name_node and node.type == "impl_item":
            name_node = node.child_by_field_name("type")
        if name_node:
            name = name_node.text.decode("utf8")
            if node_type == "class":
                class_name = name
            elif node_type in ("function", "method"):
                function_name = name

        # For methods inside classes, track parent
        if node_type == "function" and parent_class:
            node_type = "method"

        # Handle decorated definitions (Python)
        actual_node = node
        if node.type == "decorated_definition":
            # Find the actual function/class inside
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    actual_node = child
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        name = name_node.text.decode("utf8")
                        node_type = _normalize_node_type(child.type, language)
                        if node_type == "class":
                            class_name = name
                        else:
                            function_name = name
                    break

        chunk = CodeChunk(
            content=node.text.decode("utf8"),
            start_line=node.start_point[0] + 1,  # 1-indexed
            end_line=node.end_point[0] + 1,
            node_type=node_type,
            function_name=function_name,
            class_name=class_name,
            language=language,
            filepath=filepath,
        )
        chunks.append(chunk)

        # For classes, continue recursion to find methods
        if node_type == "class":
            for child in actual_node.children:
                _extract_chunks_recursive(
                    child, chunks, content, filepath, language,
                    target_types, parent_class=class_name
                )
            return  # Don't double-process children

    # Recurse into children
    for child in node.children:
        _extract_chunks_recursive(
            child, chunks, content, filepath, language,
            target_types, parent_class
        )


def _normalize_node_type(ast_type: str, language: str) -> str:  # noqa: ARG001
    """Normalize AST node types to canonical categories.

    Args:
        ast_type: Raw AST node type
        language: Programming language (reserved for future language-specific rules)

    Returns:
        Normalized type: function, class, method, interface, type, or module
    """
    _ = language  # Reserved for future language-specific normalization
    ast_lower = ast_type.lower()

    # Rust impl blocks should be treated as class containers
    if "impl" in ast_lower:
        return "class"
    if "class" in ast_lower or "struct" in ast_lower:
        return "class"
    if "interface" in ast_lower or "trait" in ast_lower:
        return "interface"
    if "type" in ast_lower and "alias" in ast_lower:
        return "type"
    if "method" in ast_lower:
        return "method"
    if "function" in ast_lower:
        return "function"
    if "enum" in ast_lower:
        return "class"  # Treat enums as class-like

    return "module"


def _fallback_line_chunks(
    content: str,
    filepath: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[CodeChunk]:
    """Fallback to line-based chunking for unsupported languages.

    Splits content into chunks at line boundaries, respecting
    target size and overlap constraints.

    Args:
        content: File content
        filepath: Source file path
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of CodeChunk objects
    """
    lines = content.split("\n")
    if not lines or (len(lines) == 1 and not lines[0].strip()):
        return []

    # Detect language for metadata (even if AST parsing failed)
    language = detect_language(filepath) or "unknown"

    # Estimate lines per chunk
    avg_line_len = len(content) / max(len(lines), 1)
    lines_per_chunk = max(10, int(chunk_size / max(avg_line_len, 20)))
    overlap_lines = max(2, int(overlap / max(avg_line_len, 20)))

    chunks: list[CodeChunk] = []
    i = 0

    while i < len(lines):
        end = min(i + lines_per_chunk, len(lines))
        chunk_content = "\n".join(lines[i:end])

        # Only create chunk if it has meaningful content
        if len(chunk_content.strip()) >= 50:
            chunks.append(CodeChunk(
                content=chunk_content,
                start_line=i + 1,  # 1-indexed
                end_line=end,
                node_type="module",
                function_name=None,
                class_name=None,
                language=language,
                filepath=filepath,
            ))

        # Move forward with overlap
        i = end - overlap_lines if end < len(lines) else end

    log.debug(f"Line-based fallback for {filepath}: {len(chunks)} chunks")
    return chunks


def chunk_file_iterator(content: str, filepath: str) -> Iterator[CodeChunk]:
    """Iterator version of chunk_file for API completeness.

    NOTE: This is a simple wrapper that materializes all chunks first.
    For true streaming of very large files (>10k lines), a cursor-based
    AST traversal would be needed. For SimpleMem Lite, the current
    approach is sufficient as most indexed files are reasonably sized.

    Args:
        content: File content as string
        filepath: File path

    Yields:
        CodeChunk objects one at a time
    """
    yield from chunk_file(content, filepath)
