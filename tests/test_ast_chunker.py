"""Tests for AST-aware code chunking.

Tests Tree-sitter based parsing for multiple languages
and fallback to line-based chunking.
"""

import pytest

from simplemem_lite.ast_chunker import (
    CodeChunk,
    chunk_file,
    detect_language,
    _fallback_line_chunks,
    _normalize_node_type,
)


class TestLanguageDetection:
    """Tests for file extension to language mapping."""

    def test_python_detection(self):
        assert detect_language("foo.py") == "python"
        assert detect_language("path/to/bar.py") == "python"
        assert detect_language("stub.pyi") == "python"

    def test_javascript_detection(self):
        assert detect_language("app.js") == "javascript"
        assert detect_language("module.mjs") == "javascript"
        assert detect_language("common.cjs") == "javascript"
        assert detect_language("component.jsx") == "javascript"

    def test_typescript_detection(self):
        assert detect_language("app.ts") == "typescript"
        assert detect_language("component.tsx") == "tsx"

    def test_rust_detection(self):
        assert detect_language("main.rs") == "rust"

    def test_go_detection(self):
        assert detect_language("main.go") == "go"

    def test_cpp_detection(self):
        assert detect_language("main.cpp") == "cpp"
        assert detect_language("main.cc") == "cpp"
        assert detect_language("header.hpp") == "cpp"

    def test_c_detection(self):
        assert detect_language("main.c") == "c"
        assert detect_language("header.h") == "c"

    def test_unknown_extension(self):
        assert detect_language("file.xyz") is None
        assert detect_language("README.md") is None
        assert detect_language("Makefile") is None


class TestNodeTypeNormalization:
    """Tests for AST node type normalization."""

    def test_function_normalization(self):
        assert _normalize_node_type("function_definition", "python") == "function"
        assert _normalize_node_type("function_declaration", "javascript") == "function"
        assert _normalize_node_type("function_item", "rust") == "function"

    def test_class_normalization(self):
        assert _normalize_node_type("class_definition", "python") == "class"
        assert _normalize_node_type("class_declaration", "javascript") == "class"
        assert _normalize_node_type("struct_item", "rust") == "class"

    def test_method_normalization(self):
        assert _normalize_node_type("method_definition", "javascript") == "method"
        assert _normalize_node_type("method_declaration", "java") == "method"

    def test_interface_normalization(self):
        assert _normalize_node_type("interface_declaration", "typescript") == "interface"
        assert _normalize_node_type("trait_item", "rust") == "interface"

    def test_type_alias_normalization(self):
        assert _normalize_node_type("type_alias_declaration", "typescript") == "type"


class TestPythonChunking:
    """Tests for Python code chunking."""

    def test_function_extraction(self):
        code = '''
def foo():
    """Docstring."""
    pass

def bar(x: int) -> int:
    return x * 2
'''
        chunks = chunk_file(code, "test.py")
        assert len(chunks) == 2

        foo_chunk = next(c for c in chunks if c.function_name == "foo")
        assert foo_chunk.node_type == "function"
        assert foo_chunk.language == "python"

        bar_chunk = next(c for c in chunks if c.function_name == "bar")
        assert bar_chunk.node_type == "function"

    def test_class_extraction(self):
        code = '''
class MyClass:
    """A class."""

    def method(self):
        pass
'''
        chunks = chunk_file(code, "test.py")

        # Should get class + method
        class_chunks = [c for c in chunks if c.node_type == "class"]
        method_chunks = [c for c in chunks if c.node_type == "method"]

        assert len(class_chunks) == 1
        assert class_chunks[0].class_name == "MyClass"

        assert len(method_chunks) == 1
        assert method_chunks[0].function_name == "method"
        assert method_chunks[0].class_name == "MyClass"

    def test_decorated_function(self):
        code = '''
@decorator
def decorated_func():
    pass

@decorator1
@decorator2
def multi_decorated():
    return True
'''
        chunks = chunk_file(code, "test.py")
        assert len(chunks) >= 2

        names = [c.function_name for c in chunks if c.function_name]
        assert "decorated_func" in names
        assert "multi_decorated" in names

    def test_nested_class(self):
        code = '''
class Outer:
    class Inner:
        def inner_method(self):
            pass

    def outer_method(self):
        pass
'''
        chunks = chunk_file(code, "test.py")

        # Should extract outer class and its methods
        assert any(c.class_name == "Outer" for c in chunks)

    def test_async_function(self):
        code = '''
async def async_handler():
    await something()
    return result
'''
        chunks = chunk_file(code, "test.py")
        assert len(chunks) >= 1
        assert any(c.function_name == "async_handler" for c in chunks)


class TestJavaScriptChunking:
    """Tests for JavaScript code chunking."""

    def test_function_declaration(self):
        code = '''
function greet(name) {
    return `Hello, ${name}!`;
}
'''
        chunks = chunk_file(code, "app.js")
        assert len(chunks) >= 1
        assert chunks[0].function_name == "greet"
        assert chunks[0].language == "javascript"

    def test_class_declaration(self):
        code = '''
class Calculator {
    constructor() {
        this.value = 0;
    }

    add(x) {
        this.value += x;
    }
}
'''
        chunks = chunk_file(code, "calc.js")
        class_chunks = [c for c in chunks if c.node_type == "class"]
        assert len(class_chunks) >= 1
        assert class_chunks[0].class_name == "Calculator"

    def test_arrow_function_in_export(self):
        code = '''
export const add = (a, b) => a + b;
'''
        chunks = chunk_file(code, "utils.js")
        # Export statement should be captured
        assert len(chunks) >= 1


class TestTypeScriptChunking:
    """Tests for TypeScript code chunking."""

    def test_interface_extraction(self):
        code = '''
interface User {
    id: number;
    name: string;
}
'''
        chunks = chunk_file(code, "types.ts")
        assert len(chunks) >= 1
        # Should extract interface
        interface_chunks = [c for c in chunks if c.node_type == "interface"]
        assert len(interface_chunks) >= 1

    def test_type_alias_extraction(self):
        code = '''
type Status = 'pending' | 'active' | 'completed';
'''
        chunks = chunk_file(code, "types.ts")
        assert len(chunks) >= 1


class TestFallbackChunking:
    """Tests for line-based fallback chunking."""

    def test_unknown_language_fallback(self):
        code = "Some content\n" * 50
        chunks = chunk_file(code, "file.xyz")
        assert len(chunks) >= 1
        assert all(c.language == "unknown" for c in chunks)
        assert all(c.node_type == "module" for c in chunks)

    def test_empty_file(self):
        chunks = chunk_file("", "empty.py")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = chunk_file("   \n\n   \t", "whitespace.py")
        assert chunks == []

    def test_file_without_functions(self):
        code = '''
# Just comments
x = 1
y = 2
'''
        chunks = chunk_file(code, "constants.py")
        # Should fall back to module-level chunk
        assert len(chunks) >= 1

    def test_fallback_chunk_overlap(self):
        """Verify chunks have proper overlap in fallback mode."""
        # Create content that will need multiple chunks
        code = "\n".join([f"line {i}" for i in range(100)])
        chunks = _fallback_line_chunks(code, "large.xyz", chunk_size=200, overlap=50)

        if len(chunks) > 1:
            # Check that consecutive chunks overlap
            for i in range(len(chunks) - 1):
                # End of chunk i should overlap with start of chunk i+1
                assert chunks[i].end_line >= chunks[i + 1].start_line - 5


class TestChunkMetadata:
    """Tests for chunk metadata accuracy."""

    def test_line_numbers_are_1_indexed(self):
        code = '''def first():
    pass

def second():
    pass
'''
        chunks = chunk_file(code, "test.py")
        # First function starts at line 1
        first_chunk = next(c for c in chunks if c.function_name == "first")
        assert first_chunk.start_line == 1

    def test_filepath_preserved(self):
        code = "def test(): pass"
        filepath = "path/to/my/file.py"
        chunks = chunk_file(code, filepath)
        assert all(c.filepath == filepath for c in chunks)

    def test_content_matches_lines(self):
        code = '''def foo():
    return 1

def bar():
    return 2
'''
        chunks = chunk_file(code, "test.py")
        for chunk in chunks:
            # Content should contain the function/class definition
            assert "def " in chunk.content or "class " in chunk.content


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_malformed_syntax_fallback(self):
        """Malformed code should trigger fallback."""
        code = '''
def incomplete(
    # Missing closing paren and body
'''
        # Should not raise, should return some chunks
        chunks = chunk_file(code, "broken.py")
        # May get partial results or fallback
        assert isinstance(chunks, list)

    def test_very_long_function(self):
        """Very long functions should be handled."""
        lines = ["def long_func():"]
        lines.extend([f"    x = {i}" for i in range(500)])
        code = "\n".join(lines)

        chunks = chunk_file(code, "long.py")
        assert len(chunks) >= 1
        # Function should be kept as single chunk
        func_chunk = next(c for c in chunks if c.function_name == "long_func")
        assert func_chunk.end_line > func_chunk.start_line + 400

    def test_unicode_content(self):
        """Unicode in code should be handled."""
        code = '''
def greet():
    return "Hello, 世界! 🌍"
'''
        chunks = chunk_file(code, "unicode.py")
        assert len(chunks) >= 1
        assert "世界" in chunks[0].content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
