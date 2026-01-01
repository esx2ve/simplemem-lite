"""Code indexing tests for SimpleMem Lite.

Tests critical code indexing logic:
- Python entity extraction patterns
- JS/TS entity extraction patterns
- Built-in ignore patterns
"""

import pytest


class TestPythonPatterns:
    """Test Python entity extraction regex patterns."""

    def test_import_pattern_simple(self):
        """Should match simple imports."""
        from simplemem_lite.code_index import _PYTHON_PATTERNS

        pattern = _PYTHON_PATTERNS["import"]
        code = "import os"
        match = pattern.search(code)

        assert match is not None
        assert match.group(2).strip() == "os"

    def test_import_pattern_from(self):
        """Should match from imports."""
        from simplemem_lite.code_index import _PYTHON_PATTERNS

        pattern = _PYTHON_PATTERNS["import"]
        code = "from pathlib import Path"
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "pathlib"
        assert match.group(2).strip() == "Path"

    def test_import_pattern_multiple(self):
        """Should match imports with multiple names."""
        from simplemem_lite.code_index import _PYTHON_PATTERNS

        pattern = _PYTHON_PATTERNS["import"]
        code = "from typing import Dict, List, Optional"
        match = pattern.search(code)

        assert match is not None
        assert "Dict" in match.group(2)

    def test_class_pattern(self):
        """Should match class definitions."""
        from simplemem_lite.code_index import _PYTHON_PATTERNS

        pattern = _PYTHON_PATTERNS["class"]
        code = """
class MyClass:
    pass
"""
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "MyClass"

    def test_class_pattern_with_inheritance(self):
        """Should match class with inheritance."""
        from simplemem_lite.code_index import _PYTHON_PATTERNS

        pattern = _PYTHON_PATTERNS["class"]
        code = "class ChildClass(ParentClass):"
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "ChildClass"

    def test_function_pattern(self):
        """Should match function definitions."""
        from simplemem_lite.code_index import _PYTHON_PATTERNS

        pattern = _PYTHON_PATTERNS["function"]
        code = """
def my_function(x, y):
    return x + y
"""
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "my_function"

    def test_async_function_pattern(self):
        """Should match async function definitions."""
        from simplemem_lite.code_index import _PYTHON_PATTERNS

        pattern = _PYTHON_PATTERNS["async_function"]
        code = """
async def fetch_data(url):
    pass
"""
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "fetch_data"


class TestJSPatterns:
    """Test JavaScript/TypeScript entity extraction regex patterns."""

    def test_import_pattern_default(self):
        """Should match default imports."""
        from simplemem_lite.code_index import _JS_PATTERNS

        pattern = _JS_PATTERNS["import"]
        code = "import React from 'react'"
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "react"

    def test_import_pattern_named(self):
        """Should match named imports."""
        from simplemem_lite.code_index import _JS_PATTERNS

        pattern = _JS_PATTERNS["import"]
        code = "import { useState, useEffect } from 'react'"
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "react"

    def test_class_pattern(self):
        """Should match class definitions."""
        from simplemem_lite.code_index import _JS_PATTERNS

        pattern = _JS_PATTERNS["class"]
        code = "class MyComponent {"
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "MyComponent"

    def test_class_pattern_export(self):
        """Should match exported class definitions."""
        from simplemem_lite.code_index import _JS_PATTERNS

        pattern = _JS_PATTERNS["class"]
        code = "export class MyComponent {"
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "MyComponent"

    def test_function_pattern(self):
        """Should match function definitions."""
        from simplemem_lite.code_index import _JS_PATTERNS

        pattern = _JS_PATTERNS["function"]
        code = "function handleClick() {"
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "handleClick"

    def test_function_pattern_export_async(self):
        """Should match exported async functions."""
        from simplemem_lite.code_index import _JS_PATTERNS

        pattern = _JS_PATTERNS["function"]
        code = "export async function fetchData() {"
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "fetchData"

    def test_const_func_arrow(self):
        """Should match arrow function declarations."""
        from simplemem_lite.code_index import _JS_PATTERNS

        pattern = _JS_PATTERNS["const_func"]
        code = "const handleClick = () => {"
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "handleClick"

    def test_const_func_arrow_with_params(self):
        """Should match arrow function with parameters."""
        from simplemem_lite.code_index import _JS_PATTERNS

        pattern = _JS_PATTERNS["const_func"]
        code = "const handleClick = (event) => {"
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "handleClick"

    def test_const_func_export_async(self):
        """Should match exported async arrow functions."""
        from simplemem_lite.code_index import _JS_PATTERNS

        pattern = _JS_PATTERNS["const_func"]
        code = "export const fetchData = async () => {"
        match = pattern.search(code)

        assert match is not None
        assert match.group(1) == "fetchData"


class TestBuiltinIgnorePatterns:
    """Test built-in ignore patterns."""

    def test_ignore_patterns_defined(self):
        """Built-in ignore patterns should be defined."""
        from simplemem_lite.code_index import _BUILTIN_IGNORE_PATTERNS

        assert _BUILTIN_IGNORE_PATTERNS is not None
        assert len(_BUILTIN_IGNORE_PATTERNS) > 0

    def test_common_patterns_included(self):
        """Common ignore patterns should be included."""
        from simplemem_lite.code_index import _BUILTIN_IGNORE_PATTERNS

        patterns = _BUILTIN_IGNORE_PATTERNS

        # Version control
        assert ".git/" in patterns

        # Python
        assert "__pycache__/" in patterns
        assert "*venv*/" in patterns

        # Node.js
        assert "node_modules/" in patterns

        # Build outputs
        assert "dist/" in patterns
        assert "build/" in patterns


class TestPatternMatching:
    """Test pattern matching with multiple entities."""

    def test_find_all_python_functions(self):
        """Should find all function definitions in Python code."""
        from simplemem_lite.code_index import _PYTHON_PATTERNS

        code = """
def func1():
    pass

def func2(x):
    return x

def func3(x, y):
    return x + y
"""
        pattern = _PYTHON_PATTERNS["function"]
        matches = pattern.findall(code)

        assert len(matches) == 3
        assert "func1" in matches
        assert "func2" in matches
        assert "func3" in matches

    def test_find_all_js_imports(self):
        """Should find all imports in JS code."""
        from simplemem_lite.code_index import _JS_PATTERNS

        code = """
import React from 'react';
import { useState } from 'react';
import axios from 'axios';
import { helper } from './utils';
"""
        pattern = _JS_PATTERNS["import"]
        matches = pattern.findall(code)

        assert len(matches) == 4
        assert "react" in matches
        assert "axios" in matches
        assert "./utils" in matches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
