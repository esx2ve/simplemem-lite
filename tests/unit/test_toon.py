"""Unit tests for TOON (Token-Optimized Object Notation) utilities.

Tests the tab-separated format optimized for LLM token efficiency:
- List parsing and rendering
- Table parsing and rendering
- Scratchpad validation and conversion
"""

import pytest
import time

from simplemem_lite.toon import (
    toon_list_parse,
    toon_list_render,
    toon_table_parse,
    toon_table_render,
    scratchpad_validate,
    scratchpad_to_markdown,
    scratchpad_expand_json,
    scratchpad_compact_json,
    TOON_LIST_FIELDS,
    TOON_TABLE_FIELDS,
)


# =============================================================================
# List Parsing Tests
# =============================================================================


class TestToonListParse:
    """Tests for toon_list_parse function."""

    def test_parse_simple_list(self):
        """Parse a simple tab-separated list."""
        result = toon_list_parse("item1\titem2\titem3")
        assert result == ["item1", "item2", "item3"]

    def test_parse_empty_string(self):
        """Return empty list for empty string."""
        assert toon_list_parse("") == []
        assert toon_list_parse("   ") == []

    def test_parse_none(self):
        """Return empty list for None."""
        assert toon_list_parse(None) == []

    def test_parse_single_item(self):
        """Parse a single item (no tabs)."""
        result = toon_list_parse("single")
        assert result == ["single"]

    def test_parse_strips_whitespace(self):
        """Whitespace around items is stripped."""
        result = toon_list_parse("  item1  \t  item2  \t  item3  ")
        assert result == ["item1", "item2", "item3"]

    def test_parse_skips_empty_items(self):
        """Empty items between tabs are skipped."""
        result = toon_list_parse("item1\t\t\titem2")
        assert result == ["item1", "item2"]

    def test_parse_file_paths(self):
        """Parse file paths with spaces preserved."""
        result = toon_list_parse("src/main.py\ttests/test_foo.py\tconfig.yaml")
        assert result == ["src/main.py", "tests/test_foo.py", "config.yaml"]


# =============================================================================
# List Rendering Tests
# =============================================================================


class TestToonListRender:
    """Tests for toon_list_render function."""

    def test_render_simple_list(self):
        """Render a simple list to tab-separated string."""
        result = toon_list_render(["item1", "item2", "item3"])
        assert result == "item1\titem2\titem3"

    def test_render_empty_list(self):
        """Return empty string for empty list."""
        assert toon_list_render([]) == ""

    def test_render_none(self):
        """Return empty string for None."""
        assert toon_list_render(None) == ""

    def test_render_single_item(self):
        """Render a single item."""
        result = toon_list_render(["single"])
        assert result == "single"

    def test_render_escapes_tabs(self):
        """Tabs in items are escaped to spaces."""
        result = toon_list_render(["item\twith\ttabs", "normal"])
        assert "\t\t" not in result  # No double tabs
        # Tabs replaced with spaces
        assert "item    with    tabs" in result

    def test_render_escapes_newlines(self):
        """Newlines in items are escaped to spaces."""
        result = toon_list_render(["item\nwith\nnewlines", "normal"])
        assert "\n" not in result.split("\t")[0]  # Newlines replaced

    def test_roundtrip(self):
        """Parse and render should roundtrip cleanly."""
        original = ["item1", "item2", "item3"]
        rendered = toon_list_render(original)
        parsed = toon_list_parse(rendered)
        assert parsed == original


# =============================================================================
# Table Parsing Tests
# =============================================================================


class TestToonTableParse:
    """Tests for toon_table_parse function."""

    def test_parse_simple_table(self):
        """Parse a simple table with headers and data."""
        table = "col1\tcol2\tcol3\nval1\tval2\tval3\nval4\tval5\tval6"
        result = toon_table_parse(table)
        assert len(result) == 2
        assert result[0] == {"col1": "val1", "col2": "val2", "col3": "val3"}
        assert result[1] == {"col1": "val4", "col2": "val5", "col3": "val6"}

    def test_parse_empty_string(self):
        """Return empty list for empty string."""
        assert toon_table_parse("") == []
        assert toon_table_parse("   ") == []

    def test_parse_none(self):
        """Return empty list for None."""
        assert toon_table_parse(None) == []

    def test_parse_headers_only(self):
        """Return empty list for headers only (no data rows)."""
        result = toon_table_parse("col1\tcol2\tcol3")
        assert result == []

    def test_parse_pads_missing_values(self):
        """Missing values are padded with empty strings."""
        table = "col1\tcol2\tcol3\nval1\tval2"  # Missing val3
        result = toon_table_parse(table)
        assert len(result) == 1
        assert result[0] == {"col1": "val1", "col2": "val2", "col3": ""}

    def test_parse_strips_whitespace(self):
        """Whitespace is stripped from headers and values."""
        table = "  col1  \t  col2  \n  val1  \t  val2  "
        result = toon_table_parse(table)
        assert result[0] == {"col1": "val1", "col2": "val2"}

    def test_parse_decisions_format(self):
        """Parse decisions table in TOON format."""
        table = "what\twhy\trejected\nUse Redis\tFast caching\tMemcached"
        result = toon_table_parse(table)
        assert len(result) == 1
        assert result[0]["what"] == "Use Redis"
        assert result[0]["why"] == "Fast caching"
        assert result[0]["rejected"] == "Memcached"

    def test_parse_attached_memories_format(self):
        """Parse attached_memories table in TOON format."""
        table = "uuid\treason\nabc-123\tSource of fix\ndef-456\tDebug context"
        result = toon_table_parse(table)
        assert len(result) == 2
        assert result[0]["uuid"] == "abc-123"
        assert result[0]["reason"] == "Source of fix"


# =============================================================================
# Table Rendering Tests
# =============================================================================


class TestToonTableRender:
    """Tests for toon_table_render function."""

    def test_render_simple_table(self):
        """Render a simple table with specified columns."""
        rows = [
            {"col1": "val1", "col2": "val2"},
            {"col1": "val3", "col2": "val4"},
        ]
        result = toon_table_render(rows, ["col1", "col2"])
        lines = result.split("\n")
        assert lines[0] == "col1\tcol2"
        assert lines[1] == "val1\tval2"
        assert lines[2] == "val3\tval4"

    def test_render_empty_list(self):
        """Return empty string for empty list."""
        assert toon_table_render([]) == ""
        assert toon_table_render(None) == ""

    def test_render_infers_columns(self):
        """Infer columns from first row if not specified."""
        rows = [{"a": "1", "b": "2"}]
        result = toon_table_render(rows)
        assert "a\tb" in result or "b\ta" in result  # Order may vary

    def test_render_handles_missing_values(self):
        """Missing values in rows become empty strings."""
        rows = [
            {"col1": "val1", "col2": "val2"},
            {"col1": "val3"},  # Missing col2
        ]
        result = toon_table_render(rows, ["col1", "col2"])
        lines = result.split("\n")
        assert lines[2] == "val3\t"

    def test_render_escapes_special_chars(self):
        """Tabs and newlines in values are escaped."""
        rows = [{"col1": "value\twith\ttabs", "col2": "value\nwith\nnewlines"}]
        result = toon_table_render(rows, ["col1", "col2"])
        lines = result.split("\n")
        assert len(lines) == 2  # Header + 1 data row (not split by embedded newlines)

    def test_roundtrip(self):
        """Parse and render should roundtrip cleanly."""
        original = [
            {"what": "Decision A", "why": "Reason A", "rejected": "Alt A"},
            {"what": "Decision B", "why": "Reason B", "rejected": "Alt B"},
        ]
        rendered = toon_table_render(original, ["what", "why", "rejected"])
        parsed = toon_table_parse(rendered)
        assert parsed == original


# =============================================================================
# Scratchpad Validation Tests
# =============================================================================


class TestScratchpadValidate:
    """Tests for scratchpad_validate function."""

    def test_valid_minimal_scratchpad(self):
        """Minimal valid scratchpad passes validation."""
        scratchpad = {
            "task_id": "test-task",
            "current_focus": "Working on feature X",
        }
        errors = scratchpad_validate(scratchpad)
        assert errors == []

    def test_missing_required_fields(self):
        """Missing required fields produce errors."""
        errors = scratchpad_validate({})
        assert "Missing required field: task_id" in errors
        assert "Missing required field: current_focus" in errors

    def test_empty_required_fields(self):
        """Empty required fields produce errors."""
        scratchpad = {"task_id": "", "current_focus": ""}
        errors = scratchpad_validate(scratchpad)
        assert "Missing required field: task_id" in errors
        assert "Missing required field: current_focus" in errors

    def test_valid_version(self):
        """Valid versions pass validation."""
        for version in ["1.0", "1.1"]:
            scratchpad = {
                "task_id": "test",
                "current_focus": "focus",
                "version": version,
            }
            errors = scratchpad_validate(scratchpad)
            assert errors == []

    def test_invalid_version(self):
        """Invalid version produces error."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "version": "2.0",
        }
        errors = scratchpad_validate(scratchpad)
        assert "Unknown version: 2.0" in errors

    def test_toon_list_fields_must_be_strings(self):
        """TOON list fields must be strings, not lists."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "active_constraints": ["constraint1", "constraint2"],  # Wrong: should be string
        }
        errors = scratchpad_validate(scratchpad)
        assert any("active_constraints" in e and "TOON string" in e for e in errors)

    def test_toon_table_fields_must_be_strings(self):
        """TOON table fields must be strings, not lists."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "decisions": [{"what": "x", "why": "y"}],  # Wrong: should be string
        }
        errors = scratchpad_validate(scratchpad)
        assert any("decisions" in e and "TOON string" in e for e in errors)

    def test_valid_toon_fields(self):
        """Valid TOON string fields pass validation."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "active_constraints": "constraint1\tconstraint2",
            "decisions": "what\twhy\trejected\nDecision 1\tReason 1\tAlt 1",
        }
        errors = scratchpad_validate(scratchpad)
        assert errors == []


# =============================================================================
# Scratchpad to Markdown Tests
# =============================================================================


class TestScratchpadToMarkdown:
    """Tests for scratchpad_to_markdown function."""

    def test_minimal_scratchpad(self):
        """Render minimal scratchpad to markdown."""
        scratchpad = {
            "task_id": "test-task",
            "current_focus": "Working on feature X",
        }
        result = scratchpad_to_markdown(scratchpad)
        assert "# Task: test-task" in result
        assert "## Current Focus" in result
        assert "Working on feature X" in result

    def test_includes_timestamp(self):
        """Updated timestamp is included."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "updated_at": int(time.time()),
        }
        result = scratchpad_to_markdown(scratchpad)
        assert "*Updated:" in result

    def test_constraints_as_bullet_list(self):
        """Constraints are rendered as bullet list."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "active_constraints": "constraint1\tconstraint2\tconstraint3",
        }
        result = scratchpad_to_markdown(scratchpad)
        assert "## Constraints" in result
        assert "- constraint1" in result
        assert "- constraint2" in result
        assert "- constraint3" in result

    def test_active_files_as_code_list(self):
        """Active files are rendered with backticks."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "active_files": "main.py\tconfig.yaml",
        }
        result = scratchpad_to_markdown(scratchpad)
        assert "## Active Files" in result
        assert "- `main.py`" in result
        assert "- `config.yaml`" in result

    def test_pending_verification_as_checklist(self):
        """Pending verification rendered as checklist."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "pending_verification": "check1\tcheck2",
        }
        result = scratchpad_to_markdown(scratchpad)
        assert "## Pending Verification" in result
        assert "- [ ] check1" in result
        assert "- [ ] check2" in result

    def test_decisions_as_markdown_table(self):
        """Decisions are rendered as markdown table."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "decisions": "what\twhy\trejected\nUse Redis\tFast\tMemcached",
        }
        result = scratchpad_to_markdown(scratchpad)
        assert "## Decisions" in result
        assert "| What | Why | Rejected |" in result
        assert "| Use Redis | Fast | Memcached |" in result

    def test_attached_memories(self):
        """Attached memories are rendered with short UUIDs."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "attached_memories": "uuid\treason\n12345678-abcd-efgh\tSource of fix",
        }
        result = scratchpad_to_markdown(scratchpad)
        assert "## Referenced Memories" in result
        assert "[12345678]" in result  # Short UUID
        assert "Source of fix" in result


# =============================================================================
# Scratchpad Expand/Compact Tests
# =============================================================================


class TestScratchpadExpandCompact:
    """Tests for scratchpad_expand_json and scratchpad_compact_json."""

    def test_expand_toon_lists(self):
        """Expand TOON list strings to Python lists."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "active_constraints": "constraint1\tconstraint2",
            "active_files": "file1.py\tfile2.ts",
        }
        result = scratchpad_expand_json(scratchpad)
        assert result["active_constraints"] == ["constraint1", "constraint2"]
        assert result["active_files"] == ["file1.py", "file2.ts"]

    def test_expand_toon_tables(self):
        """Expand TOON table strings to list of dicts."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "decisions": "what\twhy\trejected\nDecision 1\tReason 1\tAlt 1",
        }
        result = scratchpad_expand_json(scratchpad)
        assert result["decisions"] == [
            {"what": "Decision 1", "why": "Reason 1", "rejected": "Alt 1"}
        ]

    def test_compact_lists(self):
        """Compact Python lists to TOON strings."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "active_constraints": ["constraint1", "constraint2"],
            "active_files": ["file1.py", "file2.ts"],
        }
        result = scratchpad_compact_json(scratchpad)
        assert result["active_constraints"] == "constraint1\tconstraint2"
        assert result["active_files"] == "file1.py\tfile2.ts"

    def test_compact_tables(self):
        """Compact list of dicts to TOON table strings."""
        scratchpad = {
            "task_id": "test",
            "current_focus": "focus",
            "decisions": [{"what": "D1", "why": "R1", "rejected": "A1"}],
        }
        result = scratchpad_compact_json(scratchpad)
        assert "what\twhy\trejected" in result["decisions"]
        assert "D1\tR1\tA1" in result["decisions"]

    def test_roundtrip_expand_compact(self):
        """Expand and compact should roundtrip cleanly."""
        original = {
            "task_id": "test",
            "current_focus": "focus",
            "active_constraints": "c1\tc2",
            "decisions": "what\twhy\trejected\nD1\tR1\tA1",
        }
        expanded = scratchpad_expand_json(original)
        compacted = scratchpad_compact_json(expanded)
        # Re-expand to compare (since string representation may differ)
        re_expanded = scratchpad_expand_json(compacted)
        assert re_expanded["active_constraints"] == expanded["active_constraints"]
        assert re_expanded["decisions"] == expanded["decisions"]


# =============================================================================
# Constants Tests
# =============================================================================


class TestToonConstants:
    """Tests for TOON format constants."""

    def test_toon_list_fields(self):
        """TOON list fields are defined correctly."""
        assert "active_constraints" in TOON_LIST_FIELDS
        assert "active_files" in TOON_LIST_FIELDS
        assert "pending_verification" in TOON_LIST_FIELDS

    def test_toon_table_fields(self):
        """TOON table fields are defined with correct columns."""
        assert "decisions" in TOON_TABLE_FIELDS
        assert TOON_TABLE_FIELDS["decisions"] == ["what", "why", "rejected"]

        assert "attached_memories" in TOON_TABLE_FIELDS
        assert TOON_TABLE_FIELDS["attached_memories"] == ["uuid", "reason"]

        assert "attached_sessions" in TOON_TABLE_FIELDS
        assert TOON_TABLE_FIELDS["attached_sessions"] == ["session_id", "description"]
