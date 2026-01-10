"""TOON (Token-Optimized Object Notation) utilities for SimpleMem.

TOON is a tab-separated format optimized for LLM consumption:
- Headers declared once (first row for tables)
- Tab-delimited values (no quotes, braces, or brackets)
- Newline-separated records

Achieves 30-60% token savings compared to JSON.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


# =============================================================================
# List Operations (tab-separated values)
# =============================================================================


def toon_list_parse(s: str | None) -> list[str]:
    """Parse a TOON list (tab-separated string) into a Python list.

    Args:
        s: Tab-separated string, e.g. "item1\titem2\titem3"

    Returns:
        List of strings, e.g. ["item1", "item2", "item3"]
        Empty list if input is None or empty.
    """
    if not s or not s.strip():
        return []
    return [item.strip() for item in s.split("\t") if item.strip()]


def toon_list_render(items: list[str] | None) -> str:
    """Render a Python list as a TOON list (tab-separated string).

    Args:
        items: List of strings

    Returns:
        Tab-separated string, e.g. "item1\titem2\titem3"
        Empty string if input is None or empty.
    """
    if not items:
        return ""
    # Escape any tabs or newlines in items
    cleaned = [_escape_toon_value(str(item)) for item in items]
    return "\t".join(cleaned)


# =============================================================================
# Table Operations (TOON format with headers)
# =============================================================================


def toon_table_parse(s: str | None) -> list[dict[str, str]]:
    """Parse a TOON table into a list of dictionaries.

    TOON table format:
        header1\theader2\theader3
        value1\tvalue2\tvalue3
        value4\tvalue5\tvalue6

    Args:
        s: TOON table string

    Returns:
        List of dicts, e.g. [{"header1": "value1", ...}, ...]
        Empty list if input is None or empty.
    """
    if not s or not s.strip():
        return []

    lines = [line for line in s.strip().split("\n") if line.strip()]
    if len(lines) < 1:
        return []

    # First line is headers
    headers = [h.strip() for h in lines[0].split("\t")]

    if len(lines) < 2:
        return []  # Headers only, no data

    rows = []
    for line in lines[1:]:
        values = [v.strip() for v in line.split("\t")]
        # Pad with empty strings if fewer values than headers
        while len(values) < len(headers):
            values.append("")
        row = {headers[i]: values[i] for i in range(len(headers))}
        rows.append(row)

    return rows


def toon_table_render(
    rows: list[dict[str, Any]] | None,
    columns: list[str] | None = None,
) -> str:
    """Render a list of dictionaries as a TOON table.

    Args:
        rows: List of dicts to render
        columns: Column order (optional, defaults to keys from first row)

    Returns:
        TOON table string with headers on first line.
        Empty string if input is None or empty.
    """
    if not rows:
        return ""

    # Determine columns from first row if not specified
    if columns is None:
        columns = list(rows[0].keys())

    # Build header line
    header_line = "\t".join(columns)

    # Build data lines
    data_lines = []
    for row in rows:
        values = [_escape_toon_value(str(row.get(col, ""))) for col in columns]
        data_lines.append("\t".join(values))

    return header_line + "\n" + "\n".join(data_lines)


# =============================================================================
# Scratchpad Operations
# =============================================================================

# Fields that use TOON list format
TOON_LIST_FIELDS = {"active_constraints", "active_files", "pending_verification"}

# Fields that use TOON table format
TOON_TABLE_FIELDS = {
    "decisions": ["what", "why", "rejected"],
    "attached_memories": ["uuid", "reason"],
    "attached_sessions": ["session_id", "description"],
}

# Required fields
REQUIRED_FIELDS = {"task_id", "current_focus"}


def scratchpad_validate(scratchpad: dict[str, Any]) -> list[str]:
    """Validate a scratchpad structure.

    Args:
        scratchpad: Scratchpad dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in scratchpad or not scratchpad[field]:
            errors.append(f"Missing required field: {field}")

    # Validate version if present
    if "version" in scratchpad:
        version = scratchpad["version"]
        if version not in ("1.0", "1.1"):
            errors.append(f"Unknown version: {version}")

    # Validate TOON list fields (should be strings)
    for field in TOON_LIST_FIELDS:
        if field in scratchpad and scratchpad[field] is not None:
            if not isinstance(scratchpad[field], str):
                errors.append(f"Field '{field}' should be a TOON string, got {type(scratchpad[field]).__name__}")

    # Validate TOON table fields (should be strings)
    for field in TOON_TABLE_FIELDS:
        if field in scratchpad and scratchpad[field] is not None:
            if not isinstance(scratchpad[field], str):
                errors.append(f"Field '{field}' should be a TOON string, got {type(scratchpad[field]).__name__}")

    return errors


def scratchpad_to_markdown(scratchpad: dict[str, Any]) -> str:
    """Render a scratchpad as human-readable markdown.

    Args:
        scratchpad: Scratchpad dictionary (JSON+TOON hybrid)

    Returns:
        Markdown string
    """
    lines = []

    # Title
    task_id = scratchpad.get("task_id", "Unknown")
    lines.append(f"# Task: {task_id}")

    # Updated timestamp
    updated_at = scratchpad.get("updated_at")
    if updated_at:
        if isinstance(updated_at, (int, float)):
            dt = datetime.fromtimestamp(updated_at)
            lines.append(f"*Updated: {dt.strftime('%Y-%m-%d %H:%M:%S')}*")
        else:
            lines.append(f"*Updated: {updated_at}*")

    lines.append("")

    # Current focus
    if scratchpad.get("current_focus"):
        lines.append("## Current Focus")
        lines.append(scratchpad["current_focus"])
        lines.append("")

    # Constraints (TOON list)
    if scratchpad.get("active_constraints"):
        lines.append("## Constraints")
        for item in toon_list_parse(scratchpad["active_constraints"]):
            lines.append(f"- {item}")
        lines.append("")

    # Active files (TOON list)
    if scratchpad.get("active_files"):
        lines.append("## Active Files")
        for item in toon_list_parse(scratchpad["active_files"]):
            lines.append(f"- `{item}`")
        lines.append("")

    # Pending verification (TOON list as checklist)
    if scratchpad.get("pending_verification"):
        lines.append("## Pending Verification")
        for item in toon_list_parse(scratchpad["pending_verification"]):
            lines.append(f"- [ ] {item}")
        lines.append("")

    # Decisions (TOON table)
    if scratchpad.get("decisions"):
        lines.append("## Decisions")
        rows = toon_table_parse(scratchpad["decisions"])
        if rows:
            # Markdown table
            lines.append("| What | Why | Rejected |")
            lines.append("|------|-----|----------|")
            for row in rows:
                what = row.get("what", "")
                why = row.get("why", "")
                rejected = row.get("rejected", "")
                lines.append(f"| {what} | {why} | {rejected} |")
        lines.append("")

    # Notes
    if scratchpad.get("notes"):
        lines.append("## Notes")
        lines.append(scratchpad["notes"])
        lines.append("")

    # Attached memories (TOON table)
    if scratchpad.get("attached_memories"):
        lines.append("## Referenced Memories")
        rows = toon_table_parse(scratchpad["attached_memories"])
        for row in rows:
            uuid = row.get("uuid", "")[:8]  # Short UUID
            reason = row.get("reason", "")
            lines.append(f"- [{uuid}] {reason}")
        lines.append("")

    # Attached sessions (TOON table)
    if scratchpad.get("attached_sessions"):
        lines.append("## Session History")
        rows = toon_table_parse(scratchpad["attached_sessions"])
        for row in rows:
            session_id = row.get("session_id", "")
            description = row.get("description", "")
            lines.append(f"- [{session_id}] {description}")
        lines.append("")

    return "\n".join(lines)


def scratchpad_expand_json(scratchpad: dict[str, Any]) -> dict[str, Any]:
    """Expand a JSON+TOON scratchpad to full JSON (for debugging/viewing).

    Converts TOON strings to native Python lists/dicts.

    Args:
        scratchpad: Scratchpad dictionary (JSON+TOON hybrid)

    Returns:
        Fully expanded JSON dictionary
    """
    result = dict(scratchpad)

    # Expand TOON list fields
    for field in TOON_LIST_FIELDS:
        if field in result and isinstance(result[field], str):
            result[field] = toon_list_parse(result[field])

    # Expand TOON table fields
    for field in TOON_TABLE_FIELDS:
        if field in result and isinstance(result[field], str):
            result[field] = toon_table_parse(result[field])

    return result


def scratchpad_compact_json(scratchpad: dict[str, Any]) -> dict[str, Any]:
    """Compact a full JSON scratchpad to JSON+TOON hybrid.

    Converts native Python lists/dicts to TOON strings.

    Args:
        scratchpad: Scratchpad dictionary (full JSON)

    Returns:
        JSON+TOON hybrid dictionary
    """
    result = dict(scratchpad)

    # Compact list fields to TOON
    for field in TOON_LIST_FIELDS:
        if field in result and isinstance(result[field], list):
            result[field] = toon_list_render(result[field])

    # Compact table fields to TOON
    for field, columns in TOON_TABLE_FIELDS.items():
        if field in result and isinstance(result[field], list):
            result[field] = toon_table_render(result[field], columns)

    return result


# =============================================================================
# Helper Functions
# =============================================================================


def _escape_toon_value(value: str) -> str:
    """Escape tabs and newlines in a TOON value.

    Tabs and newlines are structural in TOON, so they must be escaped
    if they appear in values.
    """
    return value.replace("\t", "    ").replace("\n", " ")


__all__ = [
    "toon_list_parse",
    "toon_list_render",
    "toon_table_parse",
    "toon_table_render",
    "scratchpad_validate",
    "scratchpad_to_markdown",
    "scratchpad_expand_json",
    "scratchpad_compact_json",
    "TOON_LIST_FIELDS",
    "TOON_TABLE_FIELDS",
]
