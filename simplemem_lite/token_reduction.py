"""Token reduction utilities for SimpleMem MCP responses.

Provides multiple strategies to minimize token consumption:
1. Compact JSON serialization (orjson)
2. Field pruning (remove null/empty fields)
3. Content summarization (LLM-powered)
4. TOON format (Token-Optimized Object Notation)
"""

import asyncio
import os
from enum import Enum
from pathlib import Path
from typing import Any

import orjson

# Load .env before reading env vars
try:
    from dotenv import load_dotenv
    _pkg_dir = Path(__file__).parent
    load_dotenv(_pkg_dir / ".env") or load_dotenv(_pkg_dir.parent / ".env")
except ImportError:
    pass

from simplemem_lite.log_config import get_logger

log = get_logger("token_reduction")

# Configuration from environment
SUMMARY_MODEL = os.environ.get("SIMPLEMEM_SUMMARY_MODEL", "gemini/gemini-2.5-flash-lite")
SUMMARY_MAX_CHARS = int(os.environ.get("SIMPLEMEM_SUMMARY_MAX_CHARS", "500"))
OUTPUT_FORMAT = os.environ.get("SIMPLEMEM_OUTPUT_FORMAT", "compact")


class OutputFormat(Enum):
    """Output format options for MCP responses."""

    JSON = "json"  # Standard JSON (verbose)
    COMPACT = "compact"  # Minified JSON via orjson (default)
    TOON = "toon"  # Token-Optimized Object Notation


# ═══════════════════════════════════════════════════════════════════════════════
# COMPACT JSON SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def compact_json(data: Any) -> str:
    """Serialize data to compact JSON using orjson.

    - No indentation or whitespace
    - Sorted keys for consistency
    - Native datetime/UUID support

    Args:
        data: Data to serialize

    Returns:
        Compact JSON string
    """
    return orjson.dumps(
        data,
        option=orjson.OPT_SORT_KEYS,
    ).decode("utf-8")


def prune_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Remove null values and empty dicts from a dictionary.

    Args:
        d: Dictionary to prune

    Returns:
        Pruned dictionary without null/empty values
    """
    result = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict) and not v:
            continue
        if isinstance(v, dict):
            v = prune_dict(v)
            if not v:
                continue
        result[k] = v
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LLM SUMMARIZATION
# ═══════════════════════════════════════════════════════════════════════════════


async def summarize_content(
    content: str,
    context: str = "",
    max_chars: int = SUMMARY_MAX_CHARS,
) -> str:
    """Summarize content using a fast LLM model.

    Uses gemini-2.5-flash-lite for fast, cheap summarization.
    Always enabled - provides intelligent compression of verbose content.

    Args:
        content: Content to summarize
        context: Optional context (e.g., search query) to guide summarization
        max_chars: Target maximum characters (soft limit, won't truncate)

    Returns:
        Summarized content (or original if short enough)
    """
    # Skip if already short enough
    if len(content) <= max_chars:
        return content

    try:
        import litellm

        context_hint = f" Focus on relevance to: {context}" if context else ""
        prompt = f"""Summarize this content concisely, preserving key insights and actionable information.{context_hint}

Content:
{content}

Provide a clear, informative summary. Do not truncate important details."""

        response = await litellm.acompletion(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_chars // 3,  # Rough token estimate
            temperature=0.1,
        )

        summary = response.choices[0].message.content.strip()
        log.debug(f"Summarized {len(content)} chars -> {len(summary)} chars")
        return summary

    except Exception as e:
        log.warning(f"Summarization failed, using truncation: {e}")
        # Fallback to truncation
        return content[:max_chars] + "..." if len(content) > max_chars else content


async def summarize_code(
    code: str,
    query: str = "",
    max_lines: int = 15,
) -> str:
    """Summarize code chunk, focusing on relevance to query.

    Args:
        code: Code content to summarize
        query: Search query for context
        max_lines: Maximum lines in summary

    Returns:
        Summarized code with context
    """
    lines = code.split("\n")

    # If short enough, return as-is
    if len(lines) <= max_lines:
        return code

    try:
        import litellm

        prompt = f"""Summarize this code in {max_lines} lines or less.
Keep the most important parts that relate to: "{query}"
Include function signatures, key logic, and relevant comments.

Code:
```
{code[:2000]}  # Limit input to avoid token overflow
```

Return only the summarized code, no explanations."""

        response = await litellm.acompletion(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1,
        )

        summary = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if summary.startswith("```"):
            summary = "\n".join(summary.split("\n")[1:])
        if summary.endswith("```"):
            summary = "\n".join(summary.split("\n")[:-1])

        log.debug(f"Summarized code {len(lines)} lines -> ~{len(summary.split(chr(10)))} lines")
        return summary

    except Exception as e:
        log.warning(f"Code summarization failed: {e}")
        # Fallback: return first + last lines with context
        if len(lines) > max_lines:
            half = max_lines // 2
            return "\n".join(lines[:half] + ["...", f"# ({len(lines) - max_lines} lines omitted)"] + lines[-half:])
        return code


async def batch_summarize(
    items: list[dict[str, Any]],
    content_key: str = "content",
    context: str = "",
    max_chars: int = SUMMARY_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Summarize multiple items efficiently.

    Uses asyncio.gather for parallel summarization.

    Args:
        items: List of dicts containing content to summarize
        content_key: Key containing the content to summarize
        context: Optional context for summarization
        max_chars: Target max chars per summary

    Returns:
        Items with content summarized
    """
    if not items:
        return items

    async def summarize_item(item: dict) -> dict:
        if content_key not in item:
            return item
        content = item[content_key]
        if len(content) <= max_chars:
            return item
        summary = await summarize_content(content, context, max_chars)
        result = item.copy()
        result[content_key] = summary
        result["_summarized"] = True
        return result

    # Run summarizations in parallel
    results = await asyncio.gather(*[summarize_item(item) for item in items])
    return list(results)


# ═══════════════════════════════════════════════════════════════════════════════
# TOON FORMAT (Token-Optimized Object Notation)
# ═══════════════════════════════════════════════════════════════════════════════


def to_toon(records: list[dict[str, Any]], headers: list[str] | None = None) -> str:
    """Convert list of dicts to TOON format.

    TOON removes JSON's syntactic overhead:
    - Headers define structure once
    - Values separated by tabs
    - Records separated by newlines

    Example:
        JSON: [{"uuid":"a1","type":"lesson","score":0.9}]
        TOON: uuid\ttype\tscore
              a1\tlesson\t0.9

    Args:
        records: List of dicts to convert
        headers: Optional header order (auto-detected if not provided)

    Returns:
        TOON formatted string
    """
    if not records:
        return ""

    # Auto-detect headers from first record
    if headers is None:
        headers = list(records[0].keys())

    lines = ["\t".join(headers)]
    for r in records:
        values = []
        for h in headers:
            val = r.get(h, "")
            # Escape tabs and newlines in values
            if isinstance(val, str):
                val = val.replace("\t", " ").replace("\n", " ")
            values.append(str(val) if val is not None else "")
        lines.append("\t".join(values))

    return "\n".join(lines)


def from_toon(toon_str: str) -> list[dict[str, Any]]:
    """Parse TOON format back to list of dicts.

    Args:
        toon_str: TOON formatted string

    Returns:
        List of dicts
    """
    lines = toon_str.strip().split("\n")
    if len(lines) < 2:
        return []

    headers = lines[0].split("\t")
    records = []
    for line in lines[1:]:
        values = line.split("\t")
        record = {}
        for i, h in enumerate(headers):
            record[h] = values[i] if i < len(values) else ""
        records.append(record)

    return records


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════


def format_response(
    data: Any,
    output_format: str | None = None,
) -> str:
    """Format response based on configured output format.

    Args:
        data: Data to format
        output_format: Override format (uses env var if not specified)

    Returns:
        Formatted string
    """
    fmt = output_format or OUTPUT_FORMAT

    if fmt == "toon" and isinstance(data, list) and data and isinstance(data[0], dict):
        return to_toon(data)
    elif fmt == "compact":
        return compact_json(data)
    else:
        import json

        return json.dumps(data)


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN COUNTING (for ablation testing)
# ═══════════════════════════════════════════════════════════════════════════════


_encoder = None


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding (GPT-4/Claude compatible).

    Args:
        text: Text to count tokens for

    Returns:
        Token count
    """
    global _encoder
    if _encoder is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            log.warning("tiktoken not installed, using char estimate")
            return len(text) // 4  # Rough estimate

    return len(_encoder.encode(text))


def log_token_savings(original: str, formatted: str, label: str = "") -> dict[str, Any]:
    """Log token savings for ablation testing.

    Args:
        original: Original content
        formatted: Formatted content
        label: Optional label for logging

    Returns:
        Dict with token counts and savings percentage
    """
    original_tokens = count_tokens(original)
    formatted_tokens = count_tokens(formatted)

    if original_tokens > 0:
        savings_pct = (original_tokens - formatted_tokens) / original_tokens * 100
    else:
        savings_pct = 0

    result = {
        "original_tokens": original_tokens,
        "formatted_tokens": formatted_tokens,
        "savings_tokens": original_tokens - formatted_tokens,
        "savings_percent": round(savings_pct, 1),
    }

    log.info(f"Token savings{f' ({label})' if label else ''}: {original_tokens} -> {formatted_tokens} ({savings_pct:.1f}% reduction)")

    return result
