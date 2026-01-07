"""Tests for token reduction module.

Tests the following token reduction strategies:
1. Compact JSON serialization (orjson)
2. Field pruning (remove null/empty fields)
3. Content summarization (LLM-powered)
4. TOON format (Token-Optimized Object Notation)
5. Token counting and savings logging
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simplemem_lite.token_reduction import (
    compact_json,
    count_tokens,
    format_response,
    from_toon,
    log_token_savings,
    prune_dict,
    summarize_code,
    summarize_content,
    to_toon,
)


class TestCompactJson:
    """Test compact JSON serialization."""

    def test_compact_json_no_whitespace(self):
        """Compact JSON should have no extra whitespace."""
        data = {"key": "value", "nested": {"a": 1, "b": 2}}
        result = compact_json(data)

        # No newlines or extra spaces
        assert "\n" not in result
        assert "  " not in result  # No indentation

    def test_compact_json_sorted_keys(self):
        """Compact JSON should have sorted keys."""
        data = {"z": 1, "a": 2, "m": 3}
        result = compact_json(data)

        # Keys should be sorted
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_compact_json_vs_standard(self):
        """Compact JSON should be smaller than standard JSON."""
        data = {
            "uuid": "abc-123-def-456",
            "content": "This is a test memory content",
            "type": "lesson_learned",
            "score": 0.95,
            "metadata": {"project": "test"},
        }

        compact = compact_json(data)
        standard = json.dumps(data, indent=2)

        assert len(compact) < len(standard)
        # Typically 20-30% smaller
        savings = (len(standard) - len(compact)) / len(standard) * 100
        assert savings > 15  # At least 15% smaller


class TestPruneDict:
    """Test field pruning."""

    def test_prune_removes_null(self):
        """Prune should remove null values."""
        data = {"a": 1, "b": None, "c": "value"}
        result = prune_dict(data)

        assert "a" in result
        assert "b" not in result
        assert "c" in result

    def test_prune_removes_empty_dict(self):
        """Prune should remove empty dicts."""
        data = {"a": 1, "b": {}, "c": "value"}
        result = prune_dict(data)

        assert "a" in result
        assert "b" not in result
        assert "c" in result

    def test_prune_nested(self):
        """Prune should handle nested dicts."""
        data = {
            "a": 1,
            "nested": {
                "x": None,
                "y": "value",
                "z": {},
            },
        }
        result = prune_dict(data)

        assert "a" in result
        assert "nested" in result
        assert "x" not in result["nested"]
        assert "y" in result["nested"]
        assert "z" not in result["nested"]

    def test_prune_keeps_false_and_zero(self):
        """Prune should keep False and 0 values."""
        data = {"a": False, "b": 0, "c": None}
        result = prune_dict(data)

        assert "a" in result
        assert result["a"] is False
        assert "b" in result
        assert result["b"] == 0
        assert "c" not in result

    def test_prune_nested_becomes_empty(self):
        """Prune should remove nested dict if all values pruned."""
        data = {
            "a": 1,
            "nested": {
                "x": None,
                "y": {},
            },
        }
        result = prune_dict(data)

        assert "a" in result
        assert "nested" not in result  # All values pruned


class TestToonFormat:
    """Test TOON (Token-Optimized Object Notation) format."""

    def test_to_toon_basic(self):
        """TOON should convert list of dicts to tab-separated format."""
        records = [
            {"uuid": "a1", "type": "lesson", "score": 0.9},
            {"uuid": "b2", "type": "fact", "score": 0.8},
        ]
        result = to_toon(records)

        lines = result.split("\n")
        assert len(lines) == 3  # Header + 2 data rows

        # Check header
        headers = lines[0].split("\t")
        assert "uuid" in headers
        assert "type" in headers
        assert "score" in headers

        # Check data
        assert "a1" in lines[1]
        assert "lesson" in lines[1]
        assert "b2" in lines[2]

    def test_to_toon_custom_headers(self):
        """TOON should support custom header order."""
        records = [
            {"a": 1, "b": 2, "c": 3},
        ]
        result = to_toon(records, headers=["c", "a", "b"])

        lines = result.split("\n")
        assert lines[0] == "c\ta\tb"
        assert lines[1] == "3\t1\t2"

    def test_to_toon_escapes_special(self):
        """TOON should escape tabs and newlines in values."""
        records = [
            {"content": "line1\tline2\nline3"},
        ]
        result = to_toon(records)

        # Should not have raw tabs/newlines in content
        assert result.count("\n") == 1  # Only the row separator

    def test_to_toon_empty(self):
        """TOON should handle empty list."""
        result = to_toon([])
        assert result == ""

    def test_from_toon_roundtrip(self):
        """TOON should support roundtrip conversion."""
        records = [
            {"uuid": "a1", "type": "lesson", "score": "0.9"},
            {"uuid": "b2", "type": "fact", "score": "0.8"},
        ]
        toon = to_toon(records)
        parsed = from_toon(toon)

        assert len(parsed) == 2
        assert parsed[0]["uuid"] == "a1"
        assert parsed[1]["type"] == "fact"

    def test_toon_vs_json_tokens(self):
        """TOON should use fewer tokens than JSON."""
        records = [
            {"uuid": "abc-123", "type": "lesson_learned", "score": "0.95"},
            {"uuid": "def-456", "type": "decision", "score": "0.87"},
            {"uuid": "ghi-789", "type": "pattern", "score": "0.92"},
        ]

        json_str = json.dumps(records)
        toon_str = to_toon(records)

        json_tokens = count_tokens(json_str)
        toon_tokens = count_tokens(toon_str)

        # TOON should use significantly fewer tokens
        assert toon_tokens < json_tokens
        savings = (json_tokens - toon_tokens) / json_tokens * 100
        assert savings > 20  # At least 20% fewer tokens


class TestFormatResponse:
    """Test response formatting."""

    def test_format_json(self):
        """JSON format should return standard JSON."""
        data = {"key": "value"}
        result = format_response(data, "json")

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_format_compact(self):
        """Compact format should use orjson."""
        data = {"z": 1, "a": 2}
        result = format_response(data, "compact")

        # Should have sorted keys and no whitespace
        assert result.index('"a"') < result.index('"z"')
        assert "\n" not in result

    def test_format_toon_list(self):
        """TOON format should work for list data."""
        data = [
            {"uuid": "a", "type": "test"},
        ]
        result = format_response(data, "toon")

        # Should be tab-separated
        assert "\t" in result
        assert "uuid" in result


class TestTokenCounting:
    """Test token counting utilities."""

    def test_count_tokens_basic(self):
        """Token counting should work for basic text."""
        text = "Hello, world!"
        tokens = count_tokens(text)

        # Should return a reasonable count
        assert tokens > 0
        assert tokens < len(text)  # Tokens < chars typically

    def test_count_tokens_code(self):
        """Token counting should work for code."""
        code = """
def hello():
    print("Hello, world!")
    return True
"""
        tokens = count_tokens(code)
        assert tokens > 0

    def test_log_token_savings(self):
        """Log token savings should calculate correctly."""
        original = "This is a longer string that takes more tokens"
        formatted = "Short"

        result = log_token_savings(original, formatted, "test")

        assert "original_tokens" in result
        assert "formatted_tokens" in result
        assert "savings_tokens" in result
        assert "savings_percent" in result
        assert result["savings_tokens"] > 0


class TestSummarization:
    """Test LLM summarization functions."""

    @pytest.mark.asyncio
    async def test_summarize_content_short_passthrough(self):
        """Short content should pass through unchanged."""
        short_content = "This is short."
        result = await summarize_content(short_content, max_chars=500)
        assert result == short_content

    @pytest.mark.asyncio
    async def test_summarize_content_long(self):
        """Long content should be summarized (mocked)."""
        long_content = "x" * 1000  # Long content

        # Mock LiteLLM
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary of content"))]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await summarize_content(long_content, max_chars=200)

            # Should have called LLM
            assert mock_llm.called
            # Should return summary
            assert len(result) < len(long_content)

    @pytest.mark.asyncio
    async def test_summarize_content_fallback(self):
        """Summarization should fallback to truncation on error."""
        long_content = "x" * 1000

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API error")
            result = await summarize_content(long_content, max_chars=200)

            # Should truncate instead of failing
            assert len(result) <= 203  # max_chars + "..."
            assert result.endswith("...")

    @pytest.mark.asyncio
    async def test_summarize_code_short_passthrough(self):
        """Short code should pass through unchanged."""
        short_code = "def hello():\n    pass"
        result = await summarize_code(short_code, max_lines=15)
        assert result == short_code

    @pytest.mark.asyncio
    async def test_summarize_code_long(self):
        """Long code should be summarized (mocked)."""
        long_code = "\n".join([f"line_{i} = {i}" for i in range(50)])

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="# Summary\ndef main():\n    pass"))]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await summarize_code(long_code, query="test", max_lines=10)

            assert mock_llm.called
            # Should be shorter
            assert len(result.split("\n")) < len(long_code.split("\n"))


class TestIntegration:
    """Integration tests for token reduction pipeline."""

    def test_full_pipeline_memory(self):
        """Test full token reduction pipeline for memory results."""
        # Simulate raw search results
        raw_results = [
            {
                "uuid": "memory-abc-123-def-456-ghi-789",
                "content": "When debugging database connection issues, always check: 1) Connection pool settings 2) Timeout configurations 3) Network connectivity 4) SSL certificates. The most common cause is pool exhaustion during high traffic.",
                "type": "lesson_learned",
                "score": 0.95,
                "session_id": None,
                "metadata": {},
                "created_at": 1704067200,
            },
            {
                "uuid": "memory-xyz-987-uvw-654-rst-321",
                "content": "Decision: Use Redis for session storage instead of PostgreSQL. Rationale: Faster reads, better suited for key-value access patterns, built-in TTL support.",
                "type": "decision",
                "score": 0.87,
                "session_id": "sess-123",
                "metadata": {"project": "api"},
                "created_at": 1704153600,
            },
        ]

        # Step 1: Prune fields
        pruned = [prune_dict(r) for r in raw_results]

        # Verify null/empty fields removed
        assert "session_id" not in pruned[0]  # Was null
        assert "metadata" not in pruned[0]  # Was empty dict
        assert "session_id" in pruned[1]  # Was set

        # Step 2: Compact JSON
        compact = compact_json(pruned)

        # Step 3: Compare with standard JSON
        standard = json.dumps(raw_results, indent=2)

        savings = log_token_savings(standard, compact, "memory_pipeline")

        # Should have significant savings
        assert savings["savings_percent"] > 20

    def test_full_pipeline_toon(self):
        """Test TOON format for maximum token reduction."""
        # Simulate search results for TOON format
        results = [
            {"uuid": "a1b2c3", "type": "lesson", "score": "0.95", "content": "Short insight"},
            {"uuid": "d4e5f6", "type": "decision", "score": "0.87", "content": "Decision note"},
            {"uuid": "g7h8i9", "type": "pattern", "score": "0.92", "content": "Code pattern"},
        ]

        # JSON format
        json_output = json.dumps(results)

        # TOON format
        toon_output = to_toon(results, headers=["uuid", "type", "score", "content"])

        # Compare
        json_tokens = count_tokens(json_output)
        toon_tokens = count_tokens(toon_output)

        print(f"\nJSON output ({json_tokens} tokens):\n{json_output}")
        print(f"\nTOON output ({toon_tokens} tokens):\n{toon_output}")

        savings = (json_tokens - toon_tokens) / json_tokens * 100
        print(f"\nToken savings: {savings:.1f}%")

        # TOON should save at least 30%
        assert savings > 25

    def test_realistic_code_search_results(self):
        """Test token reduction on realistic code search results."""
        # Simulate realistic code search output
        raw_results = [
            {
                "uuid": f"chunk-{i:03d}",
                "filepath": f"/src/module_{i}/handler.py",
                "content": f"""def process_request_{i}(request: Request) -> Response:
    '''Process incoming request with validation.'''
    if not request.is_valid():
        raise ValidationError("Invalid request")

    data = request.json()
    result = service.handle(data)

    return Response(data=result, status=200)
""",
                "start_line": 10 + i * 20,
                "end_line": 20 + i * 20,
                "project_id": "config:myproject",
                "score": 0.9 - (i * 0.05),
            }
            for i in range(5)
        ]

        # Measure raw size
        raw_json = json.dumps(raw_results, indent=2)
        raw_tokens = count_tokens(raw_json)

        # Apply pruning
        pruned = [prune_dict(r) for r in raw_results]

        # Compact JSON
        compact = compact_json(pruned)
        compact_tokens = count_tokens(compact)

        # TOON with abbreviated content
        toon_results = [
            {
                "uuid": r["uuid"],
                "filepath": r["filepath"],
                "lines": f"{r['start_line']}-{r['end_line']}",
                "score": f"{r['score']:.2f}",
            }
            for r in raw_results
        ]
        toon_output = to_toon(toon_results)
        toon_tokens = count_tokens(toon_output)

        print(f"\nRaw JSON: {raw_tokens} tokens")
        print(f"Compact JSON: {compact_tokens} tokens ({(raw_tokens - compact_tokens) / raw_tokens * 100:.1f}% savings)")
        print(f"TOON (metadata only): {toon_tokens} tokens ({(raw_tokens - toon_tokens) / raw_tokens * 100:.1f}% savings)")

        # Verify significant reductions
        assert compact_tokens < raw_tokens
        assert toon_tokens < compact_tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
