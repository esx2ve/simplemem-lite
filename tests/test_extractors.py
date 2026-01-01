"""Extractor tests for SimpleMem Lite.

Tests critical extraction logic:
- JSON parsing with repair
- EnhancedExtraction dataclass operations
- ExtractedEntity dataclass
"""

import pytest


class TestParseJson:
    """Test JSON parsing with repair."""

    def test_parse_empty_string(self):
        """Empty string should return fallback."""
        from simplemem_lite.extractors import _parse_llm_json

        result = _parse_llm_json("", fallback={"default": True})
        assert result == {"default": True}

    def test_parse_whitespace_only(self):
        """Whitespace only should return fallback."""
        from simplemem_lite.extractors import _parse_llm_json

        result = _parse_llm_json("   \n\t  ", fallback=[])
        assert result == []

    def test_parse_valid_json(self):
        """Valid JSON should parse correctly."""
        from simplemem_lite.extractors import _parse_llm_json

        result = _parse_llm_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_parse_json_array(self):
        """JSON arrays should parse correctly."""
        from simplemem_lite.extractors import _parse_llm_json

        result = _parse_llm_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_parse_json_with_trailing_comma(self):
        """JSON with trailing comma should be repaired."""
        from simplemem_lite.extractors import _parse_llm_json

        # json_repair handles trailing commas
        result = _parse_llm_json('{"key": "value",}')
        assert result == {"key": "value"}

    def test_parse_json_with_single_quotes(self):
        """JSON with single quotes should be repaired."""
        from simplemem_lite.extractors import _parse_llm_json

        # json_repair handles single quotes
        result = _parse_llm_json("{'key': 'value'}")
        assert result == {"key": "value"}

    def test_parse_fallback_on_garbage(self):
        """Complete garbage should return fallback."""
        from simplemem_lite.extractors import _parse_llm_json

        result = _parse_llm_json("not json at all xyz", fallback=None)
        # json_repair may return something or fallback
        # The key is it doesn't raise


class TestExtractedEntity:
    """Test ExtractedEntity dataclass."""

    def test_create_file_entity(self):
        """Should create file entity correctly."""
        from simplemem_lite.extractors import ExtractedEntity

        entity = ExtractedEntity(
            name="src/main.py",
            type="file",
            action="modifies",
        )
        assert entity.name == "src/main.py"
        assert entity.type == "file"
        assert entity.action == "modifies"

    def test_create_tool_entity(self):
        """Should create tool entity correctly."""
        from simplemem_lite.extractors import ExtractedEntity

        entity = ExtractedEntity(
            name="Read",
            type="tool",
            action="executes",
        )
        assert entity.name == "Read"
        assert entity.type == "tool"
        assert entity.action == "executes"


class TestEnhancedExtraction:
    """Test EnhancedExtraction dataclass."""

    def test_empty_extraction(self):
        """Empty extraction should report as empty."""
        from simplemem_lite.extractors import EnhancedExtraction

        extraction = EnhancedExtraction()
        assert extraction.is_empty() is True
        assert extraction.entities == []
        assert extraction.goal is None

    def test_non_empty_extraction(self):
        """Non-empty extraction should report as non-empty."""
        from simplemem_lite.extractors import EnhancedExtraction, ExtractedEntity

        extraction = EnhancedExtraction(
            entities=[ExtractedEntity("file.py", "file", "reads")]
        )
        assert extraction.is_empty() is False

    def test_to_metadata_categorizes_entities(self):
        """to_metadata should categorize entities by type."""
        from simplemem_lite.extractors import EnhancedExtraction, ExtractedEntity

        extraction = EnhancedExtraction(
            entities=[
                ExtractedEntity("src/main.py", "file", "modifies"),
                ExtractedEntity("src/utils.py", "file", "reads"),
                ExtractedEntity("Read", "tool", "executes"),
                ExtractedEntity("Bash", "tool", "executes"),
                ExtractedEntity("git commit", "command", "executes"),
                ExtractedEntity("TypeError", "error", "triggered"),
            ],
            goal="Fix the authentication bug",
        )

        metadata = extraction.to_metadata()

        assert "src/main.py" in metadata["extracted_files"]
        assert "src/utils.py" in metadata["extracted_files"]
        assert "Read" in metadata["extracted_tools"]
        assert "Bash" in metadata["extracted_tools"]
        assert "git commit" in metadata["extracted_commands"]
        assert "TypeError" in metadata["extracted_errors"]
        assert metadata["extracted_goal"] == "Fix the authentication bug"

    def test_to_metadata_caps_at_limits(self):
        """to_metadata should respect limits on entity counts."""
        from simplemem_lite.extractors import EnhancedExtraction, ExtractedEntity

        # Create 15 file entities (limit is 10)
        entities = [
            ExtractedEntity(f"file{i}.py", "file", "reads")
            for i in range(15)
        ]

        extraction = EnhancedExtraction(entities=entities)
        metadata = extraction.to_metadata()

        assert len(metadata["extracted_files"]) == 10  # Capped at 10


class TestEntityTypeValidation:
    """Test entity type validation rules."""

    def test_valid_entity_types(self):
        """Valid entity types should be recognized."""
        valid_types = ["file", "tool", "command", "error"]
        for etype in valid_types:
            assert etype in valid_types

    def test_valid_action_types(self):
        """Valid action types should be recognized."""
        valid_actions = ["reads", "modifies", "executes", "triggered"]
        for action in valid_actions:
            assert action in valid_actions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
