"""Entity extraction for SimpleMem Lite.

Uses lightweight LLM processing to extract structured entities from text.
"""

from dataclasses import dataclass, field

from simplemem_lite.config import Config
from simplemem_lite.logging import get_logger

log = get_logger("extractors")


@dataclass
class ExtractedEntities:
    """Structured entities extracted from text.

    Attributes:
        file_paths: List of file paths mentioned
        commands: List of CLI commands used
        errors: List of error messages/patterns
        tools: List of tools/functions called
    """

    file_paths: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict:
        """Convert to metadata dict for storage."""
        return {
            "extracted_files": self.file_paths[:10],  # Cap for storage
            "extracted_commands": self.commands[:10],
            "extracted_errors": self.errors[:5],
            "extracted_tools": self.tools[:10],
        }

    def is_empty(self) -> bool:
        """Check if no entities were extracted."""
        return not (self.file_paths or self.commands or self.errors or self.tools)


async def extract_entities(text: str, config: Config | None = None) -> ExtractedEntities:
    """Extract structured entities from text using lightweight LLM.

    Args:
        text: Text to extract entities from
        config: Optional config for LLM model

    Returns:
        ExtractedEntities with file paths, commands, errors, tools
    """
    from litellm import acompletion
    import json
    import re

    config = config or Config()

    # Truncate text for efficiency
    text_sample = text[:2000] if len(text) > 2000 else text

    try:
        response = await acompletion(
            model=config.summary_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract entities from this code interaction. Return ONLY valid JSON, no other text.

Text:
{text_sample}

Return this exact JSON structure (empty arrays if none found):
{{"file_paths": ["path1.py"], "commands": ["git commit"], "errors": ["Error: ..."], "tools": ["Edit", "Bash"]}}""",
                }
            ],
            max_tokens=200,
        )

        content = response.choices[0].message.content

        # Try to extract JSON from response (handles markdown code blocks)
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()

        data = json.loads(content)

        # Validate expected structure
        if not isinstance(data, dict):
            log.warning("Entity extraction returned non-dict")
            return ExtractedEntities()

        return ExtractedEntities(
            file_paths=data.get("file_paths", [])[:10] if isinstance(data.get("file_paths"), list) else [],
            commands=data.get("commands", [])[:10] if isinstance(data.get("commands"), list) else [],
            errors=data.get("errors", [])[:5] if isinstance(data.get("errors"), list) else [],
            tools=data.get("tools", [])[:10] if isinstance(data.get("tools"), list) else [],
        )

    except Exception as e:
        log.warning(f"Entity extraction failed: {e}")
        return ExtractedEntities()


async def extract_entities_batch(
    texts: list[str], config: Config | None = None
) -> list[ExtractedEntities]:
    """Extract entities from multiple texts in parallel.

    Args:
        texts: List of texts to process
        config: Optional config

    Returns:
        List of ExtractedEntities
    """
    import asyncio

    config = config or Config()

    results = await asyncio.gather(
        *[extract_entities(text, config) for text in texts],
        return_exceptions=True,
    )

    # Replace exceptions with empty entities
    return [
        r if isinstance(r, ExtractedEntities) else ExtractedEntities()
        for r in results
    ]
