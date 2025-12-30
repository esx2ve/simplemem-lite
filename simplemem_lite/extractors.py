"""Entity extraction for SimpleMem Lite.

Uses lightweight LLM processing to extract structured entities from text.
No hardcoded regexes - all extraction via prompt engineering.
"""

from dataclasses import dataclass, field

from simplemem_lite.config import Config
from simplemem_lite.logging import get_logger

log = get_logger("extractors")


async def _validate_entities_llm(
    entities: list[dict],
    config: Config,
) -> list[dict]:
    """Validate extracted entities using LLM to filter hallucinations.

    Uses a lightweight LLM call to semantically identify and remove:
    - Schema leakage (verb_edges.*, entity_types.*)
    - Prompt examples (path/file.py, src/main.py)
    - Placeholders (file, path, filename)
    - Python functions misclassified as tools

    Args:
        entities: List of entity dicts with name, type, action
        config: Config for LLM model

    Returns:
        Filtered list of valid entities
    """
    from litellm import acompletion
    import json

    if not entities:
        return []

    # Build entity list for validation
    entity_json = json.dumps(entities, indent=2)

    try:
        response = await acompletion(
            model=config.summary_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Review these extracted entities and REMOVE invalid ones.

REMOVE entities that are:
1. Schema/internal references (containing "verb_edges.", "entity_types.", etc.)
2. Example paths from prompts (path/file.py, src/main.py, example.py, file.py)
3. Generic placeholders (file, path, filename, filepath)
4. Python functions/methods being edited (ending with "()" like add_verb_edge())
   - KEEP actual tools: Read, Edit, Bash, Grep, Glob, Write, mcp__*, pal:*
5. Single-word CLI drivers without subcommand (just "git" without "commit")

Entities to validate:
{entity_json}

Return ONLY the valid entities as a JSON array. If all are invalid, return [].
Return ONLY JSON, no explanation.""",
                }
            ],
            max_tokens=500,
        )

        content = response.choices[0].message.content

        # Extract JSON array from response
        import re
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group()

        validated = json.loads(content)

        if not isinstance(validated, list):
            log.warning("LLM validation returned non-list, using original")
            return entities

        # Filter to only dicts with required fields
        result = []
        for item in validated:
            if isinstance(item, dict) and "name" in item and "type" in item:
                result.append(item)

        removed_count = len(entities) - len(result)
        if removed_count > 0:
            log.debug(f"LLM validation removed {removed_count} invalid entities")

        return result

    except Exception as e:
        log.warning(f"LLM entity validation failed: {e}, using unfiltered entities")
        return entities


@dataclass
class ExtractedEntity:
    """A single extracted entity with action type.

    Attributes:
        name: Entity name (file path, tool name, command, error message)
        type: Entity type (file, tool, command, error)
        action: Action performed (reads, modifies, executes, triggered)
    """

    name: str
    type: str  # file, tool, command, error
    action: str  # reads, modifies, executes, triggered


@dataclass
class EnhancedExtraction:
    """Enhanced extraction result with entities, actions, and goal.

    Attributes:
        entities: List of extracted entities with their actions
        goal: Extracted user goal/intent (if present)
    """

    entities: list[ExtractedEntity] = field(default_factory=list)
    goal: str | None = None

    def to_metadata(self) -> dict:
        """Convert to metadata dict for storage."""
        files = [e.name for e in self.entities if e.type == "file"]
        tools = [e.name for e in self.entities if e.type == "tool"]
        commands = [e.name for e in self.entities if e.type == "command"]
        errors = [e.name for e in self.entities if e.type == "error"]

        return {
            "extracted_files": files[:10],
            "extracted_tools": tools[:10],
            "extracted_commands": commands[:10],
            "extracted_errors": errors[:5],
            "extracted_goal": self.goal,
        }

    def is_empty(self) -> bool:
        """Check if no entities were extracted."""
        return not self.entities


@dataclass
class ExtractedEntities:
    """Structured entities extracted from text (legacy format).

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

    # Smart truncation: preserve start (context) and end (results/errors)
    # Errors and key results often appear at the end of long outputs
    if len(text) > 2000:
        text_sample = text[:1000] + "\n...[truncated]...\n" + text[-1000:]
    else:
        text_sample = text

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


async def extract_with_actions(
    text: str,
    config: Config | None = None,
    include_goal: bool = False,
) -> EnhancedExtraction:
    """Extract entities with action types using LLM.

    Uses prompt engineering to classify each entity's action:
    - READS: File was opened/viewed but NOT changed
    - MODIFIES: File was created/edited/deleted
    - EXECUTES: Tool or command was run
    - TRIGGERED: Error was caused by an action

    Args:
        text: Text to extract entities from
        config: Optional config for LLM model
        include_goal: Whether to extract user goal/intent

    Returns:
        EnhancedExtraction with entities and optional goal
    """
    from litellm import acompletion
    import json
    import re

    config = config or Config()

    # Smart truncation: preserve start (context) and end (results/errors)
    if len(text) > 3000:
        text_sample = text[:1500] + "\n...[truncated]...\n" + text[-1500:]
    else:
        text_sample = text

    # Build the extraction prompt
    goal_instruction = ""
    goal_field = ""
    if include_goal:
        goal_instruction = '\n- "goal": Extract the user\'s main objective/intent from this interaction (1 sentence, null if unclear)'
        goal_field = ', "goal": "Fix the authentication bug" or null'

    try:
        response = await acompletion(
            model=config.summary_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract entities from this code interaction.

Action types:
- reads: File opened/viewed but NOT changed
- modifies: File created/edited/deleted
- executes: Tool or command was run
- triggered: Error was caused

Entity types:
- file: Actual file paths from the project (NOT example paths)
- tool: Agent tools that were INVOKED (Read, Edit, Bash, Grep, Glob, Write, mcp__*, pal:*)
- command: CLI commands executed (git commit, npm install, docker build)
- error: Error messages/exceptions

IMPORTANT:
- Only extract REAL entities from the text, not example paths
- Tools are agent tools being CALLED, not Python functions in the code being edited
- Do NOT include: path/file.py, src/main.py, example.py (these are examples, not real)

Text:
{text_sample}

Return ONLY valid JSON:
{{"entities": [{{"name": "actual/file.ts", "type": "file", "action": "modifies"}}]{goal_field}}}

Rules:
- Empty array if no real entities found
- Lowercase action values
- tools: action is always "executes"
- errors: action is always "triggered"
- files: "reads" if viewed only, "modifies" if changed
- commands: action is always "executes\"""",
                }
            ],
            max_tokens=400,
        )

        content = response.choices[0].message.content

        # Try to extract JSON from response (handles markdown code blocks)
        json_match = re.search(r'\{[^{}]*"entities"[^{}]*\[.*?\][^{}]*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()

        data = json.loads(content)

        if not isinstance(data, dict):
            log.warning("Enhanced extraction returned non-dict")
            return EnhancedExtraction()

        # Parse entities - first pass: extract and normalize
        raw_entities = data.get("entities", [])
        parsed_entities = []
        if isinstance(raw_entities, list):
            for item in raw_entities[:20]:  # Cap at 20
                if isinstance(item, dict):
                    name = item.get("name", "")
                    etype = item.get("type", "")
                    action = item.get("action", "")

                    # Validate entity type
                    if etype not in ("file", "tool", "command", "error"):
                        continue

                    # Validate/default action based on type
                    if etype == "tool":
                        action = "executes"
                    elif etype == "command":
                        action = "executes"
                    elif etype == "error":
                        action = "triggered"
                    elif etype == "file" and action not in ("reads", "modifies"):
                        action = "reads"  # Default for files

                    if name:
                        parsed_entities.append({"name": name, "type": etype, "action": action})

        # LLM-based validation: filter hallucinations and schema leakage
        validated_entities = await _validate_entities_llm(parsed_entities, config)

        # Convert to ExtractedEntity objects
        entities = [
            ExtractedEntity(name=e["name"], type=e["type"], action=e.get("action", "reads"))
            for e in validated_entities
            if isinstance(e, dict) and "name" in e and "type" in e
        ]

        # Parse goal if requested
        goal = None
        if include_goal:
            goal = data.get("goal")
            if goal and not isinstance(goal, str):
                goal = None

        return EnhancedExtraction(entities=entities, goal=goal)

    except Exception as e:
        log.warning(f"Enhanced extraction failed: {e}")
        return EnhancedExtraction()


async def extract_with_actions_batch(
    texts: list[str],
    config: Config | None = None,
    include_goal: bool = False,
) -> list[EnhancedExtraction]:
    """Extract entities with actions from multiple texts in parallel.

    Args:
        texts: List of texts to process
        config: Optional config
        include_goal: Whether to extract goals

    Returns:
        List of EnhancedExtraction results
    """
    import asyncio

    config = config or Config()

    results = await asyncio.gather(
        *[extract_with_actions(text, config, include_goal) for text in texts],
        return_exceptions=True,
    )

    return [
        r if isinstance(r, EnhancedExtraction) else EnhancedExtraction()
        for r in results
    ]


async def extract_goal(text: str, config: Config | None = None) -> str | None:
    """Extract user goal/intent from text using LLM.

    Args:
        text: Text (typically first user message) to extract goal from
        config: Optional config

    Returns:
        Goal string or None if unclear
    """
    from litellm import acompletion

    config = config or Config()

    # Truncate if needed
    if len(text) > 1000:
        text_sample = text[:1000]
    else:
        text_sample = text

    try:
        response = await acompletion(
            model=config.summary_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract the user's main goal/objective from this message.
Return a single sentence describing what they want to accomplish.
If the goal is unclear or it's not a task request, return "UNCLEAR".

User message:
{text_sample}

Goal (one sentence):""",
                }
            ],
            max_tokens=100,
        )

        goal = response.choices[0].message.content.strip()

        if goal.upper() == "UNCLEAR" or len(goal) < 10:
            return None

        return goal[:500]  # Cap length

    except Exception as e:
        log.warning(f"Goal extraction failed: {e}")
        return None
