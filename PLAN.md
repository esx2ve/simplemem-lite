# SimpleMem-Lite MCP Interface Overhaul Plan

## Frontier Model Consensus Results

**3 Frontier Models Consulted** with **8.7/10 average confidence**:
- Gemini 3 Pro (FOR): 9/10
- DeepSeek R1 (AGAINST): 8/10
- Grok 4.1 (NEUTRAL): 9/10

---

## Current State (Problems)

- **Resources (4)**: Passive, UUID-centric, don't expose rich entity graph
- **Prompts (4)**: Generic, don't leverage advanced features (ask_memories, cross-session)
- **Server Instructions**: Just "Minimal hybrid memory MCP server" - no guidance on when/how to use

---

## Unanimous Agreement (All 3 Models)

| Decision | Consensus |
|----------|-----------|
| Entity-centric abstraction | **CORRECT** - hides UUID mess, enables intuitive browsing |
| Project isolation | **MANDATORY** - add `project_id` to all resources/storage |
| Prompts with examples | **YES** - add 1-2 example outputs per prompt |
| "Start with Context" | **CONDITIONAL** - user-initiated, not auto-run |
| Enhanced server instructions | **P0 PRIORITY** - trivial change, massive value |
| Direct graph exposure | **NO** - would overwhelm users |

---

## Key Disagreement Resolved

**Resource Granularity:**

| Model | Position |
|-------|----------|
| Gemini | Fine-grained URIs (`entities://files/{name}`) |
| DeepSeek | Query params (`?name=&project=`) |
| Grok | **HYBRID** (both) |

**Resolution: HYBRID APPROACH**
```
# Fine-grained for high-use entities
entities://files/{name}
entities://errors/{pattern}

# Query-based for flexible filtering
insights://?type=cross-session&project=xyz&min_score=0.3
```

---

## Implementation Phases

### P0 - Server Instructions + Prompts (1-2 days)

**Goal**: Maximum value with zero backend changes.

#### 1. Enhanced Server Instructions

Replace current minimal instructions with detailed guidance:

```python
mcp = FastMCP(
    "simplemem-lite",
    instructions="""SimpleMem Lite: Long-term structured memory with cross-session learning.

## WHEN TO USE MEMORY

**At session START**: Recall relevant prior work
- Use `ask_memories("context for {task}")` for LLM-synthesized answers
- Check what files/errors we've seen before

**When encountering ERRORS**: Check past solutions
- `ask_memories("solution for {error}")` returns cited answers
- Cross-session patterns show if error appeared in other projects

**Before complex DECISIONS**: Look for past approaches
- `reason_memories` finds conclusions via multi-hop graph traversal
- Shared entities (files, tools, errors) link across sessions

**At session END**: Store key learnings
- `store_memory` with type="lesson_learned" for valuable insights
- Link to relevant entities for future cross-session discovery

## KEY CAPABILITIES

- `ask_memories`: LLM-synthesized answers with citations [1][2]
- `reason_memories`: Multi-hop graph reasoning with proof chains
- `search_memories`: Hybrid vector + graph search
- `process_trace`: Index Claude Code sessions hierarchically
- Cross-session insights via shared entities (files, tools, errors)
"""
)
```

#### 2. New Workflow-Integrated Prompts

**"Start with Context"** - Proactive recall at session start:
```python
@mcp.prompt(title="Start with Context")
def start_with_context(task: str) -> str:
    """Gather relevant context before starting a task."""
    return f'''Before starting: "{task}"

1. Use ask_memories to find relevant prior work:
   ask_memories("context for {task}")

2. Check for related files we've worked with before

3. Look for similar problems and their solutions

Synthesize: What do we already know that helps here?

Example output format:
"Based on memories [1][2], this task relates to prior work on X.
Key insight: [specific pattern or solution that worked].
Files involved: [relevant files from past sessions]."'''
```

**"Smart Debug"** - Full memory graph debugging:
```python
@mcp.prompt(title="Smart Debug")
def smart_debug(error: str, file_context: str = "") -> str:
    """Debug using full memory graph and cross-session insights."""
    file_note = f"\nContext: Error occurred in {file_context}" if file_context else ""
    return f'''Debugging error: {error}{file_note}

Steps:
1. Use ask_memories("solutions for: {error}") to get LLM-synthesized answer with citations

2. Check for similar error patterns across sessions - cross-session insights are especially valuable

3. If file context provided, look for that file's modification history

4. Provide solution based on what worked before, citing specific memories [1][2]

Focus on actionable solutions with evidence from past sessions.'''
```

**"Store Session Learnings"** - Capture insights at session end:
```python
@mcp.prompt(title="Store Session Learnings")
def store_session_learnings() -> str:
    """Capture key insights from the current session."""
    return '''Before ending, preserve what we learned:

Identify and store each of:
1. **Problem solved**: What was the main challenge?
2. **Solution that worked**: What approach succeeded?
3. **Gotchas discovered**: Any non-obvious issues?
4. **Patterns worth remembering**: Reusable techniques?

For each insight, use store_memory with:
- Descriptive content capturing the learning
- type="lesson_learned" for cross-session discovery
- Include relevant file/tool/command context

Example:
store_memory(
    text="Fixed auth token expiry by checking refresh logic before API calls. Key: always validate token age, not just presence.",
    type="lesson_learned"
)'''
```

**"What Do We Know About..."** - Entity-centric recall:
```python
@mcp.prompt(title="What Do We Know About")
def entity_recall(entity_name: str, entity_type: str = "file") -> str:
    """Recall everything we know about a specific entity."""
    return f'''Retrieve all knowledge about {entity_type}: {entity_name}

1. Search memories mentioning this entity:
   search_memories("{entity_name}")

2. Use reason_memories for deeper connections:
   reason_memories("history and patterns for {entity_name}")

3. Check cross-session patterns - has this entity appeared in other projects?

Synthesize into a brief summary:
- What operations were performed (reads/modifies)?
- What issues were encountered?
- What solutions worked?
- Any patterns or best practices?'''
```

---

### P1 - Entity-Centric Resources (3-5 days)

**Goal**: Expose the rich entity graph through browsable resources.

#### New Resources

```python
@mcp.resource("entities://files")
def list_file_entities() -> str:
    """List all tracked files with action counts."""
    # Query graph for Entity nodes where type='file'
    # Return: [{name, sessions_count, reads, modifies, last_seen}]

@mcp.resource("entities://files/{name}")
def get_file_history(name: str) -> str:
    """Get complete history of a specific file."""
    # Query all memories linked to this file entity
    # Include: sessions, actions (READS/MODIFIES), related errors

@mcp.resource("entities://tools")
def list_tool_usage() -> str:
    """Tool usage patterns across sessions."""
    # Aggregate tool executions from verb edges

@mcp.resource("entities://errors")
def list_error_patterns() -> str:
    """Common errors and their resolutions."""
    # Query error entities with linked solutions

@mcp.resource("insights://cross-session")
def get_cross_session_insights() -> str:
    """Patterns discovered across multiple sessions."""
    # Find entities appearing in >1 session
    # Include bridge_entity info from reason_memories

@mcp.resource("insights://project/{project_id}")
def get_project_insights(project_id: str) -> str:
    """All learnings for a specific project."""
    # Filter by project_id, aggregate key insights
```

#### Backend Requirements

- Add methods to `db.py` for entity queries:
  - `get_entities_by_type(type: str) -> list[dict]`
  - `get_entity_history(name: str, type: str) -> list[dict]`
  - `get_cross_session_entities(min_sessions: int = 2) -> list[dict]`

---

### P2 - Project Isolation + Entity Extraction (3-5 days)

**Goal**: Enable safe cross-session features with proper isolation.
**Status**: IN PROGRESS

#### 1. Add `project_id` to MemoryItem

```python
@dataclass
class MemoryItem:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    relations: list[dict[str, str]] = field(default_factory=list)
    project_id: str | None = None  # NEW: For cross-project isolation
```

#### 2. Update Storage to Track Project

- `store_memory` accepts optional `project_id`
- `process_trace` extracts `project_id` from session path
- All entity resources filter by `project_id` when provided

#### 3. Entity Extraction During Ingestion

Update `process_trace` to explicitly index entities:
- Parse file paths from tool_use messages
- Extract tool names from tool invocations
- Capture error patterns from tool_results
- Store as first-class Entity nodes (already done via verb edges)

#### 4. Confidence Threshold for Insights

```python
@mcp.resource("insights://cross-session")
def get_cross_session_insights(min_score: float = 0.3) -> str:
    """Patterns across sessions with confidence filtering."""
    # Use existing min_score from reason_memories
```

---

### P3 - Codebase Refactoring (Best Practices + Simplicity)

**Goal**: Refactor codebase following best practices while maintaining SIMPLICITY. No god classes, no explosion of wrappers. Maintain observability via logs.

**Approach**: Use `pal:refactor` with grok4.1-fast / gemini3pro interchangeably for analysis.

#### Guiding Principles

1. **Simplicity First**: Prefer straightforward code over abstractions
2. **No God Classes**: Break up large classes if they have too many responsibilities
3. **No Wrapper Explosion**: Don't create excessive indirection layers
4. **Maintain Observability**: Keep logging at appropriate levels (trace/debug/info/warning/error)
5. **Consistent Patterns**: Follow existing codebase conventions

#### Focus Areas

- **db.py** (~1400 lines): Evaluate if methods can be grouped into logical sections or if class should be split
- **memory.py** (~900 lines): Review MemoryStore responsibilities
- **traces.py**: Review trace processing pipeline complexity
- **server.py**: Ensure prompts/resources/tools are well-organized
- **extractors.py**: Simplify brittle parsing patterns

#### Success Criteria

1. Each class has clear, focused responsibilities
2. No single file exceeds ~800 lines without good reason
3. Logging is consistent and meaningful
4. Code is readable without excessive abstraction layers
5. Pre-existing issues from backlog are addressed where applicable

#### Completed Work

| Change | Files | Impact |
|--------|-------|--------|
| Added `json_repair` library | pyproject.toml, extractors.py | Robust LLM JSON parsing, replaces brittle regex |
| Removed dead code (~120 lines) | extractors.py, traces.py | Cleaner codebase: ExtractedEntities, extract_entities, extract_entities_batch |
| Fixed path decoding | traces.py | Added `_decode_project_path()` with smart heuristics |
| Moved inline query | db.py, server.py | Added `get_projects()` method, consistent patterns |

**Status**: ✅ P3 COMPLETE - Analysis showed no major restructuring needed. Codebase already follows simplicity principles.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Cross-project data leakage | **Mandatory** `project_id` filtering |
| Duplicate filename collisions | Namespace: `{project_id}/{filename}` |
| Noisy "Start with Context" | User-initiated only, not auto-run |
| Entity lists grow large | Add pagination early |
| Stale entities (file moves) | Graceful "not found" + search fallback |

---

## Effort Estimate

| Phase | Effort |
|-------|--------|
| P0: Instructions + Prompts | 1-2 days |
| P1: Entity Resources | 3-5 days |
| P2: Backend (project_id, entity indexing) | 3-5 days |
| P3: Codebase Refactoring | 2-3 days |
| **Total** | **~2-3 weeks** |

---

## Success Criteria

1. **P0**: Server instructions guide Claude to use memory proactively
2. **P1**: Entity resources enable browsing files/tools/errors by name
3. **P2**: Cross-session insights work safely with project isolation
4. **P3**: Codebase is clean, maintainable, and follows best practices without over-engineering
5. **Overall**: Claude naturally uses memory at session start/end and during debugging

---

## Code Quality Issues (Backlog)

Issues found during code review that are outside current phase scope but should be tracked:

### High Priority

| File | Line | Issue | Description | Status |
|------|------|-------|-------------|--------|
| db.py | 144 | Manual string escaping | Unnecessary manual escaping for parameterized queries - should rely on driver | Open |
| db.py | 394 | Path canonicalization | May over-strip "repo/src" directories - too aggressive stripping | Open |
| ~~traces.py~~ | ~~540~~ | ~~Brittle path decoding~~ | ~~`project_dir.replace("-", "/")` corrupts project names with hyphens~~ | **FIXED in P3** |
| ~~extractors.py~~ | ~~221~~ | ~~Brittle JSON parsing~~ | ~~Regex fails on nested JSON from LLM~~ | **FIXED in P3** (json_repair) |

### Medium Priority

| File | Line | Issue | Description |
|------|------|-------|-------------|
| db.py | 399 | Broad exception handling | `except Exception:` hides bugs - should catch specific exceptions |
| db.py | 688 | Inconsistent result limiting | Hardcoded LIMIT 100 + slice [:10] - should use parameter |
| server.py | 423 | smart_debug prompt | Could suggest `reason_memories` for deeper investigation |
| server.py | 47 | Server instructions | Could add "Quick Start" example |

### Low Priority

| File | Line | Issue | Description |
|------|------|-------|-------------|
| traces.py | 537 | Variable naming | `project_dir` should be `project_id` for clarity |
| traces.py | 677 | Magic numbers | Token budgeting uses unnamed literals (3, 1.5, 99) |
| db.py | - | Case sensitivity | Blacklist check may have case mismatch |
| db.py | - | Empty name handling | Empty name returns True instead of False |

**Note**: These are pre-existing issues not introduced by P0/P1/P2 changes. Consider addressing in a dedicated cleanup phase.

### P1 Implementation Issues (Deferred)

| File | Line | Issue | Description | Status |
|------|------|-------|-------------|--------|
| db.py | 1016 | Query optimization | `get_entities_by_type` uses multiple OPTIONAL MATCH + WITH - could use single pass with conditional aggregation | Open |
| ~~server.py~~ | ~~521~~ | ~~Inline query~~ | ~~`list_projects()` has inline Cypher - should move to db.py method~~ | **FIXED in P3** |
| db.py | 1167 | Missing filter | `get_cross_session_entities` should have optional `entity_type` parameter | Open |

### P2 Implementation Issues (Deferred)

| File | Line | Issue | Description |
|------|------|-------|-------------|
| db.py | 1444 | EXISTS subquery performance | Could pre-collect descendants then filter for better scalability on large graphs |
| db.py | 1450 | Hardcoded depth | CONTAINS*1..3 assumes max 3 levels - should be configurable |
| memory.py | 293 | Arbitrary multiplier | search_limit * 3 for project filtering - consider * 5 or configurable |
