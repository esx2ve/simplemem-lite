# SimpleMem Lite Improvements Log

## Overview
Fixing issues identified in the Gemini 3 Pro quality analysis of trace processing.

**Quality Score Before:** 6.5/10

## Issues to Fix

### 1. Session Summary "Bookending" Anti-Pattern (Medium Impact)
- **Problem:** `_summarize_session` only uses first/last message, ignoring chunk summaries
- **Location:** `traces.py:477-487`
- **Solution:** Pass chunk summaries to session summarization (Map-Reduce pattern)

### 2. Deep Search Data Loss (High Impact)
- **Problem:** `message_ids` list always empty - individual messages not stored
- **Location:** `traces.py:316,357`
- **Solution:** Implement selective message storage with "interestingness filter"

### 3. Arbitrary Context Truncation (Medium Impact)
- **Problem:** Only first 10 of 20 messages per chunk summarized, 500 char limit
- **Location:** `traces.py:428-429`
- **Solution:** Switch to token-based limits, prioritize tool_result content

### 4. Lack of Structured Entity Extraction (High Impact)
- **Problem:** No extraction of file paths, commands, error codes as metadata
- **Location:** MemoryItem metadata only has generic fields
- **Solution:** Add regex-based extractor for files, commands, errors

---

## Progress Log

### Session Start: 2024-12-29

| Time | Action | Status |
|------|--------|--------|
| -- | Created improvement tracking file | DONE |
| -- | PAL planner completed with gpt-5.1-codex | DONE |
| -- | Phase 1: Create extractors.py | IN PROGRESS |

---

## Implementation Plan (from PAL Planner)

### Phase 1: Entity Extraction (Foundation)
Create `extractors.py` with regex-based extraction for:
- File paths
- CLI commands
- Error patterns

### Phase 2: Interestingness Filter
Add `_is_interesting()` method to identify messages worth storing:
- Tool results > 100 chars
- Messages with errors/exceptions
- Messages with code blocks

### Phase 3: Token-based Context
Replace hard limits with token-based approach:
- ~3000 token budget per chunk
- Prioritize tool_result > assistant > user

### Phase 4: Session Summary Map-Reduce
Change `_summarize_session` to use chunk summaries instead of first/last message

### Phase 5: Store Interesting Messages
Populate `message_ids` with filtered important messages

---

## Implementation Checklist

| # | Task | Status |
|---|------|--------|
| 1 | Create extractors.py (LLM-based) | [x] |
| 2 | Add _is_interesting method | [x] |
| 3 | Add _prepare_chunk_content method | [x] |
| 4 | Update _summarize_chunk | [x] |
| 5 | Update _summarize_session (Map-Reduce) | [x] |
| 6 | Update index_session call | [x] |
| 7 | Add interesting message storage | [x] |
| 8 | Add entity metadata to messages | [x] |
| 9 | Code review (gpt-5.1-codex) | [x] |
| 10 | Fix response_format Gemini compatibility | [x] |
| 11 | Fix _is_interesting false positives | [x] |

---

## Implementation Details

### extractors.py (NEW FILE)
- `ExtractedEntities` dataclass with file_paths, commands, errors, tools
- `extract_entities()` - LLM-based extraction (not regex!)
- `extract_entities_batch()` - parallel extraction for multiple texts
- `to_metadata()` - converts to dict for storage

### traces.py CHANGES

**New methods:**
- `_is_interesting(msg)` - filters messages worth storing (tool_result >100 chars, errors, code blocks)
- `_prepare_chunk_content(chunk, max_tokens)` - token-based content prep with priority ordering

**Modified methods:**
- `_summarize_chunk()` - now uses `_prepare_chunk_content()` instead of hard limits
- `_summarize_session()` - now accepts `chunk_summaries` list and uses Map-Reduce pattern
- `index_session()` - stores interesting messages with entity metadata, passes chunk_summaries to session summary

---

## Code Review Fixes (2024-12-29)

### Fixed Issues (from PAL codereview with gpt-5.1-codex):

**MEDIUM: response_format Gemini Compatibility**
- Removed `response_format={'type': 'json_object'}` which isn't supported by Gemini flash-lite
- Added regex-based JSON extraction to handle markdown code blocks in responses
- Added validation for expected dict structure and list fields

**MEDIUM: _is_interesting False Positives**
- Changed from substring matching to word boundary regex patterns
- Added negation detection to skip "no errors", "without exceptions", etc.
- Added additional error patterns: `traceback`, `stack trace`
