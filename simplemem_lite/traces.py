"""Claude Code trace processing for SimpleMem Lite.

Parses JSONL session traces and creates hierarchical memory indexes.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from simplemem_lite.config import Config

if TYPE_CHECKING:
    from simplemem_lite.projects import ProjectManager
from simplemem_lite.extractors import (
    extract_with_actions_batch,
    extract_goal,
    EnhancedExtraction,
)
from simplemem_lite.logging import get_logger
from simplemem_lite.memory import MemoryItem, MemoryStore

log = get_logger("traces")


def _decode_project_path(encoded: str) -> str:
    """Decode a path-encoded project directory name to human-readable form.

    Claude Code encodes paths by replacing '/' with '-', e.g.:
        -Users-shimon-repo-my-project -> /Users/shimon/repo/my-project

    This decoding is LOSSY because project names can contain hyphens.
    We use heuristics to identify path separators vs literal hyphens:

    1. Leading hyphen indicates absolute path root
    2. Known path prefixes (Users, home, var, etc.) help identify boundaries
    3. The final component (project name) preserves its hyphens

    Args:
        encoded: Path-encoded directory name (e.g., "-Users-shimon-repo-my-project")

    Returns:
        Human-readable path (e.g., "/Users/shimon/repo/my-project")
    """
    if not encoded:
        return encoded

    # Pattern matches known path prefixes that indicate path boundaries
    # These are directories that commonly appear at fixed positions in paths
    path_prefix_pattern = re.compile(
        r'^-?(Users|home|var|opt|usr|tmp|root|data|srv|mnt|Volumes|Applications|Library)'
        r'(-[^-]+)*$',
        re.IGNORECASE
    )

    # If it matches the pattern, use the old conversion for display
    # Otherwise, treat the whole thing as a project name
    if path_prefix_pattern.match(encoded):
        # Convert hyphens to slashes, but be smarter about it:
        # Split by known path components and rejoin with slashes
        result = encoded.lstrip("-")

        # Replace hyphens that follow common single-word path components
        # This preserves hyphens in multi-word directory names
        common_path_parts = r'(Users|home|var|opt|usr|tmp|root|data|srv|mnt|Volumes|Applications|Library|repo|projects|src|code|dev|work|Documents|Desktop|Downloads)'
        result = re.sub(f'{common_path_parts}-', r'\1/', result, flags=re.IGNORECASE)

        return "/" + result if encoded.startswith("-") else result
    else:
        # No recognized path pattern - treat as-is (just a project name)
        return encoded.lstrip("-")


@dataclass
class TraceMessage:
    """A single message from a Claude Code trace.

    Attributes:
        uuid: Message UUID
        session_id: Session identifier
        type: Message type (user, assistant, tool_use, tool_result)
        content: Text content
        timestamp: ISO 8601 timestamp
        parent_uuid: UUID of parent message
    """

    uuid: str
    session_id: str
    type: str
    content: str
    timestamp: str
    parent_uuid: str | None = None


@dataclass
class SessionIndex:
    """Index of a processed session.

    Attributes:
        session_id: Original session identifier
        session_summary_id: UUID of session summary memory
        chunk_summary_ids: UUIDs of chunk summary memories
        message_ids: UUIDs of individual message memories
        goal_id: UUID of extracted goal (if any)
        project_id: Project identifier (if any)
    """

    session_id: str
    session_summary_id: str
    chunk_summary_ids: list[str]
    message_ids: list[str]
    goal_id: str | None = None
    project_id: str | None = None


class TraceParser:
    """Parser for Claude Code JSONL trace files.

    Reads and parses session traces from ~/.claude/projects/
    """

    def __init__(self, traces_dir: Path | None = None):
        """Initialize trace parser.

        Args:
            traces_dir: Directory containing traces (default: ~/.claude/projects)
        """
        self.traces_dir = traces_dir or Path.home() / ".claude" / "projects"
        log.debug(f"TraceParser initialized: traces_dir={self.traces_dir}")
        log.debug(f"traces_dir.exists()={self.traces_dir.exists()}")

    def list_sessions(self) -> list[dict]:
        """List available Claude Code sessions.

        Returns:
            List of session info dicts with session_id, project, path, size_kb, modified
        """
        log.trace(f"list_sessions called, traces_dir={self.traces_dir}")
        sessions = []

        if not self.traces_dir.exists():
            log.warning(f"Traces directory does not exist: {self.traces_dir}")
            return sessions

        project_count = 0
        for project_dir in self.traces_dir.iterdir():
            if not project_dir.is_dir():
                continue
            project_count += 1

            for trace_file in project_dir.glob("*.jsonl"):
                try:
                    stat = trace_file.stat()
                    sessions.append({
                        "session_id": trace_file.stem,
                        "project": project_dir.name,
                        "path": str(trace_file),
                        "size_kb": stat.st_size // 1024,
                        "modified": stat.st_mtime,
                    })
                except (OSError, PermissionError) as e:
                    log.warning(f"Could not stat {trace_file}: {e}")
                    continue

        log.debug(f"Found {len(sessions)} sessions in {project_count} projects")
        return sorted(sessions, key=lambda x: x["modified"], reverse=True)

    def find_session(self, session_id: str) -> Path | None:
        """Find session trace file by ID.

        Args:
            session_id: Session UUID

        Returns:
            Path to trace file if found
        """
        log.debug(f"find_session called: session_id={session_id}")
        log.trace(f"Searching in traces_dir={self.traces_dir}")
        log.trace(f"traces_dir.exists()={self.traces_dir.exists()}")

        if not self.traces_dir.exists():
            log.error(f"Traces directory does not exist: {self.traces_dir}")
            return None

        project_dirs_checked = 0
        for project_dir in self.traces_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_dirs_checked += 1
            trace_file = project_dir / f"{session_id}.jsonl"
            log.trace(f"Checking: {trace_file}")

            if trace_file.exists():
                log.info(f"Found session {session_id} at {trace_file}")
                return trace_file

        log.warning(f"Session {session_id} not found (checked {project_dirs_checked} project directories)")
        return None

    def parse_session(
        self,
        session_path: Path,
        start_index: int = 0,
    ) -> Iterator[tuple[int, TraceMessage]]:
        """Parse a JSONL trace file.

        Args:
            session_path: Path to trace file
            start_index: Line index to start from (0-based)

        Yields:
            Tuple of (line_index, TraceMessage) for each relevant message
        """
        log.debug(f"Parsing session: {session_path} (start_index={start_index})")
        message_count = 0
        line_index = 0

        with open(session_path, "r", encoding="utf-8") as f:
            for line in f:
                current_index = line_index
                line_index += 1

                # Skip lines before start_index
                if current_index < start_index:
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip sidechains (parallel agent runs)
                if entry.get("isSidechain", False):
                    continue

                msg_type = entry.get("type")
                if msg_type not in ("user", "assistant", "tool_use", "tool_result"):
                    continue

                content = self._extract_content(entry)
                if not content or len(content.strip()) < 10:
                    continue

                message_count += 1
                yield current_index, TraceMessage(
                    uuid=entry.get("uuid", ""),
                    session_id=entry.get("sessionId", ""),
                    type=msg_type,
                    content=content,
                    timestamp=entry.get("timestamp", ""),
                    parent_uuid=entry.get("parentUuid"),
                )
        log.debug(f"Parsed {message_count} messages from {session_path.name} (starting from index {start_index})")

    def get_file_inode(self, session_path: Path) -> int | None:
        """Get the inode of a trace file.

        Args:
            session_path: Path to trace file

        Returns:
            Inode number, or None if file not found
        """
        try:
            return session_path.stat().st_ino
        except (OSError, FileNotFoundError):
            return None

    def count_lines(self, session_path: Path) -> int:
        """Count total lines in a trace file.

        Args:
            session_path: Path to trace file

        Returns:
            Number of lines
        """
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except (OSError, FileNotFoundError):
            return 0

    def _extract_content(self, entry: dict) -> str:
        """Extract text content from various message formats.

        Args:
            entry: Raw JSONL entry

        Returns:
            Extracted text content
        """
        # Handle tool_use and tool_result specially
        if entry.get("type") == "tool_use":
            tool_name = entry.get("name", "unknown")
            tool_input = entry.get("input", {})
            # Increased from 200 to 2000 to preserve file paths for context merge
            if isinstance(tool_input, dict):
                input_preview = str(tool_input)[:2000]
            else:
                input_preview = str(tool_input)[:2000]
            return f"[Tool: {tool_name}] {input_preview}"

        if entry.get("type") == "tool_result":
            content = entry.get("content", "")
            if isinstance(content, str):
                return f"[Tool Result] {content[:500]}"
            return "[Tool Result]"

        # Handle regular messages
        msg = entry.get("message", {})
        content = msg.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        texts.append(f"[Tool: {block.get('name', 'unknown')}]")
                elif isinstance(block, str):
                    texts.append(block)
            return "\n".join(texts)

        return ""


class HierarchicalIndexer:
    """Creates hierarchical memory indexes from Claude Code sessions.

    Hierarchy:
        SessionSummary (1)
        └── ChunkSummary (5-15)
            └── Message (many)
    """

    def __init__(self, store: MemoryStore, config: Config | None = None):
        """Initialize indexer.

        Args:
            store: Memory store for saving memories
            config: Optional configuration
        """
        log.debug("Initializing HierarchicalIndexer")
        self.store = store
        self.config = config or Config()
        self.parser = TraceParser(self.config.claude_traces_dir)
        log.info(f"HierarchicalIndexer initialized with traces_dir={self.config.claude_traces_dir}")

    async def index_session(self, session_id: str, ctx=None) -> SessionIndex | None:
        """Index a Claude Code session with hierarchical summaries.

        Uses UPSERT semantics: if session was previously indexed, all existing
        memories for that session are deleted first to prevent duplicates.

        Args:
            session_id: Session UUID to index
            ctx: Optional MCP Context for progress reporting

        Returns:
            SessionIndex with created memory IDs, or None if session not found
        """
        log.info(f"Indexing session: {session_id}")
        log.debug(f"Using parser with traces_dir={self.parser.traces_dir}")

        # UPSERT: Clean up any existing index for this session
        cleanup_result = self.store.db.delete_session_memories(session_id)
        if cleanup_result["memories_deleted"] > 0:
            log.info(f"UPSERT: Deleted {cleanup_result['memories_deleted']} existing memories for session {session_id}")

        async def report(progress: int, msg: str = ""):
            """Helper to report progress if ctx available."""
            if ctx:
                await ctx.report_progress(progress, 100)
                if msg:
                    await ctx.info(msg)

        session_path = self.parser.find_session(session_id)
        if not session_path:
            log.error(f"Session {session_id} not found by parser")
            return None

        await report(5, "Found session, parsing messages...")
        log.debug(f"Found session at {session_path}")
        # Parse all messages (ignore line indices for full indexing)
        messages = [msg for _, msg in self.parser.parse_session(session_path)]
        if not messages:
            log.warning(f"No messages found in session {session_id}")
            return None

        log.info(f"Parsed {len(messages)} messages from session {session_id}")

        # 0. Extract goal from first user message
        goal_id = None
        first_user_msg = next((m for m in messages if m.type == "user"), None)
        if first_user_msg:
            await report(7, "Extracting session goal...")
            goal_text = await extract_goal(first_user_msg.content, self.config)
            if goal_text:
                import uuid
                goal_id = str(uuid.uuid4())
                try:
                    self.store.db.add_goal_node(
                        goal_id=goal_id,
                        intent=goal_text,
                        session_id=session_id,
                        status="active",
                    )
                    log.info(f"Created Goal node: {goal_text[:50]}...")
                except Exception as e:
                    log.warning(f"Failed to create Goal node: {e}")
                    goal_id = None

        # 1. Chunk messages by tool sequences
        chunks = self._chunk_by_tool_sequences(messages)
        await report(10, f"Created {len(chunks)} chunks, starting summarization...")

        # Track which chunk each message belongs to (for message→chunk relationships)
        msg_to_chunk_idx: dict[str, int] = {}
        for chunk_idx, chunk in enumerate(chunks):
            for msg in chunk:
                msg_to_chunk_idx[msg.uuid] = chunk_idx

        # 2. Summarize all chunks in parallel using asyncio.gather
        log.info(f"Summarizing {len(chunks)} chunks in parallel")
        import asyncio

        # Process chunks in parallel batches to avoid overwhelming the API
        batch_size = 5
        chunk_summaries = []
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            progress = 10 + int((batch_start / len(chunks)) * 60)
            await report(progress, f"Summarizing chunks {batch_start+1}-{batch_end}/{len(chunks)}...")

            # Summarize batch in parallel
            batch_summaries = await asyncio.gather(
                *[self._summarize_chunk(chunk) for chunk in batch]
            )
            chunk_summaries.extend(batch_summaries)

        await report(70, f"Generated {len(chunk_summaries)} summaries, batch storing...")

        # 3. Batch store all chunk summaries (single embedding call!)
        chunk_items = [
            MemoryItem(
                content=summary,
                metadata={
                    "type": "chunk_summary",
                    "session_id": session_id,
                    "source": "claude_trace",
                    "message_count": len(chunk),
                },
            )
            for chunk, summary in zip(chunks, chunk_summaries)
        ]

        await report(75, f"Batch embedding {len(chunk_items)} chunks...")
        chunk_ids = self.store.store_batch(chunk_items)
        log.info(f"Batch stored {len(chunk_ids)} chunk summaries")

        # 3b. Create chunk→chunk (follows) relationships for temporal sequence
        follows_created = 0
        for i in range(1, len(chunk_ids)):
            try:
                self.store.relate(chunk_ids[i], chunk_ids[i - 1], "follows")
                follows_created += 1
            except Exception as e:
                log.warning(f"Failed to create chunk follows relation: {e}")
        log.info(f"Created {follows_created} chunk→chunk follows relationships")

        # 4. Store interesting individual messages
        await report(80, "Identifying interesting messages...")
        interesting_msgs = [m for m in messages if self._is_interesting(m)]
        log.info(f"Found {len(interesting_msgs)} interesting messages")

        # Build uuid→message lookup for context merging
        msg_by_uuid: dict[str, TraceMessage] = {m.uuid: m for m in messages}

        message_ids = []
        extractions_list: list[EnhancedExtraction] = []
        if interesting_msgs:
            # Cap at 50 to control costs
            interesting_msgs = interesting_msgs[:50]

            # Build extraction contents with context merge (tool_use + tool_result)
            # This solves the "0 READS" issue by giving the LLM both filename AND content
            extraction_contents = []
            for msg in interesting_msgs:
                content = msg.content
                # If this is a tool_result, prepend the tool_use context
                if msg.type == "tool_result" and msg.parent_uuid:
                    parent = msg_by_uuid.get(msg.parent_uuid)
                    if parent and parent.type == "tool_use":
                        # Merge: LLM sees tool name + args + result together
                        content = f"Tool call: {parent.content}\n\nResult:\n{msg.content}"
                extraction_contents.append(content)

            # Extract entities WITH actions using enhanced LLM extraction
            await report(82, f"Extracting entities with actions from {len(interesting_msgs)} messages...")
            extractions_list = await extract_with_actions_batch(
                extraction_contents, self.config
            )

            msg_items = []
            for msg, extraction in zip(interesting_msgs, extractions_list):
                # Build searchable content with entity context appended
                content = msg.content[:1800]  # Leave room for entity footer

                # Append entity context for better semantic search
                entity_parts = []
                files = [e.name for e in extraction.entities if e.type == "file"]
                tools = [e.name for e in extraction.entities if e.type == "tool"]
                commands = [e.name for e in extraction.entities if e.type == "command"]
                errors = [e.name for e in extraction.entities if e.type == "error"]

                if files:
                    entity_parts.append(f"Files: {', '.join(files[:5])}")
                if commands:
                    entity_parts.append(f"Commands: {', '.join(commands[:3])}")
                if errors:
                    entity_parts.append(f"Errors: {', '.join(errors[:2])}")
                if tools:
                    entity_parts.append(f"Tools: {', '.join(tools[:5])}")

                if entity_parts:
                    content += f"\n\n[Context: {' | '.join(entity_parts)}]"

                msg_items.append(MemoryItem(
                    content=content,
                    metadata={
                        "type": "message",
                        "session_id": session_id,
                        "source": "claude_trace",
                        "msg_type": msg.type,
                        **extraction.to_metadata(),
                    },
                ))

            await report(85, f"Storing {len(msg_items)} messages...")
            message_ids = self.store.store_batch(msg_items)
            log.info(f"Stored {len(message_ids)} interesting messages")

            # 4b. Create verb-specific edges for cross-session linking
            verb_edges_created = 0
            for msg_id, extraction in zip(message_ids, extractions_list):
                for entity in extraction.entities[:15]:  # Cap at 15 per message
                    try:
                        self.store.db.add_verb_edge(
                            memory_uuid=msg_id,
                            entity_name=entity.name,
                            entity_type=entity.type,
                            action=entity.action,
                        )
                        verb_edges_created += 1
                    except Exception as e:
                        log.warning(f"Failed to create verb edge: {e}")
            log.info(f"Created {verb_edges_created} verb-specific edges (READS/MODIFIES/EXECUTES/TRIGGERED)")

            # 4c. Create message relationships
            msg_relations_created = 0
            for i, (msg, msg_id) in enumerate(zip(interesting_msgs, message_ids)):
                # message→chunk (child_of) - link message to its parent chunk
                chunk_idx = msg_to_chunk_idx.get(msg.uuid)
                if chunk_idx is not None and chunk_idx < len(chunk_ids):
                    try:
                        self.store.relate(msg_id, chunk_ids[chunk_idx], "child_of")
                        msg_relations_created += 1
                    except Exception as e:
                        log.warning(f"Failed to create message→chunk relation: {e}")

                # message→message (follows) - temporal sequence
                if i > 0:
                    try:
                        self.store.relate(msg_id, message_ids[i - 1], "follows")
                        msg_relations_created += 1
                    except Exception as e:
                        log.warning(f"Failed to create message follows relation: {e}")
            log.info(f"Created {msg_relations_created} message relationships")

        # 5. Generate session summary using chunk summaries (Map-Reduce)
        await report(90, "Generating session summary...")
        session_summary = await self._summarize_session(messages, chunk_summaries)

        await report(95, "Storing session summary...")
        session_summary_id = self.store.store(
            MemoryItem(
                content=session_summary,
                metadata={
                    "type": "session_summary",
                    "session_id": session_id,
                    "source": "claude_trace",
                    "goal_id": goal_id,
                },
                relations=[{"target_id": cid, "type": "contains"} for cid in chunk_ids],
            )
        )

        # 5b. Create chunk→session (child_of) relationships for bi-directional traversal
        child_of_created = 0
        for cid in chunk_ids:
            try:
                self.store.relate(cid, session_summary_id, "child_of")
                child_of_created += 1
            except Exception as e:
                log.warning(f"Failed to create chunk→session child_of relation: {e}")
        log.info(f"Created {child_of_created} chunk→session child_of relationships")

        # 5c. Link session to goal if extracted
        if goal_id:
            try:
                self.store.db.link_session_to_goal(session_summary_id, goal_id)
                log.info(f"Linked session to Goal: {goal_id[:8]}...")

                # Also link interesting messages to goal (ACHIEVES)
                for msg_id in message_ids[:10]:  # Cap to prevent too many edges
                    try:
                        self.store.db.link_memory_to_goal(msg_id, goal_id)
                    except Exception as e:
                        log.trace(f"Failed to link message to goal: {e}")
            except Exception as e:
                log.warning(f"Failed to link session to goal: {e}")

        # 5d. Create Project node and link session (BELONGS_TO)
        # Extract project ID from session path: ~/.claude/projects/{project_dir}/{session}.jsonl
        project_dir = None
        try:
            project_dir = session_path.parent.name  # e.g., "-Users-shimon-repo-simplemem"
            if project_dir and project_dir != "projects":
                # Decode path for human-readable display (preserves hyphens in project names)
                project_path_display = _decode_project_path(project_dir)
                self.store.db.add_project_node(
                    project_id=project_dir,  # Use raw dir name as unique ID
                    project_path=project_path_display,  # Human-readable path
                )
                self.store.db.link_session_to_project(session_summary_id, project_dir)
                log.info(f"Linked session to Project: {project_path_display}")
        except Exception as e:
            log.warning(f"Failed to create/link project: {e}")
            project_dir = None

        await report(100, "Complete!")
        return SessionIndex(
            session_id=session_id,
            session_summary_id=session_summary_id,
            chunk_summary_ids=chunk_ids,
            message_ids=message_ids,
            goal_id=goal_id,
            project_id=project_dir,
        )

    async def index_session_delta(
        self,
        session_id: str,
        project_root: str,
        project_manager: "ProjectManager",
        transcript_path: str | None = None,
    ) -> dict:
        """Process only new messages in a session (delta indexing).

        Uses project state to track last processed index and detect file rotation.
        Much more efficient than full reindexing for active sessions.

        Args:
            session_id: Session UUID to index
            project_root: Project root path
            project_manager: ProjectManager for cursor tracking
            transcript_path: Optional explicit path to transcript

        Returns:
            Dict with processing results
        """
        log.info(f"Delta indexing session: {session_id} for project: {project_root}")

        # Find session file
        if transcript_path:
            session_path = Path(transcript_path)
            if not session_path.exists():
                return {"error": f"Transcript not found: {transcript_path}", "processed": 0}
        else:
            session_path = self.parser.find_session(session_id)
            if not session_path:
                return {"error": f"Session {session_id} not found", "processed": 0}

        # Get project state for cursor info
        state = project_manager.get_project_state(project_root)
        last_index = 0
        last_inode = None

        if state and state.last_session_id == session_id:
            last_index = state.last_processed_index
            last_inode = state.last_trace_inode

        # Check for file rotation (inode changed)
        current_inode = self.parser.get_file_inode(session_path)
        if last_inode is not None and current_inode != last_inode:
            log.info(f"File rotation detected (inode {last_inode} -> {current_inode}), resetting cursor")
            last_index = 0

        # Parse only new messages
        new_messages: list[tuple[int, TraceMessage]] = []
        max_index = last_index

        for idx, msg in self.parser.parse_session(session_path, start_index=last_index):
            new_messages.append((idx, msg))
            max_index = max(max_index, idx + 1)

        if not new_messages:
            log.info(f"No new messages since index {last_index}")
            return {
                "processed": 0,
                "last_index": last_index,
                "message": "No new messages to process",
            }

        log.info(f"Found {len(new_messages)} new messages (index {last_index} -> {max_index})")

        # Extract just the messages (drop indices for processing)
        messages = [msg for _, msg in new_messages]

        # Store interesting messages
        interesting = [m for m in messages if self._is_interesting(m)][:20]  # Cap at 20

        stored_count = 0
        if interesting:
            msg_items = [
                MemoryItem(
                    content=msg.content[:2000],
                    metadata={
                        "type": "message",
                        "session_id": session_id,
                        "source": "claude_trace_delta",
                        "msg_type": msg.type,
                    },
                )
                for msg in interesting
            ]
            message_ids = self.store.store_batch(msg_items)
            stored_count = len(message_ids)
            log.info(f"Stored {stored_count} interesting messages from delta")

        # Update cursor
        project_manager.update_trace_cursor(
            project_root=project_root,
            session_id=session_id,
            processed_index=max_index,
            trace_inode=current_inode,
        )

        return {
            "processed": len(new_messages),
            "stored": stored_count,
            "last_index": max_index,
            "session_id": session_id,
        }

    def _chunk_by_tool_sequences(
        self, messages: list[TraceMessage]
    ) -> list[list[TraceMessage]]:
        """Split session into logical chunks.

        Chunk boundaries:
        - After tool_result (completed action)
        - After long assistant response
        - Max 20 messages per chunk

        Args:
            messages: All messages in session

        Returns:
            List of message chunks
        """
        chunks = []
        current_chunk: list[TraceMessage] = []

        for msg in messages:
            current_chunk.append(msg)

            if self._should_split(msg, current_chunk):
                chunks.append(current_chunk)
                current_chunk = []

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _should_split(self, msg: TraceMessage, chunk: list[TraceMessage]) -> bool:
        """Determine if current chunk should be split.

        Args:
            msg: Current message
            chunk: Current chunk

        Returns:
            True if should split after this message
        """
        # Max chunk size
        if len(chunk) >= 20:
            return True

        # After completed tool action
        if msg.type == "tool_result":
            return True

        # After long assistant explanation
        if msg.type == "assistant" and len(msg.content) > 1000:
            return True

        return False

    def _is_interesting(self, msg: TraceMessage) -> bool:
        """Determine if message should be stored individually.

        Interesting messages are those likely to be useful for future debugging:
        - Tool results with substantial content
        - Messages containing errors or exceptions
        - Messages with code blocks

        Args:
            msg: Message to evaluate

        Returns:
            True if message should be stored
        """
        import re

        # Tool results with substantial content
        if msg.type == "tool_result" and len(msg.content) > 100:
            return True

        # Messages with errors - use word boundaries to avoid false positives
        # (e.g., "error-free", "no errors", etc.)
        content_lower = msg.content.lower()
        error_pattern = r'\b(error|exception|failed|traceback|stack\s*trace)\b'
        if re.search(error_pattern, content_lower):
            # Additional check: skip if negated
            negation_pattern = r'\b(no|without|zero|0)\s+(errors?|exceptions?|failures?)\b'
            if not re.search(negation_pattern, content_lower):
                return True

        # Messages with code blocks
        if "```" in msg.content:
            return True

        return False

    def _prepare_chunk_content(
        self, chunk: list[TraceMessage], max_tokens: int = 3000
    ) -> str:
        """Prepare chunk content for summarization with token budget.

        Maintains chronological order while prioritizing high-value content.
        Skips low-priority messages when budget is tight, but never reorders.

        Args:
            chunk: Messages in the chunk (chronological order)
            max_tokens: Maximum token budget (~4 chars per token)

        Returns:
            Formatted content string within token budget
        """
        # Priority scores: lower = more important (keep)
        priority_score = {"tool_result": 0, "assistant": 1, "user": 2, "tool_use": 3}

        content_parts = []
        estimated_tokens = 0

        # First pass: calculate total tokens needed
        total_needed = sum(len(m.content) // 4 for m in chunk)

        # If we're over budget, we'll skip low-priority messages
        skip_threshold = 3 if total_needed > max_tokens * 1.5 else 99

        for msg in chunk:  # Maintain chronological order!
            msg_priority = priority_score.get(msg.type, 2)
            msg_tokens = len(msg.content) // 4

            # Skip low-priority if over budget
            if msg_priority >= skip_threshold and estimated_tokens + msg_tokens > max_tokens * 0.8:
                continue

            if estimated_tokens + msg_tokens > max_tokens:
                # Add truncated version if we have budget
                remaining = (max_tokens - estimated_tokens) * 4
                if remaining > 100:
                    content_parts.append(f"[{msg.type}] {msg.content[:remaining]}...")
                break

            content_parts.append(f"[{msg.type}] {msg.content}")
            estimated_tokens += msg_tokens

        return "\n\n".join(content_parts)

    async def _summarize_chunk(self, chunk: list[TraceMessage]) -> str:
        """Generate summary for a message chunk.

        Uses token-based content preparation to maximize context within limits.

        Args:
            chunk: Messages in the chunk

        Returns:
            Summary text
        """
        from litellm import acompletion

        # Use token-based content preparation
        content = self._prepare_chunk_content(chunk, max_tokens=3000)

        try:
            response = await acompletion(
                model=self.config.summary_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Summarize this Claude Code interaction in 2-3 sentences.
Focus on: what problem was being solved, what tools were used, what was the outcome.

{content}""",
                    }
                ],
                max_tokens=150,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback: simple extraction
            log.warning(f"Chunk summarization failed: {e}")
            return f"Chunk with {len(chunk)} messages. First: {chunk[0].content[:100]}..."

    async def _summarize_session(
        self,
        messages: list[TraceMessage],
        chunk_summaries: list[str],
    ) -> str:
        """Generate session-level summary using Map-Reduce pattern.

        Synthesizes chunk summaries instead of just looking at first/last message.

        Args:
            messages: All messages in session (for stats)
            chunk_summaries: List of chunk summary texts

        Returns:
            Session summary text
        """
        from litellm import acompletion

        # Count message types for stats
        type_counts = {}
        for msg in messages:
            type_counts[msg.type] = type_counts.get(msg.type, 0) + 1

        # Select representative chunk summaries: first + middle + last
        # This ensures we capture: session start, middle context, and resolution
        n = len(chunk_summaries)
        if n <= 15:
            selected_summaries = chunk_summaries
        else:
            # First 5 (context), middle 5 (development), last 5 (resolution)
            first = chunk_summaries[:5]
            mid_start = (n - 5) // 2
            middle = chunk_summaries[mid_start:mid_start + 5]
            last = chunk_summaries[-5:]
            selected_summaries = first + middle + last

        summaries_text = "\n".join(
            f"- {s}" for s in selected_summaries
        )

        try:
            response = await acompletion(
                model=self.config.summary_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Synthesize this Claude Code session from the activity summaries below.

Session stats:
- {len(messages)} total messages
- {len(chunk_summaries)} distinct activities
- Message types: {type_counts}

Activity summaries:
{summaries_text}

Provide a 3-5 sentence overview covering: main goal, key accomplishments, final state, any unresolved issues.""",
                    }
                ],
                max_tokens=250,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback: basic summary
            log.warning(f"Session summarization failed: {e}")
            return f"Session with {len(messages)} messages across {len(chunk_summaries)} activities."
