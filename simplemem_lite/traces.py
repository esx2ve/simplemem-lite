"""Claude Code trace processing for SimpleMem Lite.

Parses JSONL session traces and creates hierarchical memory indexes.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from simplemem_lite.config import Config
from simplemem_lite.logging import get_logger
from simplemem_lite.memory import MemoryItem, MemoryStore

log = get_logger("traces")


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
    """

    session_id: str
    session_summary_id: str
    chunk_summary_ids: list[str]
    message_ids: list[str]


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

    def parse_session(self, session_path: Path) -> Iterator[TraceMessage]:
        """Parse a JSONL trace file.

        Args:
            session_path: Path to trace file

        Yields:
            TraceMessage objects for each relevant message
        """
        log.debug(f"Parsing session: {session_path}")
        message_count = 0
        with open(session_path, "r", encoding="utf-8") as f:
            for line in f:
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
                yield TraceMessage(
                    uuid=entry.get("uuid", ""),
                    session_id=entry.get("sessionId", ""),
                    type=msg_type,
                    content=content,
                    timestamp=entry.get("timestamp", ""),
                    parent_uuid=entry.get("parentUuid"),
                )
        log.debug(f"Parsed {message_count} messages from {session_path.name}")

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
            if isinstance(tool_input, dict):
                input_preview = str(tool_input)[:200]
            else:
                input_preview = str(tool_input)[:200]
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

        Args:
            session_id: Session UUID to index
            ctx: Optional MCP Context for progress reporting

        Returns:
            SessionIndex with created memory IDs, or None if session not found
        """
        log.info(f"Indexing session: {session_id}")
        log.debug(f"Using parser with traces_dir={self.parser.traces_dir}")

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
        messages = list(self.parser.parse_session(session_path))
        if not messages:
            log.warning(f"No messages found in session {session_id}")
            return None

        log.info(f"Parsed {len(messages)} messages from session {session_id}")

        # 1. Chunk messages by tool sequences
        chunks = self._chunk_by_tool_sequences(messages)
        await report(10, f"Created {len(chunks)} chunks, starting summarization...")

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
        message_ids = []

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

        # 4. Generate session summary
        await report(90, "Generating session summary...")
        session_summary = await self._summarize_session(messages, len(chunks))

        await report(95, "Storing session summary...")
        session_summary_id = self.store.store(
            MemoryItem(
                content=session_summary,
                metadata={
                    "type": "session_summary",
                    "session_id": session_id,
                    "source": "claude_trace",
                },
                relations=[{"target_id": cid, "type": "contains"} for cid in chunk_ids],
            )
        )

        await report(100, "Complete!")
        return SessionIndex(
            session_id=session_id,
            session_summary_id=session_summary_id,
            chunk_summary_ids=chunk_ids,
            message_ids=message_ids,
        )

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

    async def _summarize_chunk(self, chunk: list[TraceMessage]) -> str:
        """Generate summary for a message chunk.

        Args:
            chunk: Messages in the chunk

        Returns:
            Summary text
        """
        from litellm import acompletion

        # Prepare content for summarization
        content_parts = []
        for msg in chunk[:10]:  # Limit messages
            content_parts.append(f"[{msg.type}] {msg.content[:500]}")

        content = "\n\n".join(content_parts)

        try:
            response = await acompletion(
                model=self.config.summary_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Summarize this Claude Code interaction in 2-3 sentences.
Focus on: what problem was being solved, what tools were used, what was the outcome.

{content[:4000]}""",
                    }
                ],
                max_tokens=150,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback: simple extraction
            return f"Chunk with {len(chunk)} messages. First: {chunk[0].content[:100]}..."

    async def _summarize_session(
        self, messages: list[TraceMessage], num_chunks: int
    ) -> str:
        """Generate session-level summary.

        Args:
            messages: All messages in session
            num_chunks: Number of chunks created

        Returns:
            Session summary text
        """
        from litellm import acompletion

        # Count message types
        type_counts = {}
        for msg in messages:
            type_counts[msg.type] = type_counts.get(msg.type, 0) + 1

        try:
            response = await acompletion(
                model=self.config.summary_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Summarize this Claude Code session in 3-5 sentences.

Session stats:
- {len(messages)} total messages
- {num_chunks} distinct activities
- Message types: {type_counts}

First message: {messages[0].content[:200]}
Last message: {messages[-1].content[:200]}

Focus on: main goal, key decisions, final outcome, errors resolved.""",
                    }
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback: basic summary
            return f"Session with {len(messages)} messages across {num_chunks} activities."
