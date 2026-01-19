"""Memory store for SimpleMem Lite.

Unified memory storage with hybrid search (vector + graph).
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from simplemem_lite.config import Config
from simplemem_lite.db import DatabaseManager
from simplemem_lite.embeddings import embed, embed_batch, init_embeddings
from simplemem_lite.log_config import get_logger

log = get_logger("memory")


async def expand_query(query: str, config: Config) -> list[str]:
    """Expand a search query into semantic variations using LLM.

    Uses the configured summary model to generate 3-4 semantic variations
    of the original query to improve recall during search.

    Args:
        query: Original search query
        config: Config with LLM model settings

    Returns:
        List of query variations (always includes original query first)
    """
    from json_repair import loads as json_repair_loads
    from litellm import acompletion

    log.debug(f"Expanding query: '{query[:50]}...'")

    try:
        response = await acompletion(
            model=config.summary_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Expand the search query inside <query></query> tags into 3-4 semantic variations that would help find relevant memories.
Include synonyms, related concepts, and different phrasings.

<query>{query}</query>

Return ONLY a JSON array of strings. Include the original query as the first element.
Example output: ["original query", "variation 1", "variation 2", "variation 3"]""",
                }
            ],
            max_tokens=200,
            temperature=0.3,
        )

        # Safety check for LLM response
        if not response.choices or not response.choices[0].message.content:
            log.warning("Empty LLM response for query expansion")
            return [query]

        content = response.choices[0].message.content
        variations = json_repair_loads(content)

        if not isinstance(variations, list) or len(variations) == 0:
            log.warning(f"Invalid expansion result, using original query only")
            return [query]

        # Build unique list with original query first
        seen = {query}
        result = [query]
        for v in variations:
            v_str = str(v).strip() if v else ""
            if v_str and v_str not in seen:
                result.append(v_str)
                seen.add(v_str)

        log.info(f"Query expanded: '{query[:30]}...' -> {len(result)} variations")
        log.debug(f"Expanded queries: {result}")
        return result[:5]  # Cap at 5 variations to limit API calls

    except Exception as e:
        log.warning(f"Query expansion failed: {e}, using original query only")
        return [query]


def _escape_xml_chars(text: str) -> str:
    """Escape XML special characters to prevent prompt injection.

    Args:
        text: Raw text that may contain XML characters

    Returns:
        Text with <, >, & escaped
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def detect_contradictions(
    new_content: str,
    similar_memories: list[dict],
    config: Config,
) -> list[dict]:
    """Detect memories that new content might contradict or supersede.

    Uses LLM to analyze whether the new memory contradicts, updates, or
    supersedes any existing similar memories. This enables automatic
    deprecation of outdated information.

    Args:
        new_content: Content of the new memory being stored
        similar_memories: List of similar memory dicts (from vector search)
                         Each should have 'uuid', 'content', 'type' keys
        config: Config with LLM model settings

    Returns:
        List of contradictions found, each with:
        - uuid: UUID of the contradicted memory
        - reason: Brief explanation of the contradiction
        - confidence: high/medium/low

    Note:
        - Returns empty list on LLM errors (graceful degradation)
        - Only processes memories with score > 0.3 to avoid false positives
        - Filters out memories of different types to reduce noise
    """
    from json_repair import loads as json_repair_loads
    from litellm import acompletion

    if not similar_memories:
        return []

    # Filter to memories with reasonable similarity
    relevant = [m for m in similar_memories if m.get("score", 0) > 0.3]
    if not relevant:
        log.debug("No sufficiently similar memories for contradiction detection")
        return []

    log.debug(f"Checking {len(relevant)} memories for contradictions")

    # Format memories for LLM - escape content to prevent injection
    formatted_memories = []
    for i, m in enumerate(relevant):
        content_preview = _escape_xml_chars(
            m.get("content", "")[:CONTRADICTION_CONTENT_PREVIEW_LEN]
        )
        mem_type = m.get("type", "unknown")
        formatted_memories.append(f"[{i}] ({mem_type}): {content_preview}")

    memories_text = "\n".join(formatted_memories)
    new_content_escaped = _escape_xml_chars(new_content[:500])

    try:
        response = await acompletion(
            model=config.summary_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze whether the new memory contradicts, updates, or supersedes any existing memories.

<new_memory>
{new_content_escaped}
</new_memory>

<existing_memories>
{memories_text}
</existing_memories>

A memory is SUPERSEDED when:
- The new memory provides updated/corrected information on the same topic
- The new memory contains a decision that reverses a previous decision
- The new memory fixes a bug that was documented in the old memory
- The new memory provides more accurate facts than the old memory

Return ONLY valid JSON with this structure:
{{
  "contradictions": [
    {{"index": 0, "reason": "brief explanation", "confidence": "high"}},
    ...
  ]
}}

Rules:
- "index" must be a valid integer from 0 to {len(relevant) - 1}
- "reason" should be a brief (1-2 sentence) explanation
- "confidence" must be "high", "medium", or "low"
- Only include memories that are clearly contradicted/superseded
- If no contradictions found, return {{"contradictions": []}}
- Be conservative - only flag clear contradictions, not minor updates""",
                }
            ],
            max_tokens=400,
            temperature=0.0,  # Deterministic for consistent results
        )

        # Safety check for LLM response
        if not response.choices or not response.choices[0].message.content:
            log.warning("Empty LLM response for contradiction detection")
            return []

        content = response.choices[0].message.content
        log.info(f"LLM contradiction response: {content[:500]}")  # Debug
        parsed = json_repair_loads(content)

        if not isinstance(parsed, dict):
            log.warning("Invalid JSON structure from contradiction detection")
            return []

        contradictions_raw = parsed.get("contradictions", [])
        if not isinstance(contradictions_raw, list):
            return []

        # Validate and map to UUIDs
        result = []
        for c in contradictions_raw:
            if not isinstance(c, dict):
                continue

            idx = c.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(relevant):
                continue

            reason = str(c.get("reason", "Contradicting information"))[:200]
            confidence = c.get("confidence", "medium")
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"

            result.append({
                "uuid": relevant[idx]["uuid"],
                "reason": reason,
                "confidence": confidence,
            })

        log.info(f"Contradiction detection found {len(result)} superseded memories")
        return result

    except Exception as e:
        log.warning(f"Contradiction detection failed: {e}")
        return []


# Semantic weights for relation types (applied in Python for flexibility)
RELATION_WEIGHTS = {
    # Graph edge types (uppercase as returned by FalkorDB)
    "CONTAINS": 1.0,      # Hierarchical drill-down
    "CHILD_OF": 1.0,      # Reverse contains
    "RELATES_TO": 0.8,    # Semantic relationship (most common)
    "REFERENCES": 0.7,    # Entity reference (cross-session)
    "READS": 0.7,         # Memory reads a file
    "MODIFIES": 0.8,      # Memory modifies something
    "EXECUTES": 0.6,      # Memory executes a tool
    "TRIGGERED": 0.6,     # Memory triggered an error
    "FOLLOWS": 0.6,       # Temporal sequence
    "ACHIEVES": 0.8,      # Memory achieves a goal
    "HAS_GOAL": 0.6,      # Session has a goal
    "BELONGS_TO": 0.5,    # Belongs to project
    "SUPERSEDES": 0.2,    # New memory supersedes old
    "MERGED_INTO": 0.1,   # Old memory merged into new
    # Legacy lowercase keys (for backward compat with relation_type field)
    "contains": 1.0,
    "child_of": 1.0,
    "supports": 0.8,
    "references": 0.7,
    "follows": 0.6,
    "mentions": 0.4,
    "relates": 0.3,
    "supersedes": 0.1,
}

# Content preview length for contradiction detection
CONTRADICTION_CONTENT_PREVIEW_LEN: int = 300


@dataclass
class MemoryItem:
    """Input for storing a new memory.

    Attributes:
        content: The text content to store
        metadata: Optional metadata (type, source, session_id)
        relations: Optional list of relationships to create
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    relations: list[dict[str, str]] = field(default_factory=list)
    # relations format: [{"target_id": "uuid", "type": "contains"}]


@dataclass
class Memory:
    """Retrieved memory with score.

    Attributes:
        uuid: Unique identifier
        content: Text content
        type: Memory type (fact, session_summary, chunk_summary, message, todo)
        created_at: Unix timestamp
        score: Relevance score (higher = more relevant)
        session_id: Optional session identifier
        relations: Related memories
        metadata: Additional metadata (project_id, todo fields, etc.)
    """

    uuid: str
    content: str
    type: str
    created_at: int
    score: float = 0.0
    session_id: str | None = None
    relations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """Unified memory storage with hybrid search.

    Combines LanceDB (vector similarity) with KuzuDB (graph traversal)
    for enhanced retrieval.

    Example:
        >>> config = Config()
        >>> store = MemoryStore(config)
        >>> memory_id = store.store(MemoryItem(
        ...     content="Python's GIL prevents true multithreading",
        ...     metadata={"type": "fact", "source": "docs"}
        ... ))
        >>> results = store.search("multithreading in Python")
    """

    def __init__(self, config: Config | None = None):
        """Initialize memory store.

        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        log.trace("MemoryStore.__init__ starting")
        self.config = config or Config()
        log.debug("Initializing embeddings")
        init_embeddings(self.config)
        log.debug("Initializing database manager")
        self.db = DatabaseManager(self.config)

        # Retry queue for failed vector writes
        # Format: [{"uuid": str, "embedding": list, "content": str, "type": str, "session_id": str, "metadata": dict}]
        self._vector_retry_queue: list[dict] = []
        self._max_retry_queue_size = 100  # Cap to prevent unbounded growth
        self._retry_queue_lock = threading.Lock()  # Thread safety for queue operations
        self._retry_backoff_seconds = 30  # Delay between retry attempts
        self._last_retry_attempt = 0.0  # Timestamp of last retry

        log.info("MemoryStore initialized successfully")

    def store(self, item: MemoryItem) -> str:
        """Store a memory with embedding and optional relationships.

        Uses two-phase commit: graph first (source of truth), then vectors.
        If vector write fails, the memory is queued for retry rather than
        rolled back (preventing data loss).

        Args:
            item: Memory item to store

        Returns:
            UUID of stored memory

        Raises:
            Exception: If graph storage fails (with automatic rollback)
        """
        memory_uuid = str(uuid4())
        mem_type = item.metadata.get("type", "fact")
        source = item.metadata.get("source", "user")
        session_id = item.metadata.get("session_id")
        project_id = item.metadata.get("project_id")
        created_at = int(time.time())

        log.debug(f"Storing memory: uuid={memory_uuid[:8]}..., type={mem_type}, source={source}")
        log.trace(f"Content preview: {item.content[:100]}...")

        log.trace("Generating embedding")
        embedding = embed(item.content, self.config)
        log.trace(f"Embedding generated: dim={len(embedding)}")

        with self.db.write_lock:
            # Phase 1: Write to graph (primary - source of truth)
            log.trace("Phase 1: Writing to graph")
            try:
                self.db.add_memory_node(
                    uuid=memory_uuid,
                    content=item.content,
                    mem_type=mem_type,
                    source=source,
                    session_id=session_id,
                    created_at=created_at,
                    project_id=project_id,
                )
            except Exception as e:
                log.error(f"Graph storage failed: {e}")
                raise e  # Graph failure is fatal

            # Phase 2: Write to vectors (non-fatal - queue for retry if fails)
            log.trace("Phase 2: Writing to vectors")
            vector_write_success = False
            try:
                self.db.add_memory_vector(
                    uuid=memory_uuid,
                    vector=embedding,
                    content=item.content,
                    mem_type=mem_type,
                    session_id=session_id,
                    metadata=item.metadata,
                )
                vector_write_success = True
            except Exception as e:
                log.warning(f"Vector write failed, queuing for retry: {e}")
                self._queue_vector_retry(
                    uuid=memory_uuid,
                    embedding=embedding,
                    content=item.content,
                    mem_type=mem_type,
                    session_id=session_id,
                    metadata=item.metadata,
                )

            # Phase 3: Create relationships
            if item.relations:
                log.trace(f"Phase 3: Creating {len(item.relations)} relationships")
                for rel in item.relations:
                    try:
                        self.db.add_relationship(
                            from_uuid=memory_uuid,
                            to_uuid=rel["target_id"],
                            relation_type=rel.get("type", "relates"),
                        )
                    except Exception as e:
                        log.warning(f"Failed to create relationship: {e}")

        status = "stored" if vector_write_success else "stored (vector pending)"
        log.info(f"Memory {status}: {memory_uuid[:8]}... ({mem_type})")

        # Opportunistically process retry queue (respects backoff)
        self.maybe_process_retry_queue()

        return memory_uuid

    def _queue_vector_retry(
        self,
        uuid: str,
        embedding: list[float],
        content: str,
        mem_type: str,
        session_id: str | None,
        metadata: dict,
    ) -> None:
        """Queue a failed vector write for later retry.

        Thread-safe: uses _retry_queue_lock.

        Args:
            uuid: Memory UUID
            embedding: Pre-computed embedding vector
            content: Memory content
            mem_type: Memory type
            session_id: Session ID
            metadata: Additional metadata
        """
        with self._retry_queue_lock:
            # Avoid duplicates
            if any(item["uuid"] == uuid for item in self._vector_retry_queue):
                log.debug(f"UUID {uuid[:8]}... already in retry queue")
                return

            # Enforce max queue size
            if len(self._vector_retry_queue) >= self._max_retry_queue_size:
                log.warning(f"Retry queue full ({self._max_retry_queue_size}), dropping oldest")
                self._vector_retry_queue.pop(0)

            self._vector_retry_queue.append({
                "uuid": uuid,
                "embedding": embedding,
                "content": content,
                "type": mem_type,
                "session_id": session_id or "",
                "metadata": metadata,
            })
            log.debug(f"Queued for retry: {uuid[:8]}... (queue size: {len(self._vector_retry_queue)})")

    def process_retry_queue(self, force: bool = False) -> dict:
        """Process pending vector write retries.

        Thread-safe: uses _retry_queue_lock.
        Respects backoff unless force=True.

        Args:
            force: Skip backoff check (default: False)

        Returns:
            Dictionary with:
            - processed: Number of items processed
            - succeeded: Number of successful writes
            - failed: Number of failed writes
            - remaining: Number of items still in queue
            - skipped: True if skipped due to backoff
        """
        import json

        # Check backoff (without lock to avoid blocking)
        now = time.time()
        if not force and (now - self._last_retry_attempt) < self._retry_backoff_seconds:
            return {"processed": 0, "succeeded": 0, "failed": 0, "remaining": self.get_retry_queue_size(), "skipped": True}

        with self._retry_queue_lock:
            if not self._vector_retry_queue:
                return {"processed": 0, "succeeded": 0, "failed": 0, "remaining": 0, "skipped": False}

            self._last_retry_attempt = now
            log.info(f"Processing vector retry queue: {len(self._vector_retry_queue)} items")

            succeeded = 0
            failed = 0
            still_failed = []

            for item in self._vector_retry_queue:
                try:
                    with self.db.write_lock:
                        self.db.lance_table.add([{
                            "uuid": item["uuid"],
                            "vector": item["embedding"],
                            "content": item["content"],
                            "type": item["type"],
                            "session_id": item["session_id"],
                            "metadata": json.dumps(item["metadata"]),
                        }])
                    succeeded += 1
                    log.debug(f"Retry succeeded: {item['uuid'][:8]}...")
                except Exception as e:
                    log.warning(f"Retry failed for {item['uuid'][:8]}...: {e}")
                    still_failed.append(item)
                    failed += 1

            self._vector_retry_queue = still_failed
            processed = succeeded + failed

            log.info(f"Retry queue processed: {succeeded} succeeded, {failed} failed, {len(still_failed)} remaining")
            return {
                "processed": processed,
                "succeeded": succeeded,
                "failed": failed,
                "remaining": len(still_failed),
                "skipped": False,
            }

    def maybe_process_retry_queue(self) -> dict:
        """Attempt to process retry queue if backoff has elapsed.

        This is the method that should be called periodically (e.g., after each store).
        It will only actually process if enough time has passed since the last attempt.

        Returns:
            Result dict from process_retry_queue
        """
        if self.get_retry_queue_size() == 0:
            return {"processed": 0, "succeeded": 0, "failed": 0, "remaining": 0, "skipped": False}
        return self.process_retry_queue(force=False)

    def get_retry_queue_size(self) -> int:
        """Get the number of items in the vector retry queue.

        Thread-safe: uses _retry_queue_lock.

        Returns:
            Number of pending vector writes
        """
        with self._retry_queue_lock:
            return len(self._vector_retry_queue)

    def store_batch(self, items: list[MemoryItem]) -> list[str]:
        """Store multiple memories with batch embedding for efficiency.

        Uses a single embedding call for all items, significantly faster
        than calling store() individually.

        Args:
            items: List of MemoryItem to store

        Returns:
            List of UUIDs for stored memories

        Raises:
            Exception: If storage fails (with automatic rollback)
        """
        if not items:
            return []

        log.info(f"Batch storing {len(items)} memories")

        # 1. Prepare all metadata upfront
        batch_data = []
        contents = []
        for item in items:
            memory_uuid = str(uuid4())
            mem_type = item.metadata.get("type", "fact")
            source = item.metadata.get("source", "user")
            session_id = item.metadata.get("session_id")
            project_id = item.metadata.get("project_id")
            created_at = int(time.time())

            batch_data.append({
                "uuid": memory_uuid,
                "content": item.content,
                "type": mem_type,
                "source": source,
                "session_id": session_id,
                "project_id": project_id,
                "created_at": created_at,
                "relations": item.relations,
                "metadata": item.metadata,
            })
            contents.append(item.content)

        # 2. Batch embed all content at once
        log.debug(f"Batch embedding {len(contents)} items")
        embeddings = embed_batch(contents, self.config)
        log.debug(f"Batch embedding complete: {len(embeddings)} embeddings")

        # 3. Store all to graph + vectors
        uuids = []
        with self.db.write_lock:
            try:
                for i, (data, embedding) in enumerate(zip(batch_data, embeddings)):
                    # Phase 1: Write to graph
                    self.db.add_memory_node(
                        uuid=data["uuid"],
                        content=data["content"],
                        mem_type=data["type"],
                        source=data["source"],
                        session_id=data["session_id"],
                        created_at=data["created_at"],
                        project_id=data["project_id"],
                    )

                    # Phase 2: Write to vectors
                    self.db.add_memory_vector(
                        uuid=data["uuid"],
                        vector=embedding,
                        content=data["content"],
                        mem_type=data["type"],
                        session_id=data["session_id"],
                        metadata=data["metadata"],
                    )

                    uuids.append(data["uuid"])

                # Phase 3: Create relationships (after all nodes exist)
                for data in batch_data:
                    for rel in data["relations"]:
                        self.db.add_relationship(
                            from_uuid=data["uuid"],
                            to_uuid=rel["target_id"],
                            relation_type=rel.get("type", "relates"),
                        )

            except Exception as e:
                log.error(f"Batch storage failed: {e}, rolling back")
                # Rollback: delete all created nodes and vectors
                for uuid in uuids:
                    try:
                        self.db.delete_memory_node(uuid)
                    except Exception:
                        pass
                    try:
                        self.db.delete_memory_vector(uuid)
                    except Exception:
                        pass
                raise e

        log.info(f"Batch stored {len(uuids)} memories")
        return uuids

    def search(
        self,
        query: str,
        limit: int = 10,
        use_graph: bool = True,
        type_filter: str | None = None,
        project_id: str | None = None,
    ) -> list[Memory]:
        """Hybrid search combining vector similarity and graph expansion.

        Args:
            query: Search query text
            limit: Maximum results to return
            use_graph: Whether to expand results via graph (default: True)
            type_filter: Optional filter by memory type
            project_id: Optional filter by project (for cross-project isolation)

        Returns:
            List of matching memories, sorted by relevance
        """
        log.debug(f"Search: query='{query[:50]}...', limit={limit}, use_graph={use_graph}, type_filter={type_filter}, project={project_id}")

        log.trace("Generating query embedding")
        query_embedding = embed(query, self.config)

        # Step 1: Vector search
        log.trace("Step 1: Vector search")
        search_limit = limit * 3 if project_id else (limit * 2 if use_graph else limit)
        vector_results = self.db.search_vectors(
            query_vector=query_embedding,
            limit=search_limit,
            type_filter=type_filter,
            project_id=project_id,  # Filter in LanceDB for efficiency
        )
        log.debug(f"Vector search returned {len(vector_results)} results")

        # Optional: Filter by project_id (via graph lookup)
        if project_id and vector_results:
            log.trace(f"Filtering by project: {project_id}")
            uuids = [r["uuid"] for r in vector_results]
            project_uuids = self.db.get_memories_in_project(project_id, uuids)
            vector_results = [r for r in vector_results if r["uuid"] in project_uuids]
            log.debug(f"After project filter: {len(vector_results)} results")

        if not use_graph or not vector_results:
            log.debug(f"Returning {min(len(vector_results), limit)} vector-only results")
            return self._format_vector_results(vector_results[:limit])

        # Step 2: Graph expansion from top hits
        log.trace("Step 2: Graph expansion")
        seed_ids = [r["uuid"] for r in vector_results[:5]]
        all_results: dict[str, dict[str, Any]] = {}

        # Add vector results with similarity scores
        for r in vector_results:
            # Normalize cosine distance (0-2) to similarity (0-1)
            vector_score = max(0, 1 - r.get("_distance", 0) / 2)
            all_results[r["uuid"]] = {
                "data": r,
                "vector_score": vector_score,
                "graph_score": 0.0,
            }

        # Add graph-expanded results
        graph_expanded_count = 0
        for seed_id in seed_ids:
            related = self.db.get_related_nodes(seed_id, hops=2)
            for r in related:
                uuid = r["uuid"]
                hop_bonus = 0.3 / r["hops"]  # Closer = higher bonus

                if uuid in all_results:
                    all_results[uuid]["graph_score"] = max(
                        all_results[uuid]["graph_score"],
                        hop_bonus,
                    )
                else:
                    all_results[uuid] = {
                        "data": r,
                        "vector_score": 0.0,
                        "graph_score": hop_bonus,
                    }
                    graph_expanded_count += 1

        log.debug(f"Graph expansion added {graph_expanded_count} additional results")

        # Step 3: Rank by combined score
        log.trace("Step 3: Ranking results")
        ranked = sorted(
            all_results.values(),
            key=lambda x: x["vector_score"] + x["graph_score"],
            reverse=True,
        )

        log.info(f"Search complete: returning {min(len(ranked), limit)} results")
        return self._format_combined_results(ranked[:limit])

    def search_multi_query(
        self,
        queries: list[str],
        limit: int = 10,
        use_graph: bool = True,
        type_filter: str | None = None,
        project_id: str | None = None,
        rrf_k: int = 60,
    ) -> list[Memory]:
        """Search with multiple query variations and combine using Reciprocal Rank Fusion.

        Performs search for each query variation, then combines results using RRF
        (Reciprocal Rank Fusion), which rewards memories that appear consistently
        across query variations rather than just having a single high score.

        RRF Formula: score = Σ 1/(k + rank_i)
        - k is a constant (default 60, standard in IR literature)
        - rank_i is the 1-indexed position in each query's result list
        - Memories appearing in multiple query results accumulate higher scores

        This approach is superior to max-score deduplication because:
        - It rewards consensus: a memory ranked #3 in all queries beats one ranked #1 in only one query
        - It's robust to score miscalibration across different queries
        - It reduces sensitivity to outlier scores

        Args:
            queries: List of query variations (typically from expand_query)
            limit: Maximum results to return
            use_graph: Whether to expand results via graph
            type_filter: Optional filter by memory type
            project_id: Optional filter by project
            rrf_k: RRF constant (default 60, higher values = smoother ranking)

        Returns:
            Deduplicated list of memories sorted by RRF score
        """
        if not queries:
            return []

        log.info(f"Multi-query search with {len(queries)} variations using RRF (k={rrf_k})")

        # RRF accumulation: uuid -> cumulative RRF score
        rrf_scores: dict[str, float] = {}
        # Store Memory objects: uuid -> Memory (keep first occurrence for metadata)
        memory_data: dict[str, Memory] = {}
        # Track original vector scores for logging/debugging
        original_scores: dict[str, list[float]] = {}

        for i, query in enumerate(queries):
            log.debug(f"Searching variation {i+1}/{len(queries)}: '{query[:50]}...'")
            results = self.search(
                query=query,
                limit=limit * 3,  # Fetch more for better RRF coverage
                use_graph=use_graph,
                type_filter=type_filter,
                project_id=project_id,
            )

            # Calculate RRF contribution for each result based on rank (1-indexed)
            for rank, memory in enumerate(results, start=1):
                # RRF formula: 1 / (k + rank)
                rrf_contribution = 1.0 / (rrf_k + rank)
                rrf_scores[memory.uuid] = rrf_scores.get(memory.uuid, 0.0) + rrf_contribution

                # Store memory data (first occurrence)
                if memory.uuid not in memory_data:
                    memory_data[memory.uuid] = memory
                    original_scores[memory.uuid] = []

                # Track original scores for debugging
                original_scores[memory.uuid].append(memory.score)

        # Sort by RRF score (descending)
        sorted_uuids = sorted(
            rrf_scores.keys(),
            key=lambda uuid: rrf_scores[uuid],
            reverse=True,
        )

        # Build result list with RRF scores replacing original scores
        results: list[Memory] = []
        for uuid in sorted_uuids[:limit]:
            memory = memory_data[uuid]
            # Create new Memory with RRF score
            results.append(Memory(
                uuid=memory.uuid,
                content=memory.content,
                type=memory.type,
                created_at=memory.created_at,
                score=rrf_scores[uuid],  # Use RRF score
                session_id=memory.session_id,
            ))

        # Log statistics for debugging
        if results:
            top_uuid = results[0].uuid
            log.debug(
                f"RRF top result: {top_uuid[:8]}... "
                f"rrf_score={rrf_scores[top_uuid]:.4f}, "
                f"original_scores={original_scores[top_uuid]}, "
                f"appeared_in={len(original_scores[top_uuid])} queries"
            )

        log.info(f"Multi-query RRF search complete: {len(rrf_scores)} unique results, returning top {limit}")
        return results

    def relate(
        self,
        from_id: str,
        to_id: str,
        relation_type: str = "relates",
    ) -> bool:
        """Create a relationship between two memories.

        Args:
            from_id: Source memory UUID
            to_id: Target memory UUID
            relation_type: Type of relationship

        Returns:
            True if relationship was created
        """
        log.debug(f"Creating relation: {from_id[:8]}... --[{relation_type}]--> {to_id[:8]}...")
        try:
            self.db.add_relationship(from_id, to_id, relation_type)
            log.info(f"Relation created: {relation_type}")
            return True
        except Exception as e:
            log.error(f"Failed to create relation: {e}")
            return False

    def add_verb_edge(
        self,
        memory_id: str,
        entity_name: str,
        entity_type: str,
        action: str,
        timestamp: int | None = None,
        change_summary: str | None = None,
    ) -> bool:
        """Link a memory to an entity with a verb-specific edge.

        Creates the entity node if it doesn't exist, then creates a semantic
        edge (READS, MODIFIES, EXECUTES, or TRIGGERED) from the memory to the entity.

        Args:
            memory_id: Memory UUID
            entity_name: Entity name (e.g., "src/main.py", "Read", "ImportError")
            entity_type: Entity type (file, tool, command, error)
            action: Action type (reads, modifies, executes, triggered)
            timestamp: Optional timestamp for the action
            change_summary: Optional summary of changes (for modifies)

        Returns:
            True if edge was created
        """
        log.debug(f"Creating verb edge: {memory_id[:8]}... --[{action}]--> {entity_type}:{entity_name}")
        try:
            self.db.add_verb_edge(
                memory_uuid=memory_id,
                entity_name=entity_name,
                entity_type=entity_type,
                action=action,
                timestamp=timestamp,
                change_summary=change_summary,
            )
            log.trace(f"Verb edge created: {action.upper()} -> {entity_type}:{entity_name}")
            return True
        except Exception as e:
            log.warning(f"Failed to create verb edge: {e}")
            return False

    def get(self, uuid: str) -> Memory | None:
        """Retrieve a specific memory by UUID.

        Args:
            uuid: Memory UUID

        Returns:
            Memory if found, None otherwise
        """
        log.debug(f"Getting memory: uuid={uuid[:8]}...")
        result = self.db.execute_graph(
            """
            MATCH (m:Memory {uuid: $uuid})
            RETURN m.uuid, m.content, m.type, m.session_id, m.created_at
            """,
            {"uuid": uuid},
        )

        if result.result_set:
            row = result.result_set[0]
            log.trace(f"Found memory: type={row[2]}")
            return Memory(
                uuid=row[0],
                content=row[1],
                type=row[2],
                session_id=row[3] if row[3] else None,
                created_at=row[4],
            )

        log.debug(f"Memory not found: uuid={uuid[:8]}...")
        return None

    def get_related(
        self,
        uuid: str,
        hops: int = 1,
        direction: str = "both",
    ) -> list[Memory]:
        """Get memories connected via graph relationships.

        Args:
            uuid: Starting memory UUID
            hops: Number of hops to traverse (1-3)
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of related memories
        """
        log.debug(f"Getting related: uuid={uuid[:8]}..., hops={hops}, direction={direction}")
        related = self.db.get_related_nodes(uuid, hops, direction)
        log.debug(f"Found {len(related)} related memories")

        return [
            Memory(
                uuid=r["uuid"],
                content=r["content"],
                type=r["type"],
                session_id=r["session_id"] if r["session_id"] else None,
                created_at=r["created_at"],
                score=1.0 / r["hops"],  # Score based on proximity
            )
            for r in related
        ]

    def list_recent(self, limit: int = 20) -> list[Memory]:
        """List most recent memories.

        Args:
            limit: Maximum number of memories to return

        Returns:
            List of memories, most recent first
        """
        log.debug(f"Listing recent memories: limit={limit}")
        result = self.db.execute_graph(
            """
            MATCH (m:Memory)
            RETURN m.uuid, m.content, m.type, m.session_id, m.created_at
            ORDER BY m.created_at DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )

        memories = []
        for row in result.result_set:
            memories.append(
                Memory(
                    uuid=row[0],
                    content=row[1],
                    type=row[2],
                    session_id=row[3] if row[3] else None,
                    created_at=row[4],
                )
            )

        log.debug(f"Found {len(memories)} recent memories")
        return memories

    def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics.

        Returns:
            Dictionary with stats (lightweight, O(1) operations only)

        Note: For sync health metrics, use get_sync_health() directly.
              It's not included here due to O(N) performance cost.
        """
        log.debug("Getting memory store stats")

        # Use db.get_stats() which properly handles FalkorDB QueryResult
        db_stats = self.db.get_stats()

        log.info(f"Stats: {db_stats['memories']} memories, {db_stats['relations']} relations")
        return {
            "total_memories": db_stats["memories"],
            "total_relations": db_stats["relations"],
            "types_breakdown": db_stats.get("entity_types", {}),
            "entities": db_stats.get("entities", 0),
            "retry_queue_size": self.get_retry_queue_size(),
        }

    def _parse_metadata(self, metadata_str: str | None) -> dict[str, Any]:
        """Parse JSON metadata string to dict.

        Args:
            metadata_str: JSON string or None

        Returns:
            Parsed metadata dict (empty dict if parsing fails)
        """
        if not metadata_str:
            return {}
        try:
            import json
            return json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _format_vector_results(self, results: list[dict]) -> list[Memory]:
        """Format vector search results as Memory objects.

        Note: LanceDB returns L2/cosine distance. For normalized embeddings,
        cosine distance ranges 0-2 (0=identical, 2=opposite).
        We convert to similarity: sim = max(0, 1 - distance/2)
        """
        return [
            Memory(
                uuid=r["uuid"],
                content=r["content"],
                type=r["type"],
                session_id=r["session_id"] if r.get("session_id") else None,
                created_at=0,  # Not stored in vector table
                score=max(0, 1 - r.get("_distance", 0) / 2),  # Normalize cosine distance to 0-1
                metadata=self._parse_metadata(r.get("metadata")),
            )
            for r in results
        ]

    def _format_combined_results(self, results: list[dict]) -> list[Memory]:
        """Format combined vector + graph results as Memory objects."""
        memories = []
        for r in results:
            data = r["data"]
            score = r["vector_score"] + r["graph_score"]

            memories.append(
                Memory(
                    uuid=data["uuid"],
                    content=data["content"],
                    type=data.get("type", "unknown"),
                    session_id=data.get("session_id") if data.get("session_id") else None,
                    created_at=data.get("created_at", 0),
                    score=score,
                    metadata=self._parse_metadata(data.get("metadata")),
                )
            )

        return memories

    def reset_all(self) -> dict[str, Any]:
        """Reset all data - delete all memories and relationships.

        WARNING: This is destructive and irreversible. For debug purposes only.

        Returns:
            Dictionary with counts of deleted items
        """
        log.warning("MemoryStore.reset_all: Initiating complete data wipe")
        result = self.db.reset_all()
        log.warning(f"MemoryStore.reset_all: Complete - {result}")
        return result

    def reason(
        self,
        query: str,
        seed_limit: int = 5,
        max_hops: int = 2,
        min_score: float = 0.1,
        use_pagerank: bool = True,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Multi-hop reasoning over memory graph.

        Combines vector search with graph traversal and semantic path scoring
        to find conclusions supported by chains of evidence.

        Uses PageRank to boost scores of structurally important nodes (those
        with many high-quality incoming edges).

        Args:
            query: Natural language query
            seed_limit: Number of seed nodes from vector search
            max_hops: Maximum path length for traversal
            min_score: Minimum score threshold for results
            use_pagerank: Whether to incorporate PageRank scores (default: True)
            project_id: Optional filter by project (for cross-project isolation)

        Returns:
            List of conclusions with scores and proof chains
        """
        log.info(f"Reasoning: query='{query[:50]}...', max_hops={max_hops}, pagerank={use_pagerank}, project={project_id}")

        # Step 1: Vector search for seed nodes
        log.trace("Step 1: Finding seed nodes via vector search")
        query_embedding = embed(query, self.config)
        # Fetch more seeds when filtering by project to ensure adequate results
        fetch_limit = seed_limit * 3 if project_id else seed_limit
        seeds = self.db.search_vectors(query_embedding, limit=fetch_limit, project_id=project_id)
        log.debug(f"Found {len(seeds)} seed nodes")

        # Optional: Filter by project_id (via graph lookup)
        if project_id and seeds:
            log.trace(f"Filtering seeds by project: {project_id}")
            uuids = [s["uuid"] for s in seeds]
            project_uuids = self.db.get_memories_in_project(project_id, uuids)
            seeds = [s for s in seeds if s["uuid"] in project_uuids][:seed_limit]
            log.debug(f"After project filter: {len(seeds)} seeds")

        if not seeds:
            return []

        # Step 2: Graph traversal from each seed
        log.trace("Step 2: Graph traversal from seeds")
        all_paths: dict[str, dict[str, Any]] = {}  # end_uuid -> best path info

        for seed in seeds:
            seed_uuid = seed["uuid"]
            # Normalize cosine distance (0-2) to similarity (0-1)
            seed_score = max(0, 1 - seed.get("_distance", 0) / 2)

            # Get paths from this seed (use "both" for bi-directional traversal)
            paths = self.db.get_paths(seed_uuid, max_hops=max_hops, direction="both")

            for path in paths:
                end_uuid = path["end_uuid"]

                # Calculate path score
                path_score = self._score_path(path, seed_score)

                # Evidence aggregation: keep best path to each node
                if end_uuid not in all_paths or path_score > all_paths[end_uuid]["score"]:
                    all_paths[end_uuid] = {
                        "uuid": end_uuid,
                        "content": path["end_content"],
                        "type": path["end_type"],
                        "session_id": path["session_id"],
                        "score": path_score,
                        "proof_chain": path["edge_types"],
                        "hops": path["hops"],
                        "seed_uuid": seed_uuid,
                    }

            # Also get cross-session paths via entity nodes
            cross_session_paths = self.db.get_cross_session_paths(seed_uuid, max_hops=max_hops)
            for path in cross_session_paths:
                end_uuid = path["end_uuid"]
                path_score = self._score_path(path, seed_score)

                # Add bridge entity info to proof chain
                bridge = path.get("bridge_entity", {})
                proof_chain = path["edge_types"] + [f"via:{bridge.get('type', 'entity')}:{bridge.get('name', '?')}"]

                if end_uuid not in all_paths or path_score > all_paths[end_uuid]["score"]:
                    all_paths[end_uuid] = {
                        "uuid": end_uuid,
                        "content": path["end_content"],
                        "type": path["end_type"],
                        "session_id": path["session_id"],
                        "score": path_score,
                        "proof_chain": proof_chain,
                        "hops": path["hops"],
                        "seed_uuid": seed_uuid,
                        "cross_session": True,
                        "bridge_entity": bridge,
                    }

        # Step 3: Also add seeds themselves (0-hop)
        for seed in seeds:
            seed_uuid = seed["uuid"]
            # Normalize cosine distance (0-2) to similarity (0-1)
            seed_score = max(0, 1 - seed.get("_distance", 0) / 2)
            if seed_uuid not in all_paths or seed_score > all_paths[seed_uuid]["score"]:
                all_paths[seed_uuid] = {
                    "uuid": seed_uuid,
                    "content": seed["content"],
                    "type": seed["type"],
                    "session_id": seed.get("session_id"),
                    "score": seed_score,
                    "proof_chain": [],
                    "hops": 0,
                    "seed_uuid": seed_uuid,
                }

        # Step 4: Apply PageRank boost if enabled
        if use_pagerank and all_paths:
            log.trace("Step 4: Applying PageRank boost")
            uuids = list(all_paths.keys())
            pagerank_scores = self.db.get_pagerank_for_nodes(uuids)

            # Normalize PageRank scores to 0-1 range for this result set
            if pagerank_scores:
                max_pr = max(pagerank_scores.values()) or 1.0
                for uuid, path_info in all_paths.items():
                    pr_score = pagerank_scores.get(uuid, 0.0) / max_pr
                    # PageRank boost: multiply by (1 + pr_score * 0.3)
                    # This gives up to 30% boost for highest-ranked nodes
                    path_info["score"] *= (1 + pr_score * 0.3)
                    path_info["pagerank"] = round(pr_score, 3)

                log.debug(f"PageRank applied to {len(pagerank_scores)} nodes")

        # Step 5: Filter and sort by score
        results = [
            p for p in all_paths.values()
            if p["score"] >= min_score
        ]
        results.sort(key=lambda x: x["score"], reverse=True)

        log.info(f"Reasoning complete: {len(results)} conclusions found")
        return results

    def _score_path(
        self,
        path: dict[str, Any],
        seed_score: float,
    ) -> float:
        """Calculate semantic score for a path.

        Combines:
        - Seed vector similarity
        - Edge type weights (multiplicative)
        - Temporal decay

        Args:
            path: Path info from get_paths()
            seed_score: Vector similarity of seed node

        Returns:
            Combined path score
        """
        score = seed_score

        # Multiply by edge weights
        for edge_type in path.get("edge_types", []):
            weight = RELATION_WEIGHTS.get(edge_type, 0.3)
            score *= weight

        # Temporal decay (λ = 1e-7 ≈ half-life of ~80 days)
        created_at = path.get("created_at", 0)
        if created_at > 0:
            now = time.time()
            age = now - created_at
            decay = 1 / (1 + 1e-7 * age)
            score *= decay

        return score

    async def reason_with_synthesis(
        self,
        query: str,
        max_memories: int = 10,
        max_hops: int = 2,
        min_score: float = 0.1,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """LLM-synthesized reasoning over memory graph.

        Performs multi-hop graph traversal, then uses an LLM to synthesize
        a condensed explanation of evidence chains and connection patterns.
        Returns reasoning summary + UUID references (not full content).

        Args:
            query: Natural language query to reason about
            max_memories: Maximum memories to analyze (default: 10)
            max_hops: Maximum graph traversal depth (default: 2)
            min_score: Minimum score threshold (default: 0.1)
            project_id: Optional filter by project

        Returns:
            Dictionary with:
            - reasoning: LLM-synthesized explanation of evidence chains
            - patterns: List of detected cross-session patterns
            - confidence: high/medium/low based on evidence quality
            - sources: List of source refs (uuid, type, score, hops, proof_chain)
        """
        from litellm import acompletion

        log.info(f"reason_with_synthesis: query='{query[:50]}...', max_memories={max_memories}, project={project_id}")

        # Step 1: Get graph-traversed memories via existing reason()
        memories = self.reason(
            query,
            seed_limit=max_memories,
            max_hops=max_hops,
            min_score=min_score,
            project_id=project_id,
        )[:max_memories]

        if not memories:
            log.info("reason_with_synthesis: No relevant memories found")
            return {
                "reasoning": "No relevant memories found for this query.",
                "patterns": [],
                "confidence": "none",
                "cross_session_count": 0,
                "sources": [],
            }

        # Step 2: Format memories for LLM (truncated content)
        formatted = self._format_memories_for_reasoning(memories)
        cross_session_count = sum(1 for m in memories if m.get("cross_session"))

        # Step 3: Calculate confidence
        avg_score = sum(m["score"] for m in memories) / len(memories)
        if avg_score >= 0.5:
            confidence = "high"
        elif avg_score >= 0.2:
            confidence = "medium"
        else:
            confidence = "low"

        # Step 4: LLM prompt focused on REASONING structure, not answering
        prompt = f'''You are analyzing evidence chains from a memory graph. Your task is to explain HOW memories are connected, not to answer the query directly.

## Query Being Analyzed
{query}

## Retrieved Evidence (via graph traversal with PageRank)
{formatted}

## Your Task
Provide a concise reasoning summary (2-4 sentences) that:
1. Explains the CONNECTION STRUCTURE - how memories relate via proof chains
2. Highlights CROSS-SESSION PATTERNS if any (memories from different sessions linked by shared entities)
3. Identifies the STRONGEST evidence paths (highest scores, shortest hops)
4. References memories as [1], [2], etc.

Also extract any notable patterns as a bullet list.

## Format
REASONING: <your 2-4 sentence synthesis explaining the evidence structure>

PATTERNS:
- <pattern 1>
- <pattern 2>
(or "None detected" if no clear patterns)'''

        # Step 5: Call LLM
        try:
            response = await acompletion(
                model=self.config.summary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2,
            )
            raw_response = response.choices[0].message.content or ""

            # Parse response
            reasoning = raw_response
            patterns = []

            if raw_response and "REASONING:" in raw_response:
                parts = raw_response.split("PATTERNS:")
                reasoning_part = parts[0].replace("REASONING:", "").strip()
                reasoning = reasoning_part

                if len(parts) > 1:
                    patterns_part = parts[1].strip()
                    if patterns_part.lower() != "none detected":
                        # Extract bullet points
                        for line in patterns_part.split("\n"):
                            line = line.strip()
                            if line.startswith("-"):
                                patterns.append(line[1:].strip())

        except Exception as e:
            log.error(f"reason_with_synthesis LLM call failed: {e}")
            reasoning = f"Error synthesizing reasoning: {e}"
            patterns = []
            confidence = "error"

        log.info(f"reason_with_synthesis complete: {len(memories)} memories, {len(patterns)} patterns, confidence={confidence}")

        return {
            "reasoning": reasoning,
            "patterns": patterns,
            "confidence": confidence,
            "cross_session_count": cross_session_count,
            "sources": [
                {
                    "uuid": m["uuid"],
                    "type": m["type"],
                    "score": round(m["score"], 3),
                    "hops": m["hops"],
                    "proof_chain": m.get("proof_chain", []),
                    "cross_session": m.get("cross_session", False),
                }
                for m in memories
            ],
        }

    def _format_memories_for_reasoning(self, memories: list[dict[str, Any]]) -> str:
        """Format memories for reasoning LLM prompt.

        Focuses on structure (proof chains, hops) over content.
        Content is heavily truncated since we want reasoning about structure.

        Args:
            memories: List of memory dicts from reason()

        Returns:
            Formatted string with numbered memories
        """
        formatted = []
        for i, m in enumerate(memories, 1):
            parts = [f"[{i}] type={m['type']}, score={m['score']:.2f}"]

            if m["hops"] > 0:
                parts.append(f"hops={m['hops']}")
                if m.get("proof_chain"):
                    chain = m["proof_chain"]
                    if len(chain) > 3:
                        path_str = f"{chain[0]} -> ... -> {chain[-1]}"
                    else:
                        path_str = " -> ".join(chain)
                    parts.append(f"path=[{path_str}]")

            if m.get("cross_session"):
                parts.append("[CROSS-SESSION]")
                bridge = m.get("bridge_entity", {})
                if bridge:
                    parts.append(f"via {bridge.get('type', 'entity')}:{bridge.get('name', '?')}")

            # Truncate content heavily - we care about structure
            content = m.get("content", "")[:200]
            if len(m.get("content", "")) > 200:
                content += "..."

            meta = ", ".join(parts)
            formatted.append(f"{meta}\n   Content: {content}")

        return "\n\n".join(formatted)

    async def ask_memories(
        self,
        query: str,
        max_memories: int = 8,
        max_hops: int = 2,
        include_raw: bool = False,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """LLM-powered reasoning over graph-retrieved memories.

        Retrieves relevant memories via multi-hop graph traversal, then uses
        an LLM to synthesize a coherent answer grounded in the evidence.

        Args:
            query: Natural language question
            max_memories: Maximum memories to include in context (default: 8)
            max_hops: Maximum graph traversal depth (default: 2)
            include_raw: Include raw memory data in response (default: False)
            project_id: Optional filter by project (for cross-project isolation)

        Returns:
            Dictionary with:
            - answer: LLM-synthesized answer with citations
            - memories_used: Number of memories in context
            - cross_session_insights: Count of cross-session memories
            - confidence: high/medium/low based on memory scores
            - sources: List of source memory metadata
        """
        from litellm import acompletion

        log.info(f"ask_memories: query='{query[:50]}...', max_memories={max_memories}, project={project_id}")

        # Step 1: Retrieve memories via graph reasoning
        memories = self.reason(query, max_hops=max_hops, min_score=0.05, project_id=project_id)[:max_memories]

        if not memories:
            log.info("ask_memories: No relevant memories found")
            return {
                "answer": "I don't have any relevant memories to answer this question.",
                "memories_used": 0,
                "cross_session_insights": 0,
                "confidence": "none",
                "sources": [],
            }

        # Step 2: Format memories for LLM context
        formatted_memories = self._format_memories_for_llm(memories)
        cross_session_count = sum(1 for m in memories if m.get("cross_session"))

        # Step 3: Calculate confidence based on top scores
        avg_score = sum(m["score"] for m in memories) / len(memories)
        if avg_score >= 0.5:
            confidence = "high"
        elif avg_score >= 0.2:
            confidence = "medium"
        else:
            confidence = "low"

        # Step 4: Build the reasoning prompt
        prompt = f'''You are a memory-augmented assistant that answers questions using retrieved evidence from past coding sessions.

## Retrieved Memories (ranked by relevance)
{formatted_memories}

## Rules
1. Answer ONLY using information from the memories above
2. Cite sources using [1], [2], etc. when referencing specific information
3. If the memories don't contain enough information, say "Based on available memories, I cannot fully answer this, but here's what I found: ..."
4. Cross-session memories (marked with [cross-session]) are especially valuable - they show patterns across different work sessions
5. Be concise and specific
6. Focus on actionable insights and concrete details

## Question
{query}

## Answer (with citations)'''

        # Step 5: Call LLM for synthesis
        try:
            response = await acompletion(
                model=self.config.summary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,  # Low temperature for factual, grounded answers
            )
            answer = response.choices[0].message.content
        except Exception as e:
            log.error(f"ask_memories LLM call failed: {e}")
            answer = f"Error generating answer: {e}"
            confidence = "error"

        log.info(f"ask_memories complete: {len(memories)} memories, confidence={confidence}")

        result = {
            "answer": answer,
            "memories_used": len(memories),
            "cross_session_insights": cross_session_count,
            "confidence": confidence,
            "sources": [
                {
                    "uuid": m["uuid"],
                    "type": m["type"],
                    "score": round(m["score"], 3),
                    "hops": m["hops"],
                    "cross_session": m.get("cross_session", False),
                }
                for m in memories
            ],
        }

        if include_raw:
            result["raw_memories"] = memories

        return result

    def _format_memories_for_llm(self, memories: list[dict[str, Any]]) -> str:
        """Format memories with metadata for LLM context.

        Args:
            memories: List of memory dicts from reason()

        Returns:
            Formatted string with numbered memories and metadata
        """
        formatted = []
        for i, m in enumerate(memories, 1):
            # Build metadata string
            meta_parts = [f"type: {m['type']}", f"score: {m['score']:.2f}"]

            if m["hops"] > 0:
                meta_parts.append(f"hops: {m['hops']}")
                if m.get("proof_chain"):
                    proof_chain = m["proof_chain"]
                    # Show beginning and end for long chains to preserve context
                    if len(proof_chain) > 3:
                        path_str = f"{' -> '.join(proof_chain[:2])} -> ... -> {proof_chain[-1]}"
                    else:
                        path_str = " -> ".join(proof_chain)
                    meta_parts.append(f"path: {path_str}")

            if m.get("cross_session"):
                meta_parts.append("[cross-session]")
                bridge = m.get("bridge_entity", {})
                if bridge:
                    meta_parts.append(f"via {bridge.get('type', 'entity')}: {bridge.get('name', '?')}")

            meta_str = ", ".join(meta_parts)

            # Truncate content to fit more memories
            content = m["content"][:600]
            if len(m["content"]) > 600:
                content += "..."

            formatted.append(f"[MEMORY {i}] ({meta_str})\n{content}\n")

        return "\n".join(formatted)

    def reindex_memories(
        self,
        project_id: str,
        batch_size: int = 50,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Re-generate embeddings for all memories in a project.

        Fetches memories from the graph database, generates new embeddings
        using the current embedding model, and writes them to LanceDB.
        This fixes embedding model mismatches where memories were embedded
        with one model but searches use a different model.

        Args:
            project_id: Project to reindex
            batch_size: Number of memories to embed per batch (default: 50)
            progress_callback: Optional callback(processed, total) for progress

        Returns:
            dict with:
            - reindexed: Number of memories successfully reindexed
            - errors: Number of embedding failures
            - project_id: The project that was reindexed
            - total: Total memories found in graph
        """
        log.info(f"Starting reindex for project: {project_id}, batch_size={batch_size}")

        # Step 1: Get total count for progress tracking
        total_count = self.db.get_memory_count(project_id)
        if total_count == 0:
            log.info(f"No memories found for project {project_id}")
            return {
                "reindexed": 0,
                "errors": 0,
                "project_id": project_id,
                "total": 0,
            }

        log.info(f"Found {total_count} memories to reindex")

        # Step 2: Fetch memories in batches from graph
        batches = self.db.get_memories_for_reindex(project_id, batch_size=batch_size)

        reindexed = 0
        errors = 0
        processed = 0

        # Step 3: Process each batch
        for batch_idx, batch in enumerate(batches):
            try:
                # Extract content for embedding
                contents = [mem["content"] for mem in batch]

                # Generate embeddings via batch API
                log.debug(f"Batch {batch_idx + 1}/{len(batches)}: embedding {len(contents)} memories")
                embeddings = embed_batch(contents, self.config)

                if len(embeddings) != len(batch):
                    log.warning(f"Batch {batch_idx + 1}: embedding count mismatch ({len(embeddings)} vs {len(batch)})")
                    errors += len(batch) - len(embeddings)
                    # Truncate to match
                    batch = batch[:len(embeddings)]

                # Write to LanceDB via upsert (include project_id for search filtering)
                written = self.db.upsert_memory_vectors(batch, embeddings, project_id=project_id)
                reindexed += written

                processed += len(batch)
                log.debug(f"Batch {batch_idx + 1}/{len(batches)}: wrote {written} vectors")

                # Progress callback
                if progress_callback:
                    try:
                        progress_callback(processed, total_count)
                    except Exception as e:
                        log.warning(f"Progress callback failed: {e}")

            except Exception as e:
                log.error(f"Batch {batch_idx + 1} failed: {e}")
                errors += len(batch)
                processed += len(batch)

        log.info(f"Reindex complete: {reindexed} reindexed, {errors} errors, {total_count} total")
        return {
            "reindexed": reindexed,
            "errors": errors,
            "project_id": project_id,
            "total": total_count,
        }
