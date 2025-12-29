"""Memory store for SimpleMem Lite.

Unified memory storage with hybrid search (vector + graph).
"""

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from simplemem_lite.config import Config
from simplemem_lite.db import DatabaseManager
from simplemem_lite.embeddings import embed, embed_batch, init_embeddings
from simplemem_lite.logging import get_logger

log = get_logger("memory")

# Semantic weights for relation types (applied in Python for flexibility)
RELATION_WEIGHTS = {
    "contains": 1.0,    # Hierarchical drill-down
    "child_of": 1.0,    # Reverse contains
    "supports": 0.8,    # Evidence for conclusion
    "follows": 0.6,     # Temporal sequence
    "mentions": 0.4,    # Reference
    "relates": 0.3,     # Generic/weak
}


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
        type: Memory type (fact, session_summary, chunk_summary, message)
        created_at: Unix timestamp
        score: Relevance score (higher = more relevant)
        session_id: Optional session identifier
        relations: Related memories
    """

    uuid: str
    content: str
    type: str
    created_at: int
    score: float = 0.0
    session_id: str | None = None
    relations: list[dict[str, Any]] = field(default_factory=list)


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
        log.info("MemoryStore initialized successfully")

    def store(self, item: MemoryItem) -> str:
        """Store a memory with embedding and optional relationships.

        Uses two-phase commit: graph first (source of truth), then vectors.

        Args:
            item: Memory item to store

        Returns:
            UUID of stored memory

        Raises:
            Exception: If storage fails (with automatic rollback)
        """
        memory_uuid = str(uuid4())
        mem_type = item.metadata.get("type", "fact")
        source = item.metadata.get("source", "user")
        session_id = item.metadata.get("session_id")
        created_at = int(time.time())

        log.debug(f"Storing memory: uuid={memory_uuid[:8]}..., type={mem_type}, source={source}")
        log.trace(f"Content preview: {item.content[:100]}...")

        log.trace("Generating embedding")
        embedding = embed(item.content, self.config)
        log.trace(f"Embedding generated: dim={len(embedding)}")

        with self.db.write_lock:
            try:
                # Phase 1: Write to graph (primary)
                log.trace("Phase 1: Writing to graph")
                self.db.add_memory_node(
                    uuid=memory_uuid,
                    content=item.content,
                    mem_type=mem_type,
                    source=source,
                    session_id=session_id,
                    created_at=created_at,
                )

                # Phase 2: Write to vectors
                log.trace("Phase 2: Writing to vectors")
                self.db.add_memory_vector(
                    uuid=memory_uuid,
                    vector=embedding,
                    content=item.content,
                    mem_type=mem_type,
                    session_id=session_id,
                )

                # Phase 3: Create relationships
                if item.relations:
                    log.trace(f"Phase 3: Creating {len(item.relations)} relationships")
                for rel in item.relations:
                    self.db.add_relationship(
                        from_uuid=memory_uuid,
                        to_uuid=rel["target_id"],
                        relation_type=rel.get("type", "relates"),
                    )

            except Exception as e:
                log.error(f"Storage failed: {e}, rolling back")
                # Rollback: delete from graph
                try:
                    self.db.delete_memory_node(memory_uuid)
                    log.debug("Rollback successful")
                except Exception as rollback_e:
                    log.warning(f"Rollback failed: {rollback_e}")
                raise e

        log.info(f"Memory stored: {memory_uuid[:8]}... ({mem_type})")
        return memory_uuid

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
            created_at = int(time.time())

            batch_data.append({
                "uuid": memory_uuid,
                "content": item.content,
                "type": mem_type,
                "source": source,
                "session_id": session_id,
                "created_at": created_at,
                "relations": item.relations,
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
                    )

                    # Phase 2: Write to vectors
                    self.db.add_memory_vector(
                        uuid=data["uuid"],
                        vector=embedding,
                        content=data["content"],
                        mem_type=data["type"],
                        session_id=data["session_id"],
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
                # Rollback: delete all created nodes
                for uuid in uuids:
                    try:
                        self.db.delete_memory_node(uuid)
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
    ) -> list[Memory]:
        """Hybrid search combining vector similarity and graph expansion.

        Args:
            query: Search query text
            limit: Maximum results to return
            use_graph: Whether to expand results via graph (default: True)
            type_filter: Optional filter by memory type

        Returns:
            List of matching memories, sorted by relevance
        """
        log.debug(f"Search: query='{query[:50]}...', limit={limit}, use_graph={use_graph}, type_filter={type_filter}")

        log.trace("Generating query embedding")
        query_embedding = embed(query, self.config)

        # Step 1: Vector search
        log.trace("Step 1: Vector search")
        vector_results = self.db.search_vectors(
            query_vector=query_embedding,
            limit=limit * 2 if use_graph else limit,
            type_filter=type_filter,
        )
        log.debug(f"Vector search returned {len(vector_results)} results")

        if not use_graph or not vector_results:
            log.debug(f"Returning {min(len(vector_results), limit)} vector-only results")
            return self._format_vector_results(vector_results[:limit])

        # Step 2: Graph expansion from top hits
        log.trace("Step 2: Graph expansion")
        seed_ids = [r["uuid"] for r in vector_results[:5]]
        all_results: dict[str, dict[str, Any]] = {}

        # Add vector results with similarity scores
        for r in vector_results:
            vector_score = 1 - r.get("_distance", 0)  # Convert distance to similarity
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

        if result.has_next():
            row = result.get_next()
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
        while result.has_next():
            row = result.get_next()
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
            Dictionary with stats
        """
        log.debug("Getting memory store stats")

        # Count memories
        result = self.db.execute_graph("MATCH (m:Memory) RETURN count(m)")
        total_memories = result.get_next()[0] if result.has_next() else 0

        # Count relationships
        result = self.db.execute_graph("MATCH ()-[r:RELATES_TO]->() RETURN count(r)")
        total_relations = result.get_next()[0] if result.has_next() else 0

        # Count by type
        result = self.db.execute_graph(
            "MATCH (m:Memory) RETURN m.type, count(m) ORDER BY count(m) DESC"
        )
        types_breakdown = {}
        while result.has_next():
            row = result.get_next()
            types_breakdown[row[0]] = row[1]

        log.info(f"Stats: {total_memories} memories, {total_relations} relations")
        return {
            "total_memories": total_memories,
            "total_relations": total_relations,
            "types_breakdown": types_breakdown,
        }

    def _format_vector_results(self, results: list[dict]) -> list[Memory]:
        """Format vector search results as Memory objects."""
        return [
            Memory(
                uuid=r["uuid"],
                content=r["content"],
                type=r["type"],
                session_id=r["session_id"] if r.get("session_id") else None,
                created_at=0,  # Not stored in vector table
                score=1 - r.get("_distance", 0),
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
    ) -> list[dict[str, Any]]:
        """Multi-hop reasoning over memory graph.

        Combines vector search with graph traversal and semantic path scoring
        to find conclusions supported by chains of evidence.

        Args:
            query: Natural language query
            seed_limit: Number of seed nodes from vector search
            max_hops: Maximum path length for traversal
            min_score: Minimum score threshold for results

        Returns:
            List of conclusions with scores and proof chains
        """
        log.info(f"Reasoning: query='{query[:50]}...', max_hops={max_hops}")

        # Step 1: Vector search for seed nodes
        log.trace("Step 1: Finding seed nodes via vector search")
        query_embedding = embed(query, self.config)
        seeds = self.db.search_vectors(query_embedding, limit=seed_limit)
        log.debug(f"Found {len(seeds)} seed nodes")

        if not seeds:
            return []

        # Step 2: Graph traversal from each seed
        log.trace("Step 2: Graph traversal from seeds")
        all_paths: dict[str, dict[str, Any]] = {}  # end_uuid -> best path info

        for seed in seeds:
            seed_uuid = seed["uuid"]
            seed_score = 1 - seed.get("_distance", 0)  # Vector similarity

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

        # Step 3: Also add seeds themselves (0-hop)
        for seed in seeds:
            seed_uuid = seed["uuid"]
            seed_score = 1 - seed.get("_distance", 0)
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

        # Step 4: Filter and sort by score
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
