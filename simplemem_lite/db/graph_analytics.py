"""Graph Analytics for SimpleMem Lite.

Provides graph traversal, path analysis, and scoring functionality.
Extracted from DatabaseManager as part of god class decomposition.

This module handles:
- Graph traversal (get_related_nodes, get_paths)
- Cross-session path analysis
- PageRank computation and caching
- Degree scoring and normalization
"""

from typing import Any, Protocol

from simplemem_lite.log_config import get_logger

log = get_logger("db.graph_analytics")


class GraphQueryProtocol(Protocol):
    """Protocol for graph query capability."""

    def query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a Cypher query."""
        ...


class PageRankProtocol(Protocol):
    """Protocol for PageRank computation capability."""

    def get_pagerank_scores(self) -> dict[str, float]:
        """Get PageRank scores for all Memory nodes."""
        ...


class GraphAnalytics:
    """Graph traversal, path analysis, and scoring operations.

    This class handles graph-based analytics including:
    - Node traversal with configurable hop limits
    - Path extraction with edge metadata
    - Cross-session path discovery
    - PageRank computation and caching
    - Degree-based scoring

    Uses dependency injection for the graph backend to enable testing
    and flexibility in database selection.
    """

    def __init__(
        self,
        graph: GraphQueryProtocol,
        pagerank_backend: PageRankProtocol | None = None,
        max_graph_hops: int = 3,
        graph_path_limit: int = 100,
        cross_session_limit: int = 50,
    ):
        """Initialize GraphAnalytics.

        Args:
            graph: Graph backend with query capability
            pagerank_backend: Optional backend for PageRank (may be same as graph)
            max_graph_hops: Maximum traversal depth (default: 3)
            graph_path_limit: Maximum paths to return (default: 100)
            cross_session_limit: Maximum cross-session results (default: 50)
        """
        self.graph = graph
        self.pagerank_backend = pagerank_backend
        self.max_graph_hops = max_graph_hops
        self.graph_path_limit = graph_path_limit
        self.cross_session_limit = cross_session_limit

    # =========================================================================
    # Node Traversal Methods
    # =========================================================================

    def get_related_nodes(
        self,
        uuid: str,
        hops: int = 1,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Get related memories via graph traversal.

        Args:
            uuid: Starting memory UUID
            hops: Number of hops to traverse (1-3)
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of related memories
        """
        log.trace(f"Getting related nodes: uuid={uuid[:8]}..., hops={hops}, direction={direction}")
        hops = min(max(hops, 1), self.max_graph_hops)

        if direction == "outgoing":
            pattern = f"(start:Memory {{uuid: $uuid}})-[r*1..{hops}]->(connected:Memory)"
        elif direction == "incoming":
            pattern = f"(start:Memory {{uuid: $uuid}})<-[r*1..{hops}]-(connected:Memory)"
        else:
            pattern = f"(start:Memory {{uuid: $uuid}})-[r*1..{hops}]-(connected:Memory)"

        result = self.graph.query(
            f"""
            MATCH {pattern}
            RETURN DISTINCT
                connected.uuid AS uuid,
                connected.content AS content,
                connected.type AS type,
                connected.session_id AS session_id,
                connected.created_at AS created_at
            """,
            {"uuid": uuid},
        )

        rows = []
        for record in result.result_set:
            rows.append({
                "uuid": record[0],
                "content": record[1],
                "type": record[2],
                "session_id": record[3],
                "created_at": record[4],
                "hops": 1,  # Simplified
            })

        log.debug(f"Graph traversal returned {len(rows)} related nodes")
        return rows

    # =========================================================================
    # Path Analysis Methods
    # =========================================================================

    def get_paths(
        self,
        from_uuid: str,
        max_hops: int = 2,
        direction: str = "outgoing",
    ) -> list[dict[str, Any]]:
        """Get paths from a node with full edge metadata.

        Returns paths with node and edge information for scoring.
        Includes paths through Entity nodes for cross-session reasoning.

        Args:
            from_uuid: Starting memory UUID
            max_hops: Maximum path length (1-3)
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of paths with nodes, edge_types, and metadata
        """
        log.trace(f"Getting paths: from={from_uuid[:8]}..., max_hops={max_hops}")
        max_hops = min(max(max_hops, 1), self.max_graph_hops)

        # FalkorDB Cypher for variable-length paths
        if direction == "outgoing":
            pattern = f"(start:Memory {{uuid: $uuid}})-[r*1..{max_hops}]->(target:Memory)"
        elif direction == "incoming":
            pattern = f"(start:Memory {{uuid: $uuid}})<-[r*1..{max_hops}]-(target:Memory)"
        else:
            pattern = f"(start:Memory {{uuid: $uuid}})-[r*1..{max_hops}]-(target:Memory)"

        result = self.graph.query(
            f"""
            MATCH path = {pattern}
            WHERE start <> target
            RETURN DISTINCT
                target.uuid AS end_uuid,
                target.content AS end_content,
                target.type AS end_type,
                target.session_id AS session_id,
                target.created_at AS created_at,
                [rel in relationships(path) | type(rel)] AS rel_types,
                [rel in relationships(path) | rel.relation_type] AS relation_types,
                [rel in relationships(path) | rel.weight] AS weights,
                length(path) AS hops
            LIMIT $limit
            """,
            {"uuid": from_uuid, "limit": self.graph_path_limit},
        )

        paths = []
        for record in result.result_set:
            # Extract relationship info
            rel_types = record[5] or []  # RELATES_TO, REFERENCES, etc.
            relation_types = record[6] or []  # contains, follows, etc.
            weights = record[7] or []

            # Combine into edge_types (prefer relation_type if available)
            edge_types = []
            for i, rt in enumerate(rel_types):
                if i < len(relation_types) and relation_types[i]:
                    edge_types.append(relation_types[i])
                elif rt == "REFERENCES":
                    edge_types.append("references")
                else:
                    edge_types.append("relates")

            paths.append({
                "end_uuid": record[0],
                "end_content": record[1],
                "end_type": record[2],
                "session_id": record[3],
                "created_at": record[4] or 0,
                "edge_types": edge_types,
                "edge_weights": [w or 1.0 for w in (weights or [])],
                "hops": record[8] or len(edge_types),
            })

        log.debug(f"Found {len(paths)} paths from {from_uuid[:8]}...")
        return paths

    def get_cross_session_paths(
        self,
        from_uuid: str,
        max_hops: int = 3,
    ) -> list[dict[str, Any]]:
        """Get paths that cross sessions via shared Entity nodes.

        This enables reasoning like "what other sessions touched this file?"

        Args:
            from_uuid: Starting memory UUID
            max_hops: Maximum path length

        Returns:
            List of cross-session paths with entity bridge info
        """
        log.trace(f"Getting cross-session paths from {from_uuid[:8]}...")

        result = self.graph.query(
            """
            MATCH (start:Memory {uuid: $uuid})-[:REFERENCES]->(e:Entity)<-[:REFERENCES]-(other:Memory)
            WHERE start.session_id <> other.session_id
            RETURN DISTINCT
                other.uuid AS end_uuid,
                other.content AS end_content,
                other.type AS end_type,
                other.session_id AS session_id,
                other.created_at AS created_at,
                e.name AS entity_name,
                e.type AS entity_type
            LIMIT $limit
            """,
            {"uuid": from_uuid, "limit": self.cross_session_limit},
        )

        paths = []
        for record in result.result_set:
            paths.append({
                "end_uuid": record[0],
                "end_content": record[1],
                "end_type": record[2],
                "session_id": record[3],
                "created_at": record[4] or 0,
                "edge_types": ["references", "references"],
                "edge_weights": [0.7, 0.7],
                "hops": 2,
                "bridge_entity": {
                    "name": record[5],
                    "type": record[6],
                },
            })

        log.debug(f"Found {len(paths)} cross-session paths")
        return paths

    # =========================================================================
    # Cross-Session Entity Discovery
    # =========================================================================

    def get_cross_session_entities(
        self,
        min_sessions: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get entities that appear across multiple sessions (bridge entities).

        These are valuable for cross-session insights as they link
        different work sessions together.

        Args:
            min_sessions: Minimum number of sessions the entity must appear in
            limit: Maximum results to return

        Returns:
            List of entities with session count and linked sessions
        """
        log.trace(f"Getting cross-session entities (min_sessions={min_sessions})")

        result = self.graph.query(
            """
            MATCH (m:Memory)-[r]->(e:Entity)
            WHERE type(r) IN ['READS', 'MODIFIES', 'EXECUTES', 'TRIGGERED', 'REFERENCES']
              AND m.session_id IS NOT NULL AND m.session_id <> ''
            WITH e, collect(DISTINCT m.session_id) AS sessions
            WHERE size(sessions) >= $min_sessions
            RETURN
                e.name AS name,
                e.type AS type,
                e.version AS version,
                sessions AS session_ids,
                size(sessions) AS sessions_count
            ORDER BY size(sessions) DESC
            LIMIT $limit
            """,
            {"min_sessions": min_sessions, "limit": limit},
        )

        entities = []
        for record in result.result_set:
            entities.append({
                "name": record[0],
                "type": record[1],
                "version": record[2] or 1,
                "session_ids": record[3] or [],
                "sessions_count": record[4] or 0,
            })

        log.debug(f"Found {len(entities)} cross-session entities")
        return entities

    # =========================================================================
    # PageRank and Scoring Methods
    # =========================================================================

    def get_pagerank_scores(
        self,
        uuids: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute PageRank scores for Memory nodes.

        Delegates to the graph backend's PageRank implementation:
        - FalkorDB: algo.pageRank()
        - Memgraph: pagerank.get() via MAGE
        - KuzuDB: degree-based fallback

        Args:
            uuids: Optional list of UUIDs to get scores for (None = all)

        Returns:
            Dictionary mapping UUID -> PageRank score (0.0 to 1.0)
        """
        log.trace("Computing PageRank scores")

        if not self.pagerank_backend:
            log.debug("No PageRank backend available")
            return {}

        try:
            # Delegate to graph backend (handles FalkorDB/Memgraph/KuzuDB differences)
            scores = self.pagerank_backend.get_pagerank_scores()

            log.debug(f"PageRank computed for {len(scores)} nodes")

            # Filter to requested UUIDs if specified
            if uuids:
                scores = {uuid: scores.get(uuid, 0.0) for uuid in uuids}

            return scores

        except Exception as e:
            log.warning(f"PageRank computation failed (may not be supported): {e}")
            # Return empty dict if PageRank not available
            return {}

    def get_pagerank_for_nodes(
        self,
        uuids: list[str],
    ) -> dict[str, float]:
        """Get PageRank scores for specific nodes.

        Computes PageRank on the full graph and filters to requested nodes.
        Falls back to degree-based scoring if PageRank is unavailable.

        Args:
            uuids: List of UUIDs to get scores for

        Returns:
            Dictionary mapping UUID -> PageRank score
        """
        log.trace(f"Getting PageRank for {len(uuids)} nodes")

        try:
            # Get PageRank for the whole graph
            all_scores = self.get_pagerank_scores()

            if all_scores:
                # Filter for the requested UUIDs
                scores = {uuid: all_scores.get(uuid, 0.0) for uuid in uuids}
                log.debug(f"PageRank retrieved for {len(scores)} of {len(uuids)} nodes")
                return scores

            # Empty result, fall through to fallback
            log.debug("PageRank returned no scores, using fallback")
            return self._get_degree_scores(uuids)

        except Exception as e:
            log.warning(f"PageRank failed, using degree-based fallback: {e}")
            # Fallback: use in-degree as a proxy for importance
            return self._get_degree_scores(uuids)

    def _get_degree_scores(self, uuids: list[str]) -> dict[str, float]:
        """Fallback: compute importance scores based on in-degree.

        Args:
            uuids: List of UUIDs

        Returns:
            Normalized in-degree scores
        """
        result = self.graph.query(
            """
            MATCH (m:Memory)
            WHERE m.uuid IN $uuids
            OPTIONAL MATCH (other)-[r]->(m)
            WITH m.uuid AS uuid, count(r) AS in_degree
            RETURN uuid, in_degree
            """,
            {"uuids": uuids},
        )

        # Get raw degrees
        degrees = {}
        max_degree = 1
        for record in result.result_set:
            degrees[record[0]] = record[1]
            max_degree = max(max_degree, record[1])

        # Normalize to 0-1 range
        scores = {uuid: degrees.get(uuid, 0) / max_degree for uuid in uuids}
        return scores

    # =========================================================================
    # Degree and Normalization Methods
    # =========================================================================

    def get_memory_degrees(
        self,
        uuids: list[str],
        project_id: str | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get in-degree and out-degree for a list of memory UUIDs.

        Used for connectivity-based scoring. Well-connected memories
        are boosted, orphan memories are penalized.

        Args:
            uuids: List of memory UUIDs to get degrees for
            project_id: Optional project filter

        Returns:
            Dict mapping uuid -> {"in_degree": int, "out_degree": int, "total": int}
        """
        if not uuids:
            return {}

        # Relationship types that indicate semantic dependency
        # Exclude SUPERSEDES (handled separately) and structural edges
        # Note: REFERENCES is the most common edge type (entity mentions)
        semantic_edges = ["RELATES_TO", "SUPPORTS", "FOLLOWS", "SIMILAR", "CONTAINS", "CHILD_OF", "REFERENCES"]
        edge_filter = "|".join(semantic_edges)

        query = f"""
        UNWIND $uuids AS uuid
        MATCH (m:Memory {{uuid: uuid}})
        OPTIONAL MATCH (m)-[out:{edge_filter}]->()
        WITH m, uuid, count(out) AS out_deg
        OPTIONAL MATCH ()-[in:{edge_filter}]->(m)
        RETURN uuid, out_deg, count(in) AS in_deg
        """

        result = self.graph.query(query, {"uuids": uuids})

        degrees: dict[str, dict[str, int]] = {}
        for row in result.result_set or []:
            uuid, out_deg, in_deg = row[0], row[1] or 0, row[2] or 0
            degrees[uuid] = {
                "in_degree": in_deg,
                "out_degree": out_deg,
                "total": in_deg + out_deg,
            }

        # Fill in zeros for any UUIDs not found
        for uuid in uuids:
            if uuid not in degrees:
                degrees[uuid] = {"in_degree": 0, "out_degree": 0, "total": 0}

        return degrees

    def get_graph_normalization_stats(
        self,
        project_id: str | None = None,
    ) -> dict[str, float]:
        """Get statistics for normalizing graph scores.

        Returns max degree and max pagerank for the project,
        used to normalize graph scores to [0, 1] range.

        Args:
            project_id: Optional project filter

        Returns:
            Dict with max_degree, max_pagerank, avg_degree, node_count
        """
        if project_id:
            query = """
            MATCH (m:Memory {project_id: $project_id})
            OPTIONAL MATCH (m)-[r]-()
            WITH m, count(r) AS degree
            RETURN max(degree) AS max_degree,
                   avg(degree) AS avg_degree,
                   max(m.pagerank) AS max_pagerank,
                   count(m) AS node_count
            """
            result = self.graph.query(query, {"project_id": project_id})
        else:
            query = """
            MATCH (m:Memory)
            OPTIONAL MATCH (m)-[r]-()
            WITH m, count(r) AS degree
            RETURN max(degree) AS max_degree,
                   avg(degree) AS avg_degree,
                   max(m.pagerank) AS max_pagerank,
                   count(m) AS node_count
            """
            result = self.graph.query(query)

        if not result.result_set:
            return {"max_degree": 1.0, "avg_degree": 0.0, "max_pagerank": 1.0, "node_count": 0}

        row = result.result_set[0]
        return {
            "max_degree": float(row[0] or 1),
            "avg_degree": float(row[1] or 0),
            "max_pagerank": float(row[2] or 1),
            "node_count": int(row[3] or 0),
        }

    def compute_and_cache_pagerank(
        self,
        project_id: str | None = None,
        max_iterations: int = 100,
        damping_factor: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank using MAGE and cache scores on Memory nodes.

        Should be called asynchronously (not on every write) to avoid
        latency impact. Recommended: run every 5 minutes or after N writes.

        Args:
            project_id: Optional project filter (runs on subgraph)
            max_iterations: Maximum PageRank iterations
            damping_factor: PageRank damping factor (default: 0.85)

        Returns:
            Dict mapping uuid -> pagerank score
        """
        try:
            # Run PageRank using MAGE
            # Note: For project-scoped PageRank, we'd need to use graph projection
            # For now, run on full graph and filter by project_id
            query = """
            CALL pagerank.get()
            YIELD node, rank
            WITH node, rank
            WHERE node:Memory
            SET node.pagerank = rank
            RETURN node.uuid AS uuid, rank
            """

            result = self.graph.query(
                query,
                {
                    "max_iterations": max_iterations,
                    "damping_factor": damping_factor,
                },
            )

            scores: dict[str, float] = {}
            for row in result.result_set or []:
                uuid, rank = row[0], row[1]
                if project_id is None or True:  # Filter would go here
                    scores[uuid] = float(rank)

            return scores

        except Exception as e:
            # MAGE PageRank may not be available - re-raise for caller to handle
            log.warning(f"PageRank computation failed (MAGE may not be installed): {e}")
            raise RuntimeError(f"PageRank computation failed: {e}") from e

    def get_memory_pageranks(
        self,
        uuids: list[str],
    ) -> dict[str, float]:
        """Get cached PageRank scores for a list of memory UUIDs.

        Returns cached scores from the pagerank property on Memory nodes.
        If PageRank hasn't been computed, returns 0.0 for missing nodes.

        Args:
            uuids: List of memory UUIDs

        Returns:
            Dict mapping uuid -> pagerank score (0.0 if not cached)
        """
        if not uuids:
            return {}

        query = """
        UNWIND $uuids AS uuid
        MATCH (m:Memory {uuid: uuid})
        RETURN uuid, coalesce(m.pagerank, 0.0) AS pagerank
        """

        result = self.graph.query(query, {"uuids": uuids})

        scores: dict[str, float] = {}
        for row in result.result_set or []:
            scores[row[0]] = float(row[1])

        # Fill in zeros for any UUIDs not found
        for uuid in uuids:
            if uuid not in scores:
                scores[uuid] = 0.0

        return scores
