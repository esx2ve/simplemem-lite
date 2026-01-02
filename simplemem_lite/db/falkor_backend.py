"""FalkorDB backend implementation for SimpleMem Lite.

FalkorDB is a Redis-based graph database that requires a running server
(typically via Docker). It provides excellent performance and full Cypher support.

This backend is preferred when Docker is available, but falls back to KuzuDB
on systems like HPC clusters where Docker isn't accessible.
"""

from typing import Any

from simplemem_lite.db.graph_protocol import BaseGraphBackend, QueryResult
from simplemem_lite.log_config import get_logger

log = get_logger("falkor")


class FalkorDBBackend(BaseGraphBackend):
    """FalkorDB-based graph backend (requires Docker/Redis).

    Wraps the FalkorDB Python client to implement the GraphBackend protocol.
    Provides full Cypher support including shortestPath() and PageRank.
    """

    def __init__(self, host: str = "localhost", port: int = 6379):
        """Initialize FalkorDB connection.

        Args:
            host: FalkorDB host address
            port: FalkorDB port (default: 6379)
        """
        from falkordb import FalkorDB

        self.host = host
        self.port = port

        log.info(f"Connecting to FalkorDB at {host}:{port}")
        self._db = FalkorDB(host=host, port=port)
        self._graph = self._db.select_graph(self.GRAPH_NAME)

        log.info(f"FalkorDB connected: graph={self.GRAPH_NAME}")

    @property
    def backend_name(self) -> str:
        return "falkordb"

    @property
    def graph(self):
        """Get the underlying FalkorDB graph object.

        Exposed for backward compatibility with existing DatabaseManager code.
        New code should use the query() method instead.
        """
        return self._graph

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> QueryResult:
        """Execute a Cypher query.

        Args:
            cypher: Cypher query string
            params: Optional query parameters

        Returns:
            QueryResult with result_set, header, and stats
        """
        log.trace(f"FalkorDB query: {cypher[:100]}...")

        try:
            if params:
                result = self._graph.query(cypher, params)
            else:
                result = self._graph.query(cypher)

            # Convert FalkorDB result to QueryResult
            result_set = list(result.result_set) if result.result_set else []

            # Extract header - FalkorDB header is list of tuples (type, name)
            header = None
            if hasattr(result, "header") and result.header:
                header = [
                    col[1] if isinstance(col, tuple) else str(col)
                    for col in result.header
                ]

            # FalkorDB provides execution stats
            stats = {
                "backend": "falkordb",
            }

            return QueryResult(
                result_set=result_set,
                header=header,
                stats=stats,
            )

        except Exception as e:
            log.error(f"FalkorDB query failed: {e}")
            log.debug(f"Query was: {cypher}")
            raise

    def health_check(self) -> bool:
        """Check if FalkorDB is healthy.

        Returns:
            True if connection is operational
        """
        try:
            self._graph.query("RETURN 1")
            return True
        except Exception as e:
            log.warning(f"FalkorDB health check failed: {e}")
            return False

    def close(self) -> None:
        """Close the FalkorDB connection."""
        log.info("Closing FalkorDB connection")
        # FalkorDB client doesn't need explicit close
        # Redis connection pool handles cleanup
        pass

    def init_schema(self) -> None:
        """Initialize indexes for efficient lookups.

        FalkorDB is schema-less - we just create indexes.
        Unlike KuzuDB, no table definitions are needed.
        """
        log.info("Creating FalkorDB indexes")

        try:
            # Memory indexes
            self._graph.query("CREATE INDEX FOR (m:Memory) ON (m.uuid)")
            self._graph.query("CREATE INDEX FOR (m:Memory) ON (m.type)")
            self._graph.query("CREATE INDEX FOR (m:Memory) ON (m.session_id)")

            # Entity indexes
            self._graph.query("CREATE INDEX FOR (e:Entity) ON (e.name)")
            self._graph.query("CREATE INDEX FOR (e:Entity) ON (e.type)")

            # CodeChunk indexes
            self._graph.query("CREATE INDEX FOR (c:CodeChunk) ON (c.uuid)")
            self._graph.query("CREATE INDEX FOR (c:CodeChunk) ON (c.filepath)")

            # ProjectIndex index
            self._graph.query("CREATE INDEX FOR (p:ProjectIndex) ON (p.project_root)")

            log.debug("FalkorDB indexes created")

        except Exception as e:
            # Indexes may already exist - this is fine
            log.trace(f"Index creation (may already exist): {e}")

    def reconnect(self) -> bool:
        """Attempt to reconnect to FalkorDB.

        Returns:
            True if reconnection successful
        """
        from falkordb import FalkorDB

        try:
            log.info("Attempting to reconnect to FalkorDB...")
            self._db = FalkorDB(host=self.host, port=self.port)
            self._graph = self._db.select_graph(self.GRAPH_NAME)

            # Verify connection
            self._graph.query("RETURN 1")
            log.info("FalkorDB reconnection successful")
            return True

        except Exception as e:
            log.error(f"FalkorDB reconnection failed: {e}")
            return False

    def _translate_cypher(self, cypher: str) -> str:
        """FalkorDB uses standard Cypher - no translation needed.

        Args:
            cypher: Cypher query

        Returns:
            Same query unchanged
        """
        return cypher

    # ══════════════════════════════════════════════════════════════════════════════════
    # FALKORDB-SPECIFIC FEATURES
    # ══════════════════════════════════════════════════════════════════════════════════

    def get_pagerank_scores(
        self,
        max_iterations: int = 20,
        damping_factor: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank scores using FalkorDB's built-in algorithm.

        Args:
            max_iterations: PageRank iterations
            damping_factor: PageRank damping factor

        Returns:
            Dictionary mapping UUID -> PageRank score
        """
        log.trace("Computing PageRank scores via FalkorDB")

        try:
            result = self._graph.query(
                """
                CALL algo.pageRank('Memory', 'RELATES_TO', {
                    maxIterations: $max_iter,
                    dampingFactor: $damping
                })
                YIELD node, score
                MATCH (m:Memory) WHERE id(m) = id(node)
                RETURN m.uuid AS uuid, score
                """,
                {
                    "max_iter": max_iterations,
                    "damping": damping_factor,
                },
            )

            scores = {}
            for record in result.result_set:
                scores[record[0]] = record[1]

            log.debug(f"PageRank computed for {len(scores)} nodes")
            return scores

        except Exception as e:
            log.warning(f"PageRank computation failed: {e}")
            return {}


def create_falkor_backend(
    host: str = "localhost",
    port: int = 6379,
) -> FalkorDBBackend:
    """Factory function to create and initialize a FalkorDB backend.

    Args:
        host: FalkorDB host address
        port: FalkorDB port

    Returns:
        Initialized FalkorDBBackend instance

    Raises:
        ConnectionError: If FalkorDB is not reachable
    """
    backend = FalkorDBBackend(host, port)
    backend.init_schema()
    return backend


def is_falkordb_available(host: str = "localhost", port: int = 6379) -> bool:
    """Check if FalkorDB is available at the given address.

    Args:
        host: FalkorDB host address
        port: FalkorDB port

    Returns:
        True if FalkorDB is reachable and healthy
    """
    try:
        from falkordb import FalkorDB

        db = FalkorDB(host=host, port=port)
        graph = db.select_graph("simplemem_test")
        graph.query("RETURN 1")
        return True
    except Exception as e:
        log.debug(f"FalkorDB not available at {host}:{port}: {e}")
        return False
