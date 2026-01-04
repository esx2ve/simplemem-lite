"""Memgraph backend implementation for SimpleMem Lite.

Memgraph is a high-performance, native C++ graph database with full Cypher support.
It uses the Bolt protocol and is compatible with the neo4j Python driver.

This backend replaces FalkorDB to eliminate SIGSEGV crashes in Schema_AddNodeToIndex
that occur after DETACH DELETE operations. Memgraph's native engine doesn't have
this index corruption issue.

Key differences from FalkorDB:
- Protocol: Bolt (7687) instead of Redis RESP (6379)
- Driver: neo4j instead of falkordb
- Index syntax: CREATE INDEX ON :Label(prop) instead of CREATE INDEX FOR (n:Label) ON (n.prop)
- PageRank: pagerank.get() via MAGE instead of algo.pageRank()
- No reinit_code_chunk_indexes() needed - native engine handles DETACH DELETE safely
"""

from typing import Any

from simplemem_lite.db.graph_protocol import BaseGraphBackend, QueryResult
from simplemem_lite.log_config import get_logger

log = get_logger("memgraph")


class MemgraphBackend(BaseGraphBackend):
    """Memgraph-based graph backend using Bolt protocol.

    Wraps the neo4j Python driver to implement the GraphBackend protocol.
    Provides full Cypher support including graph algorithms via MAGE.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        username: str = "",
        password: str = "",
        connection_timeout: float = 30.0,
    ):
        """Initialize Memgraph connection.

        Args:
            host: Memgraph host address
            port: Memgraph Bolt port (default: 7687)
            username: Optional username for authentication
            password: Optional password for authentication
            connection_timeout: Connection timeout in seconds (default: 30.0)
        """
        from neo4j import GraphDatabase

        self.host = host
        self.port = port
        self.username = username
        self.password = password

        # Build Bolt URI
        self._uri = f"bolt://{host}:{port}"

        log.info(f"Connecting to Memgraph at {self._uri} (auth={'yes' if password else 'no'})")

        # Create driver with optional auth and timeout
        if username or password:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(username, password),
                connection_timeout=connection_timeout,
            )
        else:
            self._driver = GraphDatabase.driver(
                self._uri,
                connection_timeout=connection_timeout,
            )

        # Verify connection
        self._driver.verify_connectivity()
        log.info(f"Memgraph connected: {self._uri}")

    @property
    def backend_name(self) -> str:
        return "memgraph"

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> QueryResult:
        """Execute a Cypher query.

        Args:
            cypher: Cypher query string
            params: Optional query parameters

        Returns:
            QueryResult with result_set, header, and stats
        """
        log.trace(f"Memgraph query: {cypher[:100]}...")

        try:
            with self._driver.session() as session:
                if params:
                    result = session.run(cypher, params)
                else:
                    result = session.run(cypher)

                # Consume all records
                records = list(result)

                # Extract result set as list of lists
                result_set = []
                header = None

                if records:
                    # Get keys from first record
                    header = list(records[0].keys())

                    # Convert records to list of values
                    for record in records:
                        result_set.append(list(record.values()))

                # Build stats
                stats = {
                    "backend": "memgraph",
                }

                return QueryResult(
                    result_set=result_set,
                    header=header,
                    stats=stats,
                )

        except Exception as e:
            log.error(f"Memgraph query failed: {e}")
            log.debug(f"Query was: {cypher}")
            raise

    def health_check(self) -> bool:
        """Check if Memgraph is healthy.

        Returns:
            True if connection is operational
        """
        try:
            with self._driver.session() as session:
                session.run("RETURN 1").consume()
            return True
        except Exception as e:
            log.warning(f"Memgraph health check failed: {e}")
            return False

    def close(self) -> None:
        """Close the Memgraph connection."""
        log.info("Closing Memgraph connection")
        if self._driver:
            self._driver.close()

    def init_schema(self) -> None:
        """Initialize indexes for efficient lookups.

        Memgraph uses different index syntax than FalkorDB:
        - FalkorDB: CREATE INDEX FOR (n:Label) ON (n.prop)
        - Memgraph: CREATE INDEX ON :Label(prop)
        """
        log.info("Creating Memgraph indexes")

        # Memgraph index syntax
        indexes = [
            # Memory indexes
            "CREATE INDEX ON :Memory(uuid)",
            "CREATE INDEX ON :Memory(type)",
            "CREATE INDEX ON :Memory(session_id)",
            # Entity indexes
            "CREATE INDEX ON :Entity(name)",
            "CREATE INDEX ON :Entity(type)",
            # CodeChunk indexes
            "CREATE INDEX ON :CodeChunk(uuid)",
            "CREATE INDEX ON :CodeChunk(filepath)",
            # ProjectIndex index
            "CREATE INDEX ON :ProjectIndex(project_root)",
        ]

        for index_query in indexes:
            try:
                self.query(index_query)
            except Exception as e:
                # Index may already exist - Memgraph returns error for duplicate indexes
                error_msg = str(e).lower()
                if "already exists" in error_msg or "index already" in error_msg:
                    log.trace(f"Index already exists: {index_query}")
                else:
                    log.warning(f"Index creation issue: {e}")

        log.debug("Memgraph indexes created")

    def reinit_code_chunk_indexes(self) -> None:
        """No-op for Memgraph - native engine handles DETACH DELETE safely.

        Unlike FalkorDB, Memgraph's C++ engine doesn't have index corruption
        issues after mass deletion of nodes. This method exists for protocol
        compatibility but does nothing.
        """
        log.trace("reinit_code_chunk_indexes called but not needed for Memgraph")
        pass

    def reconnect(self) -> bool:
        """Attempt to reconnect to Memgraph.

        Returns:
            True if reconnection successful
        """
        from neo4j import GraphDatabase

        try:
            log.info("Attempting to reconnect to Memgraph...")

            # Close existing driver
            if self._driver:
                try:
                    self._driver.close()
                except Exception:
                    pass

            # Create new driver
            if self.username or self.password:
                self._driver = GraphDatabase.driver(
                    self._uri,
                    auth=(self.username, self.password),
                )
            else:
                self._driver = GraphDatabase.driver(self._uri)

            # Verify connection
            self._driver.verify_connectivity()
            log.info("Memgraph reconnection successful")
            return True

        except Exception as e:
            log.error(f"Memgraph reconnection failed: {e}")
            return False

    def _translate_cypher(self, cypher: str) -> str:
        """Memgraph uses standard Cypher - no translation needed.

        Args:
            cypher: Cypher query

        Returns:
            Same query unchanged
        """
        return cypher

    # ══════════════════════════════════════════════════════════════════════════════════
    # MEMGRAPH-SPECIFIC FEATURES (MAGE Algorithms)
    # ══════════════════════════════════════════════════════════════════════════════════

    def get_pagerank_scores(
        self,
        max_iterations: int = 20,
        damping_factor: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank scores using Memgraph's MAGE library.

        MAGE (Memgraph Advanced Graph Extensions) provides graph algorithms
        including PageRank via the pagerank module.

        Args:
            max_iterations: PageRank iterations
            damping_factor: PageRank damping factor

        Returns:
            Dictionary mapping UUID -> PageRank score
        """
        log.trace("Computing PageRank scores via Memgraph MAGE")

        try:
            # MAGE PageRank syntax
            result = self.query(
                """
                CALL pagerank.get()
                YIELD node, rank
                WHERE node:Memory
                RETURN node.uuid AS uuid, rank AS score
                """,
            )

            scores = {}
            for record in result.result_set:
                if record[0]:  # uuid not null
                    scores[record[0]] = record[1]

            log.debug(f"PageRank computed for {len(scores)} nodes")
            return scores

        except Exception as e:
            log.warning(f"PageRank computation failed: {e}")
            # MAGE might not be loaded - return empty
            return {}


def create_memgraph_backend(
    host: str = "localhost",
    port: int = 7687,
    username: str = "",
    password: str = "",
) -> MemgraphBackend:
    """Factory function to create and initialize a Memgraph backend.

    Args:
        host: Memgraph host address
        port: Memgraph Bolt port (default: 7687)
        username: Optional username for authentication
        password: Optional password for authentication

    Returns:
        Initialized MemgraphBackend instance

    Raises:
        ConnectionError: If Memgraph is not reachable
    """
    backend = MemgraphBackend(host, port, username, password)
    backend.init_schema()
    return backend


def is_memgraph_available(
    host: str = "localhost",
    port: int = 7687,
    username: str = "",
    password: str = "",
    timeout: float = 2.0,
) -> bool:
    """Check if Memgraph is available at the given address.

    Uses socket-level pre-check before neo4j driver to avoid long hangs.

    Args:
        host: Memgraph host address
        port: Memgraph Bolt port
        username: Optional username for authentication
        password: Optional password for authentication
        timeout: Connection timeout in seconds (default: 2.0)

    Returns:
        True if Memgraph is reachable and healthy
    """
    import socket

    # Quick socket-level check first (avoids neo4j driver's long timeout)
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
    except (socket.timeout, socket.error, OSError) as e:
        log.debug(f"Memgraph socket check failed at {host}:{port}: {e}")
        return False

    try:
        from neo4j import GraphDatabase

        uri = f"bolt://{host}:{port}"
        if username or password:
            driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                connection_timeout=timeout,
            )
        else:
            driver = GraphDatabase.driver(uri, connection_timeout=timeout)

        driver.verify_connectivity()

        with driver.session() as session:
            session.run("RETURN 1").consume()

        driver.close()
        return True

    except Exception as e:
        log.debug(f"Memgraph not available at {host}:{port}: {e}")
        return False
