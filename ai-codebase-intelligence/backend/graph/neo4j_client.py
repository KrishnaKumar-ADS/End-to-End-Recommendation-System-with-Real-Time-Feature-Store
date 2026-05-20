"""
Neo4j client — manages the driver connection and provides session helpers.

Usage:
    from graphs.neo4j_client import get_driver, run_query, get_session

    # Run a simple read query
    results = run_query(
        "MATCH (f:Function {name: $name}) RETURN f",
        {"name": "verify_password"}
    )

    # Use a session directly for write operations
    with get_session() as session:
        session.run("MERGE (:Repo {id: $id})", {"id": repo_id})
"""

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError
from typing import Any
from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Singleton driver ──────────────────────────────────────────────────────────

_driver: Driver | None = None


def get_driver() -> Driver:
    """
    Return a singleton Neo4j driver.
    Creates the connection on first call and reuses it for the app lifetime.
    Thread-safe — the Neo4j driver manages its own connection pool internally.
    """
    global _driver
    if _driver is None:
        try:
            _driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_timeout=30,
            )
            # Immediately verify the connection works — fail fast at startup
            _driver.verify_connectivity()
            logger.info(
                "neo4j_driver_created",
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
            )
        except ServiceUnavailable as e:
            logger.error(
                "neo4j_connection_failed",
                uri=settings.neo4j_uri,
                error=str(e),
            )
            raise
        except AuthError as e:
            logger.error(
                "neo4j_auth_failed",
                user=settings.neo4j_user,
                error=str(e),
            )
            raise
    return _driver


def get_session() -> Session:
    """
    Return a new Neo4j session from the driver pool.
    Always use as a context manager to ensure it is closed properly:

        with get_session() as session:
            session.run(...)
    """
    return get_driver().session()


def close_driver() -> None:
    """
    Close the driver and release all pooled connections.
    Called during application shutdown via the FastAPI lifespan.
    """
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        logger.info("neo4j_driver_closed")


def run_query(
    cypher: str,
    params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict]:
    """
    Convenience function: run a Cypher query and return results as a list of dicts.

    Always use $param placeholders — never f-strings — to prevent Cypher injection.

    Example:
        results = run_query(
            "MATCH (f:Function {repo_id: $repo_id}) RETURN f.name AS name",
            {"repo_id": "uuid..."}
        )
        # returns: [{"name": "verify_password"}, {"name": "hash_password"}, ...]

    Args:
        cypher: Cypher query string with $param placeholders
        params: Parameter dict

    Returns:
        List of result records as plain Python dicts
    """
    merged_params: dict[str, Any] = {}
    if params:
        merged_params.update(params)
    if kwargs:
        merged_params.update(kwargs)

    with get_session() as session:
        result = session.run(cypher, merged_params)
        return [record.data() for record in result]


def ping() -> bool:
    """
    Return True if Neo4j is reachable, False otherwise.
    Used by the GET /health endpoint.
    """
    try:
        get_driver().verify_connectivity()
        return True
    except Exception:
        return False


def get_database_stats() -> dict:
    """
    Return node and relationship counts for the whole database.
    Used by /health and graph debug endpoints.
    """
    try:
        node_results = run_query("""
            MATCH (n)
            RETURN labels(n)[0] AS label, COUNT(n) AS count
            ORDER BY count DESC
        """)
        node_counts = {r["label"]: r["count"] for r in node_results if r["label"]}

        rel_results = run_query("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, COUNT(r) AS count
            ORDER BY count DESC
        """)
        rel_counts = {r["rel_type"]: r["count"] for r in rel_results}

        return {
            "node_counts": node_counts,
            "relationship_counts": rel_counts,
            "total_nodes": sum(node_counts.values()),
            "total_relationships": sum(rel_counts.values()),
        }
    except Exception as e:
        logger.warning("neo4j_stats_failed", error=str(e))
        return {}