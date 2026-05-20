"""
Graph Traversal Service — Cypher queries for navigating the call graph.

Used by the LLM RAG pipeline to expand context beyond the single function
found by Qdrant. When semantic search returns function X, this service:
  - Finds all functions that X calls (call chain downward)
  - Finds all functions that call X (callers upward)
  - Finds which files are imported by X's file

All functions return plain Python dicts — no Neo4j Record objects leak out.
This keeps the rest of the application independent of the Neo4j driver.
"""

from graph.neo4j_client import run_query
from core.logging import get_logger

logger = get_logger(__name__)


def get_call_chain(function_id: str, depth: int = 3) -> list[dict]:
    """
    Return all functions called by the given function, transitively.
    Traverses CALLS edges outward up to `depth` hops.

    Example depth=3: A → B → C → D  (returns B, C, D; not A itself)

    Args:
        function_id: UUID of the starting :Function node
        depth:       Maximum hops to follow (default 3, max recommended 6)

    Returns:
        List of dicts, ordered by depth ascending:
        [{"id": "...", "name": "hash_password", "file": "utils.py",
          "start_line": 12, "end_line": 25, "depth": 1}]
    """
    results = run_query(
        f"""
        MATCH path = (start:Function {{id: $start_id}})-[:CALLS*1..{depth}]->(callee:Function)
        RETURN
            callee.id         AS id,
            callee.name       AS name,
            callee.file_path  AS file,
            callee.start_line AS start_line,
            callee.end_line   AS end_line,
            length(path)      AS depth
        ORDER BY depth ASC
        """,
        {"start_id": function_id},
    )
    return results


def get_callers(function_id: str, depth: int = 2) -> list[dict]:
    """
    Return all functions that call the given function (reverse traversal).
    Answers: "Who depends on this function? What calls it?"

    Args:
        function_id: UUID of the :Function node
        depth:       Maximum hops upward (default 2)

    Returns:
        List of caller function dicts, ordered by depth ascending.
    """
    results = run_query(
        f"""
        MATCH path = (caller:Function)-[:CALLS*1..{depth}]->(target:Function {{id: $target_id}})
        RETURN
            caller.id         AS id,
            caller.name       AS name,
            caller.file_path  AS file,
            caller.start_line AS start_line,
            caller.end_line   AS end_line,
            length(path)      AS depth
        ORDER BY depth ASC
        """,
        {"target_id": function_id},
    )
    return results


def get_file_dependencies(file_id: str) -> list[dict]:
    """
    Return all files that the given file directly imports.
    Answers: "What does this file depend on?"

    Returns:
        List of dicts for imported files:
        [{"id": "...", "path": "utils/crypto.py", "import_name": "utils.crypto"}]
    """
    return run_query(
        """
        MATCH (source:File {id: $file_id})-[r:IMPORTS]->(target:File)
        RETURN
            target.id       AS id,
            target.path     AS path,
            target.language AS language,
            r.import_name   AS import_name
        """,
        {"file_id": file_id},
    )


def get_file_dependents(file_id: str) -> list[dict]:
    """
    Return all files that import the given file (reverse dependency).
    Answers: "What would break if I change this file?"
    """
    return run_query(
        """
        MATCH (importer:File)-[r:IMPORTS]->(target:File {id: $file_id})
        RETURN
            importer.id   AS id,
            importer.path AS path,
            r.import_name AS import_name
        """,
        {"file_id": file_id},
    )


def get_functions_in_file(file_id: str) -> list[dict]:
    """
    Return all functions/methods defined in a given file, ordered by line number.
    """
    return run_query(
        """
        MATCH (f:File {id: $file_id})-[:CONTAINS]->(func:Function)
        RETURN
            func.id           AS id,
            func.name         AS name,
            func.display_name AS display_name,
            func.start_line   AS start_line,
            func.end_line     AS end_line,
            func.is_method    AS is_method,
            func.docstring    AS docstring
        ORDER BY func.start_line ASC
        """,
        {"file_id": file_id},
    )


def get_repo_graph_for_api(repo_id: str, limit_nodes: int = 500) -> dict:
    """
    Return nodes + edges for a repository in a format ready for the API response.
    Used by GET /graph/{repo_id}.

    Limits results to avoid huge payloads. Returns Function, Class,
    and File nodes for visualization.

    Args:
        repo_id:     UUID of the repo
        limit_nodes: Maximum nodes to return (default 500)

    Returns:
        {"nodes": [...], "edges": [...]}
    """
    node_results = run_query(
        f"""
        MATCH (n {{repo_id: $repo_id}})
        WHERE n:Function OR n:Class OR n:File
        RETURN
            n.id          AS id,
            labels(n)[0]  AS label,
            n.name        AS name,
            n.file_path   AS file,
            n.start_line  AS start_line
        ORDER BY n.file_path, n.start_line
        LIMIT {limit_nodes}
        """,
        {"repo_id": repo_id},
    )

    if not node_results:
        return {"nodes": [], "edges": []}

    node_ids = [r["id"] for r in node_results if r["id"]]

    edge_results = run_query(
        """
        MATCH (source)-[r]->(target)
        WHERE source.repo_id = $repo_id
          AND target.repo_id = $repo_id
          AND type(r) IN ['CALLS', 'IMPORTS', 'CONTAINS', 'INHERITS']
          AND source.id IN $node_ids
          AND target.id IN $node_ids
        RETURN
            source.id AS source,
            target.id AS target,
            type(r)   AS type
        LIMIT 2000
        """,
        {"repo_id": repo_id, "node_ids": node_ids},
    )

    return {"nodes": node_results, "edges": edge_results}


def find_path_between_functions(
    source_id: str,
    target_id: str,
    max_depth: int = 5,
) -> list[dict]:
    """
    Find the shortest call path between two functions using Cypher shortestPath.

    Example:
        find_path_between_functions("login_handler_id", "bcrypt_id")
        Returns: [login_handler, authenticate_user, verify_password, bcrypt_check]

    Args:
        source_id:  UUID of the starting function
        target_id:  UUID of the ending function
        max_depth:  Maximum path length to search

    Returns:
        List of function dicts in path order. Empty list if no path exists.
    """
    return run_query(
        f"""
        MATCH path = shortestPath(
            (source:Function {{id: $source_id}})-[:CALLS*1..{max_depth}]->(target:Function {{id: $target_id}})
        )
        UNWIND nodes(path) AS n
        RETURN
            n.id          AS id,
            n.name        AS name,
            n.file_path   AS file,
            n.start_line  AS start_line
        """,
        {"source_id": source_id, "target_id": target_id},
    )