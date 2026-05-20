"""
Neo4j schema constants — node labels, relationship types, and property names.

Import these constants everywhere instead of hardcoding strings.
This prevents typos like "CALL" instead of "CALLS" from silently
creating a second relationship type and breaking all traversal queries.

Usage:
    from graphs.schema import NodeLabel, RelType, Props

    session.run(
        f"MERGE (f:{NodeLabel.FUNCTION} {{id: $id}})",
        {"id": chunk_id}
    )
"""


class NodeLabel:
    """Node labels — equivalent to SQL table names."""
    REPO     = "Repo"
    FILE     = "File"
    FUNCTION = "Function"
    CLASS    = "Class"


class RelType:
    """Relationship types between nodes."""
    # Structural containment
    CONTAINS_FILE   = "CONTAINS_FILE"    # Repo → File
    CONTAINS        = "CONTAINS"         # File → Function | File → Class
    CONTAINS_METHOD = "CONTAINS_METHOD"  # Class → Function (method)
    DEFINED_IN      = "DEFINED_IN"       # Function | Class → File

    # Code relationships
    CALLS           = "CALLS"            # Function → Function
    IMPORTS         = "IMPORTS"          # File → File
    INHERITS        = "INHERITS"         # Class → Class (parent)
    INHERITS_FROM   = "INHERITS_FROM"
    IMPLEMENTS      = "IMPLEMENTS"
    MIXES_IN        = "MIXES_IN"


class Props:
    """Property key names used on nodes and relationships."""
    # Common to all nodes
    ID           = "id"
    REPO_ID      = "repo_id"
    NAME         = "name"

    # File properties
    PATH         = "path"
    LANGUAGE     = "language"

    # Function / Class properties
    DISPLAY_NAME = "display_name"
    FILE_PATH    = "file_path"
    START_LINE   = "start_line"
    END_LINE     = "end_line"
    IS_METHOD    = "is_method"
    DOCSTRING    = "docstring"

    # Repo properties
    GITHUB_URL   = "github_url"

    # Relationship properties
    CALL_COUNT   = "call_count"
    IMPORT_NAME  = "import_name"


# ── Index creation statements ─────────────────────────────────────────────────
# These create indexes for fast MERGE and MATCH by id and repo_id.
# Without indexes, every MERGE does a full node scan — O(n) instead of O(log n).
# Run once at startup via create_indexes() — safe to call multiple times (IF NOT EXISTS).

INDEX_STATEMENTS = [
    f"CREATE INDEX function_id IF NOT EXISTS FOR (f:{NodeLabel.FUNCTION}) ON (f.id)",
    f"CREATE INDEX function_repo IF NOT EXISTS FOR (f:{NodeLabel.FUNCTION}) ON (f.repo_id)",
    f"CREATE INDEX function_name IF NOT EXISTS FOR (f:{NodeLabel.FUNCTION}) ON (f.name)",
    f"CREATE INDEX class_id IF NOT EXISTS FOR (c:{NodeLabel.CLASS}) ON (c.id)",
    f"CREATE INDEX class_repo IF NOT EXISTS FOR (c:{NodeLabel.CLASS}) ON (c.repo_id)",
    f"CREATE INDEX class_is_abstract IF NOT EXISTS FOR (c:{NodeLabel.CLASS}) ON (c.is_abstract)",
    f"CREATE INDEX class_is_mixin IF NOT EXISTS FOR (c:{NodeLabel.CLASS}) ON (c.is_mixin)",
    f"CREATE INDEX file_id IF NOT EXISTS FOR (f:{NodeLabel.FILE}) ON (f.id)",
    f"CREATE INDEX file_path IF NOT EXISTS FOR (f:{NodeLabel.FILE}) ON (f.path)",
    f"CREATE INDEX file_repo IF NOT EXISTS FOR (f:{NodeLabel.FILE}) ON (f.repo_id)",
    f"CREATE INDEX repo_id IF NOT EXISTS FOR (r:{NodeLabel.REPO}) ON (r.id)",
]


def create_indexes() -> None:
    """
    Create all Neo4j indexes.
    Fully idempotent — IF NOT EXISTS means safe to call on every startup.
    """
    from graph.neo4j_client import run_query
    from core.logging import get_logger
    log = get_logger(__name__)

    for stmt in INDEX_STATEMENTS:
        run_query(stmt)
    log.info("neo4j_indexes_created", count=len(INDEX_STATEMENTS))