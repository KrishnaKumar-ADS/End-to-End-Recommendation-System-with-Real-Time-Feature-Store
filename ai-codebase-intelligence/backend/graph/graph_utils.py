"""
Graph Utilities — reusable Neo4j traversal and analysis tools.

All functions accept a repo_id to scope queries to one repository.
All functions use the existing Neo4j client from graph/neo4j_client.py.
No new database connections are opened in this module.

Available utilities:
  get_subgraph()          — N-hop neighbourhood of any node
  bfs_expand()            — breadth-first expansion with depth limit
  find_shortest_path()    — Dijkstra shortest path between two nodes
  get_ancestors()         — all INHERITS_FROM ancestors of a class
  get_descendants()       — all direct + transitive subclasses
  reverse_call_lookup()   — all functions that call a given function
  get_graph_metrics()     — density, top callers, top inherited classes
  get_all_class_nodes()   — full class list with hierarchy metadata
  get_full_hierarchy_tree() — tree-structured JSON for frontend rendering
"""

from graph.neo4j_client import run_query
from core.logging import get_logger

logger = get_logger(__name__)


# ── Subgraph Extraction ────────────────────────────────────────────────────────

def get_subgraph(
    repo_id: str,
    node_id: str,
    depth: int = 2,
    relationship_types: list[str] | None = None,
) -> dict:
    """
    Return the N-hop neighbourhood around a single node.

    Traverses outbound AND inbound edges up to `depth` hops.
    Returns the same nodes+edges format as GET /graph/{repo_id}
    so the frontend D3.js visualisation can render it without changes.

    Args:
        repo_id:            Repository UUID (scope filter)
        node_id:            UUID of the centre node
        depth:              Number of hops to expand (default 2, max 5)
        relationship_types: If provided, only traverse these rel types.
                            e.g. ["CALLS", "INHERITS_FROM"]
                            If None, traverses all relationship types.

    Returns:
        {"nodes": [...], "edges": [...]}

    Example:
        get_subgraph(repo_id, func_id, depth=2, relationship_types=["CALLS"])
        → All functions that call this function, and functions they call.
    """
    depth = min(depth, 5)  # cap at 5 hops to prevent runaway queries

    if relationship_types:
        rel_filter = "|".join(relationship_types)
        rel_pattern = f"[r:{rel_filter}*1..{depth}]"
    else:
        rel_pattern = f"[r*1..{depth}]"

    # Find the centre node first
    centre_query = """
    MATCH (centre {id: $node_id, repo_id: $repo_id})
    RETURN centre, labels(centre) AS labels
    """
    centre_rows = run_query(centre_query, node_id=node_id, repo_id=repo_id)

    if not centre_rows:
        logger.warning("subgraph_centre_not_found", node_id=node_id, repo_id=repo_id)
        return {"nodes": [], "edges": []}

    # Expand N hops in both directions
    expand_query = f"""
    MATCH (centre {{id: $node_id, repo_id: $repo_id}})
    CALL {{
      WITH centre
      MATCH (centre)-{rel_pattern}-(neighbour)
      WHERE neighbour.repo_id = $repo_id
      RETURN neighbour, labels(neighbour) AS nb_labels
      UNION
      MATCH (centre)-{rel_pattern}-(neighbour)
      WHERE neighbour.repo_id = $repo_id
      RETURN neighbour, labels(neighbour) AS nb_labels
    }}
    RETURN DISTINCT neighbour, nb_labels
    """

    neighbour_rows = run_query(expand_query, node_id=node_id, repo_id=repo_id)

    # Build node list
    seen_ids: set[str] = set()
    nodes = []

    # Add centre node
    centre_data = centre_rows[0]
    centre_labels = centre_data.get("labels", [])
    centre_node = dict(centre_data.get("centre", {}))
    centre_node["_labels"] = centre_labels
    centre_node["_is_centre"] = True
    nodes.append(centre_node)
    seen_ids.add(node_id)

    for row in neighbour_rows:
        nb = dict(row.get("neighbour", {}))
        nb_id = nb.get("id")
        if nb_id and nb_id not in seen_ids:
            nb["_labels"] = row.get("nb_labels", [])
            nodes.append(nb)
            seen_ids.add(nb_id)

    # Fetch all edges between the collected nodes
    edges = _get_edges_between(repo_id, list(seen_ids))

    return {"nodes": nodes, "edges": edges}


def bfs_expand(
    repo_id: str,
    start_node_id: str,
    relationship_type: str,
    direction: str = "outbound",
    max_depth: int = 3,
) -> list[dict]:
    """
    Breadth-first expansion from a start node, following one relationship type.

    Returns a flat list of nodes in BFS order, with a `depth` field added.
    Does NOT return edges — use get_subgraph() if you need edges too.

    Args:
        repo_id:           Repository UUID
        start_node_id:     UUID of the node to start from
        relationship_type: Relationship to follow, e.g. "CALLS", "INHERITS_FROM"
        direction:         "outbound" (→), "inbound" (←), or "both" (-)
        max_depth:         Maximum BFS depth (default 3, max 6)

    Returns:
        List of node dicts, each with an added "depth" field.

    Example:
        bfs_expand(repo_id, login_func_id, "CALLS", "outbound", 3)
        → All functions reachable from login() within 3 call hops.
    """
    max_depth = min(max_depth, 6)

    if direction == "outbound":
        rel_pattern = f"-[:{relationship_type}*1..{max_depth}]->"
    elif direction == "inbound":
        rel_pattern = f"<-[:{relationship_type}*1..{max_depth}]-"
    else:
        rel_pattern = f"-[:{relationship_type}*1..{max_depth}]-"

    query = f"""
    MATCH (start {{id: $start_id, repo_id: $repo_id}})
    MATCH p = (start){rel_pattern}(target)
    WHERE target.repo_id = $repo_id AND target.id <> $start_id
    RETURN DISTINCT target, labels(target) AS target_labels,
           length(p) AS depth
    ORDER BY depth
    """

    rows = run_query(query, start_id=start_node_id, repo_id=repo_id)

    results = []
    for row in rows:
        node = dict(row.get("target", {}))
        node["depth"] = row.get("depth", 0)
        node["_labels"] = row.get("target_labels", [])
        results.append(node)

    return results


# ── Path Finding ───────────────────────────────────────────────────────────────

def find_shortest_path(
    repo_id: str,
    source_id: str,
    target_id: str,
    relationship_types: list[str] | None = None,
    max_depth: int = 8,
) -> list[dict]:
    """
    Find the shortest path between two nodes in the graph.

    Uses Neo4j's shortestPath() Cypher function.
    Returns a list of nodes in path order, or empty list if no path exists.

    Args:
        repo_id:            Repository UUID
        source_id:          UUID of the start node
        target_id:          UUID of the end node
        relationship_types: If provided, only follow these relationship types.
                            Default: ["CALLS", "INHERITS_FROM", "IMPORTS"]
        max_depth:          Maximum path length to search (default 8)

    Returns:
        List of node dicts in path order (source first, target last).
        Empty list if no path exists.

    Example:
        find_shortest_path(repo_id, login_id, verify_pw_id)
        → [login_controller, auth_service, verify_password]
    """
    if relationship_types is None:
        relationship_types = ["CALLS", "INHERITS_FROM", "IMPORTS"]

    rel_filter = "|".join(relationship_types)

    query = f"""
    MATCH (source {{id: $source_id, repo_id: $repo_id}}),
          (target {{id: $target_id, repo_id: $repo_id}})
    MATCH p = shortestPath(
        (source)-[:{rel_filter}*1..{max_depth}]->(target)
    )
    RETURN [n IN nodes(p) | {{
        id:           n.id,
        name:         n.name,
        display_name: n.display_name,
        file_path:    n.file_path,
        labels:       labels(n)
    }}] AS path_nodes,
    [r IN relationships(p) | type(r)] AS rel_types,
    length(p) AS path_length
    """

    rows = run_query(query, source_id=source_id, target_id=target_id, repo_id=repo_id)

    if not rows:
        return []

    row = rows[0]
    path_nodes = row.get("path_nodes", [])
    rel_types = row.get("rel_types", [])

    # Annotate each node with the edge type that leads to the next node
    annotated = []
    for i, node in enumerate(path_nodes):
        n = dict(node)
        if i < len(rel_types):
            n["next_edge_type"] = rel_types[i]
        annotated.append(n)

    return annotated


# ── Class Hierarchy Traversal ──────────────────────────────────────────────────

def get_ancestors(
    repo_id: str,
    class_id: str,
    include_external: bool = False,
) -> list[dict]:
    """
    Return all ancestor classes (direct + transitive) for a given class.

    Traverses INHERITS_FROM and IMPLEMENTS edges upward.
    Results are ordered from nearest ancestor (depth=1) to root (highest depth).

    Args:
        repo_id:          Repository UUID
        class_id:         UUID of the class node
        include_external: If True, include base classes not in this repo
                          (e.g. Django's Model from an installed package)
                          Default False — only return classes in this repo.

    Returns:
        List of class dicts, each with "depth" and "edge_type" fields.

    Example:
        get_ancestors(repo_id, golden_retriever_id)
        → [
            {"name": "Dog",    "depth": 1, "edge_type": "INHERITS_FROM"},
            {"name": "Mammal", "depth": 2, "edge_type": "INHERITS_FROM"},
            {"name": "Animal", "depth": 3, "edge_type": "INHERITS_FROM"},
          ]
    """
    repo_filter = "AND ancestor.repo_id = $repo_id" if not include_external else ""

    query = f"""
    MATCH (child:Class {{id: $class_id, repo_id: $repo_id}})
    MATCH p = (child)-[:INHERITS_FROM|IMPLEMENTS*1..10]->(ancestor:Class)
    WHERE ancestor.id <> $class_id {repo_filter}
    RETURN DISTINCT ancestor,
           labels(ancestor)              AS labels,
           length(p)                     AS depth,
           type(relationships(p)[-1])    AS edge_type
    ORDER BY depth
    """

    rows = run_query(query, class_id=class_id, repo_id=repo_id)

    results = []
    for row in rows:
        node = dict(row.get("ancestor", {}))
        node["depth"] = row.get("depth", 0)
        node["edge_type"] = row.get("edge_type", "INHERITS_FROM")
        node["_labels"] = row.get("labels", ["Class"])
        results.append(node)

    return results


def get_descendants(
    repo_id: str,
    class_id: str,
) -> list[dict]:
    """
    Return all descendant classes (direct subclasses + their subclasses).

    Traverses INHERITS_FROM edges DOWNWARD — opposite of get_ancestors().
    Results are ordered from closest descendant (depth=1) to deepest leaf.

    Args:
        repo_id:  Repository UUID
        class_id: UUID of the base class node

    Returns:
        List of class dicts, each with "depth" and "is_direct" fields.

    Example:
        get_descendants(repo_id, animal_id)
        → [
            {"name": "Mammal",          "depth": 1, "is_direct": True},
            {"name": "Dog",             "depth": 2, "is_direct": False},
            {"name": "GoldenRetriever", "depth": 3, "is_direct": False},
          ]
    """
    query = """
    MATCH (base:Class {id: $class_id, repo_id: $repo_id})
    MATCH p = (descendant:Class)-[:INHERITS_FROM*1..10]->(base)
    WHERE descendant.id <> $class_id AND descendant.repo_id = $repo_id
    RETURN DISTINCT descendant,
           labels(descendant)  AS labels,
           length(p)           AS depth,
           length(p) = 1       AS is_direct
    ORDER BY depth
    """

    rows = run_query(query, class_id=class_id, repo_id=repo_id)

    results = []
    for row in rows:
        node = dict(row.get("descendant", {}))
        node["depth"] = row.get("depth", 0)
        node["is_direct"] = row.get("is_direct", False)
        node["_labels"] = row.get("labels", ["Class"])
        results.append(node)

    return results


# ── Call Graph Utilities ───────────────────────────────────────────────────────

def reverse_call_lookup(
    repo_id: str,
    function_id: str,
    depth: int = 1,
) -> list[dict]:
    """
    Find all functions that call a given function.

    This is the reverse of the call chain — "who calls me?"
    Useful for impact analysis: if I change verify_password(), who breaks?

    Args:
        repo_id:     Repository UUID
        function_id: UUID of the target function
        depth:       How many reverse hops to trace (default 1 = direct callers only)

    Returns:
        List of function node dicts, each with "depth" and "call_chain" fields.

    Example:
        reverse_call_lookup(repo_id, verify_password_id, depth=2)
        → [
            {"name": "authenticate_user",  "depth": 1, "call_chain": ["authenticate_user", "verify_password"]},
            {"name": "login_controller",   "depth": 2, "call_chain": ["login_controller", "authenticate_user", "verify_password"]},
          ]
    """
    depth = min(depth, 5)

    query = f"""
    MATCH (target:Function {{id: $func_id, repo_id: $repo_id}})
    MATCH p = (caller:Function)-[:CALLS*1..{depth}]->(target)
    WHERE caller.repo_id = $repo_id AND caller.id <> $func_id
    RETURN DISTINCT caller,
           labels(caller)                                  AS labels,
           length(p)                                       AS depth,
           [n IN nodes(p) | n.name]                        AS call_chain
    ORDER BY depth, caller.name
    """

    rows = run_query(query, func_id=function_id, repo_id=repo_id)

    results = []
    for row in rows:
        node = dict(row.get("caller", {}))
        node["depth"] = row.get("depth", 1)
        node["call_chain"] = row.get("call_chain", [])
        node["_labels"] = row.get("labels", ["Function"])
        results.append(node)

    return results


# ── Graph Metrics ──────────────────────────────────────────────────────────────

def get_graph_metrics(repo_id: str) -> dict:
    """
    Compute aggregate metrics for the repository's knowledge graph.

    Metrics computed:
      - total_nodes:          Total number of all node types
      - total_edges:          Total number of all relationship types
      - function_count:       Number of :Function nodes
      - class_count:          Number of :Class nodes
      - file_count:           Number of :File nodes
      - calls_edges:          Number of CALLS relationships
      - inherits_edges:       Number of INHERITS_FROM relationships
      - imports_edges:        Number of IMPORTS relationships
      - graph_density:        edges / (nodes * (nodes - 1)) — 0.0 to 1.0
      - top_callers:          Top 5 functions by outgoing CALLS count
      - top_callees:          Top 5 functions by incoming CALLS count
      - most_inherited:       Top 5 classes by number of direct subclasses
      - deepest_hierarchy:    Maximum inheritance depth in the repo
      - abstract_class_count: Number of abstract classes
      - mixin_class_count:    Number of mixin classes

    Returns:
        Dict with all the above keys.

    Example:
        get_graph_metrics(repo_id)
        → {
            "total_nodes": 1259,
            "total_edges": 3847,
            "graph_density": 0.0024,
            "top_callers": [{"name": "dispatch_request", "calls_out": 42}, ...],
            "most_inherited": [{"name": "BaseModel", "subclass_count": 17}, ...],
            ...
          }
    """
    # --- Basic counts ---
    counts_query = """
    MATCH (n {repo_id: $repo_id})
    OPTIONAL MATCH (n)-[r]->()
    RETURN
      COUNT(DISTINCT n) AS total_nodes,
      COUNT(r)          AS total_edges,
      SUM(CASE WHEN 'Function' IN labels(n) THEN 1 ELSE 0 END) AS function_count,
      SUM(CASE WHEN 'Class'    IN labels(n) THEN 1 ELSE 0 END) AS class_count,
      SUM(CASE WHEN 'File'     IN labels(n) THEN 1 ELSE 0 END) AS file_count
    """
    counts_rows = run_query(counts_query, repo_id=repo_id)
    counts = counts_rows[0] if counts_rows else {}

    total_nodes = counts.get("total_nodes", 0)
    total_edges = counts.get("total_edges", 0)
    density = (
        total_edges / (total_nodes * (total_nodes - 1))
        if total_nodes > 1 else 0.0
    )

    # --- Edge type counts ---
    edge_counts_query = """
    MATCH (n {repo_id: $repo_id})-[r]->()
    RETURN type(r) AS rel_type, COUNT(r) AS cnt
    ORDER BY cnt DESC
    """
    edge_rows = run_query(edge_counts_query, repo_id=repo_id)
    edge_counts = {row.get("rel_type"): row.get("cnt", 0) for row in edge_rows}

    # --- Top callers (functions with most outgoing CALLS) ---
    top_callers_query = """
    MATCH (f:Function {repo_id: $repo_id})-[:CALLS]->(g:Function)
    RETURN f.name AS name, f.id AS id, f.file_path AS file_path,
           COUNT(g) AS calls_out
    ORDER BY calls_out DESC LIMIT 5
    """
    top_callers = run_query(top_callers_query, repo_id=repo_id)

    # --- Top callees (functions called by the most callers) ---
    top_callees_query = """
    MATCH (f:Function {repo_id: $repo_id})<-[:CALLS]-(g:Function)
    RETURN f.name AS name, f.id AS id, f.file_path AS file_path,
           COUNT(g) AS calls_in
    ORDER BY calls_in DESC LIMIT 5
    """
    top_callees = run_query(top_callees_query, repo_id=repo_id)

    # --- Most inherited classes ---
    most_inherited_query = """
    MATCH (child:Class {repo_id: $repo_id})-[:INHERITS_FROM]->(parent:Class)
    RETURN parent.name AS name, parent.id AS id, parent.file_path AS file_path,
           COUNT(child) AS subclass_count
    ORDER BY subclass_count DESC LIMIT 5
    """
    most_inherited = run_query(most_inherited_query, repo_id=repo_id)

    # --- Deepest inheritance chain ---
    deepest_query = """
    MATCH p = (:Class {repo_id: $repo_id})-[:INHERITS_FROM*1..20]->(:Class {repo_id: $repo_id})
    RETURN MAX(length(p)) AS max_depth
    """
    deepest_rows = run_query(deepest_query, repo_id=repo_id)
    max_hierarchy_depth = deepest_rows[0].get("max_depth", 0) if deepest_rows else 0

    # --- Abstract and mixin class counts ---
    class_flag_query = """
    MATCH (c:Class {repo_id: $repo_id})
    RETURN
      SUM(CASE WHEN c.is_abstract = true THEN 1 ELSE 0 END) AS abstract_count,
      SUM(CASE WHEN c.is_mixin   = true  THEN 1 ELSE 0 END) AS mixin_count
    """
    class_flag_rows = run_query(class_flag_query, repo_id=repo_id)
    class_flags = class_flag_rows[0] if class_flag_rows else {}

    return {
        "repo_id":             repo_id,
        "total_nodes":         total_nodes,
        "total_edges":         total_edges,
        "function_count":      counts.get("function_count", 0),
        "class_count":         counts.get("class_count", 0),
        "file_count":          counts.get("file_count", 0),
        "calls_edges":         edge_counts.get("CALLS", 0),
        "inherits_edges":      edge_counts.get("INHERITS_FROM", 0),
        "implements_edges":    edge_counts.get("IMPLEMENTS", 0),
        "mixes_in_edges":      edge_counts.get("MIXES_IN", 0),
        "imports_edges":       edge_counts.get("IMPORTS", 0),
        "graph_density":       round(density, 6),
        "top_callers":         [dict(r) for r in top_callers],
        "top_callees":         [dict(r) for r in top_callees],
        "most_inherited":      [dict(r) for r in most_inherited],
        "deepest_hierarchy":   max_hierarchy_depth or 0,
        "abstract_class_count": class_flags.get("abstract_count", 0),
        "mixin_class_count":    class_flags.get("mixin_count", 0),
    }


# ── Hierarchy Tree Builder ─────────────────────────────────────────────────────

def get_full_hierarchy_tree(repo_id: str) -> list[dict]:
    """
    Return the complete class hierarchy as a tree-structured list.

    Each entry represents one class with:
      - id, name, file_path, is_abstract, is_mixin
      - parents: list of immediate parent class names
      - children: list of immediate child class names
      - mro_list: MRO stored on the node (set in Week 5 Day 2)

    The result is a flat list — the frontend D3.js code converts it to
    a tree/force-directed layout.

    Returns:
        List of class dicts, each with "parents" and "children" keys.
    """
    query = """
    MATCH (c:Class {repo_id: $repo_id})
    OPTIONAL MATCH (c)-[:INHERITS_FROM]->(parent:Class {repo_id: $repo_id})
    OPTIONAL MATCH (child:Class {repo_id: $repo_id})-[:INHERITS_FROM]->(c)
    RETURN
      c.id           AS id,
      c.name         AS name,
      c.file_path    AS file_path,
      c.is_abstract  AS is_abstract,
      c.is_mixin     AS is_mixin,
      c.mro_list     AS mro_list,
      COLLECT(DISTINCT parent.name) AS parents,
      COLLECT(DISTINCT child.name)  AS children
    ORDER BY c.name
    """

    rows = run_query(query, repo_id=repo_id)

    result = []
    for row in rows:
        result.append({
            "id":          row.get("id"),
            "name":        row.get("name"),
            "file_path":   row.get("file_path"),
            "is_abstract": row.get("is_abstract", False),
            "is_mixin":    row.get("is_mixin", False),
            "mro_list":    row.get("mro_list") or [],
            "parents":     [p for p in (row.get("parents") or []) if p],
            "children":    [c for c in (row.get("children") or []) if c],
        })

    return result


def get_all_class_nodes(repo_id: str) -> list[dict]:
    """
    Return all class nodes for a repo with their key properties.
    Used by the /hierarchy endpoint as a lightweight listing.
    """
    query = """
    MATCH (c:Class {repo_id: $repo_id})
    RETURN c.id          AS id,
           c.name        AS name,
           c.file_path   AS file_path,
           c.start_line  AS start_line,
           c.end_line    AS end_line,
           c.is_abstract AS is_abstract,
           c.is_mixin    AS is_mixin,
           c.mro_list    AS mro_list,
           c.base_names  AS base_names
    ORDER BY c.name
    """
    rows = run_query(query, repo_id=repo_id)
    return [dict(row) for row in rows]


# ── Private helpers ────────────────────────────────────────────────────────────

def _get_edges_between(repo_id: str, node_ids: list[str]) -> list[dict]:
    """
    Return all edges where both source and target are in node_ids.
    Used by get_subgraph() to build the edge list.
    """
    if not node_ids:
        return []

    query = """
    MATCH (source)-[r]->(target)
    WHERE source.repo_id = $repo_id
      AND target.repo_id = $repo_id
      AND source.id IN $node_ids
      AND target.id IN $node_ids
    RETURN source.id AS source_id,
           target.id AS target_id,
           type(r)   AS rel_type,
           id(r)     AS edge_id
    """

    rows = run_query(query, repo_id=repo_id, node_ids=node_ids)

    return [
        {
            "source": row.get("source_id"),
            "target": row.get("target_id"),
            "type":   row.get("rel_type"),
            "id":     str(row.get("edge_id")),
        }
        for row in rows
    ]