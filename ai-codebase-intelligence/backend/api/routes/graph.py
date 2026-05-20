"""
Graph API routes — serve Neo4j graph data as REST endpoints.

Endpoints:
  GET /api/v1/graph/{repo_id}                  — full graph (nodes + edges)
  GET /api/v1/graph/{repo_id}/stats            — node/edge counts per type
  GET /api/v1/graph/{repo_id}/calls/{func_id}  — call chain for a function
  GET /api/v1/graph/{repo_id}/path             — shortest call path between 2 functions
"""

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from sqlalchemy import select
from db.database import get_db
from db.models import Repository
from api.schemas.graph_schema import (
    CallChainResponse, GraphStatsResponse,
)
from graph.traversal import (
    get_call_chain,
    get_callers,
    find_path_between_functions,
)
from graph.neo4j_client import run_query
from graph.neo4j_writer import get_repo_graph_stats
from graph.graph_utils import (
    get_subgraph,
    get_ancestors,
    get_descendants,
    get_graph_metrics,
    get_full_hierarchy_tree,
    get_all_class_nodes,
)
from core.logging import get_logger

router = APIRouter(prefix="/api/v1/graph", tags=["Graph"])
logger = get_logger(__name__)


async def _get_repo_or_404(repo_id: str, db: AsyncSession) -> Repository:
    """Verify repo exists in PostgreSQL. Raises 404 if not found."""
    result = await db.execute(select(Repository).where(Repository.id == repo_id))
    repo = result.scalar_one_or_none()
    if repo is None:
        raise HTTPException(status_code=404, detail=f"Repository {repo_id} not found.")
    return repo


@router.get(
    "/{repo_id}",
    summary="Get repository graph data",
    description=(
        "Returns dependency graph nodes and edges for a repository. "
        "Supports filtering by node type, relationship expansion, and bounded subgraph depth."
    ),
    responses={
        200: {"description": "Graph payload returned successfully."},
        400: {"description": "Invalid node_type value."},
        404: {"description": "Repository id not found."},
    },
)
async def get_repo_graph(
    repo_id: str,
    # ── existing Week 4 params ───────────────────────────────────────
    limit: int = Query(500, ge=1, le=5000),
    # ── new Week 5 params ────────────────────────────────────────────
    node_type: str | None = Query(
        None,
        description="Filter by node type: Function, Class, File. Default: all."
    ),
    include_hierarchy: bool = Query(
        False,
        description="If true, include INHERITS_FROM and IMPLEMENTS edges "
                    "alongside CALLS and IMPORTS edges."
    ),
    depth: int | None = Query(
        None,
        ge=1,
        le=5,
        description="If provided, return only nodes within this many hops "
                    "of the most-connected node. Default: return all."
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Return graph nodes and edges for frontend D3.js visualisation.

    New in Week 5:
      ?node_type=Function  — return only Function nodes (and their edges)
      ?node_type=Class     — return only Class nodes and hierarchy edges
      ?include_hierarchy=true — also return INHERITS_FROM, IMPLEMENTS edges
      ?depth=2             — return the 2-hop subgraph around the most-connected node

    All Week 4 behaviour is unchanged when no new params are provided.
    """
    await _get_repo_or_404(repo_id, db)

    if node_type and node_type not in {"Function", "Class", "File"}:
        raise HTTPException(
            status_code=400,
            detail="node_type must be one of: Function, Class, File",
        )

    # Build relationship filter based on params
    rel_types = ["CALLS", "IMPORTS", "CONTAINS", "CONTAINS_METHOD"]
    if include_hierarchy:
        rel_types += ["INHERITS_FROM", "IMPLEMENTS", "MIXES_IN"]

    # Build node label filter
    if node_type:
        node_label_filter = f"AND '{node_type}' IN labels(n)"
    else:
        node_label_filter = ""

    rel_type_str = "|".join(rel_types)

    # Fetch nodes
    nodes_query = f"""
    MATCH (n {{repo_id: $repo_id}})
    WHERE NOT 'Repo' IN labels(n) {node_label_filter}
    RETURN n, labels(n) AS node_labels
    LIMIT $limit
    """
    node_rows = run_query(nodes_query, repo_id=repo_id, limit=limit)  # type: ignore

    node_ids = [row.get("n", {}).get("id") for row in node_rows if row.get("n", {}).get("id")]

    # Fetch edges between those nodes
    edges_query = f"""
    MATCH (source)-[r:{rel_type_str}]->(target)
    WHERE source.repo_id = $repo_id
      AND target.repo_id = $repo_id
      AND source.id IN $node_ids
      AND target.id   IN $node_ids
    RETURN source.id  AS source_id,
           target.id  AS target_id,
           type(r)    AS rel_type,
           id(r)      AS edge_id
    """
    edge_rows = run_query(edges_query, repo_id=repo_id, node_ids=node_ids)  # type: ignore

    # Format nodes
    nodes_out = []
    for row in node_rows:
        n = dict(row.get("n", {}))
        n["_type"] = row.get("node_labels", ["Unknown"])[0]
        nodes_out.append(n)

    edges_out = [
        {
            "source": row.get("source_id"),
            "target": row.get("target_id"),
            "type":   row.get("rel_type"),
            "id":     str(row.get("edge_id")),
        }
        for row in edge_rows
    ]

    return {
        "repo_id":     repo_id,
        "node_count":  len(nodes_out),
        "edge_count":  len(edges_out),
        "nodes":       nodes_out,
        "edges":       edges_out,
    }


@router.get("/{repo_id}/stats", response_model=GraphStatsResponse)
async def get_graph_stats(repo_id: str, db: AsyncSession = Depends(get_db)):
    """Return node and edge counts by type for a repository's graph."""
    await _get_repo_or_404(repo_id, db)
    stats = get_repo_graph_stats(repo_id)
    return GraphStatsResponse(**stats)


@router.get("/{repo_id}/calls/{function_id}", response_model=CallChainResponse)
async def get_function_call_chain(
    repo_id: str,
    function_id: str,
    depth: int = Query(default=3, le=6, description="Max call chain depth"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the full call chain for a function — everything it calls, transitively.
    Also returns callers (who calls this function).
    """
    await _get_repo_or_404(repo_id, db)

    call_chain = get_call_chain(function_id, depth=depth)
    callers = get_callers(function_id, depth=2)
    func_name = call_chain[0]["name"] if call_chain else function_id

    return CallChainResponse(
        function_id=function_id,
        function_name=func_name,
        call_chain=call_chain,
        callers=callers,
    )
@router.get("/{repo_id}/subgraph/{node_id}")
async def get_node_subgraph(
    repo_id: str,
    node_id: str,
    depth: int = Query(2, ge=1, le=5, description="Number of hops to expand (1-5)"),
    rel_types: str | None = Query(
        None,
        description="Comma-separated relationship types to follow. "
                    "Default: all. Example: CALLS,INHERITS_FROM"
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the N-hop neighbourhood subgraph around a single node.

    The node can be any type: Function, Class, File, or Repo.
    The response is in the same nodes+edges format as GET /graph/{repo_id},
    so the frontend can render it without any changes.

    Path params:
      repo_id: Repository UUID
      node_id: UUID of the centre node

    Query params:
      depth:     1-5 hops (default 2)
      rel_types: Comma-separated list to filter relationship types.
                 e.g. ?rel_types=CALLS,INHERITS_FROM
                 Default: all relationship types.

    Response:
    {
      "repo_id": "...",
      "centre_node_id": "...",
      "depth": 2,
      "nodes": [...],
      "edges": [...]
    }
    """
    await _get_repo_or_404(repo_id, db)

    rel_type_list = None
    if rel_types:
        rel_type_list = [r.strip().upper() for r in rel_types.split(",") if r.strip()]

    subgraph = get_subgraph(
        repo_id=repo_id,
        node_id=node_id,
        depth=depth,
        relationship_types=rel_type_list,
    )

    return {
        "repo_id":        repo_id,
        "centre_node_id": node_id,
        "depth":          depth,
        "nodes":          subgraph["nodes"],
        "edges":          subgraph["edges"],
        "node_count":     len(subgraph["nodes"]),
        "edge_count":     len(subgraph["edges"]),
    }

@router.get("/{repo_id}/metrics")
async def get_repo_graph_metrics(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Return computed graph metrics for a repository's knowledge graph.

    Includes:
      - Node and edge counts by type
      - Graph density (edge saturation)
      - Top 5 most-calling functions
      - Top 5 most-called functions
      - Top 5 most-inherited base classes
      - Deepest inheritance chain length
      - Abstract and mixin class counts

    This endpoint is more expensive than /stats — it runs several
    Cypher aggregation queries. Results are not cached yet (Week 14 adds caching).
    Typical response time: 200-800ms depending on repo size.

    Response:
    {
      "repo_id": "...",
      "total_nodes": 1259,
      "total_edges": 3847,
      "graph_density": 0.002413,
      "function_count": 412,
      "class_count": 89,
      "file_count": 847,
      "calls_edges": 1203,
      "inherits_edges": 147,
      "top_callers": [
        {"name": "dispatch_request", "file_path": "flask/app.py", "calls_out": 42},
        ...
      ],
      "most_inherited": [
        {"name": "BaseModel", "file_path": "models/base.py", "subclass_count": 17},
        ...
      ],
      "deepest_hierarchy": 5,
      "abstract_class_count": 12,
      "mixin_class_count": 8
    }
    """
    await _get_repo_or_404(repo_id, db)
    metrics = get_graph_metrics(repo_id)
    return metrics

@router.get("/{repo_id}/ancestors/{class_id}")
async def get_class_ancestors(
    repo_id: str,
    class_id: str,
    include_external: bool = Query(
        False,
        description="Include base classes from outside this repo (e.g. Django Model)"
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the full ancestor chain for a class, in depth order.

    Traverses INHERITS_FROM and IMPLEMENTS edges upward until
    there are no more parents (or max_depth=10 is reached).

    Response:
    {
      "repo_id": "...",
      "class_id": "...",
      "ancestors": [
        {"name": "Dog",    "depth": 1, "edge_type": "INHERITS_FROM", "file_path": "..."},
        {"name": "Mammal", "depth": 2, "edge_type": "INHERITS_FROM", "file_path": "..."},
        {"name": "Animal", "depth": 3, "edge_type": "INHERITS_FROM", "file_path": "..."},
      ],
      "ancestor_count": 3
    }
    """
    await _get_repo_or_404(repo_id, db)
    ancestors = get_ancestors(
        repo_id=repo_id,
        class_id=class_id,
        include_external=include_external,
    )
    return {
        "repo_id":        repo_id,
        "class_id":       class_id,
        "ancestors":      ancestors,
        "ancestor_count": len(ancestors),
    }

@router.get("/{repo_id}/descendants/{class_id}")
async def get_class_descendants(
    repo_id: str,
    class_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Return all subclasses (direct and transitive) of a given class.

    Traverses INHERITS_FROM edges downward — the mirror image of /ancestors.
    Useful for answering: "if I change this base class, which subclasses are affected?"

    Response:
    {
      "repo_id": "...",
      "class_id": "...",
      "descendants": [
        {"name": "Mammal",          "depth": 1, "is_direct": true},
        {"name": "Dog",             "depth": 2, "is_direct": false},
        {"name": "GoldenRetriever", "depth": 3, "is_direct": false},
      ],
      "total_descendants": 3,
      "direct_subclass_count": 1
    }
    """
    await _get_repo_or_404(repo_id, db)
    descendants = get_descendants(repo_id=repo_id, class_id=class_id)
    direct_count = sum(1 for d in descendants if d.get("is_direct"))
    return {
        "repo_id":             repo_id,
        "class_id":            class_id,
        "descendants":         descendants,
        "total_descendants":   len(descendants),
        "direct_subclass_count": direct_count,
    }



@router.get("/{repo_id}/hierarchy")
async def get_class_hierarchy(
    repo_id: str,
    format: str = Query("tree", description="Response format: 'tree' (default) or 'flat'"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the complete class inheritance hierarchy for a repository.

    Query params:
      format=tree  → Returns list of class dicts, each with 'parents' and 'children' keys.
                     Best for D3.js force-directed layouts and graph visualisation.
      format=flat  → Returns a compact list with just id, name, is_abstract, base_names.
                     Best for dropdowns, search, and listing.

    Response (format=tree):
    {
      "repo_id": "...",
      "class_count": 47,
      "hierarchy": [
        {
          "id": "uuid",
          "name": "GoldenRetriever",
          "file_path": "animals/dog.py",
          "is_abstract": false,
          "is_mixin": false,
          "mro_list": ["GoldenRetriever", "Dog", "Mammal", "Animal", "object"],
          "parents": ["Dog"],
          "children": []
        },
        ...
      ]
    }
    """
    repo = await _get_repo_or_404(repo_id, db)

    if format == "flat":
        classes = get_all_class_nodes(repo_id)
        return {
            "repo_id":     repo_id,
            "class_count": len(classes),
            "classes":     classes,
        }

    # Default: tree format
    hierarchy = get_full_hierarchy_tree(repo_id)
    return {
        "repo_id":     repo_id,
        "class_count": len(hierarchy),
        "hierarchy":   hierarchy,
    }

@router.get("/{repo_id}/path")
async def get_path_between_functions(
    repo_id: str,
    source_id: str = Query(..., description="UUID of the source function"),
    target_id: str = Query(..., description="UUID of the target function"),
    max_depth: int = Query(default=5, le=8),
    db: AsyncSession = Depends(get_db),
):
    """
    Find the shortest call path between two functions.
    Returns the list of intermediate functions in order.
    """
    await _get_repo_or_404(repo_id, db)
    path = find_path_between_functions(source_id, target_id, max_depth=max_depth)
    return {
        "source_id": source_id,
        "target_id": target_id,
        "path": path,
        "length": len(path),
    }
