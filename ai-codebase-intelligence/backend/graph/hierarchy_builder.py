"""
Hierarchy Graph Builder — converts ClassHierarchyData into Neo4j edges.

This module:
  1. Loads all class CodeChunks from PostgreSQL (for a given repo_id)
  2. Runs extract_class_hierarchy() on each one (Day 1 extractor)
  3. Computes MRO for every class
  4. Writes INHERITS_FROM, IMPLEMENTS, MIXES_IN edges to Neo4j
  5. Updates :Class node properties: is_abstract, is_mixin, mro_list, base_names

Why separate from the Week 4 graph builder?
  Week 4's builder (graph/builder.py) handles structural relationships:
  CONTAINS, CALLS, IMPORTS. Those are extracted at the chunk level from the
  call extractor. Hierarchy relationships need a second pass because resolving
  inheritance requires comparing class names across ALL files in the repo —
  you can't resolve `class Dog(Animal)` without knowing which file defines Animal.
  This builder does that cross-file resolution.
"""

from dataclasses import dataclass, field
from sqlalchemy import select
from sqlalchemy.orm import Session
from db.models import CodeChunk, SourceFile
from graph.class_extractor import (
    ClassHierarchyData,
    extract_class_hierarchy,
    compute_mro,
)
from graph.neo4j_client import get_session
from graph.schema import RelType, NodeLabel
from core.logging import get_logger

logger = get_logger(__name__)

# ── Cypher queries ─────────────────────────────────────────────────────────────

# Update :Class node with hierarchy-aware properties
_UPDATE_CLASS_PROPS_CYPHER = """
UNWIND $nodes AS n
MATCH (c:Class {id: n.id})
SET
  c.is_abstract   = n.is_abstract,
  c.is_mixin      = n.is_mixin,
  c.mro_list      = n.mro_list,
  c.base_names    = n.base_names
"""

# Write INHERITS_FROM edges
_WRITE_INHERITS_FROM_CYPHER = """
UNWIND $edges AS e
MATCH (child:Class  {repo_id: $repo_id, name: e.child_name})
MATCH (parent:Class {repo_id: $repo_id, name: e.parent_name})
MERGE (child)-[r:INHERITS_FROM]->(parent)
SET
  r.depth     = e.depth,
  r.is_direct = e.is_direct
"""

# Write IMPLEMENTS edges (for ABC / abstract base class patterns)
_WRITE_IMPLEMENTS_CYPHER = """
UNWIND $edges AS e
MATCH (child:Class  {repo_id: $repo_id, name: e.child_name})
MATCH (parent:Class {repo_id: $repo_id, name: e.parent_name})
MERGE (child)-[r:IMPLEMENTS]->(parent)
SET r.is_abstract = e.is_abstract
"""

# Write MIXES_IN edges (for Mixin patterns)
_WRITE_MIXES_IN_CYPHER = """
UNWIND $edges AS e
MATCH (child:Class  {repo_id: $repo_id, name: e.child_name})
MATCH (mixin:Class  {repo_id: $repo_id, name: e.mixin_name})
MERGE (child)-[r:MIXES_IN]->(mixin)
SET r.position = e.position
"""


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class HierarchyBuildResult:
    """Summary of what was written to Neo4j by build_class_hierarchy()."""
    repo_id: str
    classes_processed: int = 0
    inherits_from_edges: int = 0
    implements_edges: int = 0
    mixes_in_edges: int = 0
    classes_updated: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_edges(self) -> int:
        return self.inherits_from_edges + self.implements_edges + self.mixes_in_edges

    def summary(self) -> str:
        return (
            f"Hierarchy built — {self.classes_processed} classes, "
            f"{self.inherits_from_edges} INHERITS_FROM edges, "
            f"{self.implements_edges} IMPLEMENTS edges, "
            f"{self.mixes_in_edges} MIXES_IN edges"
        )


# ── Public API ──────────────────────────────────────────────────────────────────

def build_class_hierarchy(repo_id: str, db: Session) -> HierarchyBuildResult:
    """
    Full hierarchy build pipeline for one repository.

    Steps:
      1. Load all class chunks from PostgreSQL
      2. Extract hierarchy data (parents, bases, abstract flag)
      3. Compute MRO for each class
      4. Update :Class node properties in Neo4j
      5. Write INHERITS_FROM, IMPLEMENTS, MIXES_IN edges in batches

    Args:
        repo_id: UUID string of the repository to process
        db:      SQLAlchemy synchronous Session

    Returns:
        HierarchyBuildResult with counts of what was written
    """
    result = HierarchyBuildResult(repo_id=repo_id)

    # ── Step 1: load class chunks ─────────────────────────────────────────────
    class_chunks = _load_class_chunks(repo_id, db)
    logger.info(
        "hierarchy_builder_loaded_chunks",
        repo_id=repo_id,
        count=len(class_chunks),
    )

    if not class_chunks:
        logger.warning("hierarchy_builder_no_class_chunks", repo_id=repo_id)
        return result

    # ── Step 2: extract hierarchy data from each chunk ────────────────────────
    all_hierarchy: list[ClassHierarchyData] = []

    for chunk in class_chunks:
        try:
            language = chunk.get("language", "python")

            if language == "python":
                data = extract_class_hierarchy(
                    source_code=chunk["content"],
                    chunk_id=chunk["id"],
                    class_name=chunk["name"],
                    file_path=chunk["file_path"],
                    repo_id=repo_id,
                )
            elif language in ("javascript", "typescript"):
                from graph.treesitter_class_extractor import extract_js_class_hierarchy

                data = extract_js_class_hierarchy(
                    source_code=chunk["content"],
                    chunk_id=chunk["id"],
                    class_name=chunk["name"],
                    file_path=chunk["file_path"],
                    repo_id=repo_id,
                    language=language,
                )
            else:
                # Go, Rust, Java — no class hierarchy support yet.
                data = None

            if data is not None:
                all_hierarchy.append(data)
                result.classes_processed += 1
            else:
                result.errors.append(
                    f"Failed to extract hierarchy for chunk {chunk['id']}"
                )
        except Exception as exc:
            msg = f"Failed to extract hierarchy for chunk {chunk['id']}: {exc}"
            logger.warning("hierarchy_extraction_error", chunk_id=chunk["id"], error=str(exc))
            result.errors.append(msg)

    logger.info(
        "hierarchy_builder_extracted",
        repo_id=repo_id,
        extracted=result.classes_processed,
    )

    # ── Step 3: compute MRO for each class ────────────────────────────────────
    mro_map: dict[str, list[str]] = {}
    for data in all_hierarchy:
        mro_map[data.class_name] = compute_mro(data.class_name, all_hierarchy)

    # ── Step 4: update :Class node properties ─────────────────────────────────
    node_updates = []
    for data in all_hierarchy:
        node_updates.append({
            "id": data.chunk_id,
            "is_abstract": data.is_abstract,
            "is_mixin": data.is_mixin,
            "mro_list": mro_map.get(data.class_name, [data.class_name, "object"]),
            "base_names": [b.name for b in data.bases],
        })

    if node_updates:
        _run_in_batches(
            query=_UPDATE_CLASS_PROPS_CYPHER,
            items=node_updates,
            batch_size=500,
            params={"repo_id": repo_id},
        )
        result.classes_updated = len(node_updates)

    # ── Step 5a: write INHERITS_FROM edges ────────────────────────────────────
    inherits_edges = []
    for data in all_hierarchy:
        if not data.has_bases:
            continue
        for base in data.bases:
            if base.name in ("object", "type"):
                continue
            if base.is_mixin or base.is_abstract:
                continue  # handled by MIXES_IN / IMPLEMENTS below
            inherits_edges.append({
                "child_name": data.class_name,
                "parent_name": base.name,
                "depth": 1,
                "is_direct": True,
            })

    if inherits_edges:
        _run_in_batches(
            query=_WRITE_INHERITS_FROM_CYPHER,
            items=inherits_edges,
            batch_size=500,
            params={"repo_id": repo_id},
        )
        result.inherits_from_edges = len(inherits_edges)

    # ── Step 5b: write IMPLEMENTS edges ───────────────────────────────────────
    implements_edges = []
    for data in all_hierarchy:
        for base in data.bases:
            if base.is_abstract and base.name not in ("object", "type"):
                implements_edges.append({
                    "child_name": data.class_name,
                    "parent_name": base.name,
                    "is_abstract": True,
                })

    if implements_edges:
        _run_in_batches(
            query=_WRITE_IMPLEMENTS_CYPHER,
            items=implements_edges,
            batch_size=500,
            params={"repo_id": repo_id},
        )
        result.implements_edges = len(implements_edges)

    # ── Step 5c: write MIXES_IN edges ─────────────────────────────────────────
    mixin_edges = []
    for data in all_hierarchy:
        for base in data.bases:
            if base.is_mixin and base.name not in ("object", "type"):
                mixin_edges.append({
                    "child_name": data.class_name,
                    "mixin_name": base.name,
                    "position": base.position,
                })

    if mixin_edges:
        _run_in_batches(
            query=_WRITE_MIXES_IN_CYPHER,
            items=mixin_edges,
            batch_size=500,
            params={"repo_id": repo_id},
        )
        result.mixes_in_edges = len(mixin_edges)

    logger.info(
        "hierarchy_builder_complete",
        repo_id=repo_id,
        result=result.summary(),
    )
    return result


def delete_class_hierarchy(repo_id: str) -> None:
    """
    Remove all INHERITS_FROM, IMPLEMENTS, and MIXES_IN edges for a repo.

    Called before re-running the hierarchy builder to avoid duplicates,
    and when a repo is deleted from the platform.
    """
    with get_session() as session:
        session.run(
            "MATCH (:Class {repo_id: $repo_id})-[r:INHERITS_FROM|IMPLEMENTS|MIXES_IN]->() "
            "DELETE r",
            repo_id=repo_id,
        )
    logger.info("hierarchy_deleted", repo_id=repo_id)


def get_hierarchy_stats(repo_id: str) -> dict:
    """
    Return counts of hierarchy edges for a repo.
    Used by GET /graph/{repo_id}/stats.
    """
    with get_session() as session:
        result = session.run(
            """
            MATCH (c:Class {repo_id: $repo_id})
            OPTIONAL MATCH (c)-[inh:INHERITS_FROM]->()
            OPTIONAL MATCH (c)-[imp:IMPLEMENTS]->()
            OPTIONAL MATCH (c)-[mix:MIXES_IN]->()
            RETURN
              COUNT(DISTINCT c)   AS total_classes,
              COUNT(DISTINCT inh) AS inherits_from_edges,
              COUNT(DISTINCT imp) AS implements_edges,
              COUNT(DISTINCT mix) AS mixes_in_edges
            """,
            repo_id=repo_id,
        )
        record = result.single()
        if record is None:
            return {}
        return dict(record)


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_class_chunks(repo_id: str, db: Session) -> list[dict]:
    """
    Load all class CodeChunks for a repository from PostgreSQL.
    Includes language so the builder can route to the right extractor.
    """
    rows = db.execute(
        select(
            CodeChunk.id,
            CodeChunk.name,
            CodeChunk.content,
            SourceFile.language.label("language"),
            SourceFile.file_path.label("file_path"),
        )
        .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
        .where(SourceFile.repository_id == repo_id)
        .where(CodeChunk.chunk_type == "class")
    ).fetchall()

    return [
        {
            "id":        str(row.id),
            "name":      row.name,
            "content":   row.content,
            "language":  row.language if isinstance(row.language, str) and row.language else "python",
            "file_path": row.file_path,
        }
        for row in rows
    ]


def _run_in_batches(
    query: str,
    items: list[dict],
    batch_size: int,
    params: dict,
) -> None:
    """
    Run a Cypher UNWIND query in batches of `batch_size`.

    Args:
        query:      Cypher query that uses UNWIND $nodes or UNWIND $edges
        items:      List of dicts to pass as $nodes or $edges
        batch_size: How many items per Neo4j transaction
        params:     Additional static params (e.g. repo_id)
    """
    # Detect whether the query uses $nodes or $edges
    param_key = "edges" if "$edges" in query else "nodes"

    with get_session() as session:
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            session.run(query, **{param_key: batch}, **params)