"""
Graph Builder — orchestrates the full graph construction pipeline.

Pipeline:
  1. Load all code chunks for a repo from PostgreSQL
  2. Build File, Function, Class nodes
  3. Build structural edges: CONTAINS, DEFINED_IN, CONTAINS_FILE, CONTAINS_METHOD
  4. Extract CALLS edges via Python AST analysis of chunk content
  5. Extract IMPORTS edges via Python import statement analysis
  6. Return a RepoGraph with all nodes + edges, ready for Neo4j writing

This module intentionally has NO Neo4j dependency — it only reads PostgreSQL
and produces in-memory Python objects. The neo4j_writer.py handles the writing.
"""

from dataclasses import dataclass, field
from core.logging import get_logger

logger = get_logger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """
    A node to be written to Neo4j.
    Maps 1:1 with a node label in the schema.
    """
    node_id: str             # UUID string — same as PostgreSQL primary key
    label: str               # NodeLabel constant: "Function", "Class", "File", "Repo"
    properties: dict         # All properties to store on the node

    def __repr__(self) -> str:
        name = self.properties.get("name", self.node_id[:8])
        return f"GraphNode(label={self.label!r}, name={name!r}, node_id={self.node_id!r})"


@dataclass
class GraphEdge:
    """
    A directed relationship to be written to Neo4j.
    source_id and target_id must both be node IDs already present in the graph.
    """
    source_id: str           # UUID of the source node
    target_id: str           # UUID of the target node
    rel_type: str            # RelType constant: "CALLS", "IMPORTS", "CONTAINS", etc.
    properties: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"GraphEdge(rel_type={self.rel_type!r}, source={self.source_id!r}, target={self.target_id!r})"


@dataclass
class RepoGraph:
    """
    Complete in-memory graph for a single repository, before writing to Neo4j.
    """
    repo_id: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    @property
    def function_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes if n.label == "Function"]

    @property
    def class_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes if n.label == "Class"]

    @property
    def file_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes if n.label == "File"]

    @property
    def call_edges(self) -> list[GraphEdge]:
        return [e for e in self.edges if e.rel_type == "CALLS"]

    @property
    def import_edges(self) -> list[GraphEdge]:
        return [e for e in self.edges if e.rel_type == "IMPORTS"]

    def summary(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "total_nodes": len(self.nodes),
            "function_nodes": len(self.function_nodes),
            "class_nodes": len(self.class_nodes),
            "file_nodes": len(self.file_nodes),
            "total_edges": len(self.edges),
            "call_edges": len(self.call_edges),
            "import_edges": len(self.import_edges),
        }

# ── Continuation of graph_builder.py — append below the dataclasses ──────────

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.models import Repository, SourceFile, CodeChunk
from graph.schema import NodeLabel, RelType, Props
from graph.python_graph_extractor import (
    ChunkIndex,
    extract_calls_from_chunk,
    extract_imports_from_file,
)


async def build_repo_graph(repo_id: str, db: AsyncSession) -> RepoGraph:
    """
    Main entry point — builds the complete in-memory graph for a repository.

    Pipeline:
      1. Load Repository record from PostgreSQL
      2. Load all SourceFile records for this repo
      3. Load all CodeChunk records (joined with file paths)
      4. Build :Repo, :File, :Function, :Class nodes
      5. Build CONTAINS_FILE, CONTAINS, DEFINED_IN, CONTAINS_METHOD edges
      6. Build CALLS edges via Python AST call analysis
      7. Build IMPORTS edges via Python import statement analysis
      8. Return RepoGraph with everything in memory

    This function is async because it reads from PostgreSQL.
    The actual Neo4j writing happens synchronously in neo4j_writer.py.

    Args:
        repo_id: UUID string of the repository
        db:      AsyncSession for PostgreSQL

    Returns:
        RepoGraph — all nodes and edges ready to write to Neo4j
    """
    logger.info("graph_build_start", repo_id=repo_id)
    graph = RepoGraph(repo_id=repo_id)

    # ── 1. Load repo ──────────────────────────────────────────────────────
    repo_result = await db.execute(
        select(Repository).where(Repository.id == repo_id)
    )
    repo = repo_result.scalar_one_or_none()
    if repo is None:
        raise ValueError(f"Repository {repo_id} not found in PostgreSQL")

    graph.nodes.append(GraphNode(
        node_id=str(repo.id),
        label=NodeLabel.REPO,
        properties={
            Props.ID: str(repo.id),
            Props.NAME: repo.name,
            Props.GITHUB_URL: repo.github_url,
            Props.REPO_ID: str(repo.id),
        },
    ))

    # ── 2. Load all SourceFiles ───────────────────────────────────────────
    files_result = await db.execute(
        select(SourceFile).where(SourceFile.repository_id == repo_id)
    )
    source_files = files_result.scalars().all()
    logger.info("graph_files_loaded", repo_id=repo_id, count=len(source_files))

    file_node_map: dict[str, GraphNode] = {}

    for sf in source_files:
        file_node = GraphNode(
            node_id=str(sf.id),
            label=NodeLabel.FILE,
            properties={
                Props.ID: str(sf.id),
                Props.PATH: sf.file_path,
                Props.LANGUAGE: sf.language or "unknown",
                Props.REPO_ID: str(repo.id),
            },
        )
        graph.nodes.append(file_node)
        file_node_map[str(sf.id)] = file_node

        # Repo → File structural edge
        graph.edges.append(GraphEdge(
            source_id=str(repo.id),
            target_id=str(sf.id),
            rel_type=RelType.CONTAINS_FILE,
        ))

    # ── 3. Load all CodeChunks ────────────────────────────────────────────
    chunks_result = await db.execute(
        select(CodeChunk, SourceFile.file_path, SourceFile.language)
        .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
        .where(SourceFile.repository_id == repo_id)
    )
    chunk_rows = chunks_result.all()
    logger.info("graph_chunks_loaded", repo_id=repo_id, count=len(chunk_rows))

    # ── 4. Build Function and Class nodes ─────────────────────────────────
    for chunk, file_path, language in chunk_rows:
        if chunk.chunk_type in ("function", "method"):
            node = GraphNode(
                node_id=str(chunk.id),
                label=NodeLabel.FUNCTION,
                properties={
                    Props.ID: str(chunk.id),
                    Props.NAME: chunk.name,
                    Props.DISPLAY_NAME: chunk.display_name or chunk.name,
                    Props.FILE_PATH: file_path,
                    Props.START_LINE: chunk.start_line,
                    Props.END_LINE: chunk.end_line,
                    Props.IS_METHOD: chunk.chunk_type == "method",
                    Props.DOCSTRING: chunk.docstring or "",
                    Props.REPO_ID: str(repo.id),
                },
            )
        elif chunk.chunk_type == "class":
            node = GraphNode(
                node_id=str(chunk.id),
                label=NodeLabel.CLASS,
                properties={
                    Props.ID: str(chunk.id),
                    Props.NAME: chunk.name,
                    Props.DISPLAY_NAME: chunk.display_name or chunk.name,
                    Props.FILE_PATH: file_path,
                    Props.START_LINE: chunk.start_line,
                    Props.END_LINE: chunk.end_line,
                    Props.DOCSTRING: chunk.docstring or "",
                    Props.REPO_ID: str(repo.id),
                },
            )
        else:
            # Skip module-level fallback chunks — not meaningful graph nodes
            continue

        graph.nodes.append(node)

        # File → Function/Class containment
        graph.edges.append(GraphEdge(
            source_id=str(chunk.source_file_id),
            target_id=str(chunk.id),
            rel_type=RelType.CONTAINS,
        ))
        # Function/Class → File reverse lookup
        graph.edges.append(GraphEdge(
            source_id=str(chunk.id),
            target_id=str(chunk.source_file_id),
            rel_type=RelType.DEFINED_IN,
        ))

    # ── 5. Build ChunkIndex for name resolution ───────────────────────────
    index = ChunkIndex.build(graph.nodes)

    # ── 6. Build CONTAINS_METHOD edges (Class → method) ───────────────────
    for chunk, file_path, language in chunk_rows:
        if chunk.chunk_type == "method" and chunk.parent_name:
            parent_candidates = index.by_name.get(chunk.parent_name, [])
            # Narrow to same file (a class in a different file cannot own this method)
            same_file_parents = [
                pid for pid in parent_candidates
                if index.by_id.get(pid)
                and index.by_id[pid].properties.get(Props.FILE_PATH) == file_path
            ]
            for parent_id in same_file_parents:
                graph.edges.append(GraphEdge(
                    source_id=parent_id,
                    target_id=str(chunk.id),
                    rel_type=RelType.CONTAINS_METHOD,
                ))

    # ── 7. Extract CALLS edges from Python function chunks ────────────────
    python_func_chunks = [
        (chunk, file_path)
        for chunk, file_path, language in chunk_rows
        if language == "python" and chunk.chunk_type in ("function", "method")
    ]

    calls_extracted = 0
    for chunk, file_path in python_func_chunks:
        if not chunk.content:
            continue
        call_edges = extract_calls_from_chunk(
            chunk_id=str(chunk.id),
            chunk_content=chunk.content,
            caller_file=file_path,
            index=index,
        )
        graph.edges.extend(call_edges)
        calls_extracted += len(call_edges)

    logger.info(
        "graph_calls_extracted",
        repo_id=repo_id,
        python_chunks=len(python_func_chunks),
        call_edges=calls_extracted,
    )

    # ── 8. Extract IMPORTS edges from Python files ────────────────────────
    file_nodes_list = graph.file_nodes

    # Group chunks by file_id to reconstruct file content for import parsing
    file_chunks: dict[str, list] = {}
    for chunk, file_path, language in chunk_rows:
        if language == "python":
            file_chunks.setdefault(str(chunk.source_file_id), []).append(chunk)

    imports_extracted = 0
    for sf in source_files:
        if sf.language != "python":
            continue

        chunks_for_file = sorted(
            file_chunks.get(str(sf.id), []),
            key=lambda c: c.start_line,
        )
        if not chunks_for_file:
            continue

        # Reconstruct a rough file content from chunk content
        # (enough to parse import statements at the top of the file)
        all_content = "\n\n".join(c.content for c in chunks_for_file)

        import_edges = extract_imports_from_file(
            file_id=str(sf.id),
            file_content=all_content,
            file_path=sf.file_path,
            all_file_nodes=file_nodes_list,
        )
        graph.edges.extend(import_edges)
        imports_extracted += len(import_edges)

    logger.info(
        "graph_imports_extracted",
        repo_id=repo_id,
        import_edges=imports_extracted,
    )

    logger.info("graph_build_complete", **graph.summary())
    return graph