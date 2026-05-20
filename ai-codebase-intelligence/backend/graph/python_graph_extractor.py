"""
Python Graph Extractor — extracts CALLS and IMPORTS edges from Python AST.

Given a list of CodeChunk records (from PostgreSQL), this module:
  1. Re-parses each Python function chunk using the stdlib `ast` module
  2. Collects all function calls made inside that chunk
  3. Resolves call names to actual chunk IDs using the ChunkIndex
  4. Extracts import statements at the file level
  5. Returns GraphEdge objects ready for Neo4j writing

Key design decision — FUZZY name matching for call resolution:
  If function A calls `hash_password(...)`, we search our in-memory index
  for a function named `hash_password` in the same repo.
  Resolution order: same-file match first, then cross-file match.
  This is not 100% accurate (dynamic dispatch, monkey-patching), but it
  is correct for the vast majority of internal function calls.

External calls (e.g. `bcrypt.checkpw`, `os.path.join`) are simply dropped —
they do not correspond to nodes in our graph.
"""

import ast
from dataclasses import dataclass
from graph.graph_builder import GraphNode, GraphEdge
from graph.schema import NodeLabel, RelType, Props
from core.logging import get_logger

logger = get_logger(__name__)


# ── AST Visitors ──────────────────────────────────────────────────────────────

class CallVisitor(ast.NodeVisitor):
    """
    AST visitor that collects all function call names from a code chunk.

    Handles:
      - Simple calls: `hash_password(x)` → "hash_password"
      - Method calls: `self.db.query(x)` → "query"
      - Chained calls: `User.objects.filter(...)` → "filter"

    Does NOT handle:
      - Call arguments (not needed for graph structure)
      - Whether the call target is internal or external (resolved later)
    """

    def __init__(self):
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call):
        name = self._extract_name(node.func)
        if name:
            self.calls.append(name)
        self.generic_visit(node)

    def _extract_name(self, node) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None


class ImportVisitor(ast.NodeVisitor):
    """
    AST visitor that collects all import statements from a Python source file.

    Returns a list of imported module names:
      `import os`               → ["os"]
      `from pathlib import Path` → ["pathlib"]
      `from . import utils`     → [".utils" (relative)]
      `from ..core import cfg`  → ["..core" (relative)]
    """

    def __init__(self):
        self.imports: list[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            prefix = "." * (node.level or 0)
            self.imports.append(f"{prefix}{node.module}")
        elif node.level:
            # `from . import something` — relative import without explicit module
            self.imports.append("." * node.level)


# ── Chunk Index ────────────────────────────────────────────────────────────────

@dataclass
class ChunkIndex:
    """
    In-memory lookup index built from all nodes of a repository.
    Built once per repo before extraction begins — avoids repeated linear scans.

    Used to resolve function call names → node IDs.
    """

    # function name → list of node IDs (multiple functions can share a name across files)
    by_name: dict[str, list[str]]

    # display_name → node ID (e.g. "AuthService.verify_password" → unique ID)
    by_display_name: dict[str, str]

    # file_path → list of node IDs for all functions/classes in that file
    by_file: dict[str, list[str]]

    # node_id → GraphNode for fast reverse lookups
    by_id: dict[str, GraphNode]

    @classmethod
    def build(cls, nodes: list[GraphNode]) -> "ChunkIndex":
        """Build the index from a list of already-constructed GraphNodes."""
        by_name: dict[str, list[str]] = {}
        by_display_name: dict[str, str] = {}
        by_file: dict[str, list[str]] = {}
        by_id: dict[str, GraphNode] = {}

        for node in nodes:
            if node.label not in (NodeLabel.FUNCTION, NodeLabel.CLASS):
                continue

            nid = node.node_id
            name = node.properties.get(Props.NAME, "")
            display_name = node.properties.get(Props.DISPLAY_NAME, "")
            file_path = node.properties.get(Props.FILE_PATH, "")

            by_id[nid] = node

            if name:
                by_name.setdefault(name, []).append(nid)

            if display_name:
                by_display_name[display_name] = nid

            if file_path:
                by_file.setdefault(file_path, []).append(nid)

        return cls(
            by_name=by_name,
            by_display_name=by_display_name,
            by_file=by_file,
            by_id=by_id,
        )

    def resolve_call(self, call_name: str, caller_file: str) -> list[str]:
        """
        Try to resolve a called function name to one or more node IDs.

        Resolution order (most specific first):
          1. Same file, exact name match  →  highest confidence
          2. Any file in repo, exact name match

        Returns a list of matching IDs.
        Returns an empty list if unresolved (external library call).
        """
        # Priority 1: same-file match
        same_file_ids = self.by_file.get(caller_file, [])
        same_file_matches = [
            nid for nid in same_file_ids
            if self.by_id.get(nid) and self.by_id[nid].properties.get(Props.NAME) == call_name
        ]
        if same_file_matches:
            return same_file_matches

        # Priority 2: cross-file match
        return self.by_name.get(call_name, [])


# ── Extraction functions ───────────────────────────────────────────────────────

def extract_calls_from_chunk(
    chunk_id: str,
    chunk_content: str,
    caller_file: str,
    index: ChunkIndex,
) -> list[GraphEdge]:
    """
    Parse a single Python function/method chunk and extract CALLS edges.

    Args:
        chunk_id:      UUID of the calling function chunk
        chunk_content: Raw Python source code of the function
        caller_file:   File path the chunk belongs to (for same-file resolution)
        index:         ChunkIndex for resolving call names to node IDs

    Returns:
        List of GraphEdge objects with rel_type="CALLS"
    """
    try:
        tree = ast.parse(chunk_content)
    except SyntaxError:
        # Chunk may not parse standalone (e.g. method without class indentation context)
        # Try wrapping it in a dummy function scope
        try:
            indented = "    " + chunk_content.replace("\n", "\n    ")
            tree = ast.parse(f"def _wrapper():\n{indented}")
        except SyntaxError:
            logger.debug("call_extraction_parse_failed", chunk_id=chunk_id[:8])
            return []

    visitor = CallVisitor()
    visitor.visit(tree)

    edges = []
    seen_targets: set[str] = set()  # deduplicate calls to the same function

    for call_name in visitor.calls:
        # Skip single-character names — likely loop variables, not meaningful function names
        if len(call_name) < 2:
            continue

        target_ids = index.resolve_call(call_name, caller_file)

        for target_id in target_ids:
            if target_id == chunk_id:
                continue  # skip recursive self-calls
            if target_id in seen_targets:
                continue  # deduplicate: A calls B twice → only one CALLS edge

            seen_targets.add(target_id)
            edges.append(GraphEdge(
                source_id=chunk_id,
                target_id=target_id,
                rel_type=RelType.CALLS,
                properties={Props.CALL_COUNT: 1},
            ))

    return edges


def extract_imports_from_file(
    file_id: str,
    file_content: str,
    file_path: str,
    all_file_nodes: list[GraphNode],
) -> list[GraphEdge]:
    """
    Parse a Python file's import statements and create IMPORTS edges.

    Resolves import module names to File node IDs where the module path
    matches a file in the repository. Unresolved (external library) imports
    are dropped — they have no corresponding File node.

    Args:
        file_id:        UUID of the source File node
        file_content:   The full source text of the file (reconstructed)
        file_path:      Relative path of the file (for relative import resolution)
        all_file_nodes: All File nodes for the repo (for resolution)

    Returns:
        List of GraphEdge objects with rel_type="IMPORTS"
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return []

    visitor = ImportVisitor()
    visitor.visit(tree)

    # Build a module-name → file-path lookup for the repo
    # e.g. "auth.service" → "auth/service.py"
    module_to_path: dict[str, str] = {}
    path_to_id: dict[str, str] = {}

    for node in all_file_nodes:
        path = node.properties[Props.PATH]
        path_to_id[path] = node.node_id
        # Convert path to dotted module name for matching
        module_name = (
            path.replace("/", ".")
                .replace("\\", ".")
                .removesuffix(".py")
        )
        module_to_path[module_name] = path

    edges = []
    seen: set[str] = set()

    for import_name in visitor.imports:
        if import_name in seen:
            continue
        seen.add(import_name)

        # Strip leading dots from relative imports and try to resolve
        clean = import_name.lstrip(".")
        target_path = module_to_path.get(clean)

        if target_path:
            target_id = path_to_id.get(target_path)
            if target_id and target_id != file_id:
                edges.append(GraphEdge(
                    source_id=file_id,
                    target_id=target_id,
                    rel_type=RelType.IMPORTS,
                    properties={Props.IMPORT_NAME: import_name},
                ))

    return edges