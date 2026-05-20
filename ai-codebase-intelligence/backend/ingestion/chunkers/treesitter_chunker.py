"""
tree-sitter Chunker — parses JavaScript, TypeScript, Go, Java
using the tree-sitter library.

Produces the same CodeChunkData output format as the Python chunker
so the rest of the pipeline doesn't care which parser was used.
"""

from tree_sitter import Language, Parser
from ingestion.chunker import CodeChunkData
from core.logging import get_logger

logger = get_logger(__name__)

# ── Language setup ────────────────────────────────────────────

def _get_parser(language: str) -> Parser | None:
    """
    Build and return a tree-sitter parser for the given language.
    Handles both old (0.21.x) and new (0.22+) tree-sitter API.
    """
    try:
        if language == "javascript":
            import tree_sitter_javascript as ts_js
            lang = Language(ts_js.language(), "javascript")

        elif language == "typescript":
            import tree_sitter_typescript as ts_ts
            lang = Language(ts_ts.language_typescript(), "typescript")

        elif language == "go":
            import tree_sitter_go as ts_go
            lang = Language(ts_go.language(), "go")

        elif language == "java":
            import tree_sitter_java as ts_java
            lang = Language(ts_java.language(), "java")

        else:
            return None

        parser = Parser()
        parser.set_language(lang)
        return parser

    except Exception as e:
        logger.warning("treesitter_parser_init_failed", language=language, error=str(e))
        return None

# ── Node type mappings per language ──────────────────────────

# Maps language → set of tree-sitter node types we want to extract
FUNCTION_NODE_TYPES = {
    "javascript": {"function_declaration", "arrow_function", "function_expression", "method_definition"},
    "typescript": {"function_declaration", "arrow_function", "function_expression", "method_definition"},
    "go":         {"function_declaration", "method_declaration"},
    "java":       {"method_declaration", "constructor_declaration"},
}

CLASS_NODE_TYPES = {
    "javascript": {"class_declaration", "class_expression"},
    "typescript": {"class_declaration", "class_expression", "abstract_class_declaration"},  # ← added abstract_class_declaration
    "go":         {"type_spec"},
    "java":       {"class_declaration", "interface_declaration"},
}

# Maps language → node type for the name field
NAME_FIELD = {
    "javascript": "name",
    "typescript": "name",
    "go":         "name",
    "java":       "name",
}


def chunk_with_treesitter(
    content: str,
    file_path: str,
    language: str,
) -> list[CodeChunkData]:
    """
    Parse a source file using tree-sitter and extract functions/classes.
    Falls back to line-based chunking if tree-sitter fails.
    """
    parser = _get_parser(language)
    if parser is None:
        logger.warning("treesitter_no_parser", language=language, file=file_path)
        from ingestion.chunker import chunk_by_lines
        return chunk_by_lines(content, file_path, language)

    try:
        tree = parser.parse(bytes(content, "utf-8"))
    except Exception as e:
        logger.warning("treesitter_parse_failed", file=file_path, error=str(e))
        from ingestion.chunker import chunk_by_lines
        return chunk_by_lines(content, file_path, language)

    lines = content.splitlines()
    chunks: list[CodeChunkData] = []

    function_types = FUNCTION_NODE_TYPES.get(language, set())
    class_types = CLASS_NODE_TYPES.get(language, set())

    _walk_tree(
        node=tree.root_node,
        lines=lines,
        file_path=file_path,
        language=language,
        function_types=function_types,
        class_types=class_types,
        chunks=chunks,
        parent_name="",
    )

    return chunks


def _walk_tree(
    node,
    lines: list[str],
    file_path: str,
    language: str,
    function_types: set,
    class_types: set,
    chunks: list[CodeChunkData],
    parent_name: str,
    depth: int = 0,
) -> None:
    """
    Recursively walk the tree-sitter AST and extract chunks.
    """
    if depth > 10:  # prevent infinite recursion on malformed files
        return

    node_type = node.type
    start_line = node.start_point[0] + 1  # tree-sitter is 0-indexed
    end_line = node.end_point[0] + 1

    if node_type in function_types:
        name = _extract_name(node, language) or f"anonymous_L{start_line}"
        content = "\n".join(lines[start_line - 1 : end_line])

        chunk_type = "method" if parent_name else "function"
        chunks.append(CodeChunkData(
            name=name,
            chunk_type=chunk_type,
            content=content,
            start_line=start_line,
            end_line=end_line,
            language=language,
            file_path=file_path,
            parent_name=parent_name,
        ))

    elif node_type in class_types:
        name = _extract_name(node, language) or f"AnonymousClass_L{start_line}"
        content = "\n".join(lines[start_line - 1 : end_line])

        chunks.append(CodeChunkData(
            name=name,
            chunk_type="class",
            content=content,
            start_line=start_line,
            end_line=end_line,
            language=language,
            file_path=file_path,
        ))

        # Walk children with this class as parent
        for child in node.children:
            _walk_tree(
                child, lines, file_path, language,
                function_types, class_types, chunks,
                parent_name=name, depth=depth + 1,
            )
        return  # don't double-walk children below

    for child in node.children:
        _walk_tree(
            child, lines, file_path, language,
            function_types, class_types, chunks,
            parent_name=parent_name, depth=depth + 1,
        )


def _extract_name(node, language: str) -> str | None:
    """Extract the name identifier from a function or class node."""
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "property_identifier"):
            return child.text.decode("utf-8")
    return None