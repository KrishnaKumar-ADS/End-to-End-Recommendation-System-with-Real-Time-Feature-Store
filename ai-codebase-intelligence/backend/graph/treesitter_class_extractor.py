"""
JavaScript / TypeScript Class Hierarchy Extractor — tree-sitter.

Extracts inheritance relationships from JS/TS source files:
  - class Dog extends Animal       → INHERITS_FROM edge
  - class C extends A implements B → INHERITS_FROM + IMPLEMENTS edges
  - abstract class Shape {}        → is_abstract = True

Output format is identical to Python's class_extractor.py —
returns ClassHierarchyData objects so hierarchy_builder.py
can process both Python and JS/TS the same way.
"""

from typing import Optional
from core.logging import get_logger

logger = get_logger(__name__)

# Lazy-loaded tree-sitter parsers — only imported when needed
_JS_PARSER = None
_TS_PARSER = None


def _get_parser(language: str):
    """Return a cached tree-sitter parser for the given language."""
    global _JS_PARSER, _TS_PARSER

    try:
        from tree_sitter import Language, Parser

        def _build_parser(language_ptr: int, language_name: str):
            lang = Language(language_ptr, language_name)
            parser = Parser()
            parser.set_language(lang)
            return parser

        if language == "javascript":
            if _JS_PARSER is None:
                import tree_sitter_javascript as tsjs
                _JS_PARSER = _build_parser(tsjs.language(), "javascript")
            return _JS_PARSER

        elif language == "typescript":
            if _TS_PARSER is None:
                import tree_sitter_typescript as tsts
                _TS_PARSER = _build_parser(tsts.language_typescript(), "typescript")
            return _TS_PARSER

    except ImportError as exc:
        logger.warning(
            "treesitter_class_extractor_import_error",
            language=language,
            error=str(exc),
        )
        return None


def extract_js_class_hierarchy(
    source_code: str,
    chunk_id: str,
    class_name: str,
    file_path: str,
    repo_id: str,
    language: str = "javascript",
) -> Optional["ClassHierarchyData"]:
    """
    Parse `source_code` (a JS/TS class chunk) and extract inheritance data.

    Returns None if:
      - tree-sitter is not installed
      - No class_declaration node is found
      - Parse error

    Args:
        source_code: The source text of the class chunk
        chunk_id:    UUID from PostgreSQL code_chunks.id
        class_name:  Expected class name (for logging)
        file_path:   Relative file path within repo
        repo_id:     Repository UUID
        language:    "javascript" or "typescript"

    Returns:
        ClassHierarchyData with populated bases list, or None
    """
    from graph.class_extractor import ClassHierarchyData, BaseClassRef

    parser = _get_parser(language)
    if parser is None:
        return None

    try:
        tree = parser.parse(bytes(source_code, "utf-8"))
    except Exception as exc:
        logger.warning(
            "ts_class_extractor_parse_error",
            chunk_id=chunk_id,
            error=str(exc),
        )
        return None

    return _walk_tree_for_class(
        tree.root_node,
        source_code,
        chunk_id,
        class_name,
        file_path,
        repo_id,
    )


def _walk_tree_for_class(
    root_node,
    source_code: str,
    chunk_id: str,
    class_name: str,
    file_path: str,
    repo_id: str,
) -> Optional["ClassHierarchyData"]:
    """Walk tree-sitter nodes recursively looking for class_declaration."""
    from graph.class_extractor import ClassHierarchyData, BaseClassRef

    for node in _iter_nodes(root_node):
        if node.type not in ("class_declaration", "abstract_class_declaration",
                              "class_expression"):
            continue

        extracted_name = _get_child_text(node, "name", source_code)
        if extracted_name and extracted_name != class_name:
            continue   # wrong class — keep searching

        is_abstract = node.type == "abstract_class_declaration"
        bases: list[BaseClassRef] = []

        # extends clause — maximum ONE parent in JS/TS
        extends_node = _find_child_by_type(node, "class_heritage")
        if extends_node is None:
            extends_node = _find_child_by_type(node, "extends_clause")

        if extends_node:
            for child in _iter_nodes(extends_node):
                if child.type in (
                    "identifier",
                    "member_expression",
                    "type_identifier",
                    "nested_identifier",
                ):
                    parent_name = source_code[child.start_byte:child.end_byte].strip()
                    # strip generic type params like SomeBase<T>
                    if "<" in parent_name:
                        parent_name = parent_name[:parent_name.index("<")]
                    if parent_name and parent_name != "extends":
                        bases.append(
                            BaseClassRef(
                                name=parent_name,
                                is_abstract=False,
                                is_mixin=parent_name.endswith(("Mixin", "Base")),
                                position=0,
                            )
                        )
                        break  # JS/TS has at most one extends target

        # implements clause — TypeScript only, can be multiple
        implements_node = _find_child_by_type(node, "implements_clause")
        if implements_node:
            position = 1
            for child in implements_node.children:
                if child.type in ("type_identifier", "identifier"):
                    iface_name = source_code[child.start_byte:child.end_byte].strip()
                    if iface_name and iface_name != "implements":
                        bases.append(BaseClassRef(
                            name=iface_name,
                            is_abstract=True,   # interfaces = abstract in TS
                            is_mixin=False,
                            position=position,
                        ))
                        position += 1

        is_mixin = (
            bool(extracted_name) and extracted_name.endswith(("Mixin", "Base"))
        )

        return ClassHierarchyData(
            chunk_id=chunk_id,
            class_name=extracted_name or class_name,
            file_path=file_path,
            repo_id=repo_id,
            bases=bases,
            is_abstract=is_abstract,
            is_mixin=is_mixin,
        )

    return None


def _iter_nodes(node):
    """Depth-first iterator over all tree-sitter nodes."""
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def _find_child_by_type(node, node_type: str):
    """Return the first direct or indirect child with the given type."""
    for child in _iter_nodes(node):
        if child.type == node_type:
            return child
    return None


def _get_child_text(node, field_name: str, source_code: str) -> Optional[str]:
    """Extract the text of a named child field."""
    child = node.child_by_field_name(field_name)
    if child is None:
        return None
    return source_code[child.start_byte:child.end_byte].strip()