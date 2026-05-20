"""
Python AST Chunker — uses Python's built-in ast module to extract
functions, classes, and methods from .py files.

No third-party dependencies required.
"""

import ast
from ingestion.chunker import CodeChunkData
from core.logging import get_logger

logger = get_logger(__name__)


def chunk_python(content: str, file_path: str) -> list[CodeChunkData]:
    """
    Parse a Python source file and return one CodeChunkData per
    function, method, or class definition.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logger.warning("python_parse_failed", file=file_path, error=str(e))
        return []

    lines = content.splitlines()
    chunks: list[CodeChunkData] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            chunks.extend(_extract_class(node, lines, file_path))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Only extract top-level functions here
            # Methods inside classes are handled by _extract_class
            if _is_top_level(node, tree):
                chunks.append(_extract_function(node, lines, file_path))

    return chunks


def _is_top_level(node: ast.AST, tree: ast.Module) -> bool:
    """Check if a function is at module level (not inside a class)."""
    for child in ast.walk(tree):
        if isinstance(child, ast.ClassDef):
            for class_child in ast.walk(child):
                if class_child is node:
                    return False
    return True


def _extract_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    file_path: str,
    parent_name: str = "",
) -> CodeChunkData:
    """Extract a single function or method into a CodeChunkData."""
    start_line = node.lineno
    end_line = node.end_lineno or node.lineno

    # Extract source lines for this function
    content = "\n".join(lines[start_line - 1 : end_line])

    # Extract docstring
    docstring = ast.get_docstring(node) or ""

    # Extract decorators
    decorators = []
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            decorators.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            decorators.append(f"{decorator.value.id}.{decorator.attr}")
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                decorators.append(decorator.func.id)

    # Extract argument names
    args = [arg.arg for arg in node.args.args]

    # Determine chunk type
    chunk_type = "method" if parent_name else "function"
    if isinstance(node, ast.AsyncFunctionDef):
        chunk_type = f"async_{chunk_type}"

    return CodeChunkData(
        name=node.name,
        chunk_type=chunk_type,
        content=content,
        start_line=start_line,
        end_line=end_line,
        language="python",
        file_path=file_path,
        docstring=docstring,
        parent_name=parent_name,
        decorators=decorators,
    )


def _extract_class(
    node: ast.ClassDef,
    lines: list[str],
    file_path: str,
) -> list[CodeChunkData]:
    """
    Extract a class and all its methods as separate chunks.
    Returns: [class_chunk, method1_chunk, method2_chunk, ...]
    """
    chunks = []

    start_line = node.lineno
    end_line = node.end_lineno or node.lineno
    content = "\n".join(lines[start_line - 1 : end_line])
    docstring = ast.get_docstring(node) or ""

    # Add the class itself as a chunk
    chunks.append(CodeChunkData(
        name=node.name,
        chunk_type="class",
        content=content,
        start_line=start_line,
        end_line=end_line,
        language="python",
        file_path=file_path,
        docstring=docstring,
    ))

    # Add each method as its own chunk
    for child in node.body:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            chunks.append(_extract_function(
                child, lines, file_path, parent_name=node.name
            ))

    return chunks


def extract_imports(content: str) -> list[str]:
    """
    Extract all import statements from a Python file.
    Used by the graph builder in Week 3.
    Returns list of module names being imported.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return imports


def extract_function_calls(content: str) -> list[str]:
    """
    Extract all function calls from a Python file.
    Used by the call graph builder in Week 3.
    Returns list of called function names.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)

    return list(set(calls))  # deduplicate