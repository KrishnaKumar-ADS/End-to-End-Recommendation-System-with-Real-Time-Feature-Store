"""
Unit tests for graphs/python_graph_extractor.py.
Pure Python AST logic — no Neo4j, no PostgreSQL.
"""

import ast
import pytest
from graphs.python_graph_extractor import (
    CallVisitor, ImportVisitor, ChunkIndex,
    extract_calls_from_chunk, extract_imports_from_file,
)
from graphs.graph_builder import GraphNode
from graphs.schema import NodeLabel, Props


# ── Helpers ───────────────────────────────────────────────────────────────────

def _func_node(nid: str, name: str, file_path: str, display_name: str = "") -> GraphNode:
    return GraphNode(
        node_id=nid,
        label=NodeLabel.FUNCTION,
        properties={
            Props.ID: nid,
            Props.NAME: name,
            Props.DISPLAY_NAME: display_name or name,
            Props.FILE_PATH: file_path,
        },
    )

def _file_node(nid: str, path: str) -> GraphNode:
    return GraphNode(
        node_id=nid,
        label="File",
        properties={Props.ID: nid, Props.PATH: path},
    )


# ── CallVisitor ───────────────────────────────────────────────────────────────

def test_call_visitor_simple_call():
    tree = ast.parse("hash_password(plain)")
    v = CallVisitor(); v.visit(tree)
    assert "hash_password" in v.calls

def test_call_visitor_method_call():
    tree = ast.parse("self.db.query(User)")
    v = CallVisitor(); v.visit(tree)
    assert "query" in v.calls

def test_call_visitor_multiple_calls():
    code = "validate_input(x)\nhash_password(y)\ngenerate_token(z)"
    tree = ast.parse(code)
    v = CallVisitor(); v.visit(tree)
    assert "validate_input" in v.calls
    assert "hash_password" in v.calls
    assert "generate_token" in v.calls

def test_call_visitor_nested_calls():
    tree = ast.parse("result = outer(inner(x))")
    v = CallVisitor(); v.visit(tree)
    assert "outer" in v.calls
    assert "inner" in v.calls

def test_call_visitor_no_calls():
    tree = ast.parse("x = 1 + 2")
    v = CallVisitor(); v.visit(tree)
    assert v.calls == []


# ── ImportVisitor ─────────────────────────────────────────────────────────────

def test_import_visitor_simple():
    tree = ast.parse("import os")
    v = ImportVisitor(); v.visit(tree)
    assert "os" in v.imports

def test_import_visitor_from():
    tree = ast.parse("from pathlib import Path")
    v = ImportVisitor(); v.visit(tree)
    assert "pathlib" in v.imports

def test_import_visitor_relative():
    tree = ast.parse("from . import utils")
    v = ImportVisitor(); v.visit(tree)
    assert any(i.startswith(".") for i in v.imports)

def test_import_visitor_multiple():
    code = "import os\nimport sys\nfrom pathlib import Path"
    tree = ast.parse(code)
    v = ImportVisitor(); v.visit(tree)
    assert "os" in v.imports
    assert "sys" in v.imports
    assert "pathlib" in v.imports


# ── ChunkIndex ────────────────────────────────────────────────────────────────

def test_chunk_index_build_by_name():
    nodes = [_func_node("id1", "hash_password", "utils.py")]
    index = ChunkIndex.build(nodes)
    assert "hash_password" in index.by_name
    assert "id1" in index.by_name["hash_password"]

def test_chunk_index_same_file_priority():
    nodes = [
        _func_node("id1", "process", "auth/service.py"),
        _func_node("id2", "process", "utils/helper.py"),
    ]
    index = ChunkIndex.build(nodes)
    result = index.resolve_call("process", "auth/service.py")
    assert "id1" in result
    assert "id2" not in result

def test_chunk_index_cross_file_fallback():
    nodes = [_func_node("id1", "hash_password", "utils/crypto.py")]
    index = ChunkIndex.build(nodes)
    result = index.resolve_call("hash_password", "auth/service.py")
    assert "id1" in result

def test_chunk_index_unknown_returns_empty():
    nodes = [_func_node("id1", "known_func", "utils.py")]
    index = ChunkIndex.build(nodes)
    assert index.resolve_call("unknown_function", "auth.py") == []

def test_chunk_index_empty_nodes():
    index = ChunkIndex.build([])
    assert index.resolve_call("anything", "file.py") == []


# ── extract_calls_from_chunk ──────────────────────────────────────────────────

def test_extract_calls_returns_edge():
    nodes = [
        _func_node("id_hash", "hash_password", "utils/crypto.py"),
        _func_node("id_verify", "verify_password", "auth/service.py"),
    ]
    index = ChunkIndex.build(nodes)
    content = "def verify_password(plain, hashed):\n    return hash_password(plain) == hashed"
    edges = extract_calls_from_chunk("id_verify", content, "auth/service.py", index)
    assert len(edges) == 1
    assert edges[0].source_id == "id_verify"
    assert edges[0].target_id == "id_hash"
    assert edges[0].rel_type == "CALLS"

def test_extract_calls_no_self_loop():
    nodes = [_func_node("id1", "recurse", "module.py")]
    index = ChunkIndex.build(nodes)
    content = "def recurse(n):\n    return recurse(n-1)"
    edges = extract_calls_from_chunk("id1", content, "module.py", index)
    assert all(e.target_id != "id1" for e in edges)

def test_extract_calls_deduplicates():
    nodes = [
        _func_node("id_hash", "hash_password", "utils.py"),
        _func_node("id_caller", "caller", "auth.py"),
    ]
    index = ChunkIndex.build(nodes)
    content = "def caller():\n    x = hash_password('a')\n    y = hash_password('b')\n    return x == y"
    edges = extract_calls_from_chunk("id_caller", content, "auth.py", index)
    hash_edges = [e for e in edges if e.target_id == "id_hash"]
    assert len(hash_edges) == 1

def test_extract_calls_handles_syntax_error():
    nodes = [_func_node("id1", "func", "file.py")]
    index = ChunkIndex.build(nodes)
    edges = extract_calls_from_chunk("id1", "def broken(::::", "file.py", index)
    assert edges == []

def test_extract_calls_empty_content():
    nodes = [_func_node("id1", "func", "file.py")]
    index = ChunkIndex.build(nodes)
    edges = extract_calls_from_chunk("id1", "", "file.py", index)
    assert edges == []