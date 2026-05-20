"""
Unit tests for graphs/graph_builder.py dataclasses and RepoGraph helpers.
Pure Python — no database, no Neo4j.
"""

import pytest
from graphs.graph_builder import GraphNode, GraphEdge, RepoGraph
from graphs.schema import NodeLabel, RelType


def _node(nid: str, label: str, name: str = "test") -> GraphNode:
    return GraphNode(node_id=nid, label=label, properties={"id": nid, "name": name})

def _edge(src: str, tgt: str, rel: str) -> GraphEdge:
    return GraphEdge(source_id=src, target_id=tgt, rel_type=rel)


def test_graph_node_repr_includes_label_and_name():
    node = _node("abc", NodeLabel.FUNCTION, "verify_password")
    assert "Function" in repr(node)
    assert "verify_password" in repr(node)

def test_graph_edge_repr_includes_rel_type():
    edge = _edge("a", "b", RelType.CALLS)
    assert "CALLS" in repr(edge)

def test_repo_graph_function_nodes_filter():
    graph = RepoGraph(repo_id="r1")
    graph.nodes = [_node("f1", NodeLabel.FUNCTION), _node("c1", NodeLabel.CLASS), _node("fi1", NodeLabel.FILE)]
    assert len(graph.function_nodes) == 1
    assert graph.function_nodes[0].node_id == "f1"

def test_repo_graph_class_nodes_filter():
    graph = RepoGraph(repo_id="r1")
    graph.nodes = [_node("f1", NodeLabel.FUNCTION), _node("c1", NodeLabel.CLASS)]
    assert len(graph.class_nodes) == 1

def test_repo_graph_file_nodes_filter():
    graph = RepoGraph(repo_id="r1")
    graph.nodes = [_node("fi1", NodeLabel.FILE), _node("f1", NodeLabel.FUNCTION)]
    assert len(graph.file_nodes) == 1

def test_repo_graph_call_edges_filter():
    graph = RepoGraph(repo_id="r1")
    graph.edges = [
        _edge("a", "b", RelType.CALLS),
        _edge("a", "b", RelType.IMPORTS),
        _edge("c", "d", RelType.CALLS),
    ]
    assert len(graph.call_edges) == 2
    assert len(graph.import_edges) == 1

def test_repo_graph_summary_keys():
    graph = RepoGraph(repo_id="test-repo")
    graph.nodes = [_node("f1", NodeLabel.FUNCTION), _node("fi1", NodeLabel.FILE)]
    graph.edges = [_edge("fi1", "f1", RelType.CONTAINS)]
    s = graph.summary()
    for key in ["repo_id", "total_nodes", "function_nodes", "file_nodes", "call_edges", "import_edges"]:
        assert key in s

def test_repo_graph_summary_counts():
    graph = RepoGraph(repo_id="r1")
    graph.nodes = [_node("f1", NodeLabel.FUNCTION), _node("fi1", NodeLabel.FILE)]
    graph.edges = [_edge("f1", "f2", RelType.CALLS), _edge("fi1", "fi2", RelType.IMPORTS)]
    s = graph.summary()
    assert s["total_nodes"] == 2
    assert s["call_edges"] == 1
    assert s["import_edges"] == 1

def test_repo_graph_empty_is_valid():
    graph = RepoGraph(repo_id="empty")
    assert graph.nodes == []
    assert graph.edges == []
    assert graph.summary()["total_nodes"] == 0

def test_graph_edge_default_properties_empty():
    edge = GraphEdge(source_id="a", target_id="b", rel_type="CALLS")
    assert edge.properties == {}