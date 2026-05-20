"""
Unit tests for graphs/neo4j_writer.py — mocks the Neo4j session.
Verifies that the writer calls correct Cypher and respects batch sizes.
"""

import pytest
from unittest.mock import patch, MagicMock
from graphs.graph_builder import GraphNode, GraphEdge, RepoGraph
from graphs.schema import NodeLabel, RelType
from graphs.neo4j_writer import write_graph_to_neo4j, delete_repo_graph, BATCH_SIZE


def _func_node(nid: str) -> GraphNode:
    return GraphNode(
        node_id=nid, label=NodeLabel.FUNCTION,
        properties={"id": nid, "name": f"func_{nid}", "repo_id": "r1",
                    "file_path": "f.py", "start_line": 1, "end_line": 10,
                    "is_method": False, "docstring": "", "display_name": f"func_{nid}"},
    )

def _file_node(nid: str) -> GraphNode:
    return GraphNode(
        node_id=nid, label=NodeLabel.FILE,
        properties={"id": nid, "path": "auth.py", "language": "python", "repo_id": "r1"},
    )

def _repo_node(nid: str) -> GraphNode:
    return GraphNode(
        node_id=nid, label=NodeLabel.REPO,
        properties={"id": nid, "name": "repo", "github_url": "https://github.com/x/y", "repo_id": nid},
    )

def _mock_session():
    s = MagicMock()
    s.__enter__ = MagicMock(return_value=s)
    s.__exit__ = MagicMock(return_value=False)
    return s


@patch("graphs.neo4j_writer.get_session")
def test_write_graph_calls_session(mock_get_session):
    mock_get_session.return_value = _mock_session()
    graph = RepoGraph(repo_id="r1")
    graph.nodes = [_repo_node("r1"), _file_node("f1"), _func_node("fn1")]
    graph.edges = [GraphEdge("r1", "f1", RelType.CONTAINS_FILE), GraphEdge("f1", "fn1", RelType.CONTAINS)]
    result = write_graph_to_neo4j(graph)
    assert mock_get_session.called
    assert result["nodes_written"] >= 2

@patch("graphs.neo4j_writer.get_session")
def test_write_graph_result_has_expected_keys(mock_get_session):
    mock_get_session.return_value = _mock_session()
    graph = RepoGraph(repo_id="r1")
    graph.nodes = [_repo_node("r1"), _func_node("f1")]
    graph.edges = [GraphEdge("f1", "f2", RelType.CALLS)]
    result = write_graph_to_neo4j(graph)
    for key in ["nodes_written", "edges_written", "calls_edges", "imports_edges", "function_nodes"]:
        assert key in result

@patch("graphs.neo4j_writer.get_session")
def test_write_empty_graph_does_not_crash(mock_get_session):
    mock_get_session.return_value = _mock_session()
    graph = RepoGraph(repo_id="r1")
    result = write_graph_to_neo4j(graph)
    assert result["nodes_written"] == 0

@patch("graphs.neo4j_writer.get_session")
def test_batch_size_triggers_multiple_calls(mock_get_session):
    s = _mock_session()
    mock_get_session.return_value = s
    many_nodes = [_func_node(f"f{i}") for i in range(BATCH_SIZE + 50)]
    graph = RepoGraph(repo_id="r1")
    graph.nodes = [_repo_node("r1")] + many_nodes
    graph.edges = []
    write_graph_to_neo4j(graph)
    assert s.run.call_count >= 2

@patch("graphs.neo4j_writer.get_session")
def test_delete_repo_graph_stops_when_zero_deleted(mock_get_session):
    s = _mock_session()
    record = MagicMock()
    record.__getitem__ = MagicMock(return_value=0)
    s.run.return_value.single.return_value = record
    mock_get_session.return_value = s
    deleted = delete_repo_graph("repo-123")
    assert deleted == 0
    assert s.run.called