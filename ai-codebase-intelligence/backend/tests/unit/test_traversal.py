"""
Unit tests for graphs/traversal.py — mocks run_query.
Verifies that the correct Cypher parameters are passed.
"""

import pytest
from unittest.mock import patch
from graphs.traversal import (
    get_call_chain, get_callers, get_file_dependencies,
    get_repo_graph_for_api, find_path_between_functions,
)


@patch("graphs.traversal.run_query")
def test_get_call_chain_passes_function_id(mock_rq):
    mock_rq.return_value = []
    get_call_chain("func-id-123", depth=3)
    assert mock_rq.call_args[0][1]["start_id"] == "func-id-123"

@patch("graphs.traversal.run_query")
def test_get_callers_passes_target_id(mock_rq):
    mock_rq.return_value = []
    get_callers("target-456")
    assert mock_rq.call_args[0][1]["target_id"] == "target-456"

@patch("graphs.traversal.run_query")
def test_get_file_dependencies_passes_file_id(mock_rq):
    mock_rq.return_value = []
    get_file_dependencies("file-789")
    assert mock_rq.call_args[0][1]["file_id"] == "file-789"

@patch("graphs.traversal.run_query")
def test_get_repo_graph_returns_nodes_and_edges_keys(mock_rq):
    mock_rq.side_effect = [
        [{"id": "f1", "label": "Function", "name": "verify_password", "file": "auth.py", "start_line": 10}],
        [{"source": "f1", "target": "f2", "type": "CALLS"}],
    ]
    result = get_repo_graph_for_api("repo-123")
    assert "nodes" in result
    assert "edges" in result
    assert len(result["nodes"]) == 1

@patch("graphs.traversal.run_query")
def test_get_repo_graph_empty_repo(mock_rq):
    mock_rq.return_value = []
    result = get_repo_graph_for_api("empty-repo")
    assert result == {"nodes": [], "edges": []}

@patch("graphs.traversal.run_query")
def test_find_path_passes_both_ids(mock_rq):
    mock_rq.return_value = []
    find_path_between_functions("src-id", "tgt-id", max_depth=4)
    params = mock_rq.call_args[0][1]
    assert params["source_id"] == "src-id"
    assert params["target_id"] == "tgt-id"

@patch("graphs.traversal.run_query")
def test_get_call_chain_returns_results_passthrough(mock_rq):
    mock_rq.return_value = [
        {"id": "id2", "name": "hash_password", "file": "utils.py", "start_line": 5, "end_line": 15, "depth": 1}
    ]
    result = get_call_chain("id1")
    assert len(result) == 1
    assert result[0]["name"] == "hash_password"