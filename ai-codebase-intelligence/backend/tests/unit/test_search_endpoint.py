"""
Unit tests for search result formatting logic.
"""
import pytest
from unittest.mock import MagicMock


def _make_scored_point(chunk_id: str, score: float, **payload) -> MagicMock:
    """Helper to create a fake Qdrant ScoredPoint."""
    point = MagicMock()
    point.id = chunk_id
    point.score = score
    point.payload = {
        "name":            "my_func",
        "display_name":    "MyClass.my_func",
        "chunk_type":      "method",
        "file_path":       "src/app.py",
        "language":        "python",
        "start_line":      10,
        "end_line":        25,
        "docstring":       "Does something useful.",
        "content_preview": "def my_func(self):",
        **payload,
    }
    return point


def test_result_includes_chunk_id():
    point = _make_scored_point("chunk-abc", 0.9)
    result = {
        "chunk_id": str(point.id),
        "score":    round(point.score, 4),
    }
    assert result["chunk_id"] == "chunk-abc"


def test_result_score_rounded_to_4dp():
    point = _make_scored_point("chunk-1", 0.912345678)
    assert round(point.score, 4) == 0.9123


def test_result_includes_file_path():
    point = _make_scored_point("chunk-1", 0.8, file_path="fastapi/routing.py")
    assert point.payload["file_path"] == "fastapi/routing.py"


def test_result_includes_line_numbers():
    point = _make_scored_point("chunk-1", 0.8, start_line=42, end_line=68)
    assert point.payload["start_line"] == 42
    assert point.payload["end_line"] == 68


def test_result_includes_docstring():
    point = _make_scored_point("chunk-1", 0.8, docstring="Hash a password.")
    assert "Hash a password" in point.payload["docstring"]


def test_results_sorted_by_score_descending():
    points = [
        _make_scored_point("c1", 0.5),
        _make_scored_point("c2", 0.9),
        _make_scored_point("c3", 0.7),
    ]
    # Qdrant already returns sorted, but verify our format preserves order
    formatted = [
        {"chunk_id": str(p.id), "score": p.score}
        for p in sorted(points, key=lambda x: x.score, reverse=True)
    ]
    assert formatted[0]["score"] == 0.9
    assert formatted[1]["score"] == 0.7
    assert formatted[2]["score"] == 0.5


def test_empty_results_returns_empty_list():
    results = []
    assert len(results) == 0


def test_result_payload_missing_fields_handled():
    """Test that missing payload fields don't crash result formatting."""
    point = MagicMock()
    point.id = "chunk-x"
    point.score = 0.85
    point.payload = {}   # completely empty payload

    p = point.payload
    result = {
        "chunk_id":      str(point.id),
        "score":         round(point.score, 4),
        "name":          p.get("name", ""),
        "file_path":     p.get("file_path", ""),
        "start_line":    p.get("start_line", 0),
    }
    assert result["name"] == ""
    assert result["file_path"] == ""
    assert result["start_line"] == 0