"""
Unit tests for graph/hierarchy_builder.py.
All Neo4j calls are mocked — no real database required.
"""
import pytest
from unittest.mock import patch, MagicMock, call
from graph.hierarchy_builder import (
    build_class_hierarchy,
    delete_class_hierarchy,
    get_hierarchy_stats,
    HierarchyBuildResult,
)

REPO_ID = "repo-uuid-test"


def _make_mock_db(class_chunks: list[dict]) -> MagicMock:
    """Helper: create a mock SQLAlchemy session that returns given class chunks."""
    db = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        MagicMock(
            id=c["id"],
            name=c["name"],
            content=c["content"],
            file_path=c.get("file_path", "app/models.py"),
        )
        for c in class_chunks
    ]
    db.execute.return_value = mock_result
    return db


class TestBuildClassHierarchy:

    @patch("graph.hierarchy_builder.get_session")
    def test_empty_repo_returns_zero_counts(self, mock_session):
        db = _make_mock_db([])
        result = build_class_hierarchy(REPO_ID, db)
        assert result.classes_processed == 0
        assert result.total_edges == 0

    @patch("graph.hierarchy_builder.get_session")
    def test_no_base_class_no_edges(self, mock_session):
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        db = _make_mock_db([{
            "id": "c1", "name": "Animal",
            "content": "class Animal:\n    pass\n",
        }])
        result = build_class_hierarchy(REPO_ID, db)
        assert result.classes_processed == 1
        assert result.inherits_from_edges == 0

    @patch("graph.hierarchy_builder.get_session")
    def test_single_inheritance_creates_inherits_edge(self, mock_session):
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        db = _make_mock_db([
            {"id": "c1", "name": "Animal", "content": "class Animal:\n    pass\n"},
            {"id": "c2", "name": "Dog",    "content": "class Dog(Animal):\n    pass\n"},
        ])
        result = build_class_hierarchy(REPO_ID, db)
        assert result.classes_processed == 2
        assert result.inherits_from_edges == 1

    @patch("graph.hierarchy_builder.get_session")
    def test_mixin_creates_mixes_in_edge(self, mock_session):
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        db = _make_mock_db([
            {"id": "c1", "name": "FetchMixin", "content": "class FetchMixin:\n    pass\n"},
            {"id": "c2", "name": "Dog",        "content": "class Dog(Animal, FetchMixin):\n    pass\n"},
        ])
        result = build_class_hierarchy(REPO_ID, db)
        assert result.mixes_in_edges == 1

    @patch("graph.hierarchy_builder.get_session")
    def test_abc_creates_implements_edge(self, mock_session):
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        db = _make_mock_db([
            {"id": "c1", "name": "Shape",  "content": "from abc import ABC\nclass Shape(ABC):\n    pass\n"},
            {"id": "c2", "name": "Circle", "content": "class Circle(Shape):\n    pass\n"},
        ])
        result = build_class_hierarchy(REPO_ID, db)
        # Shape(ABC) → is_abstract=True on ABC base → implements edge
        assert result.implements_edges >= 1

    @patch("graph.hierarchy_builder.get_session")
    def test_syntax_error_chunk_skipped_not_raised(self, mock_session):
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        db = _make_mock_db([
            {"id": "c1", "name": "Broken", "content": "class Broken(\n"},
            {"id": "c2", "name": "Dog",    "content": "class Dog(Animal):\n    pass\n"},
        ])
        # Should not raise — bad chunk is skipped with an error logged
        result = build_class_hierarchy(REPO_ID, db)
        assert result.classes_processed == 1   # only Dog succeeded
        assert len(result.errors) == 1

    def test_hierarchy_build_result_summary(self):
        r = HierarchyBuildResult(
            repo_id=REPO_ID,
            classes_processed=10,
            inherits_from_edges=5,
            implements_edges=2,
            mixes_in_edges=1,
        )
        assert "10 classes" in r.summary()
        assert "5 INHERITS_FROM" in r.summary()
        assert r.total_edges == 8

    @patch("graph.hierarchy_builder.get_session")
    def test_delete_hierarchy_runs_cypher(self, mock_session):
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        delete_class_hierarchy(REPO_ID)
        mock_ctx.run.assert_called_once()

    @patch("graph.hierarchy_builder.get_session")
    def test_get_hierarchy_stats_returns_dict(self, mock_session):
        mock_ctx = MagicMock()
        mock_record = MagicMock()
        mock_record.__iter__ = MagicMock(return_value=iter([
            ("total_classes", 10),
            ("inherits_from_edges", 5),
            ("implements_edges", 2),
            ("mixes_in_edges", 1),
        ]))
        dict(mock_record)
        mock_ctx.run.return_value.single.return_value = {
            "total_classes": 10, "inherits_from_edges": 5,
            "implements_edges": 2, "mixes_in_edges": 1,
        }
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        stats = get_hierarchy_stats(REPO_ID)
        assert "total_classes" in stats