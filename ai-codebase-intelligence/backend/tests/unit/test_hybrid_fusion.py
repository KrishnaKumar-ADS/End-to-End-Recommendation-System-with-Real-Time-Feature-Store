"""Unit tests for search.hybrid_fusion."""

from search.hybrid_fusion import bm25_only, reciprocal_rank_fusion, vector_only


def _vector(doc_id: str, score: float) -> dict:
    return {"id": doc_id, "score": score, "name": f"fn_{doc_id}"}


def _bm25(doc_id: str, score: float) -> dict:
    return {"id": doc_id, "bm25_score": score, "name": f"fn_{doc_id}"}


def test_rrf_prefers_doc_in_both_lists():
    fused = reciprocal_rank_fusion([_vector("A", 0.9), _vector("B", 0.8)], [_bm25("A", 5.0), _bm25("C", 4.0)])
    assert fused[0]["id"] == "A"


def test_rrf_contains_all_unique_documents():
    fused = reciprocal_rank_fusion([_vector("A", 0.9), _vector("B", 0.8)], [_bm25("B", 5.0), _bm25("C", 4.0)])
    assert {item["id"] for item in fused} == {"A", "B", "C"}


def test_rrf_sorted_descending():
    fused = reciprocal_rank_fusion([_vector("A", 0.9), _vector("B", 0.8)], [_bm25("C", 5.0), _bm25("D", 4.0)])
    scores = [item["hybrid_score"] for item in fused]
    assert scores == sorted(scores, reverse=True)


def test_vector_only_shape():
    items = vector_only([_vector("A", 0.9), _vector("B", 0.8)], top_k=1)
    assert len(items) == 1
    assert items[0]["bm25_rank"] is None
    assert items[0]["vector_rank"] == 1


def test_bm25_only_shape():
    items = bm25_only([_bm25("A", 5.0), _bm25("B", 2.0)], top_k=1)
    assert len(items) == 1
    assert items[0]["vector_rank"] is None
    assert items[0]["bm25_rank"] == 1
