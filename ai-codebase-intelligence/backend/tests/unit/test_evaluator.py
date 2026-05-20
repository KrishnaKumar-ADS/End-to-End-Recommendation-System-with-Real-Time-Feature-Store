"""Unit tests for search.evaluator."""

import pytest

from search.evaluator import (
    BUILTIN_TEST_QUERIES,
    _compute_dcg,
    _compute_ndcg,
    evaluate_query_results,
)


def test_compute_dcg_rank_discounting():
    assert _compute_dcg([1, 0], 2) > _compute_dcg([0, 1], 2)


def test_compute_ndcg_bounds():
    value = _compute_ndcg([1, 0, 1], 3)
    assert 0.0 <= value <= 1.0


def test_evaluate_query_results_hit_rank1():
    result = evaluate_query_results(
        query="password hashing",
        relevant_names=["hash_password"],
        result_names=["hash_password", "verify_token"],
        k=10,
    )
    assert result.reciprocal_rank == pytest.approx(1.0)
    assert result.first_relevant_rank == 1


def test_evaluate_query_results_hit_rank2():
    result = evaluate_query_results(
        query="password hashing",
        relevant_names=["hash_password"],
        result_names=["verify_token", "hash_password"],
        k=10,
    )
    assert result.reciprocal_rank == pytest.approx(0.5)
    assert result.first_relevant_rank == 2


def test_evaluate_query_results_no_hit():
    result = evaluate_query_results(
        query="password hashing",
        relevant_names=["hash_password"],
        result_names=["verify_token", "database_connect"],
        k=10,
    )
    assert result.reciprocal_rank == 0.0
    assert result.first_relevant_rank is None


def test_builtin_queries_non_empty():
    assert BUILTIN_TEST_QUERIES
    assert all(query.query for query in BUILTIN_TEST_QUERIES)
