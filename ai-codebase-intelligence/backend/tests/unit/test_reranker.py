"""Unit tests for search.reranker (model calls mocked)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def candidates():
    return [
        {"id": "1", "content": "hash password code", "hybrid_score": 0.03, "hybrid_rank": 1},
        {"id": "2", "content": "jwt token code", "hybrid_score": 0.02, "hybrid_rank": 2},
        {"id": "3", "content": "db connect code", "hybrid_score": 0.01, "hybrid_rank": 3},
    ]


def test_rerank_sorts_by_predicted_score(candidates):
    from search import reranker as module

    model = MagicMock()
    model.predict.return_value = [0.1, 0.2, 0.9]

    with patch.object(module, "_reranker_model", model):
        output, was_reranked = module.rerank("database", candidates, top_k=3)

    assert was_reranked is True
    assert [item["id"] for item in output] == ["3", "2", "1"]


def test_rerank_fallback_when_model_unavailable(candidates):
    from search import reranker as module

    with patch.object(module, "_reranker_model", module._FAILED):
        output, was_reranked = module.rerank("query", candidates, top_k=2)

    assert was_reranked is False
    assert len(output) == 2
    assert output[0]["final_rank"] == 1


def test_rerank_fallback_when_predict_raises(candidates):
    from search import reranker as module

    model = MagicMock()
    model.predict.side_effect = RuntimeError("boom")

    with patch.object(module, "_reranker_model", model):
        output, was_reranked = module.rerank("query", candidates, top_k=2)

    assert was_reranked is False
    assert len(output) == 2


@pytest.mark.asyncio
async def test_rerank_async_returns_tuple(candidates):
    from search import reranker as module

    with patch.object(module, "_reranker_model", module._FAILED):
        output, was_reranked = await module.rerank_async("query", candidates, top_k=2)

    assert isinstance(output, list)
    assert isinstance(was_reranked, bool)
