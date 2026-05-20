"""Unit tests for search.search_service orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeBM25Index:
    def __init__(self, results):
        self._results = results

    def search(self, query, top_k, chunk_type=None, language=None):
        return self._results[:top_k]


@pytest.mark.asyncio
async def test_search_vector_mode_normalizes_results():
    from search import search_service as module

    point = MagicMock()
    point.id = "c1"
    point.score = 0.91
    point.payload = {
        "name": "hash_password",
        "chunk_type": "function",
        "file_path": "auth.py",
        "content": "def hash_password(...)",
        "start_line": 1,
        "end_line": 2,
    }

    with patch("search.search_service.embed_query", return_value=[0.1, 0.2]):
        with patch("search.search_service.qdrant_search", return_value=[point]):
            result = await module.search(
                query="password hashing",
                repo_id="repo",
                db=MagicMock(),
                mode="vector",
                rerank=False,
            )

    assert result.results
    assert result.results[0].id == "c1"
    assert result.mode == "vector"


@pytest.mark.asyncio
async def test_search_hybrid_calls_fusion_and_rerank():
    from search import search_service as module

    point = MagicMock()
    point.id = "c1"
    point.score = 0.8
    point.payload = {
        "name": "hash_password",
        "chunk_type": "function",
        "file_path": "auth.py",
        "content": "def hash_password(...)",
        "start_line": 1,
        "end_line": 2,
    }

    bm25_results = [
        {
            "id": "c1",
            "name": "hash_password",
            "chunk_type": "function",
            "file_path": "auth.py",
            "content": "def hash_password(...)",
            "start_line": 1,
            "end_line": 2,
            "bm25_score": 5.0,
            "bm25_rank": 1,
        }
    ]

    with patch("search.search_service.embed_query", return_value=[0.1, 0.2]):
        with patch("search.search_service.qdrant_search", return_value=[point]):
            with patch("search.search_service.get_or_build_index", AsyncMock(return_value=_FakeBM25Index(bm25_results))):
                with patch("search.search_service.rerank_async", AsyncMock(return_value=(bm25_results, True))):
                    result = await module.search(
                        query="password hashing",
                        repo_id="repo",
                        db=MagicMock(),
                        mode="hybrid",
                        rerank=True,
                    )

    assert result.reranked is True
    assert result.mode == "hybrid"


@pytest.mark.asyncio
async def test_search_expand_query_uses_expander():
    from search import search_service as module

    point = MagicMock()
    point.id = "c1"
    point.score = 0.8
    point.payload = {
        "name": "hash_password",
        "chunk_type": "function",
        "file_path": "auth.py",
        "content": "def hash_password(...)",
        "start_line": 1,
        "end_line": 2,
    }

    with patch("search.search_service.expand_query", AsyncMock(return_value=["q1", "q2", "q3"])):
        with patch("search.search_service.embed_query", return_value=[0.1, 0.2]):
            with patch("search.search_service.qdrant_search", return_value=[point]):
                result = await module.search(
                    query="password hashing",
                    repo_id="repo",
                    db=MagicMock(),
                    mode="vector",
                    rerank=False,
                    expand_query_flag=True,
                )

    assert len(result.expanded_queries) == 3
