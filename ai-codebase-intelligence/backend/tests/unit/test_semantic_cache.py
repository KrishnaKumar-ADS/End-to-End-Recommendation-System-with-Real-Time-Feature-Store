"""Unit tests for semantic_cache.answer_cache."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from semantic_cache.answer_cache import SemanticAnswerCache

FAKE_VECTOR = [0.01] * 768
FAKE_REPO = "test-repo-uuid"


def _make_hit(score: float, repo_id: str = FAKE_REPO, expires_delta: int = 3600):
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "answer": "Authentication uses bcrypt.",
        "sources": [{"file": "auth.py"}],
        "quality_score": {"faithfulness": 0.9, "overall": 0.88},
        "repo_id": repo_id,
        "provider_used": "openrouter",
        "model_used": "qwen/qwen-2.5-coder-32b-instruct",
        "cached_at": int(time.time()),
        "expires_at": int(time.time()) + expires_delta,
    }
    return hit


class TestSemanticAnswerCacheLookup:
    @pytest.mark.asyncio
    async def test_returns_hit_when_similarity_above_threshold(self):
        cache = SemanticAnswerCache()
        cache._collection_ready = True

        mock_client = MagicMock()
        mock_client.search.return_value = [_make_hit(score=0.95)]
        cache._client = mock_client

        result = await cache.lookup(FAKE_VECTOR, FAKE_REPO, threshold=0.92)

        assert result.found is True
        assert result.similarity == 0.95
        assert result.cached_answer is not None
        assert result.cached_answer.answer == "Authentication uses bcrypt."

    @pytest.mark.asyncio
    async def test_returns_miss_when_similarity_below_threshold(self):
        cache = SemanticAnswerCache()
        cache._collection_ready = True

        mock_client = MagicMock()
        mock_client.search.return_value = [_make_hit(score=0.85)]
        cache._client = mock_client

        result = await cache.lookup(FAKE_VECTOR, FAKE_REPO, threshold=0.92)

        assert result.found is False

    @pytest.mark.asyncio
    async def test_returns_miss_when_no_results(self):
        cache = SemanticAnswerCache()
        cache._collection_ready = True

        mock_client = MagicMock()
        mock_client.search.return_value = []
        cache._client = mock_client

        result = await cache.lookup(FAKE_VECTOR, FAKE_REPO)

        assert result.found is False
        assert result.similarity == 0.0

    @pytest.mark.asyncio
    async def test_returns_miss_on_qdrant_exception(self):
        cache = SemanticAnswerCache()
        cache._collection_ready = True

        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Qdrant connection refused")
        cache._client = mock_client

        result = await cache.lookup(FAKE_VECTOR, FAKE_REPO)

        assert result.found is False


class TestSemanticAnswerCacheStore:
    @pytest.mark.asyncio
    async def test_store_calls_qdrant_upsert(self):
        cache = SemanticAnswerCache()
        cache._collection_ready = True

        mock_client = MagicMock()
        cache._client = mock_client

        await cache.store(
            question_vector=FAKE_VECTOR,
            answer="Login uses verify_password().",
            sources=[{"file": "auth.py"}],
            quality_score={"faithfulness": 0.9},
            repo_id=FAKE_REPO,
        )

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args
        assert call_kwargs.kwargs["collection_name"] == "semantic_answer_cache"

    @pytest.mark.asyncio
    async def test_store_does_not_raise_on_qdrant_failure(self):
        cache = SemanticAnswerCache()
        cache._collection_ready = True

        mock_client = MagicMock()
        mock_client.upsert.side_effect = Exception("Qdrant write failed")
        cache._client = mock_client

        await cache.store(
            question_vector=FAKE_VECTOR,
            answer="Some answer that is long enough to be stored in the cache.",
            sources=[],
            quality_score={},
            repo_id=FAKE_REPO,
        )

    @pytest.mark.asyncio
    async def test_invalidate_repo_calls_delete(self):
        cache = SemanticAnswerCache()
        cache._collection_ready = True

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.deleted_count = 7
        mock_client.delete.return_value = mock_result
        cache._client = mock_client

        deleted = await cache.invalidate_repo(FAKE_REPO)
        mock_client.delete.assert_called_once()
        assert deleted == 7


class TestSemanticAnswerCacheGetStats:
    @pytest.mark.asyncio
    async def test_get_stats_returns_dict(self):
        cache = SemanticAnswerCache()
        cache._collection_ready = True

        mock_client = MagicMock()
        mock_info = MagicMock()
        mock_info.points_count = 42
        mock_info.status = "green"
        mock_client.get_collection.return_value = mock_info
        cache._client = mock_client

        stats = await cache.get_stats()
        assert stats["total_points"] == 42
        assert stats["collection_name"] == "semantic_answer_cache"

    @pytest.mark.asyncio
    async def test_get_stats_returns_error_on_failure(self):
        cache = SemanticAnswerCache()
        cache._collection_ready = True

        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Qdrant down")
        cache._client = mock_client

        stats = await cache.get_stats()
        assert "error" in stats["status"]
