"""Qdrant-backed semantic cache for previously generated answers."""

from __future__ import annotations

import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from core.config import get_settings
from core.logging import get_logger
from semantic_cache.models import CacheLookupResult, CachedAnswer

logger = get_logger(__name__)

COLLECTION_NAME = "semantic_answer_cache"
DEFAULT_THRESHOLD = 0.92
DEFAULT_TTL_SECONDS = 60 * 60


class SemanticAnswerCache:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=10,
        )
        self._vector_size = settings.embedding_vector_dim
        self._collection_ready = False

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return

        try:
            existing = [item.name for item in self._client.get_collections().collections]
            if COLLECTION_NAME not in existing:
                self._client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
                )
            self._collection_ready = True
        except Exception as exc:
            logger.warning("semantic_cache_collection_unavailable", error=str(exc))
            self._collection_ready = False

    async def lookup(
        self,
        question_vector: list[float],
        repo_id: str,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> CacheLookupResult:
        self._ensure_collection()
        if not self._collection_ready:
            return CacheLookupResult(found=False, reason="collection_unavailable")

        try:
            hits = self._client.search(
                collection_name=COLLECTION_NAME,
                query_vector=question_vector,
                limit=1,
                query_filter=Filter(
                    must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]
                ),
                with_payload=True,
                with_vectors=False,
            )

            if not hits:
                return CacheLookupResult(found=False, similarity=0.0)

            top_hit = hits[0]
            similarity = float(top_hit.score)
            if similarity < threshold:
                return CacheLookupResult(found=False, similarity=similarity)

            payload = dict(top_hit.payload or {})
            now_ts = int(time.time())
            expires_at = int(payload.get("expires_at") or 0)
            if expires_at and expires_at <= now_ts:
                return CacheLookupResult(found=False, similarity=similarity, reason="expired")

            cached_answer = CachedAnswer(
                answer=str(payload.get("answer") or ""),
                sources=payload.get("sources") if isinstance(payload.get("sources"), list) else [],
                quality_score=payload.get("quality_score") if isinstance(payload.get("quality_score"), dict) else None,
                repo_id=str(payload.get("repo_id") or repo_id),
                provider_used=str(payload.get("provider_used") or ""),
                model_used=str(payload.get("model_used") or ""),
                cached_at=int(payload.get("cached_at") or 0),
                expires_at=expires_at,
            )
            return CacheLookupResult(
                found=True,
                similarity=similarity,
                cached_answer=cached_answer,
            )
        except Exception as exc:
            logger.warning("semantic_cache_lookup_failed", error=str(exc), repo_id=repo_id)
            return CacheLookupResult(found=False, reason=str(exc))

    async def store(
        self,
        question_vector: list[float],
        answer: str,
        sources: list[dict],
        quality_score: dict,
        repo_id: str,
        provider_used: str = "openrouter",
        model_used: str = "",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._ensure_collection()
        if not self._collection_ready:
            return

        now_ts = int(time.time())
        payload = {
            "answer": answer,
            "sources": sources,
            "quality_score": quality_score,
            "repo_id": repo_id,
            "provider_used": provider_used,
            "model_used": model_used,
            "cached_at": now_ts,
            "expires_at": now_ts + ttl_seconds,
        }

        try:
            self._client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=question_vector,
                        payload=payload,
                    )
                ],
                wait=False,
            )
        except Exception as exc:
            logger.warning("semantic_cache_store_failed", error=str(exc), repo_id=repo_id)

    async def invalidate_repo(self, repo_id: str) -> int:
        self._ensure_collection()
        if not self._collection_ready:
            return 0

        try:
            result = self._client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]
                ),
                wait=True,
            )
            return int(getattr(result, "deleted_count", 0) or 0)
        except Exception as exc:
            logger.warning("semantic_cache_invalidate_failed", repo_id=repo_id, error=str(exc))
            return 0

    async def get_stats(self) -> dict:
        self._ensure_collection()
        if not self._collection_ready:
            return {
                "collection_name": COLLECTION_NAME,
                "status": "error: unavailable",
                "total_points": 0,
            }

        try:
            info = self._client.get_collection(COLLECTION_NAME)
            return {
                "collection_name": COLLECTION_NAME,
                "status": str(info.status),
                "total_points": int(info.points_count or 0),
            }
        except Exception as exc:
            return {
                "collection_name": COLLECTION_NAME,
                "status": f"error: {exc}",
                "total_points": 0,
            }


_semantic_answer_cache: SemanticAnswerCache | None = None


def get_semantic_answer_cache() -> SemanticAnswerCache:
    global _semantic_answer_cache
    if _semantic_answer_cache is None:
        _semantic_answer_cache = SemanticAnswerCache()
    return _semantic_answer_cache
