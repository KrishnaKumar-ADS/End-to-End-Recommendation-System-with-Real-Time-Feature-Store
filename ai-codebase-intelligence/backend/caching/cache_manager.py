"""Unified Redis cache manager with stats for embeddings, search, and graph expansion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import redis

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class _CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.total) if self.total else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 3),
        }


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:40]


class CacheManager:
    """Centralized cache interface with per-cache hit/miss accounting."""

    EMBEDDING_TTL_SECONDS = 60 * 60 * 24 * 7
    SEARCH_TTL_SECONDS = 60 * 5
    GRAPH_TTL_SECONDS = 60 * 10

    def __init__(self) -> None:
        self._redis = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
        self._stats = {
            "embedding": _CacheStats(),
            "search": _CacheStats(),
            "graph": _CacheStats(),
        }

    def get_embedding(self, text: str) -> list[float] | None:
        key = f"emb:{_hash_text(text)}"
        try:
            raw = self._redis.get(key)
            if raw is None:
                self._stats["embedding"].misses += 1
                return None
            self._stats["embedding"].hits += 1
            data = json.loads(raw)
            return data if isinstance(data, list) else None
        except Exception as exc:
            logger.debug("embedding_cache_get_failed", error=str(exc))
            self._stats["embedding"].misses += 1
            return None

    def set_embedding(self, text: str, vector: list[float]) -> None:
        key = f"emb:{_hash_text(text)}"
        try:
            self._redis.setex(key, self.EMBEDDING_TTL_SECONDS, json.dumps(vector))
        except Exception as exc:
            logger.debug("embedding_cache_set_failed", error=str(exc))

    def _search_key(self, query: str, repo_id: str, **kwargs: Any) -> str:
        payload = {"query": query, "repo_id": repo_id, **kwargs}
        encoded = json.dumps(payload, sort_keys=True)
        return f"search:{_hash_text(encoded)}"

    def get_search_results(self, query: str, repo_id: str, **kwargs: Any) -> list[dict] | None:
        key = self._search_key(query, repo_id, **kwargs)
        try:
            raw = self._redis.get(key)
            if raw is None:
                self._stats["search"].misses += 1
                return None
            self._stats["search"].hits += 1
            data = json.loads(raw)
            return data if isinstance(data, list) else None
        except Exception as exc:
            logger.debug("search_cache_get_failed", error=str(exc))
            self._stats["search"].misses += 1
            return None

    def set_search_results(self, query: str, repo_id: str, results: list[dict], **kwargs: Any) -> None:
        key = self._search_key(query, repo_id, **kwargs)
        try:
            self._redis.setex(key, self.SEARCH_TTL_SECONDS, json.dumps(results))
            self._index_repo_key(repo_id, key)
        except Exception as exc:
            logger.debug("search_cache_set_failed", error=str(exc))

    def _graph_key(self, node_ids: list[str], repo_id: str, max_depth: int) -> str:
        payload = {
            "node_ids": sorted(node_ids),
            "repo_id": repo_id,
            "max_depth": max_depth,
        }
        encoded = json.dumps(payload, sort_keys=True)
        return f"graph:{_hash_text(encoded)}"

    def get_graph_expansion(self, node_ids: list[str], repo_id: str, max_depth: int) -> list[dict] | None:
        key = self._graph_key(node_ids, repo_id, max_depth)
        try:
            raw = self._redis.get(key)
            if raw is None:
                self._stats["graph"].misses += 1
                return None
            self._stats["graph"].hits += 1
            data = json.loads(raw)
            return data if isinstance(data, list) else None
        except Exception as exc:
            logger.debug("graph_cache_get_failed", error=str(exc))
            self._stats["graph"].misses += 1
            return None

    def set_graph_expansion(self, node_ids: list[str], repo_id: str, max_depth: int, nodes: list[dict]) -> None:
        key = self._graph_key(node_ids, repo_id, max_depth)
        try:
            self._redis.setex(key, self.GRAPH_TTL_SECONDS, json.dumps(nodes))
            self._index_repo_key(repo_id, key)
        except Exception as exc:
            logger.debug("graph_cache_set_failed", error=str(exc))

    def invalidate_repo(self, repo_id: str) -> int:
        """Invalidate search+graph cache entries tracked for a repo."""
        index_key = self._repo_index_key(repo_id)
        try:
            keys = list(self._redis.smembers(index_key))
            deleted = 0
            if keys:
                deleted += int(self._redis.delete(*keys))
            deleted += int(self._redis.delete(index_key))
            return deleted
        except Exception as exc:
            logger.warning("cache_invalidate_failed", repo_id=repo_id, error=str(exc))
            return 0

    def get_stats(self) -> dict[str, Any]:
        try:
            emb_count = len(self._redis.keys("emb:*"))
            search_count = len(self._redis.keys("search:*"))
            graph_count = len(self._redis.keys("graph:*"))
            mem_info = self._redis.info("memory")
            memory_bytes = int(mem_info.get("used_memory", 0))
        except Exception:
            emb_count = -1
            search_count = -1
            graph_count = -1
            memory_bytes = -1

        return {
            "embedding": {**self._stats["embedding"].to_dict(), "key_count": emb_count},
            "search": {**self._stats["search"].to_dict(), "key_count": search_count},
            "graph": {**self._stats["graph"].to_dict(), "key_count": graph_count},
            "redis_memory_bytes": memory_bytes,
        }

    @staticmethod
    def _repo_index_key(repo_id: str) -> str:
        return f"cache:index:{repo_id}"

    def _index_repo_key(self, repo_id: str, key: str) -> None:
        index_key = self._repo_index_key(repo_id)
        self._redis.sadd(index_key, key)
        self._redis.expire(index_key, self.EMBEDDING_TTL_SECONDS)


_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
