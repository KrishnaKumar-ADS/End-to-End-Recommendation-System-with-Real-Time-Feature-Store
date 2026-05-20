"""Redis-backed caching helpers for embeddings, search, and graph expansion."""

from caching.cache_manager import CacheManager, get_cache_manager

__all__ = ["CacheManager", "get_cache_manager"]
