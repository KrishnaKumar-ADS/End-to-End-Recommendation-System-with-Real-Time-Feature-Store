"""Thin embedding-cache wrapper around the shared CacheManager."""

from __future__ import annotations

from caching.cache_manager import get_cache_manager


def get_cached_embedding(text: str) -> list[float] | None:
    return get_cache_manager().get_embedding(text)


def set_cached_embedding(text: str, vector: list[float]) -> None:
    get_cache_manager().set_embedding(text, vector)
