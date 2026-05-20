"""Thin search-cache wrapper around the shared CacheManager."""

from __future__ import annotations

from typing import Any

from caching.cache_manager import get_cache_manager


def get_cached_search(query: str, repo_id: str, **kwargs: Any) -> list[dict] | None:
    return get_cache_manager().get_search_results(query=query, repo_id=repo_id, **kwargs)


def set_cached_search(query: str, repo_id: str, results: list[dict], **kwargs: Any) -> None:
    get_cache_manager().set_search_results(query=query, repo_id=repo_id, results=results, **kwargs)
