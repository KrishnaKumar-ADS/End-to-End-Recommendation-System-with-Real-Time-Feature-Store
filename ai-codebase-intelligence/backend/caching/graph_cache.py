"""Thin graph-expansion cache wrapper around the shared CacheManager."""

from __future__ import annotations

from caching.cache_manager import get_cache_manager


def get_cached_graph(node_ids: list[str], repo_id: str, max_depth: int) -> list[dict] | None:
    return get_cache_manager().get_graph_expansion(node_ids=node_ids, repo_id=repo_id, max_depth=max_depth)


def set_cached_graph(node_ids: list[str], repo_id: str, max_depth: int, nodes: list[dict]) -> None:
    get_cache_manager().set_graph_expansion(node_ids=node_ids, repo_id=repo_id, max_depth=max_depth, nodes=nodes)
