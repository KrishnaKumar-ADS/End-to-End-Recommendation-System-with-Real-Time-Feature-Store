"""Cache inspection and invalidation endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from caching.cache_manager import get_cache_manager
from semantic_cache.answer_cache import get_semantic_answer_cache

router = APIRouter(prefix="/api/v1/cache", tags=["cache"])


@router.get("/stats", summary="Get cache stats for embedding/search/graph caches.")
async def cache_stats() -> dict:
    return get_cache_manager().get_stats()


@router.delete("/{repo_id}", summary="Invalidate search/graph cache entries for a repo.")
async def invalidate_repo_cache(repo_id: str) -> dict:
    deleted = get_cache_manager().invalidate_repo(repo_id)
    semantic_deleted = await get_semantic_answer_cache().invalidate_repo(repo_id)
    return {
        "repo_id": repo_id,
        "keys_deleted": deleted + semantic_deleted,
        "search_graph_deleted": deleted,
        "semantic_deleted": semantic_deleted,
        "message": f"Invalidated {deleted + semantic_deleted} cache keys for repo {repo_id}.",
    }
