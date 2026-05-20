"""Cross-encoder reranker with graceful fallback when model is unavailable."""

from __future__ import annotations

import asyncio
from typing import Any

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_PASSAGE_CHARS = 2048

_NOT_LOADED = object()
_FAILED = object()
_reranker_model: Any = _NOT_LOADED


def _load_reranker() -> None:
    global _reranker_model
    if _reranker_model is not _NOT_LOADED:
        return

    try:
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    except Exception:
        _reranker_model = _FAILED


def _get_reranker():
    if _reranker_model is _NOT_LOADED:
        _load_reranker()
    if _reranker_model is _FAILED:
        return None
    return _reranker_model


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    content_field: str = "content",
) -> tuple[list[dict[str, Any]], bool]:
    if not candidates:
        return [], False

    model = _get_reranker()
    if model is None:
        fallback = []
        for rank, item in enumerate(candidates[:top_k], start=1):
            out = dict(item)
            out["rerank_score"] = float(out.get("hybrid_score", 0.0))
            out["final_rank"] = rank
            fallback.append(out)
        return fallback, False

    pairs = [(query, str(item.get(content_field, ""))[:MAX_PASSAGE_CHARS]) for item in candidates]

    try:
        scores = model.predict(pairs)
        scored = []
        for idx, item in enumerate(candidates):
            out = dict(item)
            out["rerank_score"] = float(scores[idx])
            scored.append(out)

        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        for rank, item in enumerate(scored[:top_k], start=1):
            item["final_rank"] = rank
        return scored[:top_k], True
    except Exception:
        fallback = []
        for rank, item in enumerate(candidates[:top_k], start=1):
            out = dict(item)
            out["rerank_score"] = float(out.get("hybrid_score", 0.0))
            out["final_rank"] = rank
            fallback.append(out)
        return fallback, False


async def rerank_async(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    content_field: str = "content",
) -> tuple[list[dict[str, Any]], bool]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, rerank, query, candidates, top_k, content_field)
