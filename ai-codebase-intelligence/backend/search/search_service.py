"""End-to-end search pipeline: expansion, retrieval, fusion, reranking."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from caching.cache_manager import get_cache_manager
from embeddings.gemini_embedder import embed_query
from embeddings.vector_store import search as qdrant_search
from search.bm25_index import BM25Index
from search.bm25_store import get_or_build_index
from search.hybrid_fusion import bm25_only, reciprocal_rank_fusion, vector_only
from search.query_expander import expand_query
from search.reranker import rerank_async

SearchMode = Literal["vector", "bm25", "hybrid"]


@dataclass
class TimingBreakdown:
    embed_ms: int = 0
    expand_ms: int = 0
    vector_ms: int = 0
    bm25_ms: int = 0
    fusion_ms: int = 0
    rerank_ms: int = 0
    total_ms: int = 0


@dataclass
class SearchResult:
    id: str
    name: str
    file_path: str
    chunk_type: str
    start_line: int
    end_line: int
    content: str
    docstring: str | None
    language: str | None
    parent_name: str | None
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    hybrid_rank: int = 0
    final_rank: int = 0


@dataclass
class SearchResponse:
    query: str
    expanded_queries: list[str]
    repo_id: str
    mode: SearchMode
    reranked: bool
    results: list[SearchResult]
    total_results: int
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _normalize_qdrant_result(item) -> dict:
    payload = dict(item.payload or {})
    payload["id"] = str(item.id)
    payload["score"] = float(item.score)
    return payload


def _to_result(item: dict) -> SearchResult:
    return SearchResult(
        id=str(item.get("id", "")),
        name=str(item.get("name") or item.get("display_name") or ""),
        file_path=str(item.get("file_path") or ""),
        chunk_type=str(item.get("chunk_type") or ""),
        start_line=int(item.get("start_line") or 0),
        end_line=int(item.get("end_line") or 0),
        content=str(item.get("content") or item.get("content_preview") or ""),
        docstring=item.get("docstring"),
        language=item.get("language"),
        parent_name=item.get("parent_name"),
        vector_score=float(item.get("vector_score") or item.get("score") or 0.0),
        bm25_score=float(item.get("bm25_score") or 0.0),
        hybrid_score=float(item.get("hybrid_score") or 0.0),
        rerank_score=(float(item.get("rerank_score")) if item.get("rerank_score") is not None else None),
        vector_rank=item.get("vector_rank"),
        bm25_rank=item.get("bm25_rank"),
        hybrid_rank=int(item.get("hybrid_rank") or 0),
        final_rank=int(item.get("final_rank") or item.get("hybrid_rank") or 0),
    )


async def search(
    query: str,
    repo_id: str,
    db: AsyncSession,
    mode: SearchMode = "hybrid",
    top_k: int = 5,
    rerank: bool = True,
    expand_query_flag: bool = False,
    chunk_type: str | None = None,
    language: str | None = None,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
) -> SearchResponse:
    cache = get_cache_manager()
    cache_kwargs = {
        "mode": mode,
        "top_k": top_k,
        "rerank": rerank,
        "expand_query_flag": expand_query_flag,
        "chunk_type": chunk_type,
        "language": language,
        "vector_top_k": vector_top_k,
        "bm25_top_k": bm25_top_k,
    }

    cached_payload = cache.get_search_results(query=query, repo_id=repo_id, **cache_kwargs)
    if cached_payload is not None:
        # Legacy cache entries stored only a list of results. For expanded-query mode
        # that payload drops expansion metadata, so skip it and recompute once.
        if not (expand_query_flag and isinstance(cached_payload, list)):
            if isinstance(cached_payload, dict):
                raw_results = cached_payload.get("results", [])
                cached_results = raw_results if isinstance(raw_results, list) else []

                raw_expanded = cached_payload.get("expanded_queries", [query])
                expanded_from_cache = (
                    [str(item) for item in raw_expanded]
                    if isinstance(raw_expanded, list)
                    else [query]
                )
                cached_reranked = bool(cached_payload.get("reranked", rerank))
            else:
                cached_results = cached_payload
                expanded_from_cache = [query]
                cached_reranked = rerank

            results = [_to_result(item) for item in cached_results]
            return SearchResponse(
                query=query,
                expanded_queries=expanded_from_cache,
                repo_id=repo_id,
                mode=mode,
                reranked=cached_reranked,
                results=results,
                total_results=len(results),
                timing=TimingBreakdown(total_ms=1),
            )

    t_total = time.perf_counter()
    timing = TimingBreakdown()

    t0 = time.perf_counter()
    expanded_queries = await expand_query(query, n_expansions=2) if expand_query_flag else [query]
    timing.expand_ms = _elapsed_ms(t0)

    t0 = time.perf_counter()
    query_vectors: list[list[float]] = []
    for q in expanded_queries:
        vector = await asyncio.get_event_loop().run_in_executor(None, embed_query, q)
        query_vectors.append(vector)
    timing.embed_ms = _elapsed_ms(t0)

    vector_results: list[dict] = []
    if mode in ("vector", "hybrid"):
        t0 = time.perf_counter()
        seen_ids: set[str] = set()
        for vector in query_vectors:
            scored_points = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda vec=vector: qdrant_search(
                    query_vector=vec,
                    repo_id=repo_id,
                    top_k=vector_top_k,
                    language=language,
                    chunk_type=chunk_type,
                    score_threshold=0.2,
                ),
            )
            for point in scored_points:
                normalized = _normalize_qdrant_result(point)
                if normalized["id"] not in seen_ids:
                    seen_ids.add(normalized["id"])
                    vector_results.append(normalized)
        timing.vector_ms = _elapsed_ms(t0)

    bm25_results: list[dict] = []
    if mode in ("bm25", "hybrid"):
        t0 = time.perf_counter()
        bm25_index: BM25Index | None = await get_or_build_index(db=db, repo_id=repo_id)
        if bm25_index is not None:
            seen_ids: set[str] = set()
            for q in expanded_queries:
                found = bm25_index.search(
                    query=q,
                    top_k=bm25_top_k,
                    chunk_type=chunk_type,
                    language=language,
                )
                for item in found:
                    doc_id = str(item.get("id"))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        bm25_results.append(item)
        timing.bm25_ms = _elapsed_ms(t0)

    t0 = time.perf_counter()
    if mode == "vector":
        fused = vector_only(vector_results, top_k=vector_top_k)
    elif mode == "bm25":
        fused = bm25_only(bm25_results, top_k=bm25_top_k)
    else:
        if vector_results and bm25_results:
            fused = reciprocal_rank_fusion(vector_results, bm25_results)
        elif vector_results:
            fused = vector_only(vector_results, top_k=vector_top_k)
        elif bm25_results:
            fused = bm25_only(bm25_results, top_k=bm25_top_k)
        else:
            fused = []
    timing.fusion_ms = _elapsed_ms(t0)

    reranked = False
    candidates = fused[:vector_top_k]
    if rerank and candidates:
        t0 = time.perf_counter()
        final_candidates, reranked = await rerank_async(query=query, candidates=candidates, top_k=top_k)
        timing.rerank_ms = _elapsed_ms(t0)
    else:
        final_candidates = []
        for rank, item in enumerate(candidates[:top_k], start=1):
            out = dict(item)
            out.setdefault("final_rank", rank)
            out.setdefault("rerank_score", out.get("hybrid_score", 0.0))
            final_candidates.append(out)

    timing.total_ms = _elapsed_ms(t_total)

    results = [_to_result(item) for item in final_candidates]
    response = SearchResponse(
        query=query,
        expanded_queries=expanded_queries,
        repo_id=repo_id,
        mode=mode,
        reranked=reranked,
        results=results,
        total_results=len(results),
        timing=timing,
    )

    cache_payload = {
        "results": [result.__dict__ for result in results],
        "expanded_queries": expanded_queries,
        "reranked": reranked,
    }
    cache.set_search_results(query=query, repo_id=repo_id, results=cache_payload, **cache_kwargs)
    return response
