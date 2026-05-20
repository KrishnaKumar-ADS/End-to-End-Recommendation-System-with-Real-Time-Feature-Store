"""Week 7 search API: vector, BM25, hybrid, rerank, expansion, evaluation."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import IngestionStatus, Repository
from search.bm25_store import get_index_meta, get_or_build_index, invalidate_index, is_index_cached
from search.evaluator import BUILTIN_TEST_QUERIES, run_evaluation
from search.search_service import SearchMode, search as run_search

router = APIRouter(prefix="/api/v1", tags=["Query"])


class TimingResponse(BaseModel):
    embed_ms: int
    expand_ms: int
    vector_ms: int
    bm25_ms: int
    fusion_ms: int
    rerank_ms: int
    total_ms: int


class SearchResultResponse(BaseModel):
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
    vector_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: float | None
    vector_rank: int | None
    bm25_rank: int | None
    hybrid_rank: int
    final_rank: int


class SearchResponse(BaseModel):
    query: str
    expanded_queries: list[str]
    repo_id: str
    mode: str
    reranked: bool
    results: list[SearchResultResponse]
    total_results: int
    timing: TimingResponse


class IndexStatusResponse(BaseModel):
    repo_id: str
    is_cached: bool
    built_at: float | None
    chunk_count: int | None
    vocab_size: int | None
    size_kb: float | None


class QueryEvalResultResponse(BaseModel):
    query: str
    relevant_names: list[str]
    returned_names: list[str]
    reciprocal_rank: float
    ndcg_at_10: float
    precision_at_1: float
    precision_at_5: float
    first_relevant_rank: int | None


class EvaluationResponse(BaseModel):
    repo_id: str
    mode: str
    reranked: bool
    mrr_at_10: float
    avg_ndcg_at_10: float
    avg_precision_at_1: float
    avg_precision_at_5: float
    num_queries: int
    num_queries_with_hit: int
    passes_threshold: bool
    query_results: list[QueryEvalResultResponse]


async def _get_repo_or_404(repo_id: UUID, db: AsyncSession) -> Repository:
    repo = (await db.execute(select(Repository).where(Repository.id == repo_id))).scalar_one_or_none()
    if repo is None:
        raise HTTPException(status_code=404, detail=f"Repository '{repo_id}' not found")
    return repo


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Semantic and hybrid search across repository code",
    description=(
        "Runs vector and/or BM25 retrieval, optional query expansion, optional reranking, "
        "and returns ranked code chunks with timing breakdown."
    ),
    responses={
        200: {"description": "Search completed successfully."},
        404: {"description": "Repository id does not exist."},
        422: {"description": "Repository is not fully indexed or query params are invalid."},
        500: {"description": "Search pipeline failed unexpectedly."},
    },
)
async def search_endpoint(
    q: str = Query(..., min_length=1, max_length=500),
    repo_id: UUID = Query(...),
    top_k: int = Query(default=5, ge=1, le=50),
    mode: SearchMode = Query(default="hybrid"),
    rerank: bool = Query(default=True),
    expand_query: bool = Query(default=False),
    chunk_type: Optional[str] = Query(default=None),
    language: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    repo = await _get_repo_or_404(repo_id, db)
    if repo.status != IngestionStatus.COMPLETED:
        raise HTTPException(
            status_code=422,
            detail=f"Repository is not fully indexed. Current status: {repo.status.value}",
        )

    try:
        response = await run_search(
            query=q,
            repo_id=str(repo_id),
            db=db,
            mode=mode,
            top_k=top_k,
            rerank=rerank,
            expand_query_flag=expand_query,
            chunk_type=chunk_type,
            language=language,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return SearchResponse(
        query=response.query,
        expanded_queries=response.expanded_queries,
        repo_id=response.repo_id,
        mode=response.mode,
        reranked=response.reranked,
        total_results=response.total_results,
        results=[SearchResultResponse(**result.__dict__) for result in response.results],
        timing=TimingResponse(**response.timing.__dict__),
    )


@router.get("/search/index/status", response_model=IndexStatusResponse)
async def index_status(repo_id: UUID = Query(...), db: AsyncSession = Depends(get_db)):
    await _get_repo_or_404(repo_id, db)
    meta = get_index_meta(str(repo_id))
    return IndexStatusResponse(
        repo_id=str(repo_id),
        is_cached=is_index_cached(str(repo_id)),
        built_at=meta.get("built_at") if meta else None,
        chunk_count=meta.get("chunk_count") if meta else None,
        vocab_size=meta.get("vocab_size") if meta else None,
        size_kb=meta.get("size_kb") if meta else None,
    )


@router.post("/search/index/rebuild")
async def rebuild_index(repo_id: UUID = Query(...), db: AsyncSession = Depends(get_db)):
    await _get_repo_or_404(repo_id, db)
    invalidate_index(str(repo_id))
    await get_or_build_index(db=db, repo_id=str(repo_id), force_rebuild=True)
    return {"status": "rebuild_triggered", "repo_id": str(repo_id)}


@router.get("/search/evaluate", response_model=EvaluationResponse)
async def evaluate_search(
    repo_id: UUID = Query(...),
    mode: SearchMode = Query(default="hybrid"),
    rerank: bool = Query(default=True),
    db: AsyncSession = Depends(get_db),
):
    repo = await _get_repo_or_404(repo_id, db)
    if repo.status != IngestionStatus.COMPLETED:
        raise HTTPException(status_code=422, detail="Repository not fully indexed")

    report = await run_evaluation(
        repo_id=str(repo_id),
        db=db,
        queries=BUILTIN_TEST_QUERIES,
        mode=mode,
        rerank=rerank,
    )

    return EvaluationResponse(
        repo_id=str(repo_id),
        mode=mode,
        reranked=rerank,
        mrr_at_10=round(report.mrr_at_10, 4),
        avg_ndcg_at_10=round(report.avg_ndcg_at_10, 4),
        avg_precision_at_1=round(report.avg_precision_at_1, 4),
        avg_precision_at_5=round(report.avg_precision_at_5, 4),
        num_queries=report.num_queries,
        num_queries_with_hit=report.num_queries_with_hit,
        passes_threshold=report.mrr_at_10 >= 0.60,
        query_results=[
            QueryEvalResultResponse(
                query=item.query,
                relevant_names=item.relevant_names,
                returned_names=item.returned_names,
                reciprocal_rank=round(item.reciprocal_rank, 4),
                ndcg_at_10=round(item.ndcg_at_10, 4),
                precision_at_1=round(item.precision_at_1, 4),
                precision_at_5=round(item.precision_at_5, 4),
                first_relevant_rank=item.first_relevant_rank,
            )
            for item in report.query_results
        ],
    )