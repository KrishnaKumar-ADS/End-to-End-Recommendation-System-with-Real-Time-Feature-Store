"""Search evaluation utilities: MRR@10, NDCG@10, Precision@k."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from search.search_service import SearchResponse, search


@dataclass
class TestQuery:
    query: str
    relevant_names: list[str]
    description: str = ""


BUILTIN_TEST_QUERIES: list[TestQuery] = [
    TestQuery(
        query="password hashing",
        relevant_names=["hash_password", "verify_password", "check_password", "make_password"],
        description="Password hashing and verification",
    ),
    TestQuery(
        query="database connection setup",
        relevant_names=["connect", "create_engine", "get_db", "init_db"],
        description="Database connection",
    ),
    TestQuery(
        query="user authentication login",
        relevant_names=["login", "authenticate", "verify_user"],
        description="Authentication flow",
    ),
    TestQuery(
        query="token generation JWT",
        relevant_names=["create_token", "generate_token", "create_access_token", "encode_token"],
        description="JWT creation",
    ),
    TestQuery(
        query="error handling exception middleware",
        relevant_names=["exception_handler", "error_handler", "handle_error"],
        description="Error handling",
    ),
]


@dataclass
class QueryEvalResult:
    query: str
    relevant_names: list[str]
    returned_names: list[str]
    reciprocal_rank: float
    ndcg_at_10: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    first_relevant_rank: int | None


@dataclass
class EvaluationReport:
    mrr_at_10: float
    avg_ndcg_at_10: float
    avg_precision_at_1: float
    avg_precision_at_5: float
    avg_precision_at_10: float
    num_queries: int
    num_queries_with_hit: int
    query_results: list[QueryEvalResult] = field(default_factory=list)


def _compute_dcg(relevances: list[int], k: int) -> float:
    total = 0.0
    for idx, rel in enumerate(relevances[:k]):
        total += rel / math.log2(idx + 2)
    return total


def _compute_ndcg(relevances: list[int], k: int) -> float:
    dcg = _compute_dcg(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = _compute_dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_query_results(
    query: str,
    relevant_names: list[str],
    result_names: list[str],
    k: int = 10,
) -> QueryEvalResult:
    relevant = [name.lower() for name in relevant_names]
    truncated = result_names[:k]

    relevances: list[int] = []
    for result_name in truncated:
        name_lower = (result_name or "").lower()
        match = any(rel in name_lower or name_lower in rel for rel in relevant)
        relevances.append(1 if match else 0)

    reciprocal_rank = 0.0
    first_rank: int | None = None
    for idx, rel in enumerate(relevances, start=1):
        if rel == 1:
            reciprocal_rank = 1.0 / idx
            first_rank = idx
            break

    def _precision(cutoff: int) -> float:
        if cutoff <= 0:
            return 0.0
        return sum(relevances[:cutoff]) / cutoff

    return QueryEvalResult(
        query=query,
        relevant_names=relevant_names,
        returned_names=truncated,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_10=_compute_ndcg(relevances, k),
        precision_at_1=_precision(1),
        precision_at_5=_precision(5),
        precision_at_10=_precision(k),
        first_relevant_rank=first_rank,
    )


async def run_evaluation(
    repo_id: str,
    db: AsyncSession,
    queries: list[TestQuery] | None = None,
    mode: str = "hybrid",
    rerank: bool = True,
    top_k: int = 10,
) -> EvaluationReport:
    suite = queries or BUILTIN_TEST_QUERIES
    per_query: list[QueryEvalResult] = []

    for test_query in suite:
        try:
            response: SearchResponse = await search(
                query=test_query.query,
                repo_id=repo_id,
                db=db,
                mode=mode,  # type: ignore[arg-type]
                top_k=top_k,
                rerank=rerank,
                expand_query_flag=False,
            )
            names = [result.name for result in response.results]
        except Exception:
            names = []

        per_query.append(
            evaluate_query_results(
                query=test_query.query,
                relevant_names=test_query.relevant_names,
                result_names=names,
                k=top_k,
            )
        )

    if not per_query:
        return EvaluationReport(
            mrr_at_10=0.0,
            avg_ndcg_at_10=0.0,
            avg_precision_at_1=0.0,
            avg_precision_at_5=0.0,
            avg_precision_at_10=0.0,
            num_queries=0,
            num_queries_with_hit=0,
            query_results=[],
        )

    num = len(per_query)
    return EvaluationReport(
        mrr_at_10=sum(item.reciprocal_rank for item in per_query) / num,
        avg_ndcg_at_10=sum(item.ndcg_at_10 for item in per_query) / num,
        avg_precision_at_1=sum(item.precision_at_1 for item in per_query) / num,
        avg_precision_at_5=sum(item.precision_at_5 for item in per_query) / num,
        avg_precision_at_10=sum(item.precision_at_10 for item in per_query) / num,
        num_queries=num,
        num_queries_with_hit=sum(1 for item in per_query if item.first_relevant_rank is not None),
        query_results=per_query,
    )
