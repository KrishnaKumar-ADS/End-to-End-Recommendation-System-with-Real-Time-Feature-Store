"""Batch quality evaluation endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from evaluation.quality_evaluator import QualityEvaluator

router = APIRouter(prefix="/api/v1", tags=["evaluation"])


class EvaluateItem(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    answer: str = Field(..., min_length=1, max_length=20000)
    context_chunks: list[str] = Field(default_factory=list)
    item_id: str | None = None


class EvaluateRequest(BaseModel):
    items: list[EvaluateItem] = Field(..., min_length=1, max_length=50)


class EvaluateResult(BaseModel):
    item_id: str
    question: str
    score: dict


class EvaluateAggregate(BaseModel):
    avg_faithfulness: float
    avg_relevance: float
    avg_completeness: float
    avg_overall: float
    total_items: int
    evaluated_items: int
    skipped_items: int
    failed_items: int
    total_eval_ms: float


class EvaluateResponse(BaseModel):
    results: list[EvaluateResult]
    aggregate: EvaluateAggregate


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate answer quality for up to 50 items.",
)
async def evaluate_batch(request: EvaluateRequest) -> EvaluateResponse:
    started = time.perf_counter()
    evaluator = QualityEvaluator()

    results: list[EvaluateResult] = []

    evaluated_items = 0
    skipped_items = 0
    failed_items = 0

    total_faithfulness = 0.0
    total_relevance = 0.0
    total_completeness = 0.0
    total_overall = 0.0

    for index, item in enumerate(request.items, start=1):
        item_id = item.item_id or f"item-{index}"
        try:
            score = await evaluator.score(
                question=item.question,
                answer=item.answer,
                context_chunks=item.context_chunks,
            )
            if score.skipped:
                skipped_items += 1
            else:
                evaluated_items += 1
                total_faithfulness += score.faithfulness
                total_relevance += score.relevance
                total_completeness += score.completeness
                total_overall += score.overall

            payload = score.model_dump()
        except Exception as exc:
            failed_items += 1
            payload = {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "completeness": 0.0,
                "overall": 0.0,
                "critique": str(exc),
                "skipped": True,
                "skip_reason": f"Evaluation failed: {exc}",
                "judge_model": "",
            }

        results.append(
            EvaluateResult(
                item_id=item_id,
                question=item.question,
                score=payload,
            )
        )

    divisor = evaluated_items if evaluated_items > 0 else 1
    total_eval_ms = round((time.perf_counter() - started) * 1000, 3)

    aggregate = EvaluateAggregate(
        avg_faithfulness=round(total_faithfulness / divisor, 6) if evaluated_items else 0.0,
        avg_relevance=round(total_relevance / divisor, 6) if evaluated_items else 0.0,
        avg_completeness=round(total_completeness / divisor, 6) if evaluated_items else 0.0,
        avg_overall=round(total_overall / divisor, 6) if evaluated_items else 0.0,
        total_items=len(request.items),
        evaluated_items=evaluated_items,
        skipped_items=skipped_items,
        failed_items=failed_items,
        total_eval_ms=total_eval_ms,
    )

    return EvaluateResponse(results=results, aggregate=aggregate)
