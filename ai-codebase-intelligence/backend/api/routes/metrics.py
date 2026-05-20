"""System metrics endpoints for Week 10 observability."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import redis
from fastapi import APIRouter
from fastapi.responses import Response

from caching.cache_manager import get_cache_manager
from core.config import get_settings
from cost_tracking.tracker import CostTracker
from reasoning.circuit_breaker import CircuitBreaker
from semantic_cache.answer_cache import get_semantic_answer_cache

router = APIRouter(prefix="/api/v1", tags=["metrics"])
_tracker = CostTracker()


async def _build_metrics_payload() -> dict:
    summary = await _tracker.get_daily_summary()
    cache_stats = get_cache_manager().get_stats()
    semantic_cache_stats = await get_semantic_answer_cache().get_stats()

    settings = get_settings()
    eval_scores = {
        "avg_faithfulness": 0.0,
        "avg_relevance": 0.0,
        "avg_completeness": 0.0,
        "avg_overall": 0.0,
        "count": 0,
    }

    try:
        redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        eval_scores = {
            "avg_faithfulness": float(redis_client.get("eval:avg:faithfulness") or 0.0),
            "avg_relevance": float(redis_client.get("eval:avg:relevance") or 0.0),
            "avg_completeness": float(redis_client.get("eval:avg:completeness") or 0.0),
            "avg_overall": float(redis_client.get("eval:avg:overall") or 0.0),
            "count": int(redis_client.get("eval:count") or 0),
        }
    except Exception:
        pass

    try:
        breaker = CircuitBreaker().get_all_statuses()
    except Exception:
        breaker = {}

    circuit_breakers = {
        provider: {
            "state": status.state.value,
            "consecutive_failures": status.failure_count,
            "last_failure_time": status.last_failure_time,
            "last_success_time": status.last_success_time,
        }
        for provider, status in breaker.items()
    }

    return {
        "token_usage": summary.per_provider,
        "budget": {
            "daily_limit_usd": summary.budget_limit_usd,
            "used_today_usd": summary.total_cost_usd,
            "remaining_usd": summary.remaining_usd,
            "used_pct": summary.budget_used_pct,
            "over_budget": summary.over_budget,
        },
        "cache": {
            **cache_stats,
            "semantic": semantic_cache_stats,
        },
        "circuit_breakers": circuit_breakers,
        "eval_scores": eval_scores,
    }


@router.get("/metrics", summary="Get runtime metrics for cost, cache, providers, and eval.")
async def get_metrics() -> dict:
    return await _build_metrics_payload()


@router.get("/metrics/export", summary="Export runtime metrics as downloadable JSON.")
async def export_metrics() -> Response:
    payload = await _build_metrics_payload()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_export_{ts}.json"

    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
