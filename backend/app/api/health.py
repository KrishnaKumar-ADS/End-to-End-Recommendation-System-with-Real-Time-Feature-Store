import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from backend.app.metrics import CACHE_HIT_RATIO
from backend.app.schemas.request_response import HealthResponse
from backend.app.services.cache_service import CacheService
from backend.app.services.pipeline_service import PipelineService

router = APIRouter()

# Server start time (for uptime calculation)
SERVER_START_TIME = time.time()


def get_pipeline_service() -> PipelineService:
    """FastAPI dependency — returns singleton PipelineService."""
    raise RuntimeError("PipelineService not initialized. Check startup.")


def get_cache_service() -> CacheService:
    """FastAPI dependency — returns singleton CacheService."""
    raise RuntimeError("CacheService not initialized. Check startup.")


def _models_component(pipeline: PipelineService) -> Dict[str, Any]:
    """Build model health summary for /health response."""
    try:
        model_loader = pipeline.ml
        checks = {
            "two_tower": model_loader.two_tower_model is not None,
            "faiss": model_loader.faiss_index is not None,
            "lgbm": model_loader.lgbm_model is not None,
        }
        missing = [name for name, ok in checks.items() if not ok]
        if missing:
            return {
                "status": "degraded",
                "detail": f"Missing: {', '.join(missing)}",
            }
        return {
            "status": "ok",
            "detail": "two_tower, faiss, lgbm loaded",
        }
    except Exception as e:
        return {"status": "down", "detail": str(e)}


def _redis_component(cache: CacheService) -> Dict[str, Any]:
    """Build Redis/cache health summary for /health response."""
    try:
        if not cache.requested_enabled:
            return {
                "status": "disabled",
                "detail": cache.disabled_reason or "redis cache disabled",
            }
        if cache.client is None:
            return {
                "status": "down",
                "detail": cache.connection_error or "redis client unavailable",
            }
        start = time.perf_counter()
        cache.client.ping()
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "status": "ok",
            "latency_ms": round(latency_ms, 2),
            "detail": "redis reachable",
        }
    except Exception as e:
        return {"status": "down", "detail": str(e)}


def _feast_component(pipeline: PipelineService) -> Dict[str, Any]:
    """Report feature system status.

    The current backend uses in-memory precomputed features by default.
    """
    try:
        n_users = len(pipeline.ml.user_features_dict or {})
        n_items = len(pipeline.ml.item_features_dict or {})
        return {
            "status": "ok",
            "detail": f"in-memory feature store active ({n_users:,} users, {n_items:,} items)",
        }
    except Exception as e:
        return {"status": "down", "detail": str(e)}


@router.get(
    "/health",
    response_model = HealthResponse,
    summary        = "System health check",
    description    = """
    Returns the status of all backend components:
    - Two-Tower model (GPU/CPU)
    - FAISS index
    - LightGBM ranker
    - Redis cache
    - Feature store (in-memory)

    Used by monitoring systems (Prometheus in Week 10).
    """,
    tags=["system"],
)
async def health_check(
    pipeline: PipelineService = Depends(get_pipeline_service),
    cache: CacheService = Depends(get_cache_service),
) -> HealthResponse:
    """
    Check the health of all backend components.

    Example response:
        {
            "status": "healthy",
            "components": [
                {"name": "two_tower", "status": "healthy", "detail": "cuda:0"},
                ...
            ],
            "uptime_s": 3600.5
        }
    """
    redis_component = _redis_component(cache)
    feast_component = _feast_component(pipeline)
    models_component = _models_component(pipeline)

    component_states = [
        redis_component.get("status"),
        feast_component.get("status"),
        models_component.get("status"),
    ]

    if "down" in component_states:
        overall = "degraded"
    else:
        acceptable_states = {"ok", "disabled"}
        overall = (
            "healthy"
            if all(state in acceptable_states for state in component_states)
            else "degraded"
        )

    uptime = round(time.time() - SERVER_START_TIME, 1)
    cache_hit_ratio = float(CACHE_HIT_RATIO._value.get())

    body = HealthResponse(
        status=overall,
        version="9.0.0",
        components={
            "redis": redis_component,
            "feast": feast_component,
            "models": models_component,
        },
        metrics={
            "cache_hit_ratio": round(cache_hit_ratio, 4),
            "uptime_hours": round(uptime / 3600, 2),
        },
        uptime_seconds=uptime,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    logger.debug(f"/health → {overall} | uptime={uptime:.1f}s | cache_hit_ratio={cache_hit_ratio:.4f}")
    return body


@router.get("/ready", tags=["system"])
async def readiness_probe(
    cache: CacheService = Depends(get_cache_service),
):
    """Readiness probe: return ready when app can serve traffic."""
    try:
        if not cache.requested_enabled:
            return {"status": "ready", "detail": "redis cache disabled"}
        if cache.client is None:
            raise RuntimeError(cache.connection_error or "redis client unavailable")
        cache.client.ping()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"not ready: {e}",
        )


@router.get("/live", tags=["system"])
async def liveness_probe():
    """Liveness probe: process is alive if this endpoint responds."""
    return {
        "status": "alive",
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
    }