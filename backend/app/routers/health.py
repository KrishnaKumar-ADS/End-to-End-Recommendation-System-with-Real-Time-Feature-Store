import time
import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any

logger = structlog.get_logger()
router = APIRouter(tags=["monitoring"])

# Track startup time for uptime calculation
_STARTUP_TIME = time.time()


class ComponentStatus(BaseModel):
    status: str              # "ok" | "degraded" | "down"
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str              # "healthy" | "degraded" | "unhealthy"
    uptime_seconds: float
    version: str
    components: Dict[str, ComponentStatus]
    metrics: Dict[str, Any]


async def _check_redis(redis_client) -> ComponentStatus:
    """Ping Redis and measure latency."""
    try:
        start = time.perf_counter()
        redis_client.ping()
        latency = (time.perf_counter() - start) * 1000
        return ComponentStatus(status="ok", latency_ms=round(latency, 2))
    except Exception as e:
        return ComponentStatus(status="down", detail=str(e))


async def _check_feast(feast_store) -> ComponentStatus:
    """Check if Feast feature store is accessible."""
    try:
        start = time.perf_counter()
        feast_store.list_feature_views()
        latency = (time.perf_counter() - start) * 1000
        return ComponentStatus(status="ok", latency_ms=round(latency, 2))
    except Exception as e:
        return ComponentStatus(status="degraded", detail=str(e))


async def _check_models(model_store) -> ComponentStatus:
    """Check if all models are loaded into memory."""
    try:
        loaded = model_store.get_loaded_models()
        required = {"mf", "sasrec", "two_tower", "lgbm"}
        if required.issubset(set(loaded)):
            return ComponentStatus(
                status="ok",
                detail=f"Loaded: {', '.join(loaded)}"
            )
        else:
            missing = required - set(loaded)
            return ComponentStatus(
                status="degraded",
                detail=f"Missing models: {missing}"
            )
    except Exception as e:
        return ComponentStatus(status="down", detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Rich health check endpoint.
    Returns system status, component statuses, and live metrics.
    Used by: monitoring dashboards, ops teams, Grafana alerts.
    """
    from app.main import get_redis, get_feast_store, get_model_store

    redis_status = await _check_redis(get_redis())
    feast_status = await _check_feast(get_feast_store())
    model_status = await _check_models(get_model_store())

    # Determine overall status
    statuses = [redis_status.status, feast_status.status, model_status.status]
    if "down" in statuses:
        overall = "unhealthy"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"

    from app.metrics import CACHE_HIT_RATIO
    cache_ratio = CACHE_HIT_RATIO._value.get()  # read current gauge value

    response = HealthResponse(
        status=overall,
        uptime_seconds=round(time.time() - _STARTUP_TIME, 1),
        version="9.0.0",
        components={
            "redis": redis_status,
            "feast": feast_status,
            "models": model_status,
        },
        metrics={
            "cache_hit_ratio": round(cache_ratio, 3),
            "uptime_hours": round((time.time() - _STARTUP_TIME) / 3600, 2),
        }
    )

    logger.info(
        "health_check",
        overall_status=overall,
        redis=redis_status.status,
        feast=feast_status.status,
        models=model_status.status,
    )

    return response


@router.get("/ready", status_code=200)
async def readiness_probe():
    """
    Kubernetes readiness probe.
    Returns 200 only if ALL critical dependencies are up.
    If this returns 503, the load balancer stops sending traffic here.
    """
    from app.main import get_redis

    try:
        get_redis().ping()
        return {"status": "ready"}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis unavailable — not ready to serve traffic"
        )


@router.get("/live", status_code=200)
async def liveness_probe():
    """
    Kubernetes liveness probe.
    Returns 200 as long as the process is alive and not deadlocked.
    If this returns 503, K8s restarts the container.
    """
    return {
        "status": "alive",
        "uptime_seconds": round(time.time() - _STARTUP_TIME, 1),
    }