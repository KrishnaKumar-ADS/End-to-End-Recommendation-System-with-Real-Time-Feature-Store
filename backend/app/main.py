"""
DS19 — FastAPI Main Application
Entry point for the backend service.
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from loguru import logger
import uvicorn

from backend.app.api import ab as ab_module
from backend.app.api import feedback as feedback_module
from backend.app.api import health as health_module
from backend.app.api import recommend as recommend_module
from backend.app.core.config import API_HOST, API_PORT, API_WORKERS, BASE_DIR, REDIS_HOST, REDIS_PORT
from backend.app.core.model_loader import ModelLoader as CoreModelLoader
from backend.app.metrics import ACTIVE_AB_EXPERIMENTS
from backend.app.middleware.logging_middleware import LatencyLoggingMiddleware
from backend.app.services.cache_service import CacheService
from backend.app.services.feature_service import FeatureService
from backend.app.services.pipeline_service import PipelineService
from backend.app.services.ranking_service import RankingService
from backend.app.services.retrieval_service import RetrievalService
from mlops.ab_testing.ab_logger import ABLogger
from mlops.ab_testing.ab_router import ABRouter as ABTestRouter
from mlops.bandits.bandit_service import BanditService
from mlops.mlflow_setup.model_loader import ModelLoader as MLflowModelLoader

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except Exception:
    Instrumentator = None

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


app_state: dict = {}
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"
API_PATH_PREFIXES = (
    "/recommend",
    "/feedback",
    "/health",
    "/ready",
    "/live",
    "/ab",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
)


def _frontend_available() -> bool:
    return FRONTEND_INDEX_FILE.exists()


def get_pipeline_service() -> PipelineService:
    return app_state["pipeline_service"]


def get_cache_service() -> CacheService:
    return app_state["cache_service"]


def get_feature_service() -> FeatureService:
    return app_state["feature_service"]


def get_bandit_service() -> BanditService:
    return app_state["bandit_service"]


def get_ab_router() -> ABTestRouter:
    return app_state["ab_router"]


def get_ab_logger() -> ABLogger:
    return app_state["ab_logger"]


def get_mlflow_loader() -> MLflowModelLoader:
    return app_state["mlflow_loader"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  DS19 RecSys Backend — Starting Up...")
    logger.info("=" * 60)

    startup_start = time.time()

    model_loader = CoreModelLoader()
    model_loader.load_all()
    app_state["model_loader"] = model_loader

    cache_service = CacheService()
    feature_service = FeatureService(model_loader)
    retrieval_service = RetrievalService(model_loader)
    ranking_service = RankingService(model_loader)

    app_state["cache_service"] = cache_service
    app_state["feature_service"] = feature_service
    app_state["retrieval_service"] = retrieval_service
    app_state["ranking_service"] = ranking_service

    ab_router = ABTestRouter()
    ab_logger = ABLogger()
    bandit_service = BanditService(
        n_items=model_loader.n_items,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_db=1,
        blend_weight=0.3,
    )
    bandit_info = {"status": "degraded", "detail": "not initialized"}
    try:
        bandit_info = bandit_service.startup()
    except Exception as e:
        logger.warning(f"Bandit startup warning: {e}")

    mlflow_loader = MLflowModelLoader()

    app_state["ab_router"] = ab_router
    app_state["ab_logger"] = ab_logger
    app_state["bandit_service"] = bandit_service
    app_state["bandit_info"] = bandit_info
    app_state["mlflow_loader"] = mlflow_loader

    pipeline_service = PipelineService(
        model_loader=model_loader,
        cache_service=cache_service,
        feature_service=feature_service,
        retrieval_service=retrieval_service,
        ranking_service=ranking_service,
    )
    app_state["pipeline_service"] = pipeline_service

    try:
        _ = mlflow_loader.get_production_model_info("lgbm_ranker")
        logger.info("MLflow registry loader initialized")
    except Exception as e:
        logger.warning(f"MLflow registry initialization warning: {e}")

    ACTIVE_AB_EXPERIMENTS.set(1)

    startup_elapsed = time.time() - startup_start
    logger.info(f"Server ready in {startup_elapsed:.1f}s")
    logger.info(f"API: http://{API_HOST}:{API_PORT}")
    logger.info(f"Docs: http://{API_HOST}:{API_PORT}/docs")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down DS19 backend...")
    bandit = app_state.get("bandit_service")
    if bandit is not None:
        try:
            bandit.bulk_save()
        except Exception as e:
            logger.warning(f"Bandit save on shutdown failed: {e}")

    cache = app_state.get("cache_service")
    if cache is not None and cache.client is not None:
        cache.client.close()

    logger.info("Shutdown complete.")


app = FastAPI(
    title="DS19 — Real-Time Recommendation API",
    description="""
## DS19 Recommendation System API

A production-grade recommendation API built with:
- Two-Tower Model for fast candidate retrieval via FAISS
- LightGBM LambdaRank for precision re-ranking
- Redis for response caching
- FastAPI for async HTTP serving
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


app.dependency_overrides[recommend_module.get_pipeline_service] = get_pipeline_service
app.dependency_overrides[feedback_module.get_cache_service] = get_cache_service
app.dependency_overrides[health_module.get_pipeline_service] = get_pipeline_service
app.dependency_overrides[health_module.get_cache_service] = get_cache_service


cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3001",
    "http://localhost:80",
]

env_origins = os.getenv("CORS_ORIGINS", "").strip()
if env_origins:
    cors_origins.extend([o.strip() for o in env_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(cors_origins)),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-AB-Variant", "X-Cache-Status", "X-Cache", "X-Latency-Ms"],
)

app.add_middleware(LatencyLoggingMiddleware)

app.include_router(
    recommend_module.router,
    prefix="",
    tags=["recommendations"],
    dependencies=[Depends(get_pipeline_service)],
)

app.include_router(
    feedback_module.router,
    prefix="",
    tags=["feedback"],
    dependencies=[Depends(get_cache_service)],
)

app.include_router(
    health_module.router,
    prefix="",
    tags=["system"],
)

app.include_router(
    ab_module.router,
    prefix="",
    tags=["monitoring"],
)

if Instrumentator is not None:
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health", "/ready", "/live"],
        inprogress_name="ds19_http_requests_in_progress",
        inprogress_labels=True,
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
else:
    logger.warning("prometheus_fastapi_instrumentator not available; using fallback /metrics endpoint")

    @app.get("/metrics", include_in_schema=False, tags=["monitoring"])
    async def metrics_fallback() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["system"], summary="API root")
async def root():
    if _frontend_available():
        return FileResponse(FRONTEND_INDEX_FILE)

    return {
        "name": "DS19 Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    if not _frontend_available():
        raise HTTPException(status_code=404, detail="Not Found")

    request_path = (f"/{full_path}").rstrip("/") or "/"
    for prefix in API_PATH_PREFIXES:
        if request_path == prefix or request_path.startswith(f"{prefix}/"):
            raise HTTPException(status_code=404, detail="Not Found")

    frontend_root = FRONTEND_DIST_DIR.resolve()
    target_path = (FRONTEND_DIST_DIR / full_path).resolve()
    try:
        target_path.relative_to(frontend_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not Found")

    if target_path.is_file():
        return FileResponse(target_path)

    return FileResponse(FRONTEND_INDEX_FILE)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred. Check server logs.",
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
        reload=False,
        log_level="info",
    )
