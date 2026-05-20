import os
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from api.routes import analysis, graph, ingest, search
from api.routes.cache import router as cache_router
from api.routes.explain import router as explain_router
from api.routes.analyze import router as analyze_router
from api.routes.ask import router as ask_router
from api.routes.evaluate import router as evaluate_router
from api.routes.metrics import router as metrics_router
from api.routes.providers import router as providers_router
from api.routes.webhook import router as webhook_router
from core.logging import setup_logging, get_logger
from core.config import ConfigurationError, get_settings, validate_production_config
from api.middleware import add_middleware
from cost_tracking.models import BudgetExceededError
from graph.neo4j_client import get_driver, close_driver, ping as neo4j_ping
from graph.schema import create_indexes

setup_logging()
settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown handler."""
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("app_startup_begin")

    # Fail early when required production env vars are missing.
    if os.getenv("APP_ENV", settings.app_env).lower() == "production":
        try:
            validate_production_config(settings)
            logger.info("production_config_valid", env="production")
        except ConfigurationError as e:
            logger.error("startup_config_error", error=str(e))
            raise SystemExit(1) from e

    # Validate LLM API keys (existing — logs warnings for missing keys)

    # Initialize Qdrant collection (existing — from Week 3)


    # ── NEW in Week 4: Initialize Neo4j ────────────────────────────────────
    try:
        get_driver()       # establishes connection pool + verifies connectivity
        create_indexes()   # creates all 9 graph indexes (idempotent)
        logger.info("neo4j_initialized")
    except Exception as e:
        # Non-fatal — app still works without Neo4j (Qdrant search still works)
        logger.warning("neo4j_startup_warning", error=str(e))

    logger.info("app_startup_complete")
    yield

    # ── Shutdown ────────────────────────────────────────────────────────────
    close_driver()
    logger.info("app_shutdown_complete")


app = FastAPI(
    title="AI Codebase Intelligence Platform",
    description=(
        "Ingest repositories, run semantic search, query with RAG, and inspect dependency graphs. "
        "The platform routes questions to specialized LLM models and returns grounded answers with citations."
    ),
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Ingestion",
            "description": "Repository ingestion and ingestion-status endpoints.",
        },
        {
            "name": "Query",
            "description": "Question-answering, provider inspection, and semantic search endpoints.",
        },
        {
            "name": "Graph",
            "description": "Dependency and call-graph retrieval and traversal endpoints.",
        },
        {
            "name": "Analysis",
            "description": "Static analysis and evaluation endpoints.",
        },
        {
            "name": "System",
            "description": "System health and operational endpoints.",
        },
    ],
    lifespan=lifespan,
)


def custom_openapi():
    """Inject Week 16 API metadata into the generated OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    info = schema.setdefault("info", {})
    info["contact"] = {
        "name": "AI Codebase Intelligence Team",
        "url": "https://github.com/your-username/ai-codebase-intelligence",
        "email": "maintainers@example.com",
    }
    info["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
    schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Local development",
        },
        {
            "url": "https://your-app.example.com",
            "description": "Production",
        },
    ]

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

add_middleware(app)

# ── Routers ───────────────────────────────────────────────
app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(graph.router)
app.include_router(analysis.router)
app.include_router(ask_router, prefix="/api/v1")
app.include_router(analyze_router)
app.include_router(providers_router)
app.include_router(cache_router)
app.include_router(explain_router)
app.include_router(evaluate_router)
app.include_router(metrics_router)
app.include_router(webhook_router)


@app.exception_handler(BudgetExceededError)
async def budget_exceeded_handler(request, exc: BudgetExceededError):
    return JSONResponse(
        status_code=429,
        content={
            "error": "daily_budget_exceeded",
            "message": str(exc),
            "daily_limit_usd": exc.daily_limit_usd,
            "used_usd": exc.used_usd,
        },
    )


@app.get(
    "/health",
    tags=["System"],
    summary="System health check",
    description=(
        "Returns high-level service health and available LLM providers. "
        "Use this endpoint for smoke checks and uptime monitoring."
    ),
)
async def health():
    from reasoning.llm_router import get_available_providers
    available = [p.value for p in get_available_providers()]

    qdrant_status = "ok"
    qdrant_vectors = 0
    try:
        from embeddings.vector_store import get_collection_info
        info = get_collection_info()
        qdrant_vectors = info["vectors_count"]
    except Exception as e:
        qdrant_status = f"unavailable: {e}"

    return JSONResponse({
        "status": "ok",
        "version": "1.0.0",
        "llm_providers_available": available,
        "qdrant": {
            "status": qdrant_status,
            "total_vectors": qdrant_vectors,
        },
    })