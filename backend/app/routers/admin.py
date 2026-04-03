import os
import structlog
from fastapi import APIRouter, HTTPException, Header
from typing import Optional

logger = structlog.get_logger()
router = APIRouter(prefix="/admin", tags=["admin"])

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "ds19-admin-secret")  # CHANGE IN PROD


def verify_admin(x_admin_token: Optional[str] = Header(None)):
    """Verify admin token. Blocks unauthorized access."""
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")


@router.post("/reload-models")
async def reload_models(x_admin_token: Optional[str] = Header(None)):
    """
    Hot-reload all models from MLflow Production registry.
    Called by retraining pipeline after promoting new models.
    Zero-downtime: loads new models into memory, then swaps atomically.
    """
    verify_admin(x_admin_token)
    logger.info("model_reload_requested")

    from app.main import app
    model_store = app.state.model_store

    try:
        # Load new models (this takes 10-30 seconds, but old models still serve)
        await model_store.reload_from_registry()

        logger.info("model_reload_complete", models=model_store.get_loaded_models())
        return {
            "status": "ok",
            "message": "Models reloaded successfully",
            "loaded_models": model_store.get_loaded_models(),
        }
    except Exception as e:
        logger.error("model_reload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@router.get("/pipeline-status")
async def pipeline_status(x_admin_token: Optional[str] = Header(None)):
    """
    Returns the status of the last retraining pipeline run.
    Used by GitHub Actions to verify deployment succeeded.
    """
    verify_admin(x_admin_token)

    import json
    from pathlib import Path

    results_dir = Path("mlops/retraining")
    result_files = sorted(results_dir.glob("run_*.json"), reverse=True)

    if not result_files:
        return {"status": "no_runs_found"}

    with open(result_files[0]) as f:
        last_run = json.load(f)

    return {
        "status": "ok",
        "last_run": last_run,
    }