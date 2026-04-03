
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from backend.app.schemas.request_response import FeedbackRequest, FeedbackResponse
from backend.app.services.cache_service import CacheService
from backend.app.core.config import API_LOG_FILE, LOGS_DIR
from backend.app.metrics import FEEDBACK_EVENTS

router = APIRouter()


# Ensure logs directory exists
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)


def get_cache_service() -> CacheService:
    """FastAPI dependency — returns singleton CacheService."""
    raise RuntimeError("CacheService not initialized. Check startup.")


@router.post(
    "/feedback",
    response_model = FeedbackResponse,
    summary        = "Submit user click feedback",
    description    = """
    Log a user's click on a recommended item.

    This endpoint:
    1. Appends the feedback event to logs/api_requests.jsonl
    2. Invalidates the user's recommendation cache (so fresh recs are generated)
    3. Returns confirmation

    In Week 8, this feedback is consumed by the Thompson Sampling bandit.
    """,
    tags=["feedback"],
)
async def submit_feedback(
    feedback: FeedbackRequest,
    cache:    CacheService = Depends(get_cache_service),
) -> FeedbackResponse:
    """
    Submit click feedback.

    Body:
        user_id:    int    - User who clicked
        item_id:    int    - Item that was clicked
        reward:     float  - 1.0=click, 0.0=skip (default: 1.0)
        session_id: str    - Optional session identifier
        context:    dict   - Optional metadata (position, page, etc.)

    Example:
        POST /feedback
        {
            "user_id": 123,
            "item_id": 4886,
            "reward": 1.0,
            "session_id": "sess_abc"
        }
    """
    logger.info(
        f"→ /feedback user_id={feedback.user_id} "
        f"item_id={feedback.item_id} reward={feedback.reward}"
    )

    clicked = feedback.reward > 0.0
    context = feedback.context or {}
    experiment_id = str(feedback.experiment_id or context.get("experiment_id", "retrieval_v1"))
    variant = feedback.variant or context.get("variant")
    bandit_updated = False
    ab_logged = False

    try:
        FEEDBACK_EVENTS.labels(
            variant=str(variant or "unknown"),
            feedback_type="positive" if clicked else "negative",
        ).inc()
    except Exception as e:
        logger.warning(f"Feedback metric increment failed: {e}")

    # ── Write to log file ────────────────────────────────────
    event = {
        "event_type":  "feedback",
        "user_id":     feedback.user_id,
        "item_id":     feedback.item_id,
        "reward":      feedback.reward,
        "session_id":  feedback.session_id,
        "context":     feedback.context,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "unix_ts":     time.time(),
    }

    try:
        with open(API_LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.error(f"Failed to write feedback log: {e}")
        raise HTTPException(status_code=500, detail="Failed to log feedback")

    # ── Invalidate cache ─────────────────────────────────────
    # After a user clicks, their taste signal has updated.
    # Invalidate their cached recommendations so the next request
    # triggers a fresh pipeline run.
    cache.invalidate(user_id=feedback.user_id)

    # ── Week 8: update bandit state in Redis ────────────────
    try:
        from backend.app.main import get_bandit_service
        bandit_service = get_bandit_service()
        bandit_service.record_feedback(item_idx=feedback.item_id, clicked=clicked)
        bandit_updated = True
    except Exception as e:
        logger.warning(f"Bandit feedback update failed for item_id={feedback.item_id}: {e}")

    # ── Week 8: log A/B conversion event ────────────────────
    if clicked:
        try:
            from backend.app.main import get_ab_router, get_ab_logger
            ab_router = get_ab_router()
            ab_logger = get_ab_logger()

            if not variant:
                variant = ab_router.assign(feedback.user_id, experiment_id)

            ab_logger.log_conversion(
                user_id=feedback.user_id,
                item_id=feedback.item_id,
                variant=variant,
                experiment_id=experiment_id,
                extra={"reward": feedback.reward},
            )
            ab_logged = True
        except Exception as e:
            logger.warning(f"A/B conversion log failed for user_id={feedback.user_id}: {e}")

    logger.info(
        f"  ✅ Feedback logged + cache invalidated for user_id={feedback.user_id} "
        f"| bandit_updated={bandit_updated} ab_logged={ab_logged}"
    )

    return FeedbackResponse(
        status  = "ok",
        user_id = feedback.user_id,
        item_id = feedback.item_id,
        message = "Feedback logged successfully. Cache invalidated.",
        bandit_updated=bandit_updated,
        ab_logged=ab_logged,
        variant=variant,
    )