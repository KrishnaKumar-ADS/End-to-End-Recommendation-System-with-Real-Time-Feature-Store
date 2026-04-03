"""
DS19 — Week 8/9 — Updated /feedback Endpoint
Integrates:
  1. Thompson Sampling Bandit (Week 8)
  2. A/B Testing Logger (Week 8)
  3. Prometheus Metrics (Week 9)
  4. Rate Limiting (Week 9)
"""

import time
import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

# Week 9: Prometheus metrics + rate limiting
from app.metrics import FEEDBACK_EVENTS
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)
feedback_router = APIRouter()

# Rate limiter setup (Week 9)
limiter = Limiter(key_func=get_remote_address)


# ──────────────────────────────────────────────────────────────────────
# REQUEST/RESPONSE MODELS
# ──────────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """Week 9: Use request body instead of query params for better validation"""
    user_id: int
    item_id: int
    clicked: bool
    variant: Optional[str] = "unknown"
    experiment_id: Optional[str] = "retrieval_v1"
    session_id: Optional[str] = None
    timestamp: Optional[float] = None


class FeedbackResponse(BaseModel):
    success: bool
    user_id: int
    item_id: int
    clicked: bool
    bandit_updated: bool
    ab_logged: bool
    metrics_logged: bool
    processing_ms: float


# ──────────────────────────────────────────────────────────────────────
# FEEDBACK ENDPOINT
# ──────────────────────────────────────────────────────────────────────

@feedback_router.post("/feedback", response_model=FeedbackResponse)
@limiter.limit("200/minute")  # generous — users click fast
async def submit_feedback(
    request: Request,
    feedback: FeedbackRequest,
):
    """
    Receives user click signal. Updates bandit + A/B logger + Prometheus metrics.
    
    Called by the frontend when a user clicks (or explicitly ignores) an item.
    
    Updates:
      - Thompson Sampling bandit arm for the item (Week 8)
      - A/B testing conversion log (Week 8)
      - Prometheus metrics for monitoring (Week 9)
      - Redis: persists new bandit arm state
    
    This endpoint should complete in < 10ms.
    Non-critical failures (Redis down, log write fail) are logged but don't fail the request.
    """
    t0 = time.time()

    bandit_updated = False
    ab_logged = False
    metrics_logged = False

    # Week 9: Record to Prometheus
    variant = feedback.variant or "unknown"
    feedback_type = "positive" if feedback.clicked else "negative"

    try:
        FEEDBACK_EVENTS.labels(
            variant=variant,
            feedback_type=feedback_type,
        ).inc()
        metrics_logged = True
    except Exception as e:
        logger.warning(f"Prometheus metrics logging failed: {e}")

    # Week 8: Update Thompson Sampling bandit
    try:
        from backend.app.main import bandit_service
        bandit_service.record_feedback(item_idx=feedback.item_id, clicked=feedback.clicked)
        bandit_updated = True
    except Exception as e:
        logger.warning(f"Bandit update failed for item={feedback.item_id}: {e}")

    # Week 8: Log A/B conversion (only for actual clicks)
    if feedback.clicked:
        try:
            from backend.app.main import ab_logger, ab_router as router

            # Look up this user's variant for this experiment
            assigned_variant = router.assign(feedback.user_id, feedback.experiment_id)
            ab_logger.log_conversion(
                user_id=feedback.user_id,
                item_id=feedback.item_id,
                variant=assigned_variant,
                experiment_id=feedback.experiment_id,
            )
            ab_logged = True
        except Exception as e:
            logger.warning(f"A/B conversion log failed for user={feedback.user_id}: {e}")

    processing_ms = (time.time() - t0) * 1000
    logger.info(
        f"Feedback recorded | "
        f"user={feedback.user_id} item={feedback.item_id} clicked={feedback.clicked} | "
        f"bandit_updated={bandit_updated} ab_logged={ab_logged} metrics_logged={metrics_logged} | "
        f"{processing_ms:.2f}ms"
    )

    return FeedbackResponse(
        success=True,
        user_id=feedback.user_id,
        item_id=feedback.item_id,
        clicked=feedback.clicked,
        bandit_updated=bandit_updated,
        ab_logged=ab_logged,
        metrics_logged=metrics_logged,
        processing_ms=round(processing_ms, 2),
    )