"""Code explanation endpoint for structured function/class analysis."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from explanation.explainer import CodeExplainer
from explanation.schemas import ExplainRequest, ExplainResponse
from core.exceptions import ChunkNotFoundError
from core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["explanation"])


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Generate structured explanation for one function/class.",
)
async def explain_function(
    request: ExplainRequest,
    db: AsyncSession = Depends(get_db),
) -> ExplainResponse:
    if not request.function_name and not request.chunk_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either function_name or chunk_id must be provided.",
        )

    try:
        service = CodeExplainer(db_session=db)
        return await service.explain(request)
    except ChunkNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("explain_endpoint_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Explanation generation failed: {exc}",
        ) from exc
