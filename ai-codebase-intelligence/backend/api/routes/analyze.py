"""Analysis endpoints for bug localization and security deep scan."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from analysis.bug_localizer import BugLocalizer
from analysis.security_chain import SecurityChain
from analysis.security_scanner import SecurityScanner
from core.exceptions import LLMProviderError
from core.logging import get_logger
from db.database import get_db
from db.models import IngestionStatus, Repository

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["analysis"])


class BugAnalysisRequest(BaseModel):
    repo_id: str = Field(..., description="UUID of the ingested repository")
    error_description: str = Field(..., min_length=10, max_length=2000)
    max_hops: int = Field(default=4, ge=1, le=8)


class SecurityAnalysisRequest(BaseModel):
    repo_id: str = Field(..., description="UUID of the ingested repository")
    file_filter: str = Field(default="", max_length=200)
    max_llm_calls: int = Field(default=10, ge=1, le=20)


class BugAnalysisResponse(BaseModel):
    error_signal: str
    call_chain: list[str]
    callers: list[str]
    callees: list[str]
    root_cause_file: str
    root_cause_function: str
    root_cause_line: int | None
    explanation: str
    fix_suggestion: str
    confidence: str
    provider_used: str
    model_used: str
    graph_nodes_explored: int


class SecurityFindingResponse(BaseModel):
    file_path: str
    function: str
    line_number: int
    severity: str
    category: str
    description: str
    matched_text: str
    rule_id: str
    cwe_id: str
    llm_analysis: str
    false_positive: bool


class SecurityAnalysisResponse(BaseModel):
    repo_id: str
    findings: list[SecurityFindingResponse]
    false_positives_removed: int
    summary_stats: dict
    scan_duration_ms: int
    static_findings_count: int
    chunks_scanned: int
    file_filter: str
    provider_used: str
    model_used: str


async def _get_completed_repo(repo_id: str, db: AsyncSession) -> Repository:
    try:
        repo_uuid = uuid.UUID(repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid repository id '{repo_id}'.") from exc

    repo = await db.get(Repository, repo_uuid)
    if repo is None:
        raise HTTPException(status_code=404, detail=f"Repository '{repo_id}' not found.")

    if repo.status != IngestionStatus.COMPLETED:
        raise HTTPException(
            status_code=422,
            detail=f"Repository not ready (status: {repo.status.value}). Wait for ingestion to complete.",
        )

    return repo


@router.post(
    "/analyze/bug",
    response_model=BugAnalysisResponse,
    summary="Localize a bug using graph traversal and LLM reasoning",
)
async def analyze_bug(
    request: BugAnalysisRequest,
    db: AsyncSession = Depends(get_db),
) -> BugAnalysisResponse:
    await _get_completed_repo(request.repo_id, db)

    logger.info(
        "bug_analysis_request",
        repo_id=request.repo_id,
        error=request.error_description[:80],
    )

    localizer = BugLocalizer()
    try:
        result = await localizer.localize(
            repo_id=request.repo_id,
            error_description=request.error_description,
            db=db,
            max_hops=request.max_hops,
        )
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=f"LLM analysis failed: {exc}") from exc
    except Exception as exc:
        logger.error("bug_analysis_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Internal analysis error: {exc}") from exc

    return BugAnalysisResponse(
        error_signal=result.error_signal,
        call_chain=result.call_chain,
        callers=result.callers,
        callees=result.callees,
        root_cause_file=result.root_cause_file,
        root_cause_function=result.root_cause_function,
        root_cause_line=result.root_cause_line,
        explanation=result.explanation,
        fix_suggestion=result.fix_suggestion,
        confidence=result.confidence,
        provider_used=result.provider_used,
        model_used=result.model_used,
        graph_nodes_explored=result.graph_nodes_explored,
    )


@router.post(
    "/analyze/security",
    response_model=SecurityAnalysisResponse,
    summary="Run static + LLM security analysis",
)
async def analyze_security(
    request: SecurityAnalysisRequest,
    db: AsyncSession = Depends(get_db),
) -> SecurityAnalysisResponse:
    await _get_completed_repo(request.repo_id, db)

    logger.info(
        "security_analysis_request",
        repo_id=request.repo_id,
        file_filter=request.file_filter or "(entire repo)",
        max_llm_calls=request.max_llm_calls,
    )

    chain = SecurityChain()
    try:
        report = await chain.analyze(
            repo_id=request.repo_id,
            db=db,
            file_filter=request.file_filter,
            max_llm_calls=request.max_llm_calls,
        )
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=f"LLM analysis failed: {exc}") from exc
    except Exception as exc:
        logger.error("security_analysis_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Internal analysis error: {exc}") from exc

    return SecurityAnalysisResponse(
        repo_id=report.repo_id,
        findings=[
            SecurityFindingResponse(
                file_path=finding.file_path,
                function=finding.function,
                line_number=finding.line_number,
                severity=finding.severity.value,
                category=finding.category,
                description=finding.description,
                matched_text=finding.matched_text,
                rule_id=finding.rule_id,
                cwe_id=finding.cwe_id,
                llm_analysis=finding.llm_analysis,
                false_positive=finding.false_positive,
            )
            for finding in report.findings
        ],
        false_positives_removed=report.false_positives_removed,
        summary_stats=report.summary_stats,
        scan_duration_ms=report.scan_duration_ms,
        static_findings_count=report.static_findings_count,
        chunks_scanned=report.chunks_scanned,
        file_filter=report.file_filter,
        provider_used=report.provider_used,
        model_used=report.model_used,
    )


@router.get(
    "/analyze/security/rules",
    summary="List static security scanner rules",
)
async def list_security_rules() -> dict:
    scanner = SecurityScanner()
    return {
        "rule_count": 15,
        "rules": scanner.get_rules_summary(),
    }
