from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from analysis.bug_detector import detect_bug_patterns
from analysis.complexity_analyzer import analyze_chunk_complexity
from analysis.security_scanner import scan_security_patterns
from db.database import get_db
from db.models import CodeChunk, Repository, SourceFile


router = APIRouter(prefix="/api/v1", tags=["analysis"])


class AnalyzeRequest(BaseModel):
	repo_id: UUID
	mode: str = Field(default="all", description="all | bugs | security | complexity")
	top_k: int = Field(default=200, ge=1, le=2000)


@router.post("/analyze")
async def analyze_repository(request: AnalyzeRequest, db: AsyncSession = Depends(get_db)):
	repo_result = await db.execute(select(Repository).where(Repository.id == request.repo_id))
	repo = repo_result.scalar_one_or_none()
	if repo is None:
		raise HTTPException(status_code=404, detail="Repository not found.")

	rows_result = await db.execute(
		select(
			CodeChunk.id,
			CodeChunk.name,
			CodeChunk.content,
			SourceFile.file_path,
		)
		.join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
		.where(SourceFile.repository_id == request.repo_id)
		.limit(request.top_k)
	)

	rows = rows_result.all()

	findings_bugs: list[dict] = []
	findings_security: list[dict] = []
	complexities: list[dict] = []

	mode = request.mode.lower().strip()
	run_all = mode == "all"

	for row in rows:
		source = row.content or ""
		symbol = row.name or "unknown_symbol"
		file_path = row.file_path or ""

		if run_all or mode == "bugs":
			findings_bugs.extend(detect_bug_patterns(source, file_path, symbol))
		if run_all or mode == "security":
			findings_security.extend(scan_security_patterns(source, file_path, symbol))
		if run_all or mode == "complexity":
			complexities.append(analyze_chunk_complexity(symbol, file_path, source))

	complexities.sort(key=lambda x: x["cyclomatic_complexity"], reverse=True)

	return {
		"repo_id": str(request.repo_id),
		"mode": request.mode,
		"analyzed_chunks": len(rows),
		"bug_findings": findings_bugs,
		"security_findings": findings_security,
		"complexity": complexities[:50],
		"summary": {
			"bug_count": len(findings_bugs),
			"security_count": len(findings_security),
			"high_complexity_count": len([c for c in complexities if c["risk"] == "high"]),
		},
	}
