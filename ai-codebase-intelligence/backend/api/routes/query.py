from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.query_schema import QueryRequest, QueryResponse, SourceReference
from db.database import get_db
from db.models import IngestionStatus, Repository
from reasoning.chain import run_rag_chain_async


router = APIRouter(prefix="/api/v1", tags=["query"])


def _chunk_text(text: str, size: int = 120):
	for i in range(0, len(text), size):
		yield text[i : i + size]


@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest, db: AsyncSession = Depends(get_db)):
	repo_result = await db.execute(select(Repository).where(Repository.id == request.repo_id))
	repo = repo_result.scalar_one_or_none()

	if repo is None:
		raise HTTPException(status_code=404, detail="Repository not found.")

	if repo.status != IngestionStatus.COMPLETED:
		raise HTTPException(
			status_code=400,
			detail=f"Repository is not ready for Q&A. Current status: {repo.status.value}",
		)

	try:
		rag = await run_rag_chain_async(repo_id=str(request.repo_id), question=request.question, db=db)
	except RuntimeError as exc:
		raise HTTPException(status_code=503, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

	if request.stream:
		def event_stream():
			for chunk in _chunk_text(rag["answer"]):
				yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

			final_payload = {
				"type": "final",
				"provider_used": rag["provider"],
				"model_used": rag["model"],
				"sources": rag["sources"],
				"graph_path": rag["graph_path"],
			}
			yield f"data: {json.dumps(final_payload)}\n\n"

		return StreamingResponse(event_stream(), media_type="text/event-stream")

	return QueryResponse(
		answer=rag["answer"],
		provider_used=rag["provider"],
		model_used=rag["model"],
		sources=[SourceReference(**s) for s in rag["sources"]],
		graph_path=rag["graph_path"],
	)
