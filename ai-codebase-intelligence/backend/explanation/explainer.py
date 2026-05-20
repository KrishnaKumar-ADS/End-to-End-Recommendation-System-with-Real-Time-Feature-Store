"""Code explanation service built on PostgreSQL chunks + Neo4j call neighbors + OpenRouter LLM."""

from __future__ import annotations

import json
import time
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import ChunkNotFoundError
from core.logging import get_logger
from db.models import CodeChunk, SourceFile
from explanation.schemas import (
    CallerCalleeInfo,
    ExplainRequest,
    ExplainResponse,
    ParameterInfo,
    ReturnInfo,
)
from graph.neo4j_client import run_query
from reasoning.openrouter_client import OpenRouterClient

logger = get_logger(__name__)

_EXPLAIN_SYSTEM_PROMPT = """You are an expert code analyst.
Return JSON only with keys: summary, parameters, returns, side_effects, complexity_score.
- parameters: list of {name, type_annotation, default_value, description}
- returns: {type_annotation, description}
- side_effects: list[str]
- complexity_score: integer 1-10
No markdown. JSON only.
"""


class CodeExplainer:
    """Generate structured function/class explanations from repository code."""

    MODEL = "qwen/qwen-2.5-coder-32b-instruct"

    def __init__(
        self,
        db_session: AsyncSession,
        llm_client: OpenRouterClient | None = None,
    ) -> None:
        self._db = db_session
        self._llm = llm_client or OpenRouterClient(model=self.MODEL)

    async def explain(self, request: ExplainRequest) -> ExplainResponse:
        started = time.perf_counter()

        chunk = await self._lookup_chunk(request)
        if chunk is None:
            target_name = request.function_name or request.chunk_id or "unknown"
            raise ChunkNotFoundError(f"No code chunk found for '{target_name}' in repo '{request.repo_id}'.")

        callers = self._fetch_callers(chunk_id=chunk["id"], repo_id=request.repo_id, limit=request.max_callers)
        callees = self._fetch_callees(chunk_id=chunk["id"], repo_id=request.repo_id, limit=request.max_callees)

        prompt = self._build_prompt(chunk=chunk, callers=callers, callees=callees)
        raw = await self._llm.complete(
            prompt=prompt,
            system_prompt=_EXPLAIN_SYSTEM_PROMPT,
            model=self.MODEL,
            temperature=0.1,
            max_tokens=1600,
        )
        parsed = self._parse_json(raw)

        elapsed_ms = (time.perf_counter() - started) * 1000

        params_raw = parsed.get("parameters", [])
        return_raw = parsed.get("returns", {})
        side_effects = parsed.get("side_effects", [])
        complexity = parsed.get("complexity_score", 1)

        parameters: list[ParameterInfo] = []
        if isinstance(params_raw, list):
            for item in params_raw:
                if isinstance(item, dict) and item.get("name"):
                    parameters.append(
                        ParameterInfo(
                            name=str(item.get("name")),
                            type_annotation=(str(item.get("type_annotation")) if item.get("type_annotation") is not None else None),
                            default_value=(str(item.get("default_value")) if item.get("default_value") is not None else None),
                            description=(str(item.get("description")) if item.get("description") is not None else None),
                        )
                    )

        if not isinstance(return_raw, dict):
            return_raw = {}

        try:
            complexity_int = int(complexity)
        except (TypeError, ValueError):
            complexity_int = 1
        complexity_int = min(max(complexity_int, 1), 10)

        if not isinstance(side_effects, list):
            side_effects = []

        return ExplainResponse(
            function_name=chunk["name"],
            file_path=chunk["file_path"],
            start_line=int(chunk.get("start_line", 0) or 0),
            end_line=int(chunk.get("end_line", 0) or 0),
            summary=str(parsed.get("summary") or "No summary available."),
            parameters=parameters,
            returns=ReturnInfo(
                type_annotation=(str(return_raw.get("type_annotation")) if return_raw.get("type_annotation") is not None else None),
                description=str(return_raw.get("description") or ""),
            ),
            side_effects=[str(item) for item in side_effects if str(item).strip()],
            callers=[
                CallerCalleeInfo(
                    function_name=str(row.get("name") or ""),
                    file_path=str(row.get("file_path") or ""),
                    node_id=str(row.get("node_id") or ""),
                )
                for row in callers
                if row.get("node_id")
            ],
            callees=[
                CallerCalleeInfo(
                    function_name=str(row.get("name") or ""),
                    file_path=str(row.get("file_path") or ""),
                    node_id=str(row.get("node_id") or ""),
                )
                for row in callees
                if row.get("node_id")
            ],
            complexity_score=complexity_int,
            provider_used="openrouter",
            model_used=self.MODEL,
            explanation_ms=elapsed_ms,
        )

    async def _lookup_chunk(self, request: ExplainRequest) -> dict | None:
        repo_uuid: uuid.UUID | str
        try:
            repo_uuid = uuid.UUID(request.repo_id)
        except ValueError:
            repo_uuid = request.repo_id

        if request.chunk_id:
            try:
                chunk_uuid = uuid.UUID(request.chunk_id)
            except ValueError:
                chunk_uuid = request.chunk_id

            stmt = (
                select(
                    CodeChunk.id,
                    CodeChunk.name,
                    CodeChunk.content,
                    CodeChunk.start_line,
                    CodeChunk.end_line,
                    SourceFile.file_path,
                )
                .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
                .where(SourceFile.repository_id == repo_uuid)
                .where(CodeChunk.id == chunk_uuid)
                .limit(1)
            )
            row = (await self._db.execute(stmt)).first()
            if row:
                return {
                    "id": str(row.id),
                    "name": row.name,
                    "content": row.content,
                    "start_line": row.start_line,
                    "end_line": row.end_line,
                    "file_path": row.file_path,
                }
            return None

        if not request.function_name:
            return None

        stmt = (
            select(
                CodeChunk.id,
                CodeChunk.name,
                CodeChunk.content,
                CodeChunk.start_line,
                CodeChunk.end_line,
                SourceFile.file_path,
            )
            .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
            .where(SourceFile.repository_id == repo_uuid)
            .where(CodeChunk.name == request.function_name)
        )

        if request.file_path:
            stmt = stmt.where(SourceFile.file_path.ilike(f"%{request.file_path}%"))

        stmt = stmt.order_by(CodeChunk.start_line.asc()).limit(1)

        row = (await self._db.execute(stmt)).first()
        if not row:
            return None

        return {
            "id": str(row.id),
            "name": row.name,
            "content": row.content,
            "start_line": row.start_line,
            "end_line": row.end_line,
            "file_path": row.file_path,
        }

    @staticmethod
    def _fetch_callers(chunk_id: str, repo_id: str, limit: int) -> list[dict]:
        query = """
        MATCH (caller:Function)-[:CALLS]->(target:Function {id: $chunk_id, repo_id: $repo_id})
        WHERE caller.repo_id = $repo_id
        RETURN caller.id AS node_id,
               coalesce(caller.display_name, caller.name, caller.id) AS name,
               coalesce(caller.file_path, '') AS file_path
        LIMIT $limit
        """
        try:
            return run_query(query, chunk_id=chunk_id, repo_id=repo_id, limit=limit)
        except Exception:
            return []

    @staticmethod
    def _fetch_callees(chunk_id: str, repo_id: str, limit: int) -> list[dict]:
        query = """
        MATCH (source:Function {id: $chunk_id, repo_id: $repo_id})-[:CALLS]->(callee:Function)
        WHERE callee.repo_id = $repo_id
        RETURN callee.id AS node_id,
               coalesce(callee.display_name, callee.name, callee.id) AS name,
               coalesce(callee.file_path, '') AS file_path
        LIMIT $limit
        """
        try:
            return run_query(query, chunk_id=chunk_id, repo_id=repo_id, limit=limit)
        except Exception:
            return []

    @staticmethod
    def _build_prompt(chunk: dict, callers: list[dict], callees: list[dict]) -> str:
        callers_text = ", ".join(str(item.get("name") or "") for item in callers) if callers else "none"
        callees_text = ", ".join(str(item.get("name") or "") for item in callees) if callees else "none"

        return (
            f"File: {chunk['file_path']}\n"
            f"Symbol: {chunk['name']}\n"
            f"Lines: {chunk.get('start_line', 0)}-{chunk.get('end_line', 0)}\n"
            f"Callers: {callers_text}\n"
            f"Callees: {callees_text}\n\n"
            f"Source:\n```python\n{chunk['content']}\n```\n"
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = (raw or "").strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
