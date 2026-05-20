"""Hybrid security analysis chain: static scan + optional LLM triage."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass

import redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from analysis.security_scanner import SecurityFinding, SecurityScanner, Severity
from core.config import get_settings
from core.logging import get_logger
from db.models import CodeChunk, SourceFile
from reasoning.llm_router import TaskType, ask

logger = get_logger(__name__)
settings = get_settings()

SECURITY_CACHE_TTL = 60 * 60


@dataclass
class SecurityReport:
    repo_id: str
    findings: list[SecurityFinding]
    false_positives_removed: int
    summary_stats: dict
    scan_duration_ms: int
    static_findings_count: int
    chunks_scanned: int
    file_filter: str
    provider_used: str
    model_used: str


class SecurityChain:
    """Run static scanner and optionally validate findings with LLM."""

    def __init__(self, scanner: SecurityScanner | None = None) -> None:
        self._scanner = scanner or SecurityScanner()
        self._redis = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )

    async def analyze(
        self,
        repo_id: str,
        db: AsyncSession,
        file_filter: str = "",
        max_llm_calls: int = 10,
    ) -> SecurityReport:
        started = time.perf_counter()

        chunks = await self._load_chunks(repo_id=repo_id, db=db, file_filter=file_filter)
        static_findings = self._scanner.scan_chunks_batch(chunks)
        deduped_findings = self._dedupe_findings(static_findings)

        provider_used = ""
        model_used = ""
        llm_candidates = deduped_findings[: max(1, max_llm_calls)] if deduped_findings else []

        for finding in llm_candidates:
            llm_analysis, false_positive, provider, model = await self._llm_triage(finding)
            finding.llm_analysis = llm_analysis
            finding.false_positive = false_positive
            if provider:
                provider_used = provider
            if model:
                model_used = model

        final_findings = [finding for finding in deduped_findings if not finding.false_positive]
        final_findings.sort(
            key=lambda finding: (
                -self._severity_weight(finding.severity),
                finding.file_path,
                finding.line_number,
            )
        )

        stats = self._build_summary_stats(final_findings)
        duration_ms = int((time.perf_counter() - started) * 1000)

        return SecurityReport(
            repo_id=repo_id,
            findings=final_findings,
            false_positives_removed=len([finding for finding in deduped_findings if finding.false_positive]),
            summary_stats=stats,
            scan_duration_ms=duration_ms,
            static_findings_count=len(static_findings),
            chunks_scanned=len(chunks),
            file_filter=file_filter,
            provider_used=provider_used,
            model_used=model_used,
        )

    async def _load_chunks(
        self,
        repo_id: str,
        db: AsyncSession,
        file_filter: str,
    ) -> list[dict]:
        try:
            repo_uuid = uuid.UUID(repo_id)
        except ValueError:
            repo_uuid = repo_id

        query = (
            select(
                CodeChunk.content,
                CodeChunk.name,
                CodeChunk.start_line,
                SourceFile.file_path,
            )
            .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
            .where(SourceFile.repository_id == repo_uuid)
        )

        if file_filter:
            query = query.where(SourceFile.file_path.like(f"{file_filter}%"))

        rows = (await db.execute(query)).all()
        return [
            {
                "content": row.content or "",
                "name": row.name or "",
                "start_line": row.start_line or 1,
                "file_path": row.file_path or "",
            }
            for row in rows
        ]

    def _dedupe_findings(self, findings: list[SecurityFinding]) -> list[SecurityFinding]:
        seen: set[tuple[str, str, int, str]] = set()
        deduped: list[SecurityFinding] = []

        for finding in findings:
            key = (finding.rule_id, finding.file_path, finding.line_number, finding.function)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(finding)

        return deduped

    async def _llm_triage(
        self,
        finding: SecurityFinding,
    ) -> tuple[str, bool, str, str]:
        cache_key = self._cache_key_for_finding(finding)
        cached = self._read_cache(cache_key)
        if cached:
            return (
                cached.get("llm_analysis", ""),
                bool(cached.get("false_positive", False)),
                cached.get("provider", ""),
                cached.get("model", ""),
            )

        prompt = self._build_llm_prompt(finding)
        try:
            answer_text, provider, model = await ask(
                task_type=TaskType.SECURITY,
                prompt=prompt,
                system_prompt="Respond in JSON with fields: false_positive, explanation, fix.",
                temperature=0.1,
                max_tokens=700,
            )
        except Exception as exc:
            logger.warning("security_chain_llm_triage_failed", error=str(exc))
            return (f"LLM triage unavailable: {exc}", False, "", "")

        parsed = self._parse_llm_response(answer_text)
        explanation = parsed.get("explanation") or answer_text[:1000]
        fix = parsed.get("fix") or ""
        merged = explanation if not fix else f"{explanation}\nFix: {fix}"
        false_positive = bool(parsed.get("false_positive", False))

        payload = {
            "llm_analysis": merged,
            "false_positive": false_positive,
            "provider": provider.value,
            "model": model,
        }
        self._write_cache(cache_key, payload)

        return merged, false_positive, provider.value, model

    @staticmethod
    def _build_llm_prompt(finding: SecurityFinding) -> str:
        return (
            "Evaluate this static security finding. Return compact JSON.\n"
            "{\"false_positive\": bool, \"explanation\": str, \"fix\": str}\n\n"
            f"Rule: {finding.rule_id} ({finding.category})\n"
            f"Severity: {finding.severity.value}\n"
            f"CWE: {finding.cwe_id}\n"
            f"File: {finding.file_path}:{finding.line_number}\n"
            f"Function: {finding.function}\n"
            f"Description: {finding.description}\n"
            f"Matched code: {finding.matched_text}\n"
        )

    @staticmethod
    def _parse_llm_response(raw: str) -> dict:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        return {"false_positive": False, "explanation": text, "fix": ""}

    @staticmethod
    def _severity_weight(severity: Severity) -> int:
        if severity == Severity.CRITICAL:
            return 4
        if severity == Severity.HIGH:
            return 3
        if severity == Severity.MEDIUM:
            return 2
        return 1

    @staticmethod
    def _build_summary_stats(findings: list[SecurityFinding]) -> dict:
        summary = {"total": len(findings), "critical": 0, "high": 0, "medium": 0, "low": 0}
        for finding in findings:
            summary[finding.severity.value] += 1
        return summary

    def _cache_key_for_finding(self, finding: SecurityFinding) -> str:
        raw = "|".join(
            [
                finding.rule_id,
                finding.file_path,
                str(finding.line_number),
                finding.function,
                finding.matched_text,
            ]
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return f"security:llm:{digest}"

    def _read_cache(self, cache_key: str) -> dict | None:
        try:
            raw = self._redis.get(cache_key)
            if not raw:
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except (redis.RedisError, json.JSONDecodeError):
            return None

    def _write_cache(self, cache_key: str, payload: dict) -> None:
        try:
            self._redis.setex(cache_key, SECURITY_CACHE_TTL, json.dumps(payload))
        except redis.RedisError:
            return
