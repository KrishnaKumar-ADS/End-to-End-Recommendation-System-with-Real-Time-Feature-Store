"""Bug localization pipeline using graph context and LLM reasoning."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import LLMProviderError
from core.logging import get_logger
from db.models import CodeChunk, SourceFile
from graph.neo4j_client import run_query
from reasoning.llm_router import TaskType, ask

logger = get_logger(__name__)


@dataclass
class BugLocalizationResult:
    error_signal: str
    call_chain: list[str] = field(default_factory=list)
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    root_cause_file: str = ""
    root_cause_function: str = ""
    root_cause_line: int | None = None
    explanation: str = ""
    fix_suggestion: str = ""
    confidence: str = "unknown"
    provider_used: str = ""
    model_used: str = ""
    graph_nodes_explored: int = 0


class ErrorSignalParser:
    """Extract structured hints from a free-form bug report."""

    _exception_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b")
    _file_line_re = re.compile(r"(?:at\s+)?([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+)[:\s]+(\d+)")
    _file_line_keyword_re = re.compile(r"\(([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+)\s+line\s+(\d+)\)")
    _func_in_re = re.compile(r"\bin\s+([A-Za-z_][A-Za-z0-9_]*)\b")

    def parse(self, error_description: str) -> dict:
        text = (error_description or "").strip()
        exception_match = self._exception_re.search(text)

        file_path = None
        line_number = None
        file_match = self._file_line_re.search(text)
        if not file_match:
            file_match = self._file_line_keyword_re.search(text)

        if file_match:
            file_path = file_match.group(1).replace("\\", "/")
            try:
                line_number = int(file_match.group(2))
            except ValueError:
                line_number = None

        func_match = self._func_in_re.search(text)
        function_name = func_match.group(1) if func_match else None

        return {
            "raw": text,
            "exception_type": exception_match.group(1) if exception_match else None,
            "file_path": file_path,
            "line_number": line_number,
            "function_name": function_name,
        }


def _strip_markdown_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def _build_bug_analysis_prompt(
    error_description: str,
    parsed_error: dict,
    callers: list[str],
    callees: list[str],
    call_chain: list[str],
    code_by_function: dict[str, str],
) -> str:
    chain_text = " -> ".join(call_chain) if call_chain else "(no call graph data)"

    code_blocks: list[str] = []
    for function_name in call_chain:
        snippet = code_by_function.get(function_name)
        if not snippet:
            continue
        code_blocks.append(
            f'<function name="{function_name}">\n{snippet}\n</function>'
        )

    parsed_json = json.dumps(parsed_error, indent=2)
    code_section = "\n\n".join(code_blocks) if code_blocks else "(no source snippets available)"

    return (
        "You are a senior debugging assistant.\n"
        "Analyze the bug signal and identify likely root cause.\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "root_cause_file": "...",\n'
        '  "root_cause_function": "...",\n'
        '  "root_cause_line": 0,\n'
        '  "explanation": "...",\n'
        '  "fix_suggestion": "...",\n'
        '  "confidence": "high|medium|low|unknown"\n'
        "}\n\n"
        f"Error description:\n{error_description}\n\n"
        f"Parsed signal:\n{parsed_json}\n\n"
        f"Callers: {', '.join(callers) if callers else '(none)'}\n"
        f"Callees: {', '.join(callees) if callees else '(none)'}\n"
        f"Call chain: {chain_text}\n\n"
        f"Source snippets:\n{code_section}"
    )


def _parse_llm_bug_response(
    raw_text: str,
    result: BugLocalizationResult,
    parsed_error: dict,
) -> BugLocalizationResult:
    cleaned = _strip_markdown_fence(raw_text)
    payload: dict = {}

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = {}

    fallback_file = parsed_error.get("file_path") or ""
    fallback_function = parsed_error.get("function_name") or ""
    fallback_line = parsed_error.get("line_number")

    result.root_cause_file = str(payload.get("root_cause_file") or fallback_file)
    result.root_cause_function = str(payload.get("root_cause_function") or fallback_function)

    line_value = payload.get("root_cause_line", fallback_line)
    if isinstance(line_value, int):
        result.root_cause_line = line_value
    else:
        try:
            result.root_cause_line = int(line_value)
        except (TypeError, ValueError):
            result.root_cause_line = fallback_line if isinstance(fallback_line, int) else None

    result.explanation = str(
        payload.get("explanation")
        or "Unable to parse structured LLM output; using fallback error signal analysis."
    )
    result.fix_suggestion = str(
        payload.get("fix_suggestion")
        or "Add guards, input validation, and focused unit tests around the failing path."
    )

    confidence = str(payload.get("confidence") or "low").lower().strip()
    if confidence not in {"high", "medium", "low", "unknown"}:
        confidence = "unknown"
    result.confidence = confidence

    return result


class BugLocalizer:
    """High-level bug localization service."""

    def __init__(self) -> None:
        self._parser = ErrorSignalParser()

    async def localize(
        self,
        repo_id: str,
        error_description: str,
        db: AsyncSession,
        max_hops: int = 4,
    ) -> BugLocalizationResult:
        parsed_error = self._parser.parse(error_description)
        result = BugLocalizationResult(error_signal=error_description)

        callers: list[str] = []
        callees: list[str] = []
        call_chain: list[str] = []

        function_name = parsed_error.get("function_name") or ""
        if function_name:
            callers, callees = self._query_call_graph(
                repo_id=repo_id,
                function_name=function_name,
                max_hops=max_hops,
            )

            ordered = list(reversed(callers[:max_hops])) + [function_name] + callees[:max_hops]
            seen: set[str] = set()
            for name in ordered:
                if name and name not in seen:
                    seen.add(name)
                    call_chain.append(name)
        else:
            call_chain = []

        code_by_function = await self._load_source_snippets(
            repo_id=repo_id,
            db=db,
            function_names=call_chain,
        )

        prompt = _build_bug_analysis_prompt(
            error_description=error_description,
            parsed_error=parsed_error,
            callers=callers,
            callees=callees,
            call_chain=call_chain,
            code_by_function=code_by_function,
        )

        answer_text, provider, model = await ask(
            task_type=TaskType.CODE_QA,
            prompt=prompt,
            system_prompt="Respond with strict JSON only.",
            temperature=0.1,
            max_tokens=1200,
        )

        result.callers = callers
        result.callees = callees
        result.call_chain = call_chain
        result.provider_used = provider.value
        result.model_used = model
        result.graph_nodes_explored = len(set(callers + callees + call_chain))

        return _parse_llm_bug_response(answer_text, result, parsed_error)

    def _query_call_graph(
        self,
        repo_id: str,
        function_name: str,
        max_hops: int,
    ) -> tuple[list[str], list[str]]:
        depth = max(1, min(max_hops, 8))

        callers_query = f"""
        MATCH path = (caller:Function)-[:CALLS*1..{depth}]->(target:Function)
        WHERE target.repo_id = $repo_id
          AND (target.name = $fn OR target.display_name ENDS WITH $fn)
          AND caller.repo_id = $repo_id
        RETURN DISTINCT caller.name AS name, length(path) AS depth
        ORDER BY depth ASC
        LIMIT 30
        """

        callees_query = f"""
        MATCH path = (target:Function)-[:CALLS*1..{depth}]->(callee:Function)
        WHERE target.repo_id = $repo_id
          AND (target.name = $fn OR target.display_name ENDS WITH $fn)
          AND callee.repo_id = $repo_id
        RETURN DISTINCT callee.name AS name, length(path) AS depth
        ORDER BY depth ASC
        LIMIT 30
        """

        try:
            callers_rows = run_query(callers_query, repo_id=repo_id, fn=function_name)
            callees_rows = run_query(callees_query, repo_id=repo_id, fn=function_name)
        except Exception as exc:
            logger.warning("bug_localizer_graph_lookup_failed", error=str(exc))
            return [], []

        callers = [row.get("name", "") for row in callers_rows if row.get("name")]
        callees = [row.get("name", "") for row in callees_rows if row.get("name")]
        return callers, callees

    async def _load_source_snippets(
        self,
        repo_id: str,
        db: AsyncSession,
        function_names: list[str],
    ) -> dict[str, str]:
        if not function_names:
            return {}

        try:
            repo_uuid = uuid.UUID(repo_id)
        except ValueError:
            repo_uuid = repo_id

        query = (
            select(
                CodeChunk.name,
                CodeChunk.content,
                SourceFile.file_path,
                CodeChunk.start_line,
                CodeChunk.end_line,
            )
            .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
            .where(SourceFile.repository_id == repo_uuid)
            .where(CodeChunk.name.in_(function_names))
            .limit(100)
        )
        rows = (await db.execute(query)).all()

        code_by_function: dict[str, str] = {}
        for row in rows:
            name = row.name or ""
            if not name or name in code_by_function:
                continue

            file_path = row.file_path or ""
            start_line = row.start_line or 0
            end_line = row.end_line or 0
            snippet = (row.content or "")[:3000]
            header = f"# {file_path}:{start_line}-{end_line}\n"
            code_by_function[name] = header + snippet

        return code_by_function
