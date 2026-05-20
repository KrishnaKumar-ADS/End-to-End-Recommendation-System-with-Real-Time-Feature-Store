"""Ask endpoint with session-aware responses and step-level streaming SSE."""

from __future__ import annotations

import asyncio
import json as _json
import re as _re
import time as _time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.ask_schema import (
    AskRequest,
    AskResponse,
    DeepAskRequest,
    DeepAskResponse,
    SourceReference,
)
from conversation.history_builder import build_history_block
from conversation.session_store import SessionStore
from cost_tracking.models import BudgetExceededError
from cost_tracking.tracker import CostTracker
from core.exceptions import LLMProviderError
from core.logging import get_logger
from db.database import AsyncSessionLocal, get_db
from db.models import IngestionStatus, Repository
from decomposition.query_decomposer import QueryDecomposer
from decomposition.synthesizer import AnswerSynthesizer
from evaluation.quality_evaluator import QualityEvaluator, update_running_averages
from intent.classifier import IntentClassifier
from intent.prompt_engine import (
    get_model_for_intent,
    get_system_prompt,
    get_temperature_for_intent,
)
from reasoning.context_assembler import ContextAssembler, ContextAssemblerConfig
from reasoning.circuit_breaker import CircuitBreaker
from reasoning.llm_router import TaskType, get_available_providers, stream_ask
from reasoning.prompt_templates import build_prompt, get_task_type_from_question
from reasoning.rag_chain import RAGChain
from semantic_cache.answer_cache import get_semantic_answer_cache

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency fallback
    tiktoken = None

logger = get_logger(__name__)
router = APIRouter(tags=["Query"])
_rag_chain = RAGChain(default_top_k=8, include_graph=True)
_session_store = SessionStore()
_quality_evaluator = QualityEvaluator()
_semantic_cache = get_semantic_answer_cache()
_cost_tracker = CostTracker()
_REPLY_WORD_MIN = 100
_REPLY_WORD_MAX = 200
_ASK_REPLY_MAX_TOKENS = 280
_WORD_PATTERN = _re.compile(r"\S+")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _count_words(text: str) -> int:
    if not text:
        return 0
    return len(_WORD_PATTERN.findall(text))


def _truncate_to_max_words(text: str, max_words: int = _REPLY_WORD_MAX) -> str:
    if not text:
        return text
    if max_words <= 0:
        return ""

    matches = list(_WORD_PATTERN.finditer(text))
    if len(matches) <= max_words:
        return text

    cutoff = matches[max_words].start()
    truncated = text[:cutoff].rstrip()
    if truncated and truncated[-1] not in ".!?":
        truncated += "."
    return truncated


def _trim_chunk_to_remaining_words(
    chunk: str,
    remaining_words: int,
) -> tuple[str, int, bool]:
    """Return (trimmed_chunk, emitted_words, hit_limit)."""
    if not chunk or remaining_words <= 0:
        return "", 0, True

    matches = list(_WORD_PATTERN.finditer(chunk))
    chunk_words = len(matches)
    if chunk_words <= remaining_words:
        return chunk, chunk_words, False

    cutoff = matches[remaining_words].start()
    trimmed = chunk[:cutoff].rstrip()
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed, remaining_words, True


def _ensure_fallback_word_floor(text: str, min_words: int = _REPLY_WORD_MIN) -> str:
    if _count_words(text) >= min_words:
        return text

    supplement = (
        "Interim plan while generation is unavailable: I can summarize a specific file section you name, "
        "extract likely responsibilities of key modules, map probable call paths from indexed chunks, "
        "and list concrete verification steps to run locally. I can also draft focused follow-up questions "
        "to narrow scope quickly and improve retrieval quality for the next attempt. This gives actionable "
        "progress now, and we can refine with a full LLM answer once quotas reset."
    )
    combined = text.strip()
    while _count_words(combined) < min_words:
        combined = f"{combined}\n\n{supplement}".strip()
    return combined


def _build_fallback_sources(context_chunks: list, limit: int = 5) -> list[dict]:
    sources: list[dict] = []
    for chunk in context_chunks[:limit]:
        sources.append(
            {
                "file": chunk.file_path,
                "function": chunk.display_name,
                "start_line": int(chunk.start_line),
                "end_line": int(chunk.end_line),
                "lines": f"{chunk.start_line}-{chunk.end_line}",
                "score": float(chunk.score),
                "chunk_type": chunk.chunk_type,
                "snippet": chunk.content,
            }
        )
    return sources


def _build_provider_unavailable_fallback_answer(
    question: str,
    sources: list[dict],
    graph_path: list[str],
    failure_detail: str | None = None,
) -> str:
    lines = [
        "LLM providers are temporarily unavailable due to quota or rate-limit constraints, so I cannot produce a full generated explanation right now.",
        f"I still searched indexed project context for your question: \"{question}\".",
    ]

    if sources:
        lines.append("Closest indexed evidence:")
        for index, source in enumerate(sources[:2], start=1):
            lines.append(
                f"{index}. {source.get('function', 'unknown')} in {source.get('file', 'unknown')}:{source.get('lines', '0-0')} (score {float(source.get('score', 0.0)):.2f})"
            )
    else:
        lines.append("No closely matching indexed chunks were found for this exact query.")

    if graph_path:
        lines.append(f"Potential call path: {' -> '.join(graph_path[:8])}")

    lines.append(
        "If you share a concrete filename or function name, I can provide a more targeted retrieval-based explanation immediately."
    )

    detail = (failure_detail or "").strip()
    detail_lower = detail.lower()
    if detail:
        if "free-models-per-day" in detail_lower or "x-ratelimit-remaining': '0" in detail_lower:
            lines.append("OpenRouter free-model daily quota is exhausted.")
            lines.append("Add OpenRouter credits or switch to a paid model to restore full answers immediately.")
        elif "insufficient balance" in detail_lower:
            lines.append("A provider account has insufficient balance.")
            lines.append("Add balance/credits for that provider to restore full answers.")
        elif "resourceexhausted" in detail_lower or "quota" in detail_lower:
            lines.append("An upstream provider quota is exhausted.")
            lines.append("Retry after quota reset or use another provider with available quota.")
        else:
            lines.append("Retry in about a minute for a full LLM-generated answer once provider limits recover.")
    else:
        lines.append("Retry in about a minute for a full LLM-generated answer once provider limits recover.")

    answer = "\n".join(lines)
    answer = _ensure_fallback_word_floor(answer)
    return _truncate_to_max_words(answer)


def _chunk_text_for_sse(text: str, chunk_size: int = 180) -> list[str]:
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


_SHORT_CIRCUIT_FAILURE_WINDOW_SEC = 90.0


def _is_recently_unhealthy_provider(status, now: float) -> bool:
    if status.failure_count <= 0:
        return False
    if status.last_failure_time <= 0:
        return False
    if now - status.last_failure_time > _SHORT_CIRCUIT_FAILURE_WINDOW_SEC:
        return False

    # A newer success means this provider likely recovered.
    if status.last_success_time > status.last_failure_time:
        return False

    return True


def _should_short_circuit_llm_attempts() -> bool:
    """Return True when all configured providers recently failed and likely fail again."""
    try:
        available_providers = [provider.value for provider in get_available_providers()]
        if not available_providers:
            return True

        statuses = CircuitBreaker().get_all_statuses()
        now = _time.time()

        for provider_name in available_providers:
            status = statuses.get(provider_name)
            if status is None:
                return False
            if not _is_recently_unhealthy_provider(status, now):
                return False

        return True
    except Exception:
        return False


async def _build_non_stream_fallback_response(
    request: AskRequest,
    session_id: str,
    resolved_task: TaskType,
    db: AsyncSession,
    request_started: float,
    store: SessionStore,
    attempt_context: bool = True,
    context_timeout_sec: float = 6.0,
    failure_detail: str | None = None,
) -> AskResponse:
    fallback_context = None
    if attempt_context:
        try:
            assembler = ContextAssembler()
            fallback_context = await asyncio.wait_for(
                assembler.assemble(
                    repo_id=request.repo_id,
                    question=request.question,
                    config=ContextAssemblerConfig(
                        top_k=request.top_k,
                        include_graph=request.include_graph,
                        language_filter=request.language_filter,
                        chunk_type_filter=request.chunk_type_filter,
                    ),
                    db=db,
                ),
                timeout=context_timeout_sec,
            )
        except Exception as context_exc:
            logger.warning(
                "ask_endpoint_fallback_context_failed",
                error=str(context_exc),
                repo_id=request.repo_id,
            )

    fallback_sources = _build_fallback_sources(
        fallback_context.chunks if fallback_context is not None else []
    )
    fallback_graph_path = (
        fallback_context.graph.call_chain if fallback_context is not None else []
    )
    fallback_answer = _build_provider_unavailable_fallback_answer(
        question=request.question,
        sources=fallback_sources,
        graph_path=fallback_graph_path,
        failure_detail=failure_detail,
    )
    fallback_quality = {
        "faithfulness": 0.0,
        "relevance": 0.0,
        "completeness": 0.0,
        "overall": 0.0,
        "critique": "Quality evaluation skipped because all providers were unavailable.",
        "skipped": True,
        "skip_reason": "LLM providers unavailable",
    }

    store.append_turn(
        session_id=session_id,
        role="assistant",
        content=fallback_answer,
        sources=[
            {
                "file": source.get("file", ""),
                "function": source.get("function", ""),
                "lines": source.get("lines", "0-0"),
            }
            for source in fallback_sources
        ],
        provider_used="fallback",
        model_used="retrieval-only",
    )

    return AskResponse(
        request_id=str(uuid.uuid4())[:8],
        answer=fallback_answer,
        session_id=session_id,
        provider_used="fallback",
        model_used="retrieval-only",
        task_type=resolved_task.value,
        sources=[
            SourceReference(
                file_path=str(source.get("file") or ""),
                function_name=str(source.get("function") or ""),
                start_line=int(source.get("start_line") or 0),
                end_line=int(source.get("end_line") or 0),
                score=float(source.get("score") or 0.0),
                chunk_type=str(source.get("chunk_type") or ""),
            )
            for source in fallback_sources
        ],
        graph_path=fallback_graph_path,
        context_chunks_used=len(fallback_sources),
        estimated_tokens=_estimate_tokens(fallback_answer),
        vector_search_ms=(
            fallback_context.vector_search_ms if fallback_context is not None else 0.0
        ),
        graph_expansion_ms=(
            fallback_context.graph_expansion_ms if fallback_context is not None else 0.0
        ),
        total_latency_ms=(_time.perf_counter() - request_started) * 1000,
        top_result_score=(
            fallback_context.top_result_score if fallback_context is not None else 0.0
        ),
        quality_score=fallback_quality,
        cached=False,
        cache_similarity=0.0,
        intent=resolved_task.value,
    )


async def _get_completed_repo(repo_id: str, db: AsyncSession) -> Repository:
    try:
        repo_uuid = uuid.UUID(repo_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid repository id '{repo_id}'.",
        ) from exc

    result = await db.execute(select(Repository).where(Repository.id == repo_uuid))
    repo = result.scalar_one_or_none()

    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' not found. Ingest it first with POST /api/v1/ingest.",
        )

    if repo.status != IngestionStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Repository '{repo.name}' is not yet fully indexed. "
                f"Current status: {repo.status.value}."
            ),
        )

    return repo


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a natural language question about an ingested repository.",
    description=(
        "Runs the RAG pipeline for repository-aware Q&A. "
        "Supports streaming and non-streaming responses, citation payloads, graph-path hints, "
        "semantic caching, and provider fallback behavior."
    ),
    responses={
        200: {"description": "Answer generated successfully."},
        404: {"description": "Repository or conversation session was not found."},
        422: {"description": "Repository id or request payload is invalid."},
        429: {"description": "Daily budget exceeded for provider usage."},
        500: {"description": "Unexpected server-side generation failure."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "non_stream": {
                            "summary": "Non-streaming ask",
                            "value": {
                                "repo_id": "11111111-1111-1111-1111-111111111111",
                                "question": "How does SSL verification work?",
                                "stream": False,
                                "top_k": 8,
                                "include_graph": True,
                            },
                        },
                        "stream": {
                            "summary": "Streaming ask",
                            "value": {
                                "repo_id": "11111111-1111-1111-1111-111111111111",
                                "question": "Trace the login call path and failure points.",
                                "stream": True,
                                "task_type": "code_qa",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def ask_question(request: AskRequest, db: AsyncSession = Depends(get_db)):
    repo = await _get_completed_repo(request.repo_id, db)
    request_started = _time.perf_counter()

    task_type: TaskType | None = None
    if request.task_type:
        task_type = TaskType(request.task_type)

    logger.info(
        "ask_endpoint_called",
        repo_id=request.repo_id,
        repo_name=repo.name,
        task_type=request.task_type or "auto",
        stream=request.stream,
        question_len=len(request.question),
    )

    session_id = request.session_id or _session_store.create_session()
    if request.session_id and not _session_store.session_exists(request.session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation session '{request.session_id}' not found.",
        )

    history_turns = _session_store.get_turns(session_id)
    history_block = build_history_block(history_turns)

    _session_store.append_turn(
        session_id=session_id,
        role="user",
        content=request.question,
    )

    if request.stream:
        return StreamingResponse(
            _stream_ask_with_session(
                request=request,
                session_id=session_id,
                history_block=history_block,
                store=_session_store,
                db=db,
                task_type=task_type,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    normalized_question = request.question.strip()
    resolved_task = task_type or get_task_type_from_question(normalized_question)
    cache_similarity = 0.0
    semantic_hit = False

    if _should_short_circuit_llm_attempts():
        logger.warning(
            "ask_endpoint_short_circuit_fallback",
            repo_id=request.repo_id,
            reason="all providers recently unhealthy",
        )
        return await _build_non_stream_fallback_response(
            request=request,
            session_id=session_id,
            resolved_task=resolved_task,
            db=db,
            request_started=request_started,
            store=_session_store,
            attempt_context=False,
        )

    try:
        from embeddings.gemini_embedder import embed_query

        question_vector = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: embed_query(normalized_question),
        )
    except Exception:
        question_vector = None

    if question_vector is not None:
        cache_result = await _semantic_cache.lookup(question_vector=question_vector, repo_id=request.repo_id)
        if cache_result.found and cache_result.cached_answer is not None:
            semantic_hit = True
            cache_similarity = cache_result.similarity
            cached_answer = cache_result.cached_answer

            answer_text = _truncate_to_max_words(cached_answer.answer)
            quality_score = cached_answer.quality_score

            _session_store.append_turn(
                session_id=session_id,
                role="assistant",
                content=answer_text,
                sources=cached_answer.sources[:5],
                provider_used=cached_answer.provider_used,
                model_used=cached_answer.model_used,
            )

            return AskResponse(
                request_id=str(uuid.uuid4())[:8],
                answer=answer_text,
                session_id=session_id,
                provider_used=cached_answer.provider_used or "semantic-cache",
                model_used=cached_answer.model_used or "semantic-cache",
                task_type=resolved_task.value,
                sources=[
                    SourceReference(
                        file_path=str(source.get("file") or source.get("file_path") or ""),
                        function_name=str(source.get("function") or source.get("function_name") or ""),
                        start_line=int(source.get("start_line") or 0),
                        end_line=int(source.get("end_line") or 0),
                        score=float(source.get("score") or 0.0),
                        chunk_type=str(source.get("chunk_type") or ""),
                    )
                    for source in cached_answer.sources
                ],
                graph_path=[],
                context_chunks_used=len(cached_answer.sources),
                estimated_tokens=_estimate_tokens(answer_text),
                vector_search_ms=0.0,
                graph_expansion_ms=0.0,
                total_latency_ms=0.0,
                top_result_score=cache_similarity,
                quality_score=quality_score,
                cached=True,
                cache_similarity=cache_similarity,
                intent=resolved_task.value,
            )

    try:
        rag_response = await _rag_chain.answer(
            repo_id=request.repo_id,
            question=request.question,
            task_type=resolved_task,
            top_k=request.top_k,
            include_graph=request.include_graph,
            language_filter=request.language_filter,
            chunk_type_filter=request.chunk_type_filter,
            history_block=history_block,
            max_tokens=_ASK_REPLY_MAX_TOKENS,
            db=db,
        )
    except LLMProviderError as exc:
        logger.error("ask_endpoint_llm_failed", error=str(exc), repo_id=request.repo_id)
        return await _build_non_stream_fallback_response(
            request=request,
            session_id=session_id,
            resolved_task=resolved_task,
            db=db,
            request_started=request_started,
            store=_session_store,
            failure_detail=str(exc),
        )
    except Exception as exc:
        logger.error("ask_endpoint_error", error=str(exc), repo_id=request.repo_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {exc}",
        ) from exc

    rag_response.answer = _truncate_to_max_words(rag_response.answer)

    context_chunks = [
        f"{source.file_path}:{source.start_line}-{source.end_line}\n{source.function_name}"
        for source in rag_response.sources[:8]
    ]
    quality = await _quality_evaluator.score(
        question=request.question,
        answer=rag_response.answer,
        context_chunks=context_chunks,
    )
    quality_payload = quality.model_dump()

    if question_vector is not None:
        await _semantic_cache.store(
            question_vector=question_vector,
            answer=rag_response.answer,
            sources=[
                {
                    "file_path": source.file_path,
                    "function_name": source.function_name,
                    "start_line": source.start_line,
                    "end_line": source.end_line,
                    "score": source.score,
                    "chunk_type": source.chunk_type,
                }
                for source in rag_response.sources
            ],
            quality_score=quality_payload,
            repo_id=request.repo_id,
            provider_used=rag_response.provider_used,
            model_used=rag_response.model_used,
        )

    asyncio.create_task(update_running_averages(quality))

    try:
        await _cost_tracker.record(
            provider=rag_response.provider_used,
            model=rag_response.model_used,
            input_tokens=max(1, rag_response.estimated_tokens // 2),
            output_tokens=_estimate_tokens(rag_response.answer),
            check_budget=True,
        )
    except BudgetExceededError:
        raise
    except Exception:
        # Cost accounting should not break successful answer responses.
        pass

    _session_store.append_turn(
        session_id=session_id,
        role="assistant",
        content=rag_response.answer,
        sources=[
            {
                "file": source.file_path,
                "function": source.function_name,
                "lines": f"{source.start_line}-{source.end_line}",
            }
            for source in rag_response.sources[:5]
        ],
        provider_used=rag_response.provider_used,
        model_used=rag_response.model_used,
    )

    return AskResponse(
        request_id=rag_response.request_id,
        answer=rag_response.answer,
        session_id=session_id,
        provider_used=rag_response.provider_used,
        model_used=rag_response.model_used,
        task_type=rag_response.task_type,
        sources=[
            SourceReference(
                file_path=source.file_path,
                function_name=source.function_name,
                start_line=source.start_line,
                end_line=source.end_line,
                score=source.score,
                chunk_type=source.chunk_type,
            )
            for source in rag_response.sources
        ],
        graph_path=rag_response.graph_path,
        context_chunks_used=rag_response.context_chunks_used,
        estimated_tokens=rag_response.estimated_tokens,
        vector_search_ms=rag_response.vector_search_ms,
        graph_expansion_ms=rag_response.graph_expansion_ms,
        total_latency_ms=rag_response.total_latency_ms,
        top_result_score=rag_response.top_result_score,
        quality_score=quality_payload,
        cached=semantic_hit,
        cache_similarity=cache_similarity,
        intent=rag_response.task_type,
    )


@router.get(
    "/ask/providers",
    summary="List currently available LLM providers.",
    description="Returns provider identifiers currently available to the router based on configured API keys.",
)
async def list_providers() -> dict:
    providers = get_available_providers()
    return {"providers": [provider.value for provider in providers], "count": len(providers)}


@router.get("/conversation/{session_id}", summary="Get full conversation history")
async def get_conversation(session_id: str) -> dict:
    if not _session_store.session_exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation session '{session_id}' not found.",
        )

    turns = _session_store.get_turns(session_id)
    return {
        "session_id": session_id,
        "turn_count": len(turns),
        "turns": turns,
    }


@router.delete("/conversation/{session_id}", summary="Delete conversation history")
async def delete_conversation(session_id: str) -> dict:
    if not _session_store.delete_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation session '{session_id}' not found.",
        )

    return {
        "session_id": session_id,
        "deleted": True,
    }


@router.post(
    "/ask/deep",
    response_model=DeepAskResponse,
    summary="Multi-step deep ask for compound questions",
    description=(
        "Decomposes a complex question into sub-questions, answers each sub-question, "
        "and synthesizes a final response with merged sources."
    ),
)
async def deep_ask(request: DeepAskRequest, db: AsyncSession = Depends(get_db)) -> DeepAskResponse:
    await _get_completed_repo(request.repo_id, db)

    started = _time.perf_counter()

    decomposer = QueryDecomposer()
    classifier = IntentClassifier()
    synthesizer = AnswerSynthesizer()

    sub_questions = await decomposer.decompose(request.question)
    if not sub_questions:
        sub_questions = [request.question]

    async def _run_sub_question(sub_question: str) -> tuple[str, list[dict], str]:
        intent = await classifier.classify(sub_question)
        model = get_model_for_intent(intent)
        system_prompt = get_system_prompt(intent)
        temperature = get_temperature_for_intent(intent)

        async with AsyncSessionLocal() as sub_db:
            response = await _rag_chain.answer(
                repo_id=request.repo_id,
                question=sub_question,
                task_type=None,
                top_k=8,
                include_graph=True,
                model_override=model,
                system_prompt_override=system_prompt,
                temperature_override=temperature,
                max_tokens=_ASK_REPLY_MAX_TOKENS,
                db=sub_db,
            )

        sources = [
            {
                "file": source.file_path,
                "function": source.function_name,
                "lines": f"{source.start_line}-{source.end_line}",
            }
            for source in response.sources
        ]
        return response.answer, sources, response.provider_used

    partial_answers: list[str] = []
    all_sources: list[dict] = []
    providers_used: list[str] = []

    results = await asyncio.gather(
        *[_run_sub_question(sub_question) for sub_question in sub_questions],
        return_exceptions=True,
    )

    for sub_question, result in zip(sub_questions, results):
        if isinstance(result, Exception):
            logger.warning(
                "deep_ask_sub_question_failed",
                repo_id=request.repo_id,
                sub_question=sub_question,
                error=str(result),
            )
            partial_answers.append(f"Failed to answer sub-question: {sub_question}")
            continue

        answer_text, sources, provider_used = result
        partial_answers.append(answer_text)
        all_sources.extend(sources)
        if provider_used:
            providers_used.append(provider_used)

    final_answer = await synthesizer.synthesize(
        original_question=request.question,
        sub_questions=sub_questions,
        partial_answers=partial_answers,
    )

    deduped_sources: list[dict] = []
    seen: set[str] = set()
    for source in all_sources:
        key = f"{source.get('file', '')}:{source.get('function', '')}:{source.get('lines', '')}"
        if key in seen:
            continue
        seen.add(key)
        deduped_sources.append(source)

    total_ms = (_time.perf_counter() - started) * 1000
    return DeepAskResponse(
        answer=final_answer,
        sub_questions=sub_questions,
        partial_answers=partial_answers,
        all_sources=deduped_sources,
        providers_used=list(dict.fromkeys(providers_used)),
        decomposed=len(sub_questions) > 1,
        total_ms=total_ms,
    )


async def _stream_ask_with_session(
    request: AskRequest,
    session_id: str,
    history_block: str,
    store: SessionStore,
    db: AsyncSession,
    task_type: TaskType | None,
) -> AsyncIterator[str]:
    """Full streaming pipeline with stage events and token events."""
    t_start = _time.perf_counter()
    done_marker = "data: [DONE]\n\n"

    def emit(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {_json.dumps(payload)}\n\n"

    yield emit("step", {"stage": "searching", "message": "Searching code context..."})

    resolved_task = task_type or get_task_type_from_question(request.question)

    normalized_question = request.question.strip()
    question_vector = None
    try:
        from embeddings.gemini_embedder import embed_query

        question_vector = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: embed_query(normalized_question),
        )
    except Exception:
        question_vector = None

    if question_vector is not None:
        cache_result = await _semantic_cache.lookup(
            question_vector=question_vector,
            repo_id=request.repo_id,
        )
        if cache_result.found and cache_result.cached_answer is not None:
            cached_answer = cache_result.cached_answer
            answer_text = _truncate_to_max_words(cached_answer.answer)

            source_payload = [
                {
                    "file": str(source.get("file") or source.get("file_path") or ""),
                    "function": str(
                        source.get("function") or source.get("function_name") or ""
                    ),
                    "lines": str(
                        source.get("lines")
                        or f"{int(source.get('start_line') or 0)}-{int(source.get('end_line') or 0)}"
                    ),
                }
                for source in cached_answer.sources[:5]
            ]

            store.append_turn(
                session_id=session_id,
                role="assistant",
                content=answer_text,
                sources=source_payload,
                provider_used=cached_answer.provider_used,
                model_used=cached_answer.model_used,
            )

            yield emit(
                "step",
                {
                    "stage": "cache",
                    "message": "Using semantic cache hit for this question.",
                },
            )
            for chunk in _chunk_text_for_sse(answer_text):
                yield emit("token", {"text": chunk})
            yield emit("sources", {"sources": source_payload})
            yield emit(
                "done",
                {
                    "provider": cached_answer.provider_used or "semantic-cache",
                    "model": cached_answer.model_used or "semantic-cache",
                    "session_id": session_id,
                    "cached": True,
                    "graph_path": [],
                    "timing": {
                        "search_ms": 0,
                        "graph_ms": 0,
                        "context_ms": 0,
                        "total_ms": int((_time.perf_counter() - t_start) * 1000),
                    },
                },
            )
            yield done_marker
            return

    if _should_short_circuit_llm_attempts():
        logger.warning(
            "ask_stream_short_circuit_fallback",
            repo_id=request.repo_id,
            reason="all providers recently unhealthy",
        )

        fallback_answer = _build_provider_unavailable_fallback_answer(
            question=request.question,
            sources=[],
            graph_path=[],
        )

        store.append_turn(
            session_id=session_id,
            role="assistant",
            content=fallback_answer,
            sources=[],
            provider_used="fallback",
            model_used="retrieval-only",
        )

        yield emit(
            "step",
            {
                "stage": "fallback",
                "message": "LLM providers are unavailable; returning immediate fallback.",
            },
        )
        for chunk in _chunk_text_for_sse(fallback_answer):
            yield emit("token", {"text": chunk})
        yield emit("sources", {"sources": []})
        yield emit(
            "done",
            {
                "provider": "fallback",
                "model": "retrieval-only",
                "session_id": session_id,
                "graph_path": [],
                "timing": {
                    "search_ms": 0,
                    "graph_ms": 0,
                    "context_ms": 0,
                    "total_ms": int((_time.perf_counter() - t_start) * 1000),
                },
            },
        )
        yield done_marker
        return

    assembler = ContextAssembler()
    config = ContextAssemblerConfig(
        top_k=request.top_k,
        include_graph=request.include_graph,
        language_filter=request.language_filter,
        chunk_type_filter=request.chunk_type_filter,
    )

    try:
        context = await assembler.assemble(
            repo_id=request.repo_id,
            question=request.question,
            config=config,
            db=db,
        )
    except Exception as exc:
        logger.error("ask_stream_context_failed", error=str(exc))
        yield emit("error", {"error": f"Context assembly failed: {exc}"})
        yield done_marker
        return

    yield emit(
        "step",
        {
            "stage": "searching",
            "message": f"Found {context.chunks_used} relevant chunks in {int(context.vector_search_ms)}ms.",
        },
    )

    yield emit(
        "step",
        {
            "stage": "graph",
            "message": (
                f"Graph expanded to {len(context.graph.call_chain)} call-chain nodes "
                f"in {int(context.graph_expansion_ms)}ms."
            ),
        },
    )

    built_prompt = build_prompt(
        task_type=resolved_task,
        question=request.question,
        context_chunks=context.chunks,
        graph_context=context.graph,
        repo_name=context.repo_name,
        history_block=history_block,
    )

    yield emit("step", {"stage": "context", "message": "Context ready. Routing to LLM..."})
    yield emit("step", {"stage": "generating", "message": "Generating answer..."})

    provider_used = ""
    model_used = ""
    full_answer: list[str] = []
    emitted_word_count = 0

    if _should_short_circuit_llm_attempts():
        logger.warning(
            "ask_stream_short_circuit_fallback",
            repo_id=request.repo_id,
            reason="all providers recently unhealthy",
        )

        fallback_sources = _build_fallback_sources(context.chunks)
        fallback_graph_path = context.graph.call_chain
        fallback_answer = _build_provider_unavailable_fallback_answer(
            question=request.question,
            sources=fallback_sources,
            graph_path=fallback_graph_path,
        )

        source_payload = [
            {
                "file": source.get("file", ""),
                "function": source.get("function", ""),
                "lines": source.get("lines", "0-0"),
            }
            for source in fallback_sources
        ]

        store.append_turn(
            session_id=session_id,
            role="assistant",
            content=fallback_answer,
            sources=source_payload,
            provider_used="fallback",
            model_used="retrieval-only",
        )

        yield emit(
            "step",
            {
                "stage": "fallback",
                "message": "LLM providers are unavailable; returning retrieval-only fallback.",
            },
        )
        for chunk in _chunk_text_for_sse(fallback_answer):
            yield emit("token", {"text": chunk})
        yield emit("sources", {"sources": source_payload})
        yield emit(
            "done",
            {
                "provider": "fallback",
                "model": "retrieval-only",
                "session_id": session_id,
                "graph_path": fallback_graph_path,
                "timing": {
                    "search_ms": int(context.vector_search_ms),
                    "graph_ms": int(context.graph_expansion_ms),
                    "context_ms": 0,
                    "total_ms": int((_time.perf_counter() - t_start) * 1000),
                },
            },
        )
        yield done_marker
        return

    try:
        async for chunk, provider, model in stream_ask(
            task_type=resolved_task,
            prompt=built_prompt.user_prompt,
            system_prompt=built_prompt.system_prompt,
            max_tokens=_ASK_REPLY_MAX_TOKENS,
        ):
            if provider is not None:
                provider_used = provider.value
            if model:
                model_used = model

            remaining_words = _REPLY_WORD_MAX - emitted_word_count
            chunk_text, added_words, hit_limit = _trim_chunk_to_remaining_words(
                chunk,
                remaining_words,
            )
            emitted_word_count += added_words

            if chunk_text:
                full_answer.append(chunk_text)
                yield emit("token", {"text": chunk_text})

            if hit_limit:
                break

    except LLMProviderError as exc:
        logger.error("ask_stream_llm_failed", error=str(exc))
        fallback_sources = _build_fallback_sources(context.chunks)
        fallback_graph_path = context.graph.call_chain
        fallback_answer = _build_provider_unavailable_fallback_answer(
            question=request.question,
            sources=fallback_sources,
            graph_path=fallback_graph_path,
            failure_detail=str(exc),
        )

        source_payload = [
            {
                "file": source.get("file", ""),
                "function": source.get("function", ""),
                "lines": source.get("lines", "0-0"),
            }
            for source in fallback_sources
        ]

        store.append_turn(
            session_id=session_id,
            role="assistant",
            content=fallback_answer,
            sources=source_payload,
            provider_used="fallback",
            model_used="retrieval-only",
        )

        yield emit(
            "step",
            {
                "stage": "fallback",
                "message": "LLM providers are unavailable; returning retrieval-only fallback.",
            },
        )
        for chunk in _chunk_text_for_sse(fallback_answer):
            yield emit("token", {"text": chunk})
        yield emit("sources", {"sources": source_payload})
        yield emit(
            "done",
            {
                "provider": "fallback",
                "model": "retrieval-only",
                "session_id": session_id,
                "graph_path": fallback_graph_path,
                "timing": {
                    "search_ms": int(context.vector_search_ms),
                    "graph_ms": int(context.graph_expansion_ms),
                    "context_ms": 0,
                    "total_ms": int((_time.perf_counter() - t_start) * 1000),
                },
            },
        )
        yield done_marker
        return
    except Exception as exc:
        logger.error("ask_stream_unexpected_failed", error=str(exc))
        yield emit("error", {"error": f"Streaming failed: {exc}"})
        yield done_marker
        return

    answer_text = "".join(full_answer)
    answer_text = _truncate_to_max_words(answer_text)
    sources = _build_fallback_sources(context.chunks)
    source_payload = [
        {
            "file": source.get("file", ""),
            "function": source.get("function", ""),
            "lines": source.get("lines", "0-0"),
        }
        for source in sources
    ]

    store.append_turn(
        session_id=session_id,
        role="assistant",
        content=answer_text,
        sources=source_payload,
        provider_used=provider_used,
        model_used=model_used,
    )

    yield emit("sources", {"sources": source_payload})

    timing = {
        "search_ms": int(context.vector_search_ms),
        "graph_ms": int(context.graph_expansion_ms),
        "context_ms": 0,
        "total_ms": int((_time.perf_counter() - t_start) * 1000),
    }
    yield emit(
        "done",
        {
            "provider": provider_used,
            "model": model_used,
            "session_id": session_id,
            "graph_path": context.graph.call_chain,
            "timing": timing,
        },
    )
    yield done_marker
