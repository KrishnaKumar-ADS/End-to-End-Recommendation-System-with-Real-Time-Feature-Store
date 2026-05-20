"""RAG chain orchestration for Week 6 ask flows."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

from core.logging import get_logger
from reasoning.context_assembler import AssembledContext, ContextAssembler, ContextAssemblerConfig
from reasoning.llm_router import (
    Provider,
    TaskType,
    _MODEL_TABLE,
    ask,
    route,
    stream_ask,
)
from reasoning.prompt_templates import ContextChunkForPrompt, build_prompt, get_task_type_from_question

logger = get_logger(__name__)

ASK_REPLY_MAX_TOKENS = 280


@dataclass
class SourceCitation:
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    score: float
    chunk_type: str


@dataclass
class RAGResponse:
    request_id: str
    answer: str
    provider_used: str
    model_used: str
    task_type: str
    sources: list[SourceCitation]
    graph_path: list[str]
    context_chunks_used: int
    estimated_tokens: int
    vector_search_ms: float
    graph_expansion_ms: float
    total_latency_ms: float
    top_result_score: float


class RAGChain:
    """Stateless orchestrator for retrieval, prompt building, and LLM answering."""

    def __init__(self, default_top_k: int = 8, include_graph: bool = True) -> None:
        self._assembler = ContextAssembler()
        self._default_top_k = default_top_k
        self._include_graph = include_graph

    async def answer(
        self,
        repo_id: str,
        question: str,
        task_type: TaskType | None = None,
        top_k: int | None = None,
        include_graph: bool | None = None,
        language_filter: str | None = None,
        chunk_type_filter: str | None = None,
        history_block: str = "",
        model_override: str | None = None,
        system_prompt_override: str | None = None,
        temperature_override: float | None = None,
        max_tokens: int = ASK_REPLY_MAX_TOKENS,
        db=None,
    ) -> RAGResponse:
        request_id = str(uuid.uuid4())[:8]
        t_start = time.perf_counter()

        resolved_task = task_type or get_task_type_from_question(question)

        config = ContextAssemblerConfig(
            top_k=top_k or self._default_top_k,
            include_graph=include_graph if include_graph is not None else self._include_graph,
            language_filter=language_filter,
            chunk_type_filter=chunk_type_filter,
        )

        context = await self._assembler.assemble(
            repo_id=repo_id,
            question=question,
            config=config,
            db=db,
        )

        built_prompt = build_prompt(
            task_type=resolved_task,
            question=question,
            context_chunks=context.chunks,
            graph_context=context.graph,
            repo_name=context.repo_name,
            history_block=history_block,
        )

        final_system_prompt = system_prompt_override or built_prompt.system_prompt
        final_temperature = temperature_override

        if model_override:
            from reasoning.openrouter_client import OpenRouterClient

            client = OpenRouterClient(model=model_override)
            answer_text = await client.complete(
                prompt=built_prompt.user_prompt,
                system_prompt=final_system_prompt,
                model=model_override,
                temperature=final_temperature if final_temperature is not None else 0.2,
                max_tokens=max_tokens,
            )
            provider = Provider.OPENROUTER
            model = model_override
        else:
            answer_text, provider, model = await ask(
                task_type=resolved_task,
                prompt=built_prompt.user_prompt,
                system_prompt=final_system_prompt,
                temperature=final_temperature,
                max_tokens=max_tokens,
            )

        total_latency_ms = (time.perf_counter() - t_start) * 1000

        return RAGResponse(
            request_id=request_id,
            answer=answer_text,
            provider_used=provider.value,
            model_used=model,
            task_type=resolved_task.value,
            sources=self._build_citations(context.chunks),
            graph_path=context.graph.call_chain,
            context_chunks_used=context.chunks_used,
            estimated_tokens=built_prompt.estimated_tokens,
            vector_search_ms=context.vector_search_ms,
            graph_expansion_ms=context.graph_expansion_ms,
            total_latency_ms=total_latency_ms,
            top_result_score=context.top_result_score,
        )

    async def stream_answer(
        self,
        repo_id: str,
        question: str,
        task_type: TaskType | None = None,
        top_k: int | None = None,
        include_graph: bool | None = None,
        history_block: str = "",
        max_tokens: int = ASK_REPLY_MAX_TOKENS,
        db=None,
    ) -> AsyncIterator[dict]:
        request_id = str(uuid.uuid4())[:8]
        t_start = time.perf_counter()

        resolved_task = task_type or get_task_type_from_question(question)

        config = ContextAssemblerConfig(
            top_k=top_k or self._default_top_k,
            include_graph=include_graph if include_graph is not None else self._include_graph,
        )

        context: AssembledContext = await self._assembler.assemble(
            repo_id=repo_id,
            question=question,
            config=config,
            db=db,
        )

        built_prompt = build_prompt(
            task_type=resolved_task,
            question=question,
            context_chunks=context.chunks,
            graph_context=context.graph,
            repo_name=context.repo_name,
            history_block=history_block,
        )

        provider_name = "unknown"
        model_name = "unknown"
        try:
            provider = route(resolved_task)
            provider_name = provider.value
            model_name = _MODEL_TABLE.get(resolved_task, {}).get(provider, "unknown")
        except Exception:
            pass

        yield {
            "type": "metadata",
            "request_id": request_id,
            "provider": provider_name,
            "model": model_name,
            "task_type": resolved_task.value,
            "chunks_used": context.chunks_used,
            "sources": [
                {
                    "file_path": chunk.file_path,
                    "function": chunk.display_name,
                    "lines": f"{chunk.start_line}-{chunk.end_line}",
                    "score": round(chunk.score, 3),
                }
                for chunk in context.chunks[:5]
            ],
            "graph_path": context.graph.call_chain,
            "top_result_score": context.top_result_score,
        }

        char_count = 0
        try:
            async for chunk, provider, model in stream_ask(
                task_type=resolved_task,
                prompt=built_prompt.user_prompt,
                system_prompt=built_prompt.system_prompt,
                max_tokens=max_tokens,
            ):
                if provider is not None:
                    provider_name = provider.value
                    model_name = model
                char_count += len(chunk)
                yield {"type": "delta", "delta": chunk}
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}

        total_latency_ms = (time.perf_counter() - t_start) * 1000
        yield {
            "type": "done",
            "provider": provider_name,
            "model": model_name,
            "total_latency_ms": round(total_latency_ms, 1),
            "total_chars": char_count,
        }

    @staticmethod
    def _build_citations(chunks: list[ContextChunkForPrompt]) -> list[SourceCitation]:
        citations: list[SourceCitation] = []
        for chunk in chunks:
            citations.append(
                SourceCitation(
                    file_path=chunk.file_path,
                    function_name=chunk.display_name,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=round(chunk.score, 4),
                    chunk_type=chunk.chunk_type,
                )
            )
        return citations
