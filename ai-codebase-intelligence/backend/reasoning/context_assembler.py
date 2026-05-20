"""Context assembler combining vector search and graph expansion for RAG."""

from __future__ import annotations

import time
from dataclasses import dataclass

from caching.cache_manager import get_cache_manager
from context_manager.window_manager import ContextWindowManager
from graph_expansion.expander import GraphContextExpander
from graph_expansion.models import ExpansionConfig
from reasoning.prompt_templates import ContextChunkForPrompt, GraphContextForPrompt

SYSTEM_PROMPT_RESERVE_TOKENS = 500
QUESTION_RESERVE_TOKENS = 200
GRAPH_CONTEXT_RESERVE_TOKENS = 300
RESPONSE_RESERVE_TOKENS = 4096
MIN_CHUNK_SCORE = 0.25

MODEL_CONTEXT_WINDOWS = {
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "deepseek-coder": 128_000,
    "deepseek-chat": 65_536,
    "deepseek-reasoner": 65_536,
    "qwen/qwen-2.5-coder-32b-instruct": 32_768,
    "qwen/qwen-max": 128_000,
}

RESERVED_OVERHEAD = (
    SYSTEM_PROMPT_RESERVE_TOKENS
    + QUESTION_RESERVE_TOKENS
    + GRAPH_CONTEXT_RESERVE_TOKENS
    + RESPONSE_RESERVE_TOKENS
)


@dataclass
class AssembledContext:
    question: str
    repo_id: str
    repo_name: str
    chunks: list[ContextChunkForPrompt]
    graph: GraphContextForPrompt
    total_chunks_found: int
    chunks_used: int
    estimated_tokens: int
    vector_search_ms: float
    graph_expansion_ms: float
    include_graph: bool
    top_result_score: float


@dataclass
class ContextAssemblerConfig:
    top_k: int = 8
    include_graph: bool = True
    graph_hops: int = 2
    language_filter: str | None = None
    chunk_type_filter: str | None = None
    model_name: str = "gemini-2.0-flash"


class ContextAssembler:
    """Stateless context assembly service used by the RAG chain."""

    async def assemble(
        self,
        repo_id: str,
        question: str,
        config: ContextAssemblerConfig | None = None,
        db=None,
    ) -> AssembledContext:
        cfg = config or ContextAssemblerConfig()
        cache = get_cache_manager()
        repo_name = await self._get_repo_name(repo_id, db)

        query_vector = await self._embed_question(question)

        t_search_start = time.perf_counter()
        raw_results = await self._vector_search(
            repo_id=repo_id,
            query_vector=query_vector,
            top_k=cfg.top_k,
            language=cfg.language_filter,
            chunk_type=cfg.chunk_type_filter,
        )
        vector_search_ms = (time.perf_counter() - t_search_start) * 1000

        filtered = [result for result in raw_results if result["score"] >= MIN_CHUNK_SCORE]

        graph_context = GraphContextForPrompt()
        graph_expansion_ms = 0.0
        if cfg.include_graph and filtered:
            t_graph_start = time.perf_counter()
            seed_ids = [item.get("chunk_id", "") for item in filtered[: min(5, len(filtered))] if item.get("chunk_id")]
            graph_nodes = cache.get_graph_expansion(node_ids=seed_ids, repo_id=repo_id, max_depth=cfg.graph_hops)
            if graph_nodes is None:
                graph_context = await self._expand_graph(
                    repo_id=repo_id,
                    top_result=filtered[0],
                    hops=cfg.graph_hops,
                    seed_results=filtered,
                )
                cache.set_graph_expansion(
                    node_ids=seed_ids,
                    repo_id=repo_id,
                    max_depth=cfg.graph_hops,
                    nodes=[
                        {
                            "name": name,
                            "kind": "call_chain",
                        }
                        for name in graph_context.call_chain
                    ],
                )
            else:
                graph_context = GraphContextForPrompt(
                    call_chain=[item.get("name", "") for item in graph_nodes if item.get("kind") == "call_chain" and item.get("name")],
                )
            graph_expansion_ms = (time.perf_counter() - t_graph_start) * 1000

        context_window = MODEL_CONTEXT_WINDOWS.get(cfg.model_name, 65_536)
        available_tokens = context_window - RESERVED_OVERHEAD
        selected_chunks, total_tokens = self._apply_token_budget(filtered, available_tokens)

        # Week 9 packer: prioritize by importance score and enforce model budget.
        for chunk in selected_chunks:
            # Combine retrieval score + graph relevance into one rank signal.
            if "importance_score" not in chunk:
                chunk["importance_score"] = float(chunk.get("score", 0.0))

        manager = ContextWindowManager(model=cfg.model_name)
        packed = manager.pack(
            system_prompt="",
            history_block="",
            code_chunks=selected_chunks,
            question=question,
        )
        selected_chunks = packed.selected_chunks
        total_tokens = packed.total_tokens

        prompt_chunks = [self._to_prompt_chunk(chunk) for chunk in selected_chunks]

        top_score = filtered[0]["score"] if filtered else 0.0

        return AssembledContext(
            question=question,
            repo_id=repo_id,
            repo_name=repo_name,
            chunks=prompt_chunks,
            graph=graph_context,
            total_chunks_found=len(raw_results),
            chunks_used=len(prompt_chunks),
            estimated_tokens=total_tokens,
            vector_search_ms=vector_search_ms,
            graph_expansion_ms=graph_expansion_ms,
            include_graph=cfg.include_graph,
            top_result_score=top_score,
        )

    async def _embed_question(self, question: str) -> list[float]:
        from embeddings.gemini_embedder import embed_query

        return embed_query(question)

    async def _vector_search(
        self,
        repo_id: str,
        query_vector: list[float],
        top_k: int,
        language: str | None,
        chunk_type: str | None,
    ) -> list[dict]:
        from embeddings.vector_store import search

        scored_points = search(
            query_vector=query_vector,
            repo_id=repo_id,
            top_k=top_k,
            language=language,
            chunk_type=chunk_type,
        )

        results: list[dict] = []
        for point in scored_points:
            payload = point.payload or {}
            results.append(
                {
                    "chunk_id": str(point.id),
                    "name": payload.get("name", "unknown"),
                    "display_name": payload.get("display_name", payload.get("name", "unknown")),
                    "file_path": payload.get("file_path", "unknown"),
                    "language": payload.get("language", "unknown"),
                    "chunk_type": payload.get("chunk_type", "unknown"),
                    "content": payload.get("content", payload.get("content_preview", "")),
                    "start_line": payload.get("start_line", 0),
                    "end_line": payload.get("end_line", 0),
                    "docstring": payload.get("docstring", ""),
                    "score": float(point.score),
                }
            )

        return results

    async def _expand_graph(
        self,
        repo_id: str,
        top_result: dict,
        hops: int,
        seed_results: list[dict] | None = None,
    ) -> GraphContextForPrompt:
        seed_results = seed_results or []
        seed_ids = [item.get("chunk_id", "") for item in seed_results if item.get("chunk_id")]
        semantic_scores = {
            item.get("chunk_id", ""): float(item.get("score", 0.0))
            for item in seed_results
            if item.get("chunk_id")
        }

        if not seed_ids:
            seed_id = top_result.get("chunk_id", "")
            if seed_id:
                seed_ids = [seed_id]
                semantic_scores[seed_id] = float(top_result.get("score", 0.0))

        if not seed_ids:
            return GraphContextForPrompt()

        try:
            expander = GraphContextExpander()
            expanded = expander.expand(
                seed_node_ids=seed_ids,
                repo_id=repo_id,
                config=ExpansionConfig(max_depth=hops, max_nodes=20, include_callers=True, include_callees=True),
                semantic_scores=semantic_scores,
            )

            ordered_nodes = expanded.nodes
            callers = [node.name for node in ordered_nodes if node.hop_distance > 0][:5]
            callees = [node.name for node in ordered_nodes if node.hop_distance > 0][5:10]
            call_chain = [node.name for node in ordered_nodes[: min(8, len(ordered_nodes))] if node.name]

            related_files = []
            for node in ordered_nodes:
                if node.file_path:
                    related_files.append(node.file_path)

            return GraphContextForPrompt(
                call_chain=list(dict.fromkeys(call_chain)),
                callers=list(dict.fromkeys([name for name in callers if name]))[:5],
                callees=list(dict.fromkeys([name for name in callees if name]))[:5],
                class_parents=[],
                related_files=list(dict.fromkeys([path for path in related_files if path]))[:5],
            )
        except Exception:
            return GraphContextForPrompt()

    def _apply_token_budget(
        self,
        results: list[dict],
        available_tokens: int,
    ) -> tuple[list[dict], int]:
        selected: list[dict] = []
        total_tokens = 0

        for result in results:
            chunk_tokens = max(1, len(result.get("content", "")) // 4)

            if total_tokens + chunk_tokens > available_tokens and selected:
                break

            selected.append(result)
            total_tokens += chunk_tokens

        return selected, total_tokens

    def _to_prompt_chunk(self, result: dict) -> ContextChunkForPrompt:
        return ContextChunkForPrompt(
            file_path=result.get("file_path", "unknown"),
            name=result.get("name", "unknown"),
            display_name=result.get("display_name", result.get("name", "unknown")),
            chunk_type=result.get("chunk_type", "unknown"),
            start_line=result.get("start_line", 0),
            end_line=result.get("end_line", 0),
            score=float(result.get("score", 0.0)),
            content=result.get("content", ""),
            docstring=result.get("docstring", ""),
            language=result.get("language", "unknown"),
        )

    @staticmethod
    async def _get_repo_name(repo_id: str, db) -> str:
        if db is None:
            return "unknown-repo"

        try:
            from sqlalchemy import select

            from db.models import Repository

            result = await db.execute(
                select(Repository.name).where(Repository.id == repo_id)
            )
            name = result.scalar_one_or_none()
            return name or "unknown-repo"
        except Exception:
            return "unknown-repo"
