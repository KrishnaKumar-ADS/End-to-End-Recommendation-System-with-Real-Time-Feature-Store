"""Legacy compatibility shim for the old chain entrypoint."""

from __future__ import annotations

import asyncio

from reasoning.rag_chain import RAGChain


async def run_rag_chain_async(repo_id: str, question: str, top_k: int = 5, db=None) -> dict:
    """Compatibility async wrapper returning the legacy dictionary payload."""
    chain = RAGChain(default_top_k=top_k, include_graph=True)
    response = await chain.answer(
        repo_id=repo_id,
        question=question,
        top_k=top_k,
        db=db,
    )
    return {
        "answer": response.answer,
        "provider": response.provider_used,
        "model": response.model_used,
        "sources": [
            {
                "file": source.file_path,
                "function": source.function_name,
                "lines": f"{source.start_line}-{source.end_line}",
            }
            for source in response.sources
        ],
        "graph_path": response.graph_path,
    }


def run_rag_chain(repo_id: str, question: str, top_k: int = 5) -> dict:
    """Sync wrapper kept for legacy callers outside async contexts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_rag_chain_async(repo_id=repo_id, question=question, top_k=top_k))

    raise RuntimeError(
        "run_rag_chain() was called from an active event loop. "
        "Use run_rag_chain_async() in async call sites."
    )
