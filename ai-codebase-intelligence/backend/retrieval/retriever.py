"""Hybrid retrieval helpers used by the /ask route."""

from __future__ import annotations

from embeddings.gemini_embedder import embed_query
from embeddings.vector_store import search
from graph.traversal import find_path_between_functions
from core.logging import get_logger

logger = get_logger(__name__)


def retrieve_for_question(repo_id: str, question: str, top_k: int = 5) -> dict:
	"""
	Retrieve relevant code chunks for a natural-language question.

	Returns:
		{
		  "results": [ ... normalized hit dicts ... ],
		  "graph_path": [symbol_name, ...],
		}
	"""
	query_vector = embed_query(question)

	points = search(
		query_vector=query_vector,
		repo_id=repo_id,
		top_k=top_k,
		score_threshold=0.2,
	)

	results: list[dict] = []
	for p in points:
		payload = p.payload or {}
		results.append(
			{
				"chunk_id": str(p.id),
				"score": round(float(p.score), 4),
				"name": payload.get("name", ""),
				"display_name": payload.get("display_name", ""),
				"chunk_type": payload.get("chunk_type", ""),
				"file_path": payload.get("file_path", ""),
				"language": payload.get("language", ""),
				"start_line": payload.get("start_line", 0),
				"end_line": payload.get("end_line", 0),
				"docstring": payload.get("docstring", ""),
				"content_preview": payload.get("content_preview", ""),
			}
		)

	graph_path: list[str] = []
	# Best effort: if two function hits are available, attempt call-path lookup.
	if len(results) >= 2:
		src = results[0]["chunk_id"]
		tgt = results[1]["chunk_id"]
		try:
			path_nodes = find_path_between_functions(src, tgt, max_depth=5)
			graph_path = [n.get("name") for n in path_nodes if n.get("name")]
		except Exception as exc:
			logger.debug("retriever_graph_path_failed", repo_id=repo_id, error=str(exc))

	return {
		"results": results,
		"graph_path": graph_path,
	}
