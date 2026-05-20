"""Reciprocal Rank Fusion (RRF) helpers for vector and BM25 results."""

from __future__ import annotations

from typing import Any

RRF_K = 60
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_BM25_WEIGHT = 0.3


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = RRF_K,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    id_field: str = "id",
) -> list[dict[str, Any]]:
    total_weight = vector_weight + bm25_weight
    if total_weight == 0:
        vector_weight = 0.5
        bm25_weight = 0.5
    else:
        vector_weight /= total_weight
        bm25_weight /= total_weight

    scores: dict[str, float] = {}
    meta: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(vector_results, start=1):
        doc_id = str(item[id_field])
        scores[doc_id] = scores.get(doc_id, 0.0) + (vector_weight / (k + rank))
        if doc_id not in meta:
            meta[doc_id] = dict(item)
            meta[doc_id]["vector_rank"] = rank
            meta[doc_id]["vector_score"] = float(item.get("score", 0.0))
            meta[doc_id].setdefault("bm25_rank", None)
            meta[doc_id].setdefault("bm25_score", 0.0)
        else:
            meta[doc_id]["vector_rank"] = rank
            meta[doc_id]["vector_score"] = float(item.get("score", 0.0))

    for rank, item in enumerate(bm25_results, start=1):
        doc_id = str(item[id_field])
        scores[doc_id] = scores.get(doc_id, 0.0) + (bm25_weight / (k + rank))
        if doc_id not in meta:
            meta[doc_id] = dict(item)
            meta[doc_id]["bm25_rank"] = rank
            meta[doc_id]["bm25_score"] = float(item.get("bm25_score", 0.0))
            meta[doc_id].setdefault("vector_rank", None)
            meta[doc_id].setdefault("vector_score", 0.0)
        else:
            meta[doc_id]["bm25_rank"] = rank
            meta[doc_id]["bm25_score"] = float(item.get("bm25_score", 0.0))

    sorted_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
    fused: list[dict[str, Any]] = []
    for final_rank, doc_id in enumerate(sorted_ids, start=1):
        out = meta[doc_id]
        out["hybrid_score"] = scores[doc_id]
        out["hybrid_rank"] = final_rank
        fused.append(out)

    return fused


def vector_only(vector_results: list[dict[str, Any]], top_k: int = 20) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank, item in enumerate(vector_results[:top_k], start=1):
        out = dict(item)
        score = float(item.get("score", 0.0))
        out["hybrid_score"] = score
        out["hybrid_rank"] = rank
        out["vector_rank"] = rank
        out["vector_score"] = score
        out["bm25_rank"] = None
        out["bm25_score"] = 0.0
        output.append(out)
    return output


def bm25_only(bm25_results: list[dict[str, Any]], top_k: int = 20) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank, item in enumerate(bm25_results[:top_k], start=1):
        out = dict(item)
        score = float(item.get("bm25_score", 0.0))
        out["hybrid_score"] = score
        out["hybrid_rank"] = rank
        out["bm25_rank"] = rank
        out["bm25_score"] = score
        out["vector_rank"] = None
        out["vector_score"] = 0.0
        output.append(out)
    return output
