"""Node scoring for graph-expansion results."""

from __future__ import annotations

import math

from graph_expansion.models import ExpansionConfig, GraphNode


def score_nodes(
    nodes: list[GraphNode],
    config: ExpansionConfig,
    semantic_scores: dict[str, float] | None = None,
) -> list[GraphNode]:
    """Assign importance scores and return nodes sorted descending."""
    if not nodes:
        return nodes

    semantic_scores = semantic_scores or {}

    raw_centrality = [float(max(0, n.in_degree) + max(0, n.out_degree)) for n in nodes]
    max_centrality = max(raw_centrality) if raw_centrality else 1.0
    if max_centrality <= 0.0:
        max_centrality = 1.0
    normalized_centrality = [value / max_centrality for value in raw_centrality]

    raw_semantic = [max(0.0, float(semantic_scores.get(n.node_id, 0.0))) for n in nodes]
    max_semantic = max(raw_semantic) if raw_semantic else 1.0
    if max_semantic <= 0.0:
        max_semantic = 1.0
    normalized_semantic = [value / max_semantic for value in raw_semantic]

    hop_scores = [math.pow(config.hop_decay, max(0, n.hop_distance)) for n in nodes]

    hop_weight = 1.0 - config.centrality_weight - config.semantic_weight
    if hop_weight < 0.0:
        hop_weight = 0.0

    for idx, node in enumerate(nodes):
        node.importance_score = (
            hop_weight * hop_scores[idx]
            + config.centrality_weight * normalized_centrality[idx]
            + config.semantic_weight * normalized_semantic[idx]
        )

    nodes.sort(key=lambda item: item.importance_score, reverse=True)
    return nodes
