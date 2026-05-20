"""Data models for graph expansion."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraphNode:
    """A graph node returned by expansion."""

    node_id: str
    name: str
    file_path: str
    node_type: str
    start_line: int = 0
    end_line: int = 0
    hop_distance: int = 0
    in_degree: int = 0
    out_degree: int = 0
    importance_score: float = 0.0
    chunk_id: str | None = None
    code_snippet: str | None = None

    def __hash__(self) -> int:
        return hash(self.node_id)


@dataclass
class GraphEdge:
    """A directed relationship between two graph nodes."""

    source: str
    target: str
    edge_type: str
    weight: float = 1.0


@dataclass
class ExpansionConfig:
    """Config for one graph-expansion run."""

    max_depth: int = 3
    max_nodes: int = 25
    include_callers: bool = True
    include_callees: bool = True
    include_imports: bool = False
    include_siblings: bool = False
    centrality_weight: float = 0.4
    semantic_weight: float = 0.3
    hop_decay: float = 0.8


@dataclass
class ExpandedContext:
    """Expanded subgraph output for context assembly."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    seed_node_ids: list[str] = field(default_factory=list)
    total_hops: int = 0
    nodes_visited: int = 0
    expansion_ms: float = 0.0
