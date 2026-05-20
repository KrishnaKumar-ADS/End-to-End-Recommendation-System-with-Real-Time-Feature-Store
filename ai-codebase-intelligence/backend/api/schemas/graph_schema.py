"""Pydantic schemas for the graph API endpoints."""
from pydantic import BaseModel


class GraphNode(BaseModel):
    id: str
    label: str              # "Function", "Class", "File"
    name: str
    file: str | None = None
    start_line: int | None = None


class GraphEdge(BaseModel):
    source: str             # source node id
    target: str             # target node id
    type: str               # "CALLS", "IMPORTS", "CONTAINS", "INHERITS"


class GraphResponse(BaseModel):
    repo_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    node_count: int
    edge_count: int


class CallChainResponse(BaseModel):
    function_id: str
    function_name: str
    call_chain: list[dict]
    callers: list[dict]


class GraphStatsResponse(BaseModel):
    repo_id: str
    nodes: dict[str, int]
    relationships: dict[str, int]