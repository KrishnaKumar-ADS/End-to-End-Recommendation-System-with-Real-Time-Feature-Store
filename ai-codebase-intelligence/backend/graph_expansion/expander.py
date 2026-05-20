"""Multi-hop BFS graph context expander."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

from core.logging import get_logger
from graph.neo4j_client import run_query
from graph_expansion.models import ExpandedContext, ExpansionConfig, GraphEdge, GraphNode
from graph_expansion.scorer import score_nodes

logger = get_logger(__name__)


class GraphContextExpander:
    """Expand seed graph nodes with multi-hop traversal and ranking."""

    def __init__(self, query_runner: Callable[..., list[dict]] | None = None) -> None:
        self._run_query = query_runner or run_query

    def expand(
        self,
        seed_node_ids: list[str],
        repo_id: str,
        config: ExpansionConfig | None = None,
        semantic_scores: dict[str, float] | None = None,
    ) -> ExpandedContext:
        cfg = config or ExpansionConfig()
        semantic_scores = semantic_scores or {}

        unique_seeds = [node_id for node_id in dict.fromkeys(seed_node_ids) if node_id]
        if not unique_seeds:
            return ExpandedContext(seed_node_ids=[])

        started = time.perf_counter()
        visited: set[str] = set(unique_seeds)
        frontier: deque[tuple[str, int]] = deque((node_id, 0) for node_id in unique_seeds)
        all_nodes: dict[str, GraphNode] = {}
        all_edges: list[GraphEdge] = []

        for row in self._fetch_nodes_batch(unique_seeds, repo_id):
            node = self._row_to_node(row, hop_distance=0)
            all_nodes[node.node_id] = node

        for seed_id in unique_seeds:
            all_nodes.setdefault(
                seed_id,
                GraphNode(
                    node_id=seed_id,
                    name=seed_id,
                    file_path="",
                    node_type="function",
                    hop_distance=0,
                    chunk_id=seed_id,
                ),
            )

        max_frontier_nodes = max(cfg.max_nodes * 3, cfg.max_nodes)

        while frontier:
            current_depth = frontier[0][1]
            if current_depth >= cfg.max_depth:
                break

            current_level: list[str] = []
            while frontier and frontier[0][1] == current_depth:
                node_id, _ = frontier.popleft()
                current_level.append(node_id)

            next_hop = current_depth + 1

            if cfg.include_callees:
                rows, edges = self._fetch_callees_batch(current_level, repo_id)
                all_edges.extend(edges)
                self._ingest_rows(rows, next_hop, visited, frontier, all_nodes, max_frontier_nodes)

            if cfg.include_callers:
                rows, edges = self._fetch_callers_batch(current_level, repo_id)
                all_edges.extend(edges)
                self._ingest_rows(rows, next_hop, visited, frontier, all_nodes, max_frontier_nodes)

            if cfg.include_siblings:
                rows, edges = self._fetch_siblings_batch(current_level, repo_id)
                all_edges.extend(edges)
                self._ingest_rows(rows, next_hop, visited, frontier, all_nodes, max_frontier_nodes)

            if cfg.include_imports:
                rows, edges = self._fetch_import_related_batch(current_level, repo_id)
                all_edges.extend(edges)
                self._ingest_rows(rows, next_hop, visited, frontier, all_nodes, max_frontier_nodes)

            if len(all_nodes) >= max_frontier_nodes:
                break

        ranked = score_nodes(list(all_nodes.values()), cfg, semantic_scores=semantic_scores)
        top_nodes = ranked[: cfg.max_nodes]
        top_ids = {node.node_id for node in top_nodes}

        edge_seen: set[tuple[str, str, str]] = set()
        unique_edges: list[GraphEdge] = []
        for edge in all_edges:
            if edge.source not in top_ids or edge.target not in top_ids:
                continue
            key = (edge.source, edge.target, edge.edge_type)
            if key in edge_seen:
                continue
            edge_seen.add(key)
            unique_edges.append(edge)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return ExpandedContext(
            nodes=top_nodes,
            edges=unique_edges,
            seed_node_ids=unique_seeds,
            total_hops=max((node.hop_distance for node in top_nodes), default=0),
            nodes_visited=len(visited),
            expansion_ms=elapsed_ms,
        )

    def _ingest_rows(
        self,
        rows: list[dict],
        hop_distance: int,
        visited: set[str],
        frontier: deque[tuple[str, int]],
        all_nodes: dict[str, GraphNode],
        max_frontier_nodes: int,
    ) -> None:
        for row in rows:
            node_id = str(row.get("node_id") or "").strip()
            if not node_id:
                continue

            if node_id in all_nodes:
                existing = all_nodes[node_id]
                existing.hop_distance = min(existing.hop_distance, hop_distance)
                existing.in_degree = max(existing.in_degree, int(row.get("in_degree", 0) or 0))
                existing.out_degree = max(existing.out_degree, int(row.get("out_degree", 0) or 0))
            else:
                all_nodes[node_id] = self._row_to_node(row, hop_distance=hop_distance)

            if node_id not in visited and len(visited) < max_frontier_nodes:
                visited.add(node_id)
                frontier.append((node_id, hop_distance))

    def _fetch_nodes_batch(self, node_ids: list[str], repo_id: str) -> list[dict]:
        if not node_ids:
            return []

        cypher = """
        UNWIND $node_ids AS node_id
        MATCH (n {id: node_id, repo_id: $repo_id})
        RETURN
          n.id AS node_id,
          coalesce(n.display_name, n.name, n.path, n.id) AS name,
          coalesce(n.file_path, n.path, '') AS file_path,
          CASE
            WHEN 'Class' IN labels(n) THEN 'class'
            WHEN 'File' IN labels(n) THEN 'module'
            ELSE 'function'
          END AS node_type,
          coalesce(n.start_line, 0) AS start_line,
          coalesce(n.end_line, 0) AS end_line,
          coalesce(size((n)<-[:CALLS]-()), 0) AS in_degree,
          coalesce(size((n)-[:CALLS]->()), 0) AS out_degree,
          n.id AS chunk_id
        """
        return self._run_query(cypher, {"node_ids": node_ids, "repo_id": repo_id})

    def _fetch_callees_batch(self, node_ids: list[str], repo_id: str) -> tuple[list[dict], list[GraphEdge]]:
        if not node_ids:
            return [], []

        cypher = """
        UNWIND $node_ids AS source_id
        MATCH (source:Function {id: source_id, repo_id: $repo_id})-[:CALLS]->(target:Function)
        WHERE target.repo_id = $repo_id
        RETURN
          source.id AS source_id,
          target.id AS node_id,
          coalesce(target.display_name, target.name, target.id) AS name,
          coalesce(target.file_path, '') AS file_path,
          'function' AS node_type,
          coalesce(target.start_line, 0) AS start_line,
          coalesce(target.end_line, 0) AS end_line,
          coalesce(size((target)<-[:CALLS]-()), 0) AS in_degree,
          coalesce(size((target)-[:CALLS]->()), 0) AS out_degree,
          target.id AS chunk_id
        """
        rows = self._run_query(cypher, {"node_ids": node_ids, "repo_id": repo_id})
        edges = [
            GraphEdge(
                source=str(row.get("source_id") or ""),
                target=str(row.get("node_id") or ""),
                edge_type="CALLS",
            )
            for row in rows
            if row.get("source_id") and row.get("node_id")
        ]
        return rows, edges

    def _fetch_callers_batch(self, node_ids: list[str], repo_id: str) -> tuple[list[dict], list[GraphEdge]]:
        if not node_ids:
            return [], []

        cypher = """
        UNWIND $node_ids AS target_id
        MATCH (caller:Function)-[:CALLS]->(target:Function {id: target_id, repo_id: $repo_id})
        WHERE caller.repo_id = $repo_id
        RETURN
          target.id AS target_id,
          caller.id AS node_id,
          coalesce(caller.display_name, caller.name, caller.id) AS name,
          coalesce(caller.file_path, '') AS file_path,
          'function' AS node_type,
          coalesce(caller.start_line, 0) AS start_line,
          coalesce(caller.end_line, 0) AS end_line,
          coalesce(size((caller)<-[:CALLS]-()), 0) AS in_degree,
          coalesce(size((caller)-[:CALLS]->()), 0) AS out_degree,
          caller.id AS chunk_id
        """
        rows = self._run_query(cypher, {"node_ids": node_ids, "repo_id": repo_id})
        edges = [
            GraphEdge(
                source=str(row.get("node_id") or ""),
                target=str(row.get("target_id") or ""),
                edge_type="CALLS",
            )
            for row in rows
            if row.get("node_id") and row.get("target_id")
        ]
        return rows, edges

    def _fetch_siblings_batch(self, node_ids: list[str], repo_id: str) -> tuple[list[dict], list[GraphEdge]]:
        if not node_ids:
            return [], []

        cypher = """
        UNWIND $node_ids AS seed_id
        MATCH (seed:Function {id: seed_id, repo_id: $repo_id})<-[:CONTAINS]-(f:File)-[:CONTAINS]->(sib:Function)
        WHERE f.repo_id = $repo_id AND sib.repo_id = $repo_id AND sib.id <> seed.id
        RETURN
          seed.id AS source_id,
          sib.id AS node_id,
          coalesce(sib.display_name, sib.name, sib.id) AS name,
          coalesce(sib.file_path, '') AS file_path,
          'function' AS node_type,
          coalesce(sib.start_line, 0) AS start_line,
          coalesce(sib.end_line, 0) AS end_line,
          coalesce(size((sib)<-[:CALLS]-()), 0) AS in_degree,
          coalesce(size((sib)-[:CALLS]->()), 0) AS out_degree,
          sib.id AS chunk_id
        LIMIT 200
        """
        rows = self._run_query(cypher, {"node_ids": node_ids, "repo_id": repo_id})
        edges = [
            GraphEdge(
                source=str(row.get("source_id") or ""),
                target=str(row.get("node_id") or ""),
                edge_type="SIBLING",
            )
            for row in rows
            if row.get("source_id") and row.get("node_id")
        ]
        return rows, edges

    def _fetch_import_related_batch(self, node_ids: list[str], repo_id: str) -> tuple[list[dict], list[GraphEdge]]:
        if not node_ids:
            return [], []

        cypher = """
        UNWIND $node_ids AS seed_id
        MATCH (seed:Function {id: seed_id, repo_id: $repo_id})-[:DEFINED_IN]->(f:File {repo_id: $repo_id})-[:IMPORTS]->(dep:File {repo_id: $repo_id})-[:CONTAINS]->(fn:Function {repo_id: $repo_id})
        RETURN
          seed.id AS source_id,
          fn.id AS node_id,
          coalesce(fn.display_name, fn.name, fn.id) AS name,
          coalesce(fn.file_path, '') AS file_path,
          'function' AS node_type,
          coalesce(fn.start_line, 0) AS start_line,
          coalesce(fn.end_line, 0) AS end_line,
          coalesce(size((fn)<-[:CALLS]-()), 0) AS in_degree,
          coalesce(size((fn)-[:CALLS]->()), 0) AS out_degree,
          fn.id AS chunk_id
        LIMIT 300
        """
        rows = self._run_query(cypher, {"node_ids": node_ids, "repo_id": repo_id})
        edges = [
            GraphEdge(
                source=str(row.get("source_id") or ""),
                target=str(row.get("node_id") or ""),
                edge_type="IMPORTS",
            )
            for row in rows
            if row.get("source_id") and row.get("node_id")
        ]
        return rows, edges

    @staticmethod
    def _row_to_node(row: dict, hop_distance: int) -> GraphNode:
        return GraphNode(
            node_id=str(row.get("node_id") or ""),
            name=str(row.get("name") or row.get("node_id") or ""),
            file_path=str(row.get("file_path") or ""),
            node_type=str(row.get("node_type") or "function"),
            start_line=int(row.get("start_line", 0) or 0),
            end_line=int(row.get("end_line", 0) or 0),
            hop_distance=hop_distance,
            in_degree=int(row.get("in_degree", 0) or 0),
            out_degree=int(row.get("out_degree", 0) or 0),
            chunk_id=str(row.get("chunk_id") or row.get("node_id") or ""),
        )
