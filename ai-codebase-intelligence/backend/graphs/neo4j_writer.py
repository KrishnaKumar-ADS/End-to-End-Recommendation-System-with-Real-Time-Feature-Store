"""Compatibility Neo4j writer module for legacy graphs.* imports."""

from __future__ import annotations

from graphs.graph_builder import GraphEdge, GraphNode, RepoGraph
from graphs.neo4j_client import get_session, run_query
from graphs.schema import NodeLabel, RelType
from core.logging import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 500


def _write_nodes_batch(nodes: list[GraphNode], label: str) -> int:
	if not nodes:
		return 0
	props_list = [{"id": n.node_id, **n.properties} for n in nodes]
	cypher = f"""
	UNWIND $nodes AS props
	MERGE (n:{label} {{id: props.id}})
	SET n += props
	"""
	written = 0
	with get_session() as session:
		for i in range(0, len(props_list), BATCH_SIZE):
			batch = props_list[i : i + BATCH_SIZE]
			session.run(cypher, {"nodes": batch})
			written += len(batch)
	return written


def write_repo_node(node: GraphNode) -> None:
	with get_session() as session:
		session.run(
			f"MERGE (r:{NodeLabel.REPO} {{id: $id}}) SET r += $props",
			{"id": node.node_id, "props": node.properties},
		)


def write_file_nodes(nodes: list[GraphNode]) -> int:
	return _write_nodes_batch(nodes, NodeLabel.FILE)


def write_function_nodes(nodes: list[GraphNode]) -> int:
	return _write_nodes_batch(nodes, NodeLabel.FUNCTION)


def write_class_nodes(nodes: list[GraphNode]) -> int:
	return _write_nodes_batch(nodes, NodeLabel.CLASS)


def _write_edges_by_type(edges: list[GraphEdge], rel_type: str) -> int:
	if not edges:
		return 0
	edge_list = [
		{
			"source_id": e.source_id,
			"target_id": e.target_id,
			"props": e.properties,
		}
		for e in edges
	]

	cypher = f"""
	UNWIND $edges AS edge
	MATCH (source {{id: edge.source_id}})
	MATCH (target {{id: edge.target_id}})
	MERGE (source)-[r:{rel_type}]->(target)
	SET r += edge.props
	"""

	written = 0
	with get_session() as session:
		for i in range(0, len(edge_list), BATCH_SIZE):
			batch = edge_list[i : i + BATCH_SIZE]
			session.run(cypher, {"edges": batch})
			written += len(batch)
	return written


def write_contains_file_edges(edges: list[GraphEdge]) -> int:
	return _write_edges_by_type(edges, RelType.CONTAINS_FILE)


def write_contains_edges(edges: list[GraphEdge]) -> int:
	return _write_edges_by_type(edges, RelType.CONTAINS)


def write_contains_method_edges(edges: list[GraphEdge]) -> int:
	return _write_edges_by_type(edges, RelType.CONTAINS_METHOD)


def write_defined_in_edges(edges: list[GraphEdge]) -> int:
	return _write_edges_by_type(edges, RelType.DEFINED_IN)


def write_calls_edges(edges: list[GraphEdge]) -> int:
	return _write_edges_by_type(edges, RelType.CALLS)


def write_imports_edges(edges: list[GraphEdge]) -> int:
	return _write_edges_by_type(edges, RelType.IMPORTS)


def write_graph_to_neo4j(graph: RepoGraph) -> dict:
	logger.info("neo4j_write_start", **graph.summary())

	repo_nodes = [n for n in graph.nodes if n.label == NodeLabel.REPO]
	for rn in repo_nodes:
		write_repo_node(rn)

	file_count = write_file_nodes(graph.file_nodes)
	func_count = write_function_nodes(graph.function_nodes)
	class_count = write_class_nodes(graph.class_nodes)
	total_nodes = len(repo_nodes) + file_count + func_count + class_count

	write_contains_file_edges([e for e in graph.edges if e.rel_type == RelType.CONTAINS_FILE])
	write_contains_edges([e for e in graph.edges if e.rel_type == RelType.CONTAINS])
	write_contains_method_edges([e for e in graph.edges if e.rel_type == RelType.CONTAINS_METHOD])
	write_defined_in_edges([e for e in graph.edges if e.rel_type == RelType.DEFINED_IN])

	calls_count = write_calls_edges(graph.call_edges)
	imports_count = write_imports_edges(graph.import_edges)

	return {
		"repo_id": graph.repo_id,
		"nodes_written": total_nodes,
		"file_nodes": file_count,
		"function_nodes": func_count,
		"class_nodes": class_count,
		"edges_written": len(graph.edges),
		"calls_edges": calls_count,
		"imports_edges": imports_count,
	}


def delete_repo_graph(repo_id: str) -> int:
	deleted = 0
	with get_session() as session:
		while True:
			result = session.run(
				"""
				MATCH (n {repo_id: $repo_id})
				WITH n LIMIT 1000
				DETACH DELETE n
				RETURN COUNT(n) AS deleted
				""",
				{"repo_id": repo_id},
			)
			batch_deleted = result.single()["deleted"]
			deleted += batch_deleted
			if batch_deleted == 0:
				break

	logger.info("neo4j_repo_deleted", repo_id=repo_id, nodes_deleted=deleted)
	return deleted


def get_repo_graph_stats(repo_id: str) -> dict:
	node_results = run_query(
		"""
		MATCH (n {repo_id: $repo_id})
		RETURN labels(n)[0] AS label, COUNT(n) AS count
		""",
		{"repo_id": repo_id},
	)
	edge_results = run_query(
		"""
		MATCH (a {repo_id: $repo_id})-[r]->(b {repo_id: $repo_id})
		RETURN type(r) AS rel_type, COUNT(r) AS count
		""",
		{"repo_id": repo_id},
	)
	return {
		"repo_id": repo_id,
		"nodes": {r["label"]: r["count"] for r in node_results if r.get("label")},
		"relationships": {r["rel_type"]: r["count"] for r in edge_results},
	}
