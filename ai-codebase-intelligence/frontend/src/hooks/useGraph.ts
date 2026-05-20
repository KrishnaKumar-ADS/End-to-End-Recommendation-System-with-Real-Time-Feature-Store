"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { fetchGraph, fetchGraphSubgraph } from "@/lib/api-client";
import {
  categorizeByType,
  countConnectedComponents,
  degreeCentrality,
  extractSubgraph,
  findEntryPoints,
  findMostConnected,
  getIsolatedNodeIds,
  getNodeFilePath,
  getNodeLabel,
  mergeGraphData,
  normalizeGraphNodeType,
} from "@/lib/graph-utils";
import type {
  GraphData,
  GraphFilterState,
  GraphNode,
  GraphNodeType,
} from "@/types/api";

const DEFAULT_FILTERS: GraphFilterState = {
  nodeTypes: {
    function: true,
    class: true,
    module: true,
    file: true,
  },
  search: "",
  showIsolated: true,
  minDegree: 0,
  depth: 2,
};

type SubgraphMode = "none" | "client" | "server";

export function useGraph(repoId: string | null, limit = 1200) {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [serverSubgraph, setServerSubgraph] = useState<GraphData | null>(null);
  const [isLoading, setIsLoading] = useState(Boolean(repoId));
  const [isSubgraphLoading, setIsSubgraphLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [filters, setFilters] = useState<GraphFilterState>(DEFAULT_FILTERS);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [subgraphStartNodeId, setSubgraphStartNodeId] = useState<string | null>(null);
  const [subgraphMode, setSubgraphMode] = useState<SubgraphMode>("none");

  const refresh = useCallback(async () => {
    if (!repoId) {
      setGraph(null);
      setServerSubgraph(null);
      setIsLoading(false);
      return null;
    }

    setIsLoading(true);
    try {
      const response = await fetchGraph(repoId, { limit });
      setGraph(response);
      setServerSubgraph(null);
      setSubgraphStartNodeId(null);
      setSubgraphMode("none");
      setError(null);
      return response;
    } catch (err) {
      setGraph(null);
      setError(err instanceof Error ? err.message : "Failed to load graph.");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [limit, repoId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const baseGraph = useMemo(() => mergeGraphData(graph ?? {
    repo_id: repoId ?? "",
    nodes: [],
    edges: [],
    node_count: 0,
    edge_count: 0,
  }, serverSubgraph), [graph, repoId, serverSubgraph]);

  const subgraphSlice = useMemo(() => {
    if (!graph || subgraphMode !== "client" || !subgraphStartNodeId) {
      return null;
    }
    return extractSubgraph(graph.nodes, graph.edges, subgraphStartNodeId, filters.depth);
  }, [filters.depth, graph, subgraphMode, subgraphStartNodeId]);

  const scopedNodes = subgraphSlice?.nodes ?? baseGraph.nodes;
  const scopedEdges = subgraphSlice?.edges ?? baseGraph.edges;

  const typeCounts = useMemo(() => {
    const grouped = categorizeByType(baseGraph.nodes);
    return {
      function: grouped.function.length,
      class: grouped.class.length,
      module: grouped.module.length,
      file: grouped.file.length,
    };
  }, [baseGraph.nodes]);

  const scopedDegree = useMemo(() => {
    const centrality = degreeCentrality(scopedNodes, scopedEdges);
    const map: Record<string, number> = {};
    for (const [id, value] of Object.entries(centrality)) {
      map[id] = Math.round(value * 1000) / 1000;
    }
    return map;
  }, [scopedEdges, scopedNodes]);

  const filtered = useMemo(() => {
    let candidateNodes = scopedNodes.filter((node) => filters.nodeTypes[normalizeGraphNodeType(node)]);
    let candidateNodeIds = new Set(candidateNodes.map((node) => node.id));

    let candidateEdges = scopedEdges.filter(
      (edge) => candidateNodeIds.has(edge.source) && candidateNodeIds.has(edge.target),
    );

    const degreeMap = degreeCentrality(candidateNodes, candidateEdges);

    if (!filters.showIsolated) {
      const isolated = getIsolatedNodeIds(candidateNodes, candidateEdges);
      candidateNodes = candidateNodes.filter((node) => !isolated.has(node.id));
      candidateNodeIds = new Set(candidateNodes.map((node) => node.id));
      candidateEdges = candidateEdges.filter(
        (edge) => candidateNodeIds.has(edge.source) && candidateNodeIds.has(edge.target),
      );
    }

    if (filters.minDegree > 0) {
      candidateNodes = candidateNodes.filter((node) => {
        const normalizedDegree = degreeMap[node.id] ?? 0;
        return normalizedDegree * 10 >= filters.minDegree;
      });
      candidateNodeIds = new Set(candidateNodes.map((node) => node.id));
      candidateEdges = candidateEdges.filter(
        (edge) => candidateNodeIds.has(edge.source) && candidateNodeIds.has(edge.target),
      );
    }

    const searchTerm = filters.search.trim().toLowerCase();
    const searchMatches = new Set<string>();
    if (searchTerm) {
      for (const node of candidateNodes) {
        const haystack = `${getNodeLabel(node)} ${getNodeFilePath(node)} ${node.id}`.toLowerCase();
        if (haystack.includes(searchTerm)) {
          searchMatches.add(node.id);
        }
      }
    }

    return {
      nodes: candidateNodes,
      edges: candidateEdges,
      searchMatches,
    };
  }, [filters.minDegree, filters.nodeTypes, filters.search, filters.showIsolated, scopedEdges, scopedNodes]);

  const selectedNode = useMemo(
    () => filtered.nodes.find((node) => node.id === selectedNodeId) ?? null,
    [filtered.nodes, selectedNodeId],
  );

  const mostConnectedNode = useMemo(
    () => findMostConnected(filtered.nodes, filtered.edges),
    [filtered.edges, filtered.nodes],
  );

  const graphStats = useMemo(() => {
    const components = countConnectedComponents(filtered.nodes, filtered.edges);
    const entryPointCount = findEntryPoints(filtered.nodes, filtered.edges).length;

    return {
      totalNodes: baseGraph.nodes.length,
      totalEdges: baseGraph.edges.length,
      visibleNodes: filtered.nodes.length,
      visibleEdges: filtered.edges.length,
      connectedComponents: components,
      entryPointCount,
      mostConnectedNode,
    };
  }, [baseGraph.edges.length, baseGraph.nodes.length, filtered.edges, filtered.nodes, mostConnectedNode]);

  const toggleNodeType = useCallback((type: GraphNodeType, checked: boolean) => {
    setFilters((previous) => ({
      ...previous,
      nodeTypes: {
        ...previous.nodeTypes,
        [type]: checked,
      },
    }));
  }, []);

  const setSearch = useCallback((search: string) => {
    setFilters((previous) => ({ ...previous, search }));
  }, []);

  const setShowIsolated = useCallback((showIsolated: boolean) => {
    setFilters((previous) => ({ ...previous, showIsolated }));
  }, []);

  const setMinDegree = useCallback((minDegree: number) => {
    setFilters((previous) => ({ ...previous, minDegree }));
  }, []);

  const setDepth = useCallback((depth: number) => {
    setFilters((previous) => ({ ...previous, depth }));
  }, []);

  const resetFilters = useCallback(() => {
    setFilters(DEFAULT_FILTERS);
  }, []);

  const selectNode = useCallback((nodeId: string | null) => {
    setSelectedNodeId(nodeId);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedNodeId(null);
  }, []);

  const exploreSubgraph = useCallback((nodeId: string, depth?: number) => {
    if (depth) {
      setDepth(depth);
    }
    setSubgraphStartNodeId(nodeId);
    setSubgraphMode("client");
    setServerSubgraph(null);
  }, [setDepth]);

  const exploreSubgraphFromServer = useCallback(async (nodeId: string, depth?: number) => {
    if (!repoId) {
      return null;
    }

    const resolvedDepth = depth ?? filters.depth;
    setIsSubgraphLoading(true);
    try {
      const response = await fetchGraphSubgraph(repoId, nodeId, resolvedDepth);
      setServerSubgraph(response);
      setSubgraphStartNodeId(nodeId);
      setSubgraphMode("server");
      setError(null);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load subgraph.");
      return null;
    } finally {
      setIsSubgraphLoading(false);
    }
  }, [filters.depth, repoId]);

  const clearSubgraph = useCallback(() => {
    setServerSubgraph(null);
    setSubgraphStartNodeId(null);
    setSubgraphMode("none");
  }, []);

  return {
    graph,
    isLoading,
    isSubgraphLoading,
    error,
    refresh,

    nodes: filtered.nodes,
    edges: filtered.edges,
    searchMatches: filtered.searchMatches,
    degreeCentrality: scopedDegree,

    filters,
    typeCounts,
    toggleNodeType,
    setSearch,
    setShowIsolated,
    setMinDegree,
    setDepth,
    resetFilters,

    selectedNodeId,
    selectedNode,
    selectNode,
    clearSelection,

    subgraphStartNodeId,
    subgraphMode,
    exploreSubgraph,
    exploreSubgraphFromServer,
    clearSubgraph,

    graphStats,
  };
}
