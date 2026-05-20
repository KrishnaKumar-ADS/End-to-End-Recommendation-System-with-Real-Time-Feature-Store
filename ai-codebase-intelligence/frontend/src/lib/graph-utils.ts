import type {
  GraphData,
  GraphEdge,
  GraphNeighborInfo,
  GraphNode,
  GraphNodeType,
} from "@/types/api";

export interface SubgraphExtractionResult {
  nodes: GraphNode[];
  edges: GraphEdge[];
  nodeIds: Set<string>;
  edgeIds: Set<string>;
}

export const NODE_COLORS: Record<GraphNodeType, string> = {
  function: "#3b82f6",
  class: "#8b5cf6",
  module: "#10b981",
  file: "#6b7280",
};

export const EDGE_COLORS: Record<string, string> = {
  calls: "#f8fafc",
  imports: "#60a5fa",
  inherits: "#a78bfa",
};

export const EDGE_DASH: Record<string, string | undefined> = {
  calls: undefined,
  imports: "8 4",
  inherits: "2 4",
};

function toLower(value: unknown): string {
  return String(value ?? "").trim().toLowerCase();
}

function toEdgeTypeKey(type: string): string {
  const normalized = toLower(type);
  if (normalized.includes("import")) {
    return "imports";
  }
  if (normalized.includes("inherit") || normalized.includes("implement") || normalized.includes("mixes")) {
    return "inherits";
  }
  return "calls";
}

export function edgeTypeLabel(type: string): string {
  const key = toEdgeTypeKey(type);
  if (key === "imports") {
    return "imports";
  }
  if (key === "inherits") {
    return "inherits";
  }
  return "calls";
}

export function normalizeGraphNodeType(node: GraphNode): GraphNodeType {
  const raw = toLower(node._type ?? node.label ?? "");
  if (raw.includes("class")) {
    return "class";
  }
  if (raw.includes("module")) {
    return "module";
  }
  if (raw.includes("file")) {
    return "file";
  }
  return "function";
}

export function getNodeLabel(node: GraphNode): string {
  const candidate = node.display_name ?? node.name ?? node.id;
  return String(candidate);
}

export function getNodeFilePath(node: GraphNode): string {
  const candidate = node.file_path ?? node.file ?? node.path;
  return String(candidate ?? "");
}

export function buildDegreeMap(nodes: GraphNode[], edges: GraphEdge[]): Record<string, number> {
  const degree: Record<string, number> = {};

  for (const node of nodes) {
    degree[node.id] = 0;
  }

  for (const edge of edges) {
    if (edge.source in degree) {
      degree[edge.source] += 1;
    }
    if (edge.target in degree) {
      degree[edge.target] += 1;
    }
  }

  return degree;
}

export function degreeCentrality(nodes: GraphNode[], edges: GraphEdge[]): Record<string, number> {
  const degree = buildDegreeMap(nodes, edges);
  const values = Object.values(degree);
  const maxDegree = values.length ? Math.max(...values) : 0;

  if (maxDegree <= 0) {
    const zeroed: Record<string, number> = {};
    for (const node of nodes) {
      zeroed[node.id] = 0;
    }
    return zeroed;
  }

  const centrality: Record<string, number> = {};
  for (const [id, value] of Object.entries(degree)) {
    centrality[id] = value / maxDegree;
  }
  return centrality;
}

export function nodeRadius(nodeId: string, centralityMap: Record<string, number>): number {
  const value = centralityMap[nodeId] ?? 0;
  return 6 + value * 12;
}

export function getNeighbors(nodeId: string, edges: GraphEdge[]): GraphNeighborInfo {
  const callers: string[] = [];
  const callees: string[] = [];

  for (const edge of edges) {
    if (edge.target === nodeId && !callers.includes(edge.source)) {
      callers.push(edge.source);
    }
    if (edge.source === nodeId && !callees.includes(edge.target)) {
      callees.push(edge.target);
    }
  }

  return { callers, callees };
}

export function categorizeByType(nodes: GraphNode[]): Record<GraphNodeType, GraphNode[]> {
  const grouped: Record<GraphNodeType, GraphNode[]> = {
    function: [],
    class: [],
    module: [],
    file: [],
  };

  for (const node of nodes) {
    grouped[normalizeGraphNodeType(node)].push(node);
  }

  return grouped;
}

function buildUndirectedAdjacency(edges: GraphEdge[]): Map<string, Set<string>> {
  const adjacency = new Map<string, Set<string>>();

  for (const edge of edges) {
    if (!adjacency.has(edge.source)) {
      adjacency.set(edge.source, new Set<string>());
    }
    if (!adjacency.has(edge.target)) {
      adjacency.set(edge.target, new Set<string>());
    }
    adjacency.get(edge.source)?.add(edge.target);
    adjacency.get(edge.target)?.add(edge.source);
  }

  return adjacency;
}

export function extractSubgraph(
  nodes: GraphNode[],
  edges: GraphEdge[],
  startNodeId: string,
  maxDepth: number,
): SubgraphExtractionResult {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const normalizedDepth = Math.max(1, Math.floor(maxDepth));

  if (!nodeMap.has(startNodeId)) {
    return {
      nodes: [],
      edges: [],
      nodeIds: new Set<string>(),
      edgeIds: new Set<string>(),
    };
  }

  const adjacency = buildUndirectedAdjacency(edges);
  const nodeIds = new Set<string>([startNodeId]);
  const queue: Array<{ id: string; depth: number }> = [{ id: startNodeId, depth: 0 }];

  while (queue.length) {
    const current = queue.shift();
    if (!current) {
      break;
    }
    if (current.depth >= normalizedDepth) {
      continue;
    }

    const neighbors = adjacency.get(current.id);
    if (!neighbors) {
      continue;
    }

    for (const neighborId of neighbors) {
      if (nodeIds.has(neighborId)) {
        continue;
      }
      nodeIds.add(neighborId);
      queue.push({ id: neighborId, depth: current.depth + 1 });
    }
  }

  const edgeIds = new Set<string>();
  const subgraphEdges = edges.filter((edge, index) => {
    const include = nodeIds.has(edge.source) && nodeIds.has(edge.target);
    if (include) {
      edgeIds.add(edge.id ?? `${edge.source}->${edge.target}:${edge.type}:${index}`);
    }
    return include;
  });

  const subgraphNodes = nodes.filter((node) => nodeIds.has(node.id));

  return {
    nodes: subgraphNodes,
    edges: subgraphEdges,
    nodeIds,
    edgeIds,
  };
}

export function getIsolatedNodeIds(nodes: GraphNode[], edges: GraphEdge[]): Set<string> {
  const degree = buildDegreeMap(nodes, edges);
  const isolated = new Set<string>();

  for (const node of nodes) {
    if ((degree[node.id] ?? 0) === 0) {
      isolated.add(node.id);
    }
  }

  return isolated;
}

export function countConnectedComponents(nodes: GraphNode[], edges: GraphEdge[]): number {
  if (!nodes.length) {
    return 0;
  }

  const adjacency = buildUndirectedAdjacency(edges);
  const unvisited = new Set(nodes.map((node) => node.id));
  let components = 0;

  while (unvisited.size) {
    const start = unvisited.values().next().value as string;
    const stack = [start];
    components += 1;

    while (stack.length) {
      const current = stack.pop();
      if (!current || !unvisited.has(current)) {
        continue;
      }
      unvisited.delete(current);
      const neighbors = adjacency.get(current);
      if (!neighbors) {
        continue;
      }
      for (const neighbor of neighbors) {
        if (unvisited.has(neighbor)) {
          stack.push(neighbor);
        }
      }
    }
  }

  return components;
}

export function findEntryPoints(nodes: GraphNode[], edges: GraphEdge[]): string[] {
  const inDegree: Record<string, number> = {};
  const outDegree: Record<string, number> = {};

  for (const node of nodes) {
    inDegree[node.id] = 0;
    outDegree[node.id] = 0;
  }

  for (const edge of edges) {
    if (edge.source in outDegree) {
      outDegree[edge.source] += 1;
    }
    if (edge.target in inDegree) {
      inDegree[edge.target] += 1;
    }
  }

  return nodes
    .filter((node) => inDegree[node.id] === 0 && outDegree[node.id] > 0)
    .map((node) => node.id);
}

export function findMostConnected(nodes: GraphNode[], edges: GraphEdge[]): GraphNode | null {
  const degree = buildDegreeMap(nodes, edges);
  let best: GraphNode | null = null;
  let bestDegree = -1;

  for (const node of nodes) {
    const nodeDegree = degree[node.id] ?? 0;
    if (nodeDegree > bestDegree) {
      best = node;
      bestDegree = nodeDegree;
    }
  }

  return best;
}

export function mergeGraphData(base: GraphData, override: GraphData | null): GraphData {
  if (!override) {
    return base;
  }

  return {
    ...base,
    nodes: override.nodes,
    edges: override.edges,
    node_count: override.node_count,
    edge_count: override.edge_count,
    centre_node_id: override.centre_node_id,
    depth: override.depth,
  };
}
