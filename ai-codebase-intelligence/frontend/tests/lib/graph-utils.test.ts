import { describe, expect, it } from "vitest";

import {
  categorizeByType,
  countConnectedComponents,
  degreeCentrality,
  edgeTypeLabel,
  extractSubgraph,
  getIsolatedNodeIds,
  getNeighbors,
} from "@/lib/graph-utils";
import type { GraphEdge, GraphNode } from "@/types/api";

function node(id: string, type: string, extra: Partial<GraphNode> = {}): GraphNode {
  return {
    id,
    _type: type,
    ...extra,
  };
}

describe("graph-utils", () => {
  const nodes: GraphNode[] = [
    node("func-a", "function", { display_name: "func_a" }),
    node("class-b", "class", { display_name: "ClassB" }),
    node("module-c", "module", { display_name: "module_c" }),
    node("file-d", "file", { display_name: "file_d.py" }),
    node("isolated-e", "function", { display_name: "isolated_e" }),
  ];

  const edges: GraphEdge[] = [
    { source: "func-a", target: "class-b", type: "CALLS" },
    { source: "class-b", target: "module-c", type: "IMPORTS" },
    { source: "module-c", target: "file-d", type: "INHERITS" },
  ];

  it("extracts a BFS subgraph at depth 1", () => {
    const result = extractSubgraph(nodes, edges, "class-b", 1);

    expect(result.nodes.map((item) => item.id).sort()).toEqual(["class-b", "func-a", "module-c"]);
    expect(result.edges).toHaveLength(2);
  });

  it("extracts a BFS subgraph at depth 2", () => {
    const result = extractSubgraph(nodes, edges, "class-b", 2);

    expect(result.nodes.map((item) => item.id).sort()).toEqual(["class-b", "file-d", "func-a", "module-c"]);
    expect(result.edges).toHaveLength(3);
  });

  it("returns an empty subgraph for unknown or empty graph starts", () => {
    const missingStart = extractSubgraph(nodes, edges, "missing-node", 2);
    expect(missingStart.nodes).toEqual([]);
    expect(missingStart.edges).toEqual([]);

    const emptyGraph = extractSubgraph([], [], "anything", 3);
    expect(emptyGraph.nodes).toEqual([]);
    expect(emptyGraph.edges).toEqual([]);
  });

  it("computes normalized degree centrality", () => {
    const localNodes = [node("a", "function"), node("b", "function"), node("c", "function")];
    const localEdges: GraphEdge[] = [
      { source: "a", target: "b", type: "CALLS" },
      { source: "a", target: "c", type: "CALLS" },
    ];

    const centrality = degreeCentrality(localNodes, localEdges);

    expect(centrality.a).toBe(1);
    expect(centrality.b).toBe(0.5);
    expect(centrality.c).toBe(0.5);
  });

  it("returns zero centrality for nodes without edges", () => {
    const localNodes = [node("a", "function"), node("b", "function")];

    const centrality = degreeCentrality(localNodes, []);

    expect(centrality).toEqual({ a: 0, b: 0 });
  });

  it("returns deduplicated callers and callees", () => {
    const localEdges: GraphEdge[] = [
      { source: "caller", target: "target", type: "CALLS" },
      { source: "caller", target: "target", type: "CALLS" },
      { source: "target", target: "callee", type: "CALLS" },
      { source: "target", target: "callee", type: "CALLS" },
    ];

    const neighbors = getNeighbors("target", localEdges);

    expect(neighbors.callers).toEqual(["caller"]);
    expect(neighbors.callees).toEqual(["callee"]);
  });

  it("detects isolated nodes", () => {
    const isolated = getIsolatedNodeIds(nodes, edges);

    expect(isolated.has("isolated-e")).toBe(true);
    expect(isolated.has("func-a")).toBe(false);
  });

  it("counts connected components", () => {
    const componentCount = countConnectedComponents(nodes, edges);

    expect(componentCount).toBe(2);
  });

  it("categorizes nodes by type", () => {
    const grouped = categorizeByType(nodes);

    expect(grouped.function).toHaveLength(2);
    expect(grouped.class).toHaveLength(1);
    expect(grouped.module).toHaveLength(1);
    expect(grouped.file).toHaveLength(1);
  });

  it("maps edge labels for legend display", () => {
    expect(edgeTypeLabel("IMPORTS")).toBe("imports");
    expect(edgeTypeLabel("IMPLEMENTS")).toBe("inherits");
    expect(edgeTypeLabel("CALLS")).toBe("calls");
  });
});
