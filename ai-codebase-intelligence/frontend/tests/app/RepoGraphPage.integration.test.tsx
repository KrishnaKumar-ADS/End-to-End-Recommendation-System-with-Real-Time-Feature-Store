import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { forwardRef } from "react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import RepoGraphPage from "@/app/repos/[repoId]/graph/page";
import { useGraph } from "@/hooks/useGraph";
import { useRepo } from "@/hooks/useRepo";

const searchParamGet = vi.fn();

vi.mock("next/navigation", () => ({
  useSearchParams: () => ({
    get: searchParamGet,
  }),
}));

vi.mock("@/hooks/useGraph", () => ({
  useGraph: vi.fn(),
}));

vi.mock("@/hooks/useRepo", () => ({
  useRepo: vi.fn(),
}));

vi.mock("@/components/Graph/ForceGraph", () => ({
  ForceGraph: forwardRef(function MockForceGraph(
    { onNodeClick }: { onNodeClick?: (nodeId: string | null) => void },
    _ref,
  ) {
    return (
      <div data-testid="force-graph">
        <button onClick={() => onNodeClick?.("node-b")} type="button">
          Select Node B
        </button>
      </div>
    );
  }),
}));

vi.mock("@/components/Graph/GraphControls", () => ({
  GraphControls: () => <div data-testid="graph-controls">Graph Controls</div>,
}));

vi.mock("@/components/Graph/GraphStats", () => ({
  GraphStats: ({ visibleNodes, visibleEdges }: { visibleNodes: number; visibleEdges: number }) => (
    <div data-testid="graph-stats">{visibleNodes} / {visibleEdges}</div>
  ),
}));

vi.mock("@/components/Graph/GraphLegend", () => ({
  GraphLegend: () => <div data-testid="graph-legend">Legend</div>,
}));

vi.mock("@/components/Graph/GraphTooltip", () => ({
  GraphTooltip: ({ node }: { node: { id: string } | null }) => (
    <div data-testid="graph-tooltip">{node?.id ?? "none"}</div>
  ),
}));

vi.mock("@/components/Graph/GraphFilterPanel", () => ({
  GraphFilterPanel: ({ onSearchChange }: { onSearchChange: (value: string) => void }) => (
    <div data-testid="graph-filter-panel">
      <button onClick={() => onSearchChange("auth")} type="button">
        Trigger Search
      </button>
    </div>
  ),
}));

vi.mock("@/components/Graph/NodeDetailPanel", () => ({
  NodeDetailPanel: ({ node, onClose }: { node: { id: string } | null; onClose: () => void }) => (
    <div data-testid="node-detail-panel">
      <span>{node?.id ?? "empty"}</span>
      <button onClick={onClose} type="button">
        Close Node Panel
      </button>
    </div>
  ),
}));

beforeAll(() => {
  class ResizeObserverMock {
    callback: ResizeObserverCallback;

    constructor(callback: ResizeObserverCallback) {
      this.callback = callback;
    }

    observe() {
      this.callback(
        [{ contentRect: { width: 900, height: 600 } } as ResizeObserverEntry],
        this as unknown as ResizeObserver,
      );
    }

    unobserve() {
      return undefined;
    }

    disconnect() {
      return undefined;
    }
  }

  vi.stubGlobal("ResizeObserver", ResizeObserverMock);
});

describe("RepoGraphPage route orchestration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    searchParamGet.mockReturnValue(null);

    vi.mocked(useRepo).mockReturnValue({
      repo: {
        id: "repo-1",
        github_url: "https://github.com/example/repo-one",
      },
      isLoading: false,
      error: null,
      refresh: vi.fn(),
    } as never);
  });

  it("shows the graph initialization skeleton while loading", () => {
    vi.mocked(useGraph).mockReturnValue({
      graph: null,
      nodes: [],
      edges: [],
      isLoading: true,
      isSubgraphLoading: false,
      error: null,
      refresh: vi.fn(),
      filters: {
        nodeTypes: { function: true, class: true, module: true, file: true },
        search: "",
        showIsolated: true,
        minDegree: 0,
        depth: 2,
      },
      typeCounts: { function: 0, class: 0, module: 0, file: 0 },
      toggleNodeType: vi.fn(),
      setSearch: vi.fn(),
      setShowIsolated: vi.fn(),
      setMinDegree: vi.fn(),
      setDepth: vi.fn(),
      resetFilters: vi.fn(),
      selectedNode: null,
      selectedNodeId: null,
      selectNode: vi.fn(),
      clearSelection: vi.fn(),
      subgraphMode: "none",
      exploreSubgraph: vi.fn(),
      clearSubgraph: vi.fn(),
      graphStats: {
        totalNodes: 0,
        totalEdges: 0,
        visibleNodes: 0,
        visibleEdges: 0,
        connectedComponents: 0,
        entryPointCount: 0,
        mostConnectedNode: null,
      },
      searchMatches: new Set<string>(),
      degreeCentrality: {},
    } as never);

    render(<RepoGraphPage params={{ repoId: "repo-1" }} />);

    expect(screen.getByText("Laying out graph...")).toBeInTheDocument();
  });

  it("wires hook state with graph interactions and panels", async () => {
    const refresh = vi.fn();
    const selectNode = vi.fn();
    const clearSelection = vi.fn();
    const clearSubgraph = vi.fn();
    const setSearch = vi.fn();

    searchParamGet.mockImplementation((key: string) => (key === "file" ? "app/auth.py" : null));

    vi.mocked(useGraph).mockReturnValue({
      graph: {
        repo_id: "repo-1",
        nodes: [
          { id: "node-a", _type: "function", display_name: "auth_login", file_path: "app/auth.py" },
          { id: "node-b", _type: "class", display_name: "AuthService", file_path: "app/auth.py" },
        ],
        edges: [{ source: "node-a", target: "node-b", type: "CALLS" }],
        node_count: 2,
        edge_count: 1,
      },
      nodes: [
        { id: "node-a", _type: "function", display_name: "auth_login", file_path: "app/auth.py" },
        { id: "node-b", _type: "class", display_name: "AuthService", file_path: "app/auth.py" },
      ],
      edges: [{ source: "node-a", target: "node-b", type: "CALLS" }],
      isLoading: false,
      isSubgraphLoading: false,
      error: "temporary graph error",
      refresh,
      filters: {
        nodeTypes: { function: true, class: true, module: true, file: true },
        search: "",
        showIsolated: true,
        minDegree: 0,
        depth: 2,
      },
      typeCounts: { function: 1, class: 1, module: 0, file: 0 },
      toggleNodeType: vi.fn(),
      setSearch,
      setShowIsolated: vi.fn(),
      setMinDegree: vi.fn(),
      setDepth: vi.fn(),
      resetFilters: vi.fn(),
      selectedNode: { id: "node-a", _type: "function", display_name: "auth_login", file_path: "app/auth.py" },
      selectedNodeId: null,
      selectNode,
      clearSelection,
      subgraphMode: "client",
      exploreSubgraph: vi.fn(),
      clearSubgraph,
      graphStats: {
        totalNodes: 2,
        totalEdges: 1,
        visibleNodes: 2,
        visibleEdges: 1,
        connectedComponents: 1,
        entryPointCount: 1,
        mostConnectedNode: { id: "node-a", _type: "function", display_name: "auth_login" },
      },
      searchMatches: new Set<string>(["node-a"]),
      degreeCentrality: { "node-a": 1, "node-b": 0.5 },
    } as never);

    const user = userEvent.setup();

    render(<RepoGraphPage params={{ repoId: "repo-1" }} />);

    await waitFor(() => {
      expect(selectNode).toHaveBeenCalledWith("node-a");
    });

    await user.click(screen.getByRole("button", { name: "Retry graph load" }));
    expect(refresh).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Clear subgraph" }));
    expect(clearSubgraph).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Select Node B" }));
    expect(selectNode).toHaveBeenCalledWith("node-b");

    await user.click(screen.getAllByRole("button", { name: "Close Node Panel" })[0]);
    expect(clearSelection).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Filters" }));
    await user.click(screen.getAllByRole("button", { name: "Trigger Search" })[0]);
    expect(setSearch).toHaveBeenCalledWith("auth");
  });
});
