import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useGraph } from "@/hooks/useGraph";
import { fetchGraph, fetchGraphSubgraph } from "@/lib/api-client";
import type { GraphData } from "@/types/api";

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ...actual,
    fetchGraph: vi.fn(),
    fetchGraphSubgraph: vi.fn(),
  };
});

const fullGraph: GraphData = {
  repo_id: "repo-1",
  nodes: [
    { id: "f-auth", _type: "function", display_name: "auth_login", file_path: "app/auth.py" },
    { id: "c-auth", _type: "class", display_name: "AuthService", file_path: "app/auth.py" },
    { id: "m-auth", _type: "module", display_name: "auth", file_path: "app/auth.py" },
    { id: "file-auth", _type: "file", display_name: "auth.py", file_path: "app/auth.py" },
    { id: "isolated", _type: "function", display_name: "lonely", file_path: "app/lonely.py" },
  ],
  edges: [
    { source: "f-auth", target: "c-auth", type: "CALLS" },
    { source: "c-auth", target: "m-auth", type: "IMPORTS" },
    { source: "m-auth", target: "file-auth", type: "INHERITS" },
  ],
  node_count: 5,
  edge_count: 3,
};

const serverSubgraph: GraphData = {
  repo_id: "repo-1",
  nodes: [
    { id: "f-auth", _type: "function", display_name: "auth_login", file_path: "app/auth.py" },
    { id: "c-auth", _type: "class", display_name: "AuthService", file_path: "app/auth.py" },
  ],
  edges: [{ source: "f-auth", target: "c-auth", type: "CALLS" }],
  node_count: 2,
  edge_count: 1,
  centre_node_id: "f-auth",
  depth: 1,
};

describe("useGraph", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchGraph).mockResolvedValue(fullGraph);
  });

  it("loads graph data successfully", async () => {
    const { result } = renderHook(() => useGraph("repo-1"));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.graph?.repo_id).toBe("repo-1");
    });

    expect(fetchGraph).toHaveBeenCalledWith("repo-1", { limit: 1200 });
    expect(result.current.nodes).toHaveLength(5);
    expect(result.current.error).toBeNull();
  });

  it("captures fetch errors", async () => {
    vi.mocked(fetchGraph).mockRejectedValueOnce(new Error("Graph unavailable"));

    const { result } = renderHook(() => useGraph("repo-1"));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.graph).toBeNull();
      expect(result.current.error).toBe("Graph unavailable");
    });

    expect(result.current.nodes).toEqual([]);
    expect(result.current.edges).toEqual([]);
  });

  it("applies node type filtering", async () => {
    const { result } = renderHook(() => useGraph("repo-1"));

    await waitFor(() => expect(result.current.nodes).toHaveLength(5));

    act(() => {
      result.current.toggleNodeType("function", false);
    });

    expect(result.current.nodes.map((item) => item.id)).toEqual(["c-auth", "m-auth", "file-auth"]);
  });

  it("tracks name search matches", async () => {
    const { result } = renderHook(() => useGraph("repo-1"));

    await waitFor(() => expect(result.current.nodes).toHaveLength(5));

    act(() => {
      result.current.setSearch("auth_login");
    });

    expect(Array.from(result.current.searchMatches)).toEqual(["f-auth"]);
  });

  it("enters client subgraph mode", async () => {
    const { result } = renderHook(() => useGraph("repo-1"));

    await waitFor(() => expect(result.current.nodes).toHaveLength(5));

    act(() => {
      result.current.exploreSubgraph("c-auth", 1);
    });

    expect(result.current.subgraphMode).toBe("client");
    expect(result.current.subgraphStartNodeId).toBe("c-auth");
    expect(result.current.nodes.map((item) => item.id).sort()).toEqual(["c-auth", "f-auth", "m-auth"]);
  });

  it("loads and clears server subgraph mode", async () => {
    vi.mocked(fetchGraphSubgraph).mockResolvedValueOnce(serverSubgraph);

    const { result } = renderHook(() => useGraph("repo-1"));

    await waitFor(() => expect(result.current.nodes).toHaveLength(5));

    await act(async () => {
      await result.current.exploreSubgraphFromServer("f-auth", 1);
    });

    await waitFor(() => {
      expect(result.current.subgraphMode).toBe("server");
      expect(result.current.nodes).toHaveLength(2);
    });

    expect(fetchGraphSubgraph).toHaveBeenCalledWith("repo-1", "f-auth", 1);

    act(() => {
      result.current.clearSubgraph();
    });

    expect(result.current.subgraphMode).toBe("none");
    expect(result.current.nodes).toHaveLength(5);
  });
});
