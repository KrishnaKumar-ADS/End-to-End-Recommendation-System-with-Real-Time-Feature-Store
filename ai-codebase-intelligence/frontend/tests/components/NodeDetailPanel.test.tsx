import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { NodeDetailPanel } from "@/components/Graph/NodeDetailPanel";
import { explainSymbol } from "@/lib/api-client";
import type { GraphEdge, GraphNode } from "@/types/api";

const { pushMock } = vi.hoisted(() => ({
  pushMock: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
  }),
}));

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ...actual,
    explainSymbol: vi.fn(),
  };
});

function buildGraphFixture(suffix: string): {
  targetNode: GraphNode;
  allNodes: GraphNode[];
  allEdges: GraphEdge[];
  callerId: string;
  calleeId: string;
} {
  const targetId = `target-${suffix}`;
  const callerId = `caller-${suffix}`;
  const calleeId = `callee-${suffix}`;

  const targetNode: GraphNode = {
    id: targetId,
    _type: "function",
    display_name: `verify_password_${suffix}`,
    file_path: `auth/service_${suffix}.py`,
    start_line: 45,
    end_line: 67,
  };

  const callerNode: GraphNode = {
    id: callerId,
    _type: "function",
    display_name: `login_handler_${suffix}`,
  };

  const calleeNode: GraphNode = {
    id: calleeId,
    _type: "function",
    display_name: `bcrypt_checkpw_${suffix}`,
  };

  const allEdges: GraphEdge[] = [
    { source: callerId, target: targetId, type: "CALLS" },
    { source: targetId, target: calleeId, type: "CALLS" },
  ];

  return {
    targetNode,
    allNodes: [targetNode, callerNode, calleeNode],
    allEdges,
    callerId,
    calleeId,
  };
}

function renderPanel(suffix: string, overrides?: Partial<ComponentProps<typeof NodeDetailPanel>>) {
  const fixture = buildGraphFixture(suffix);
  const props: ComponentProps<typeof NodeDetailPanel> = {
    repoId: "repo-1",
    node: fixture.targetNode,
    allNodes: fixture.allNodes,
    allEdges: fixture.allEdges,
    depth: 2,
    onDepthChange: vi.fn(),
    onExplore: vi.fn(),
    onSelectNode: vi.fn(),
    onClose: vi.fn(),
    ...overrides,
  };

  return {
    ...fixture,
    ...render(<NodeDetailPanel {...props} />),
    props,
  };
}

describe("NodeDetailPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(explainSymbol).mockResolvedValue({
      function_name: "verify_password",
      file_path: "auth/service.py",
      start_line: 45,
      end_line: 67,
      summary: "Validates a password hash using bcrypt.",
      parameters: [],
      returns: { type_annotation: null, description: "bool" },
      side_effects: [],
      callers: [],
      callees: [],
      complexity_score: 1,
      provider_used: "openrouter",
      model_used: "qwen",
      explanation_ms: 12,
    });
  });

  it("renders selected node info and generated description", async () => {
    const { targetNode } = renderPanel("info");

    expect(screen.getByRole("heading", { name: targetNode.display_name as string })).toBeInTheDocument();
    expect(screen.getByText(`Lines: ${targetNode.start_line}-${targetNode.end_line}`)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("Validates a password hash using bcrypt.")).toBeInTheDocument();
    });

    expect(explainSymbol).toHaveBeenCalledWith({
      repo_id: "repo-1",
      file_path: targetNode.file_path,
      function_name: targetNode.display_name,
    });
  });

  it("selects a caller node from chips", async () => {
    const user = userEvent.setup();
    const onSelectNode = vi.fn();
    const { callerId } = renderPanel("caller", { onSelectNode });

    await user.click(screen.getByRole("button", { name: "login_handler_caller" }));

    expect(onSelectNode).toHaveBeenCalledWith(callerId);
  });

  it("selects a callee node from chips", async () => {
    const user = userEvent.setup();
    const onSelectNode = vi.fn();
    const { calleeId } = renderPanel("callee", { onSelectNode });

    await user.click(screen.getByRole("button", { name: "bcrypt_checkpw_callee" }));

    expect(onSelectNode).toHaveBeenCalledWith(calleeId);
  });

  it("navigates to chat for Ask about this", async () => {
    const user = userEvent.setup();
    const { targetNode } = renderPanel("ask");

    await user.click(screen.getByRole("button", { name: "Ask about this" }));

    expect(pushMock).toHaveBeenCalledWith(
      `/repos/repo-1/chat?question=${encodeURIComponent(`Explain ${targetNode.display_name as string}`)}`,
    );
  });

  it("navigates to file view for Go to file", async () => {
    const user = userEvent.setup();
    const { targetNode } = renderPanel("file");

    await user.click(screen.getByRole("button", { name: "Go to file" }));

    expect(pushMock).toHaveBeenCalledWith(
      `/repos/repo-1?file=${encodeURIComponent(targetNode.file_path as string)}`,
    );
  });
});
