import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useChat } from "@/hooks/useChat";
import { useAsk } from "@/hooks/useAsk";
import { clearSession, getSession, saveSession } from "@/lib/session";
import type { QualityScore, SourceCitation } from "@/types/api";

vi.mock("@/hooks/useAsk", () => ({
  useAsk: vi.fn(),
}));

vi.mock("@/lib/session", () => ({
  getSession: vi.fn(),
  saveSession: vi.fn(),
  clearSession: vi.fn(),
}));

type MockAskState = {
  streamingText: string;
  sources: SourceCitation[];
  qualityScore: QualityScore | null;
  graphPath: string[];
  sessionId: string | null;
  modelUsed: string | null;
  providerUsed: string | null;
  cached: boolean;
  totalMs: number | null;
  isStreaming: boolean;
  stepMessage: string | null;
  error: string | null;
  ask: ReturnType<typeof vi.fn>;
  cancel: ReturnType<typeof vi.fn>;
  reset: ReturnType<typeof vi.fn>;
};

function makeAskState(): MockAskState {
  return {
    streamingText: "",
    sources: [],
    qualityScore: null,
    graphPath: [],
    sessionId: null,
    modelUsed: null,
    providerUsed: null,
    cached: false,
    totalMs: null,
    isStreaming: false,
    stepMessage: null,
    error: null,
    ask: vi.fn().mockResolvedValue(undefined),
    cancel: vi.fn(),
    reset: vi.fn(),
  };
}

describe("useChat", () => {
  let askState: MockAskState;

  beforeEach(() => {
    vi.clearAllMocks();

    askState = makeAskState();
    vi.mocked(useAsk).mockImplementation(() => askState as never);
    vi.mocked(getSession).mockReturnValue("session-stored");
  });

  it("loads session id from storage for repo", () => {
    const { result } = renderHook(() => useChat("repo-1"));

    expect(getSession).toHaveBeenCalledWith("repo-1");
    expect(result.current.sessionId).toBe("session-stored");
  });

  it("appends user and assistant placeholder messages on ask", async () => {
    const { result } = renderHook(() => useChat("repo-1"));

    await act(async () => {
      await result.current.ask("  Explain auth flow  ");
    });

    expect(askState.ask).toHaveBeenCalledWith({
      repo_id: "repo-1",
      question: "Explain auth flow",
      include_graph: true,
      session_id: "session-stored",
    });

    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0]).toMatchObject({ role: "user", content: "Explain auth flow" });
    expect(result.current.messages[1]).toMatchObject({ role: "assistant", isStreaming: true });
  });

  it("hydrates assistant message from streaming state and saves new session", async () => {
    const { result, rerender } = renderHook(() => useChat("repo-1"));

    await act(async () => {
      await result.current.ask("Explain login");
    });

    askState.streamingText = "The login path starts in app/main.py";
    askState.sources = [{ file: "app/main.py", function: "login", lines: "10-22" }];
    askState.qualityScore = {
      faithfulness: 0.9,
      relevance: 0.8,
      completeness: 0.85,
      overall: 0.85,
      critique: "Solid answer.",
    };
    askState.graphPath = ["api.login", "service.authenticate"];
    askState.modelUsed = "qwen";
    askState.providerUsed = "openrouter";
    askState.cached = true;
    askState.totalMs = 1200;
    askState.sessionId = "session-new";
    askState.isStreaming = false;

    act(() => {
      rerender();
    });

    const assistant = result.current.messages[1];
    expect(assistant).toMatchObject({
      role: "assistant",
      content: "The login path starts in app/main.py",
      modelUsed: "qwen",
      providerUsed: "openrouter",
      cached: true,
      totalMs: 1200,
      isStreaming: false,
    });

    expect(saveSession).toHaveBeenCalledWith("repo-1", "session-new");
    expect(result.current.sessionId).toBe("session-new");
  });

  it("surfaces streaming errors and clears them", async () => {
    const { result, rerender } = renderHook(() => useChat("repo-1"));

    await act(async () => {
      await result.current.ask("Explain login");
    });

    askState.error = "LLM provider unavailable";

    act(() => {
      rerender();
    });

    expect(result.current.error).toBe("LLM provider unavailable");

    act(() => {
      result.current.clearError();
    });

    expect(result.current.error).toBeNull();
  });

  it("cancels active stream", () => {
    const { result } = renderHook(() => useChat("repo-1"));

    act(() => {
      result.current.cancelStream();
    });

    expect(askState.cancel).toHaveBeenCalledTimes(1);
  });

  it("starts a new session by clearing state and storage", async () => {
    const { result } = renderHook(() => useChat("repo-1"));

    await act(async () => {
      await result.current.ask("Explain auth");
    });

    act(() => {
      result.current.newSession();
    });

    expect(askState.reset).toHaveBeenCalledTimes(1);
    expect(clearSession).toHaveBeenCalledWith("repo-1");
    expect(result.current.sessionId).toBeNull();
    expect(result.current.messages).toEqual([]);
    expect(result.current.error).toBeNull();
  });
});
