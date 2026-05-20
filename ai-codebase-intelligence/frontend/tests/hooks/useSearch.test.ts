import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { searchCode } from "@/lib/api-client";
import { useSearch } from "@/hooks/useSearch";

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ...actual,
    searchCode: vi.fn(),
  };
});

describe("useSearch", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  async function flushMicrotasks() {
    await act(async () => {
      await Promise.resolve();
    });
  }

  it("does not search when repoId is missing", async () => {
    const { result } = renderHook(() => useSearch(null));

    act(() => {
      result.current.setQuery("auth");
      vi.advanceTimersByTime(400);
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(searchCode).not.toHaveBeenCalled();
    expect(result.current.results).toEqual([]);
  });

  it("debounces search by 300ms and trims query", async () => {
    vi.mocked(searchCode).mockResolvedValueOnce({
      query: "auth",
      expanded_queries: [],
      repo_id: "repo-1",
      mode: "hybrid",
      reranked: false,
      results: [
        {
          id: "chunk-1",
          name: "login",
          file_path: "app/auth.py",
          chunk_type: "function",
          start_line: 1,
          end_line: 20,
          content: "def login(): ...",
          docstring: null,
          language: "python",
          parent_name: null,
          vector_score: 0.9,
          bm25_score: 0.7,
          hybrid_score: 0.85,
          rerank_score: null,
          vector_rank: 1,
          bm25_rank: 2,
          hybrid_rank: 1,
          final_rank: 1,
        },
      ],
      total_results: 1,
      timing: {
        embed_ms: 1,
        expand_ms: 1,
        vector_ms: 1,
        bm25_ms: 1,
        fusion_ms: 1,
        rerank_ms: 1,
        total_ms: 6,
      },
    });

    const { result } = renderHook(() => useSearch("repo-1"));

    act(() => {
      result.current.setQuery("  auth  ");
    });

    act(() => {
      vi.advanceTimersByTime(299);
    });

    expect(searchCode).not.toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(1);
      await Promise.resolve();
    });

    await flushMicrotasks();

    expect(searchCode).toHaveBeenCalledWith({ repo_id: "repo-1", q: "auth", top_k: 8 });
    expect(result.current.results).toHaveLength(1);
  });

  it("captures errors from search API", async () => {
    vi.mocked(searchCode).mockRejectedValueOnce(new Error("Search unavailable"));

    const { result } = renderHook(() => useSearch("repo-1"));

    act(() => {
      result.current.setQuery("security");
    });

    await act(async () => {
      vi.advanceTimersByTime(300);
      await Promise.resolve();
    });

    await flushMicrotasks();

    expect(result.current.error).toBe("Search unavailable");
    expect(result.current.results).toEqual([]);
  });
});
