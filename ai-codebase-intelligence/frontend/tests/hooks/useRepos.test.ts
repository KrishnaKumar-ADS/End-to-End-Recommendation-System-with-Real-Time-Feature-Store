import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { fetchRepos } from "@/lib/api-client";
import { useRepos } from "@/hooks/useRepos";

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ...actual,
    fetchRepos: vi.fn(),
  };
});

describe("useRepos", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("loads repositories with limit and offset", async () => {
    vi.mocked(fetchRepos).mockResolvedValueOnce([
      {
        id: "repo-1",
        github_url: "https://github.com/tiangolo/fastapi",
        name: "fastapi",
        branch: "main",
        status: "completed",
        task_id: null,
        error_message: null,
        total_files: 10,
        processed_files: 10,
        total_chunks: 20,
        created_at: "",
        updated_at: "",
      },
    ]);

    const { result } = renderHook(() => useRepos(20, 40));

    await waitFor(() => {
      expect(fetchRepos).toHaveBeenCalledWith({ limit: 20, offset: 40 });
      expect(result.current.repos).toHaveLength(1);
      expect(result.current.error).toBeNull();
    });
  });

  it("returns empty list and error when API fails", async () => {
    vi.mocked(fetchRepos).mockRejectedValueOnce(new Error("Backend unavailable"));

    const { result } = renderHook(() => useRepos());

    await waitFor(() => {
      expect(result.current.repos).toEqual([]);
      expect(result.current.error).toBe("Backend unavailable");
      expect(result.current.isLoading).toBe(false);
    });
  });

  it("refresh fetches repositories again", async () => {
    vi.mocked(fetchRepos)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        {
          id: "repo-2",
          github_url: "https://github.com/pallets/flask",
          name: "flask",
          branch: "main",
          status: "completed",
          task_id: null,
          error_message: null,
          total_files: 12,
          processed_files: 12,
          total_chunks: 24,
          created_at: "",
          updated_at: "",
        },
      ]);

    const { result } = renderHook(() => useRepos());

    await waitFor(() => expect(fetchRepos).toHaveBeenCalledTimes(1));

    await act(async () => {
      await result.current.refresh();
    });

    expect(fetchRepos).toHaveBeenCalledTimes(2);
    expect(result.current.repos[0]?.name).toBe("flask");
  });
});
