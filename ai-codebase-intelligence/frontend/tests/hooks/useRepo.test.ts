import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { fetchRepo } from "@/lib/api-client";
import { useRepo } from "@/hooks/useRepo";

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ...actual,
    fetchRepo: vi.fn(),
  };
});

describe("useRepo", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("stays idle when repoId is null", () => {
    const { result } = renderHook(() => useRepo(null));

    expect(result.current.repo).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(fetchRepo).not.toHaveBeenCalled();
  });

  it("loads repository detail for a repo id", async () => {
    vi.mocked(fetchRepo).mockResolvedValueOnce({
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
      files: [],
    });

    const { result } = renderHook(() => useRepo("repo-1"));

    await waitFor(() => {
      expect(fetchRepo).toHaveBeenCalledWith("repo-1");
      expect(result.current.repo?.id).toBe("repo-1");
      expect(result.current.error).toBeNull();
    });
  });

  it("captures loading errors and clears repo data", async () => {
    vi.mocked(fetchRepo).mockRejectedValueOnce(new Error("Repository not found"));

    const { result } = renderHook(() => useRepo("missing"));

    await waitFor(() => {
      expect(result.current.repo).toBeNull();
      expect(result.current.error).toBe("Repository not found");
      expect(result.current.isLoading).toBe(false);
    });
  });

  it("refresh re-fetches current repository", async () => {
    vi.mocked(fetchRepo)
      .mockResolvedValueOnce({
        id: "repo-1",
        github_url: "",
        name: "fastapi",
        branch: "main",
        status: "queued",
        task_id: "t1",
        error_message: null,
        total_files: 0,
        processed_files: 0,
        total_chunks: 0,
        created_at: "",
        updated_at: "",
        files: [],
      })
      .mockResolvedValueOnce({
        id: "repo-1",
        github_url: "",
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
        files: [],
      });

    const { result } = renderHook(() => useRepo("repo-1"));

    await waitFor(() => expect(fetchRepo).toHaveBeenCalledTimes(1));

    await act(async () => {
      await result.current.refresh();
    });

    expect(fetchRepo).toHaveBeenCalledTimes(2);
    expect(result.current.repo?.status).toBe("completed");
  });
});
