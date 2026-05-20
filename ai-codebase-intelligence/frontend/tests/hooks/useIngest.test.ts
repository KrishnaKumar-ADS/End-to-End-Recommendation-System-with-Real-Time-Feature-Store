import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiClientError, ingestRepository } from "@/lib/api-client";
import { useIngest } from "@/hooks/useIngest";

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ...actual,
    ingestRepository: vi.fn(),
  };
});

describe("useIngest", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("starts in idle phase", () => {
    const { result } = renderHook(() => useIngest());
    expect(result.current.phase).toBe("idle");
    expect(result.current.isLoading).toBe(false);
    expect(result.current.taskId).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("transitions to submitted on success", async () => {
    vi.mocked(ingestRepository).mockResolvedValueOnce({
      repo_id: "r1",
      task_id: "t1",
      status: "queued",
      message: "queued",
    });

    const { result } = renderHook(() => useIngest());

    await act(async () => {
      await result.current.submit({ github_url: "https://github.com/a/b" });
    });

    expect(result.current.phase).toBe("submitted");
    expect(result.current.repoId).toBe("r1");
    expect(result.current.taskId).toBe("t1");
  });

  it("transitions to error when the API call fails", async () => {
    vi.mocked(ingestRepository).mockRejectedValueOnce(new Error("Network error: could not reach the backend."));

    const { result } = renderHook(() => useIngest());

    await act(async () => {
      await result.current.submit({ github_url: "https://github.com/a/b" });
    });

    expect(result.current.phase).toBe("error");
    expect(result.current.error).toBe("Network error: could not reach the backend.");
  });

  it("sets isConflict=true on 409 errors", async () => {
    vi.mocked(ingestRepository).mockRejectedValueOnce(
      new ApiClientError("Repository already ingested.", 409),
    );

    const { result } = renderHook(() => useIngest());

    await act(async () => {
      await result.current.submit({ github_url: "https://github.com/a/b" });
    });

    expect(result.current.isConflict).toBe(true);
  });

  it("reset returns the hook to the idle phase", async () => {
    vi.mocked(ingestRepository).mockRejectedValueOnce(new Error("Failed."));
    const { result } = renderHook(() => useIngest());

    await act(async () => {
      await result.current.submit({ github_url: "https://github.com/a/b" });
    });
    expect(result.current.phase).toBe("error");

    act(() => {
      result.current.reset();
    });

    expect(result.current.phase).toBe("idle");
    expect(result.current.error).toBeNull();
  });
});
