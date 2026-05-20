import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { fetchStatus } from "@/lib/api-client";
import { useStatus } from "@/hooks/useStatus";

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ...actual,
    fetchStatus: vi.fn(),
  };
});

describe("useStatus", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  it("stays idle without a task id", () => {
    const { result } = renderHook(() => useStatus(null));
    expect(result.current.data).toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  it("polls until the task reaches a terminal state", async () => {
    vi.mocked(fetchStatus)
      .mockResolvedValueOnce({
        task_id: "t1",
        status: "queued",
        progress: 10,
        message: "Queued",
        repo_id: null,
        error: null,
        total_files: 0,
        processed_files: 0,
        total_chunks: 0,
      })
      .mockResolvedValueOnce({
        task_id: "t1",
        status: "completed",
        progress: 100,
        message: "Done",
        repo_id: "r1",
        error: null,
        total_files: 1,
        processed_files: 1,
        total_chunks: 2,
      });

    const { result } = renderHook(() => useStatus("t1"));

    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.data?.status).toBe("queued");

    await act(async () => {
      vi.advanceTimersByTime(2000);
      await Promise.resolve();
    });

    expect(result.current.data?.status).toBe("completed");
    expect(fetchStatus).toHaveBeenCalledTimes(2);
  });

  it("increments elapsed seconds while polling", async () => {
    vi.mocked(fetchStatus).mockResolvedValue({
      task_id: "t1",
      status: "queued",
      progress: 10,
      message: "Queued",
      repo_id: null,
      error: null,
      total_files: 0,
      processed_files: 0,
      total_chunks: 0,
    });

    const { result } = renderHook(() => useStatus("t1"));

    await act(async () => {
      vi.advanceTimersByTime(3000);
      await Promise.resolve();
    });

    expect(result.current.elapsedSec).toBeGreaterThanOrEqual(3);
  });
});
