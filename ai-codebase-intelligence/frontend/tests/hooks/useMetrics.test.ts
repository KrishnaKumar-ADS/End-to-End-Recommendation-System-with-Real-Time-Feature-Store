import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fetchMetrics } from "@/lib/api-client";
import { useMetrics } from "@/hooks/useMetrics";

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ...actual,
    fetchMetrics: vi.fn(),
  };
});

function metricsFixture() {
  return {
    token_usage: {
      openrouter: {
        input_tokens: 10,
        output_tokens: 20,
        total_tokens: 30,
        call_count: 2,
        cost_today_usd: 0.01,
      },
    },
    budget: {
      daily_limit_usd: 10,
      used_today_usd: 1,
      remaining_usd: 9,
      used_pct: 10,
      over_budget: false,
    },
    cache: {},
    circuit_breakers: {},
    eval_scores: {},
  };
}

describe("useMetrics", () => {
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

  it("loads metrics on mount", async () => {
    vi.mocked(fetchMetrics).mockResolvedValueOnce(metricsFixture());

    const { result } = renderHook(() => useMetrics());

    await flushMicrotasks();

    expect(fetchMetrics).toHaveBeenCalledTimes(1);
    expect(result.current.metrics?.budget.remaining_usd).toBe(9);
    expect(result.current.error).toBeNull();
  });

  it("refreshes metrics every 30 seconds", async () => {
    vi.mocked(fetchMetrics)
      .mockResolvedValueOnce(metricsFixture())
      .mockResolvedValueOnce({
        ...metricsFixture(),
        budget: {
          daily_limit_usd: 10,
          used_today_usd: 2,
          remaining_usd: 8,
          used_pct: 20,
          over_budget: false,
        },
      });

    const { result } = renderHook(() => useMetrics());

    await flushMicrotasks();
    expect(fetchMetrics).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(30000);
      await Promise.resolve();
    });

    await flushMicrotasks();

    expect(fetchMetrics).toHaveBeenCalledTimes(2);
    expect(result.current.metrics?.budget.remaining_usd).toBe(8);
  });

  it("captures API errors", async () => {
    vi.mocked(fetchMetrics).mockRejectedValueOnce(new Error("Redis unavailable"));

    const { result } = renderHook(() => useMetrics());

    await flushMicrotasks();

    expect(result.current.error).toBe("Redis unavailable");
    expect(result.current.isLoading).toBe(false);
  });
});
