"use client";

import { useCallback, useEffect, useState } from "react";

import { fetchMetrics } from "@/lib/api-client";
import type { MetricsResponse } from "@/types/api";

export function useMetrics() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await fetchMetrics();
      setMetrics(response);
      setError(null);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load metrics.");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const intervalId = setInterval(() => {
      void refresh();
    }, 30000);

    return () => {
      clearInterval(intervalId);
    };
  }, [refresh]);

  return {
    metrics,
    isLoading,
    error,
    refresh,
  };
}
