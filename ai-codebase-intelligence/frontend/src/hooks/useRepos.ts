"use client";

import { useCallback, useEffect, useState } from "react";

import { fetchRepos } from "@/lib/api-client";
import type { RepositorySummary } from "@/types/api";

export function useRepos(limit = 100, offset = 0) {
  const [repos, setRepos] = useState<RepositorySummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const nextRepos = await fetchRepos({ limit, offset });
      setRepos(nextRepos);
      setError(null);
      return nextRepos;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load repositories.");
      setRepos([]);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [limit, offset]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return {
    repos,
    isLoading,
    error,
    refresh,
  };
}
