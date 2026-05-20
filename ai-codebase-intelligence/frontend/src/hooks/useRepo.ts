"use client";

import { useCallback, useEffect, useState } from "react";

import { fetchRepo } from "@/lib/api-client";
import type { RepositoryDetail } from "@/types/api";

export function useRepo(repoId: string | null) {
  const [repo, setRepo] = useState<RepositoryDetail | null>(null);
  const [isLoading, setIsLoading] = useState(Boolean(repoId));
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!repoId) {
      setRepo(null);
      setIsLoading(false);
      return null;
    }

    setIsLoading(true);
    try {
      const nextRepo = await fetchRepo(repoId);
      setRepo(nextRepo);
      setError(null);
      return nextRepo;
    } catch (err) {
      setRepo(null);
      setError(err instanceof Error ? err.message : "Failed to load repository.");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [repoId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return {
    repo,
    isLoading,
    error,
    refresh,
  };
}
