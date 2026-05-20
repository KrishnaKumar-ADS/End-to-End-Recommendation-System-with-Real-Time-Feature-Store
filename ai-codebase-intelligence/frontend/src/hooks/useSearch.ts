"use client";

import { useEffect, useMemo, useState } from "react";

import { searchCode } from "@/lib/api-client";
import type { SearchResponse } from "@/types/api";

export function useSearch(repoId: string | null, initialQuery = "", topK = 8) {
  const [query, setQuery] = useState(initialQuery);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const normalized = query.trim();
    if (!repoId || !normalized) {
      setData(null);
      setIsLoading(false);
      setError(null);
      return;
    }

    setIsLoading(true);
    const timer = setTimeout(() => {
      void searchCode({
        repo_id: repoId,
        q: normalized,
        top_k: topK,
      })
        .then((response) => {
          setData(response);
          setError(null);
        })
        .catch((err) => {
          setData(null);
          setError(err instanceof Error ? err.message : "Search failed.");
        })
        .finally(() => {
          setIsLoading(false);
        });
    }, 300);

    return () => {
      clearTimeout(timer);
    };
  }, [query, repoId, topK]);

  return {
    query,
    setQuery,
    data,
    results: useMemo(() => data?.results ?? [], [data]),
    timing: data?.timing ?? null,
    isLoading,
    error,
  };
}
