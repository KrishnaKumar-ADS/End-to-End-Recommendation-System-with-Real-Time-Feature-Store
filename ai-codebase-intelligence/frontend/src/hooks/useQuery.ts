"use client";

import { useCallback, useState } from "react";

import { askQuestion } from "@/lib/api-client";
import type { AskRequest, AskResponse } from "@/types/api";

export function useQuery() {
  const [data, setData] = useState<AskResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = useCallback(async (payload: AskRequest) => {
    setIsLoading(true);
    try {
      const response = await askQuestion(payload);
      setData(response);
      setError(null);
      return response;
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "Query failed.");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return {
    data,
    isLoading,
    error,
    submit,
    reset,
  };
}
