"use client";

import { useCallback, useState } from "react";

import { ApiClientError, ingestRepository } from "@/lib/api-client";
import type { IngestRequest } from "@/types/api";

export type IngestPhase = "idle" | "submitting" | "submitted" | "error";

interface UseIngestState {
  phase: IngestPhase;
  isLoading: boolean;
  isConflict: boolean;
  error: string | null;
  repoId: string | null;
  taskId: string | null;
}

const INITIAL_STATE: UseIngestState = {
  phase: "idle",
  isLoading: false,
  isConflict: false,
  error: null,
  repoId: null,
  taskId: null,
};

export function useIngest() {
  const [state, setState] = useState<UseIngestState>(INITIAL_STATE);

  const submit = useCallback(async (payload: IngestRequest) => {
    setState((current) => ({
      ...current,
      phase: "submitting",
      isLoading: true,
      isConflict: false,
      error: null,
    }));

    try {
      const response = await ingestRepository(payload);
      setState({
        phase: "submitted",
        isLoading: false,
        isConflict: false,
        error: null,
        repoId: response.repo_id,
        taskId: response.task_id,
      });
      return response;
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to submit repository ingestion.";
      const isConflict = error instanceof ApiClientError && error.status === 409;

      setState({
        phase: "error",
        isLoading: false,
        isConflict,
        error: message,
        repoId: null,
        taskId: null,
      });

      return null;
    }
  }, []);

  const reset = useCallback(() => {
    setState(INITIAL_STATE);
  }, []);

  return {
    ...state,
    submit,
    reset,
  };
}
