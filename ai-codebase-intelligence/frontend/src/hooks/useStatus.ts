"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { fetchStatus } from "@/lib/api-client";
import type { StatusResponse } from "@/types/api";

const TERMINAL_STATUSES = new Set(["completed", "failed"]);

export function useStatus(taskId: string | null) {
  const [data, setData] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);
  const startedAtRef = useRef<number | null>(null);

  const poll = useCallback(async () => {
    if (!taskId) {
      return null;
    }

    const next = await fetchStatus(taskId);
    setData(next);
    setError(null);
    return next;
  }, [taskId]);

  useEffect(() => {
    if (!taskId) {
      setData(null);
      setError(null);
      setElapsedSec(0);
      setIsLoading(false);
      startedAtRef.current = null;
      return;
    }

    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    setIsLoading(true);
    setElapsedSec(0);
    startedAtRef.current = Date.now();

    const run = async () => {
      try {
        const next = await poll();
        if (cancelled || !next) {
          return;
        }
        setIsLoading(false);

        if (!TERMINAL_STATUSES.has(next.status)) {
          timeoutId = setTimeout(run, 2000);
        }
      } catch (err) {
        if (cancelled) {
          return;
        }
        setIsLoading(false);
        setError(err instanceof Error ? err.message : "Failed to fetch task status.");
      }
    };

    void run();

    const elapsedInterval = setInterval(() => {
      if (startedAtRef.current) {
        setElapsedSec(Math.floor((Date.now() - startedAtRef.current) / 1000));
      }
    }, 1000);

    return () => {
      cancelled = true;
      clearInterval(elapsedInterval);
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [poll, taskId]);

  const refetch = useCallback(async () => {
    if (!taskId) {
      return null;
    }

    setIsLoading(true);
    try {
      const next = await poll();
      setIsLoading(false);
      return next;
    } catch (err) {
      setIsLoading(false);
      setError(err instanceof Error ? err.message : "Failed to fetch task status.");
      return null;
    }
  }, [poll, taskId]);

  const isTerminal = useMemo(
    () => Boolean(data && TERMINAL_STATUSES.has(data.status)),
    [data],
  );

  return {
    data,
    error,
    isLoading,
    elapsedSec,
    isTerminal,
    refetch,
  };
}
