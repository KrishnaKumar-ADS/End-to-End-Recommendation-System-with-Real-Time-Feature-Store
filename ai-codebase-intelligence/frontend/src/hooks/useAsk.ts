"use client";

import { useCallback, useRef, useState } from "react";

import { streamAsk } from "@/lib/api-client";
import { readSSEStream } from "@/lib/streaming";
import type {
  AskRequest,
  QualityScore,
  SourceCitation,
  ChatEvent,
} from "@/types/api";

interface AskState {
  streamingText: string;
  sources: SourceCitation[];
  qualityScore: QualityScore | null;
  graphPath: string[];
  sessionId: string | null;
  modelUsed: string | null;
  providerUsed: string | null;
  cached: boolean;
  totalMs: number | null;
  isStreaming: boolean;
  stepMessage: string | null;
  error: string | null;
}

const INITIAL_STATE: AskState = {
  streamingText: "",
  sources: [],
  qualityScore: null,
  graphPath: [],
  sessionId: null,
  modelUsed: null,
  providerUsed: null,
  cached: false,
  totalMs: null,
  isStreaming: false,
  stepMessage: null,
  error: null,
};

function normalizeSource(source: SourceCitation): SourceCitation {
  return {
    file: source.file,
    function: source.function,
    lines: source.lines,
    snippet: source.snippet,
  };
}

export interface UseAskReturn extends AskState {
  ask: (request: AskRequest & { session_id?: string | null }) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

export function useAsk(): UseAskReturn {
  const [state, setState] = useState<AskState>(INITIAL_STATE);
  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setState((previous) => ({
      ...previous,
      isStreaming: false,
    }));
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState(INITIAL_STATE);
  }, []);

  const handleEvent = useCallback((event: ChatEvent) => {
    switch (event.type) {
      case "step":
        setState((previous) => ({
          ...previous,
          stepMessage: event.message,
        }));
        break;
      case "token":
        setState((previous) => ({
          ...previous,
          streamingText: `${previous.streamingText}${event.content}`,
        }));
        break;
      case "sources":
        setState((previous) => ({
          ...previous,
          sources: event.sources.map(normalizeSource),
        }));
        break;
      case "done":
        setState((previous) => ({
          ...previous,
          isStreaming: false,
          sessionId: event.session_id,
          qualityScore: event.quality_score,
          graphPath: event.graph_path,
          modelUsed: event.model_used ?? event.model ?? previous.modelUsed,
          providerUsed: event.provider_used ?? event.provider ?? previous.providerUsed,
          cached: event.cached ?? false,
          totalMs: event.total_ms ?? event.timing?.total_ms ?? null,
        }));
        break;
      case "error":
        setState((previous) => ({
          ...previous,
          isStreaming: false,
          error: event.message,
        }));
        break;
      default:
        break;
    }
  }, []);

  const ask = useCallback(
    async (request: AskRequest & { session_id?: string | null }) => {
      abortRef.current?.abort();
      abortRef.current = new AbortController();
      const signal = abortRef.current.signal;

      setState({
        ...INITIAL_STATE,
        isStreaming: true,
      });

      try {
        const response = await streamAsk(request, signal);
        if (signal.aborted) {
          return;
        }

        const body = response.body;
        if (!body) {
          throw new Error("Streaming response body is unavailable.");
        }

        for await (const event of readSSEStream(body, signal)) {
          if (signal.aborted) {
            break;
          }
          handleEvent(event);
        }

        setState((previous) => ({
          ...previous,
          isStreaming: false,
        }));
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          setState((previous) => ({
            ...previous,
            isStreaming: false,
          }));
          return;
        }

        setState((previous) => ({
          ...previous,
          isStreaming: false,
          error: error instanceof Error ? error.message : "An unexpected error occurred.",
        }));
      }
    },
    [handleEvent],
  );

  return {
    ...state,
    ask,
    cancel,
    reset,
  };
}
