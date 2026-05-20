"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useAsk } from "@/hooks/useAsk";
import { clearSession, getSession, saveSession } from "@/lib/session";
import type {
  AssistantMessage,
  ChatMessage,
  UserMessage,
} from "@/types/api";

function generateId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export interface UseChatReturn {
  messages: ChatMessage[];
  isStreaming: boolean;
  sessionId: string | null;
  error: string | null;
  stepMessage: string | null;
  ask: (question: string) => Promise<void>;
  cancelStream: () => void;
  newSession: () => void;
  clearError: () => void;
}

export function useChat(repoId: string): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const streamingMessageIdRef = useRef<string | null>(null);

  const askState = useAsk();

  useEffect(() => {
    if (!repoId) {
      setSessionId(null);
      return;
    }

    const storedSession = getSession(repoId);
    setSessionId(storedSession);
  }, [repoId]);

  useEffect(() => {
    const messageId = streamingMessageIdRef.current;
    if (!messageId) {
      return;
    }

    setMessages((previous) =>
      previous.map((message): ChatMessage => {
        if (message.id !== messageId || message.role !== "assistant") {
          return message;
        }

        return {
          ...message,
          content: askState.streamingText,
          sources: askState.sources,
          qualityScore: askState.qualityScore,
          graphPath: askState.graphPath,
          modelUsed: askState.modelUsed,
          providerUsed: askState.providerUsed,
          cached: askState.cached,
          totalMs: askState.totalMs,
          isStreaming: askState.isStreaming,
        };
      }),
    );

    if (!askState.isStreaming) {
      streamingMessageIdRef.current = null;
    }

    if (askState.sessionId && askState.sessionId !== sessionId) {
      setSessionId(askState.sessionId);
      saveSession(repoId, askState.sessionId);
    }

    if (askState.error) {
      setError(askState.error);
    }
  }, [
    askState.cached,
    askState.error,
    askState.graphPath,
    askState.isStreaming,
    askState.modelUsed,
    askState.providerUsed,
    askState.qualityScore,
    askState.sessionId,
    askState.sources,
    askState.streamingText,
    askState.totalMs,
    repoId,
    sessionId,
  ]);

  const ask = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || !repoId) {
        return;
      }

      setError(null);

      const userMessage: UserMessage = {
        id: generateId(),
        role: "user",
        content: trimmed,
        timestamp: Date.now(),
      };

      const assistantMessageId = generateId();
      const assistantMessage: AssistantMessage = {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        sources: [],
        qualityScore: null,
        graphPath: [],
        providerUsed: null,
        modelUsed: null,
        cached: false,
        totalMs: null,
        isStreaming: true,
        timestamp: Date.now(),
      };

      streamingMessageIdRef.current = assistantMessageId;

      setMessages((previous) => [...previous, userMessage, assistantMessage]);

      await askState.ask({
        repo_id: repoId,
        question: trimmed,
        include_graph: true,
        session_id: sessionId ?? undefined,
      });
    },
    [askState, repoId, sessionId],
  );

  const cancelStream = useCallback(() => {
    askState.cancel();
  }, [askState]);

  const newSession = useCallback(() => {
    askState.reset();
    clearSession(repoId);
    setSessionId(null);
    setMessages([]);
    setError(null);
    streamingMessageIdRef.current = null;
  }, [askState, repoId]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    messages,
    isStreaming: askState.isStreaming,
    sessionId,
    error,
    stepMessage: askState.stepMessage,
    ask,
    cancelStream,
    newSession,
    clearError,
  };
}
