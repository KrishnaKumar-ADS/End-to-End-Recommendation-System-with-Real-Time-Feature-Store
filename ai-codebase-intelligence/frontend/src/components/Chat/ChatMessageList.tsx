"use client";

import { VirtualChatList } from "@/components/Chat/VirtualChatList";
import type { ChatMessage as ChatMessageType } from "@/types/api";

interface ChatMessageListProps {
  messages: ChatMessageType[];
  repoId: string;
  isStreaming: boolean;
}

export function ChatMessageList({
  messages,
  repoId,
  isStreaming,
}: ChatMessageListProps) {
  if (!messages.length && !isStreaming) {
    return null;
  }

  return <VirtualChatList isStreaming={isStreaming} messages={messages} repoId={repoId} />;
}
