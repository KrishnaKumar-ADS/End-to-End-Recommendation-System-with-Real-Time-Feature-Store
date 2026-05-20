"use client";

import { AnswerRenderer } from "@/components/Chat/AnswerRenderer";
import { QualityScorePanel } from "@/components/Chat/QualityScoreBar";
import { SourceCitationPanel } from "@/components/Chat/SourceCitationPanel";
import { StreamingIndicator } from "@/components/Chat/StreamingIndicator";
import { ProviderBadge } from "@/components/ui/Badge";
import { cn } from "@/lib/utils";
import type { ChatMessage as ChatMessageType } from "@/types/api";

interface ChatMessageProps {
  message: ChatMessageType;
  repoId: string;
}

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <div
        className={cn(
          "max-w-[80%] rounded-2xl rounded-tr-md px-4 py-2.5",
          "bg-brand-600 text-sm leading-relaxed text-white",
          "shadow-sm shadow-brand-900/50",
        )}
      >
        {content}
      </div>
    </div>
  );
}

function AssistantBubble({
  message,
  repoId,
}: {
  message: Extract<ChatMessageType, { role: "assistant" }>;
  repoId: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <div
        aria-hidden="true"
        className={cn(
          "flex h-7 w-7 shrink-0 items-center justify-center rounded-lg",
          "border border-violet-500/30 bg-violet-600/20 text-xs font-bold text-violet-300",
        )}
      >
        AI
      </div>

      <div className="min-w-0 flex-1 space-y-3">
        {message.isStreaming && !message.content ? (
          <StreamingIndicator
            modelName={message.modelUsed ?? "qwen/qwen-2.5-coder-32b-instruct"}
          />
        ) : (
          <AnswerRenderer content={message.content} isStreaming={message.isStreaming} />
        )}

        {!message.isStreaming && message.content ? (
          <div className="flex flex-wrap items-center gap-3 text-xs text-surface-muted">
            {message.modelUsed ? (
              <ProviderBadge model={message.modelUsed} provider={message.providerUsed ?? undefined} />
            ) : null}
            {message.cached ? <span className="font-mono text-amber-300">cached</span> : null}
            {message.totalMs ? (
              <span className="font-mono tabular-nums">{(message.totalMs / 1000).toFixed(1)}s</span>
            ) : null}
          </div>
        ) : null}

        {!message.isStreaming && message.sources.length ? (
          <SourceCitationPanel repoId={repoId} sources={message.sources} />
        ) : null}

        {!message.isStreaming && message.qualityScore ? (
          <QualityScorePanel className="border-t border-surface-border pt-1" qualityScore={message.qualityScore} />
        ) : null}

        {!message.isStreaming && message.graphPath.length ? (
          <div className="flex flex-wrap items-center gap-1 font-mono text-xs text-surface-muted">
            <span className="shrink-0">Call path:</span>
            {message.graphPath.map((node, index) => (
              <span className="flex items-center gap-1" key={`${node}-${index}`}>
                {index > 0 ? <span className="text-[10px] text-surface-muted">→</span> : null}
                <span className="text-brand-200">{node}</span>
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function ChatMessage({ message, repoId }: ChatMessageProps) {
  if (message.role === "user") {
    return <UserBubble content={message.content} />;
  }

  return <AssistantBubble message={message} repoId={repoId} />;
}
