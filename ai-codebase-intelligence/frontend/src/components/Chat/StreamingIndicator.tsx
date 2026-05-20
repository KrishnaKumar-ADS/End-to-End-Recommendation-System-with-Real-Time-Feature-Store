"use client";

import { cn } from "@/lib/utils";

interface StreamingIndicatorProps {
  modelName?: string;
  className?: string;
}

export function StreamingIndicator({
  modelName,
  className,
}: StreamingIndicatorProps) {
  return (
    <div
      aria-label="AI is generating a response"
      aria-live="polite"
      className={cn("flex items-center gap-3 text-xs text-surface-muted", className)}
    >
      <div aria-hidden="true" className="flex gap-1">
        {[0, 1, 2].map((index) => (
          <span
            key={index}
            className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand-500"
            style={{ animationDelay: `${index * 0.15}s` }}
          />
        ))}
      </div>

      {modelName ? (
        <span className="max-w-[260px] truncate font-mono text-xs text-violet-300" title={modelName}>
          {modelName}
        </span>
      ) : null}

      <span className="text-surface-muted">thinking...</span>
    </div>
  );
}
